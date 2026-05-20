"""Fallback operator application and layer output patching."""

from __future__ import annotations

import weakref
from typing import Any

import torch

from ..activation_patch_custom_op import build_activation_patch_hidden_states_op
from ..activation_patch_math import (
    ADD_DIRECTION_OPERATOR,
    INTERCHANGE_MODE_ID,
    INTERCHANGE_OPERATOR,
    PROJECT_OUT_MODE_ID,
    PROJECT_OUT_OPERATOR,
    RANDOM_CONTROL_OPERATOR,
    RESIDUAL_PATH_MODE_ID,
    RESIDUAL_PATH_OPERATOR,
    SWAP_COMPONENTS_OPERATOR,
    SWAP_MEAN_OPERATOR,
    apply_residual_path_transport,
    apply_subspace_operator,
)
from .base import (
    ActivationPatchedLayer,
    contiguous_token_span,
    debug_log,
    debug_mode_enabled,
    debug_panic,
    extract_hidden_tensor,
    replace_hidden_tensor,
    spec_from_payload,
)
from .custom_ops import (
    noop_custom_op,
    register_torch_library_interchange_batch_op,
    register_torch_library_residual_path_batch_op,
    register_torch_library_subspace_batch_op,
    register_torch_library_subspace_op,
    register_torch_library_project_out_batch_op,
    register_torch_library_project_out_op,
    run_custom_op,
)
from .state import _record_patch_stats
from .subspace_family import (
    SUBSPACE_OPERATORS,
    is_subspace_operator,
    resolve_subspace_inputs,
)


def _first_invalid_active_query_position(
    *,
    query_positions: torch.Tensor,
    token_counts: torch.Tensor,
    active: torch.Tensor,
    hidden_token_count: int,
) -> dict[str, Any] | None:
    slot_count = int(active.shape[0])
    max_tokens = int(query_positions.shape[1]) if query_positions.ndim >= 2 else 0
    for slot_idx in range(slot_count):
        if int(active[slot_idx].item()) <= 0:
            continue
        token_count = int(token_counts[slot_idx].item())
        if token_count <= 0:
            continue
        if token_count > max_tokens:
            return {
                "slot_idx": int(slot_idx),
                "token_count": int(token_count),
                "max_tokens": int(max_tokens),
                "reason": "token_count_exceeds_buffer",
            }
        positions = query_positions[slot_idx, :token_count].detach().cpu().tolist()
        invalid = [int(pos) for pos in positions if int(pos) < 0 or int(pos) >= int(hidden_token_count)]
        if invalid:
            return {
                "slot_idx": int(slot_idx),
                "token_count": int(token_count),
                "hidden_token_count": int(hidden_token_count),
                "invalid_positions": invalid[:16],
                "all_positions": [int(pos) for pos in positions[:64]],
                "reason": "query_positions_out_of_bounds",
            }
    return None


def patch_hidden_states_for_layer(
    hidden_states: Any,
    *,
    owner_model: Any,
    layer_idx: int,
    batch_spec: dict[str, Any],
) -> tuple[Any, dict[str, Any] | None]:
    spec = spec_from_payload(dict(batch_spec["patch_spec"]))
    if spec.query_span:
        query_positions = list(range(int(spec.query_span[0]), int(spec.query_span[1])))
    else:
        query_positions = list(spec.query_positions)
    if not query_positions:
        return hidden_states, None

    hidden = hidden_states
    original_dtype = hidden.dtype
    is_1d = hidden.ndim == 1
    if is_1d:
        hidden = hidden.unsqueeze(0)
    if hidden.ndim != 2:
        return hidden_states, {
            "layer": int(layer_idx),
            "status": "skipped",
            "reason": f"unsupported_hidden_ndim:{hidden.ndim}",
        }

    if max(query_positions) >= int(hidden.shape[0]):
        return hidden_states, {
            "layer": int(layer_idx),
            "status": "skipped",
            "reason": f"query_positions_out_of_bounds:{int(hidden.shape[0])}",
            "operator": spec.operator,
        }

    if spec.is_interchange():
        donor_positions = list(spec.donor_positions)
        bank = getattr(owner_model, "_v2_activation_patch_bank", {})
        layer_bank = bank.get(int(layer_idx))
        if not isinstance(layer_bank, dict):
            return hidden_states, {
                "layer": int(layer_idx),
                "status": "skipped",
                "reason": "missing_layer_bank",
                "operator": spec.operator,
            }
        donor_entry = layer_bank.get(spec.donor_example_key)
        if not isinstance(donor_entry, dict):
            return hidden_states, {
                "layer": int(layer_idx),
                "status": "skipped",
                "reason": "missing_donor_example",
                "operator": spec.operator,
            }
        donor_values = donor_entry["values"]
        if max(donor_positions) >= int(donor_values.shape[0]):
            return hidden_states, {
                "layer": int(layer_idx),
                "status": "skipped",
                "reason": f"donor_positions_out_of_bounds:{int(donor_values.shape[0])}",
                "operator": spec.operator,
            }

        patched = hidden.clone()
        patched_rows = donor_values[donor_positions].to(dtype=hidden.dtype)
        before = hidden[query_positions].to(torch.float32)
        after = patched_rows.to(torch.float32)
        patched[query_positions] = patched_rows
        stats = {
            "layer": int(layer_idx),
            "status": "ok",
            "operator": spec.operator,
            "query_positions": list(query_positions),
            "donor_positions": list(donor_positions),
            "token_count": len(query_positions),
            "delta_norm_raw": float(torch.linalg.norm(after - before).item()),
            "control_name": spec.control_name,
            "case_key": spec.case_key,
        }
    elif is_subspace_operator(spec.operator):
        subspace_inputs, skip_stats = resolve_subspace_inputs(
            owner_model=owner_model,
            spec=spec,
            layer_idx=int(layer_idx),
            hidden_dim=int(hidden.shape[-1]),
            device=hidden.device,
        )
        if subspace_inputs is None:
            return hidden_states, skip_stats

        patched, subspace_stats = apply_subspace_operator(
            hidden.to(torch.float32),
            query_positions=query_positions,
            operator=spec.operator,
            mean=subspace_inputs["mean"],
            scale=subspace_inputs["scale"],
            safe_scale=subspace_inputs["safe_scale"],
            selected_rows=subspace_inputs["selected_rows"],
            strength=float(spec.strength),
            direction_raw=subspace_inputs["direction_raw"],
            direction_std=subspace_inputs["direction_std"],
            donor_mean=subspace_inputs["donor_mean"],
            random_rows=subspace_inputs["random_rows"],
            match_projected_norm=bool(spec.match_projected_norm),
            rowwise=bool(spec.rowwise),
        )
        patched = patched.to(dtype=hidden.dtype)
        stats = {
            "layer": int(layer_idx),
            "source_layer": int(subspace_inputs["source_layer"]),
            **subspace_stats,
        }
        if spec.query_span:
            stats["query_span"] = [int(spec.query_span[0]), int(spec.query_span[1])]
        else:
            stats["query_positions"] = list(query_positions)
        if spec.covered_abs_spans:
            stats["covered_abs_spans"] = [
                [int(start), int(end)] for start, end in spec.covered_abs_spans
            ]
            stats["covered_abs_tokens"] = sum(
                max(0, int(end) - int(start)) for start, end in spec.covered_abs_spans
            )
        if spec.phase_counts:
            stats["phase_counts"] = {str(name): int(count) for name, count in spec.phase_counts}
        if spec.target_policy:
            stats["target_policy"] = dict(spec.target_policy)
    elif spec.is_residual_path():
        bank = getattr(owner_model, "_v2_activation_patch_bank", {})
        path_stats: dict[str, Any] = {
            "layer": int(layer_idx),
            "operator": spec.operator,
            "query_positions": list(query_positions),
            "control_name": spec.control_name,
            "case_key": spec.case_key,
            "transport": spec.transport,
            "path_edges": [
                {
                    "source_layer": int(source_layer),
                    "write_layer": int(write_layer),
                    "weight": float(weight),
                }
                for source_layer, write_layer, weight in spec.path_edges
                if int(write_layer) == int(layer_idx)
            ],
        }
        layer_edges = [
            (int(source_layer), float(weight))
            for source_layer, write_layer, weight in spec.path_edges
            if int(write_layer) == int(layer_idx)
        ]
        if not layer_edges:
            return hidden_states, {
                "layer": int(layer_idx),
                "status": "skipped",
                "reason": "missing_path_edges_for_write_layer",
                "operator": spec.operator,
            }
        patched = hidden.clone()
        accumulated = torch.zeros_like(hidden[query_positions].to(torch.float32))
        for source_layer, weight in layer_edges:
            layer_bank = bank.get(int(source_layer))
            if not isinstance(layer_bank, dict):
                return hidden_states, {
                    "layer": int(layer_idx),
                    "status": "skipped",
                    "reason": f"missing_layer_bank:{int(source_layer)}",
                    "operator": spec.operator,
                }
            donor_entry = layer_bank.get(spec.donor_example_key)
            if not isinstance(donor_entry, dict):
                return hidden_states, {
                    "layer": int(layer_idx),
                    "status": "skipped",
                    "reason": "missing_donor_example",
                    "operator": spec.operator,
                }
            donor_values = donor_entry["values"]
            if max(spec.donor_positions or (0,)) >= int(donor_values.shape[0]):
                return hidden_states, {
                    "layer": int(layer_idx),
                    "status": "skipped",
                    "reason": f"donor_positions_out_of_bounds:{int(donor_values.shape[0])}",
                    "operator": spec.operator,
                }
            donor_rows = donor_values[list(spec.donor_positions)].to(torch.float32)
            target_source_rows = None
            if spec.transport == "delta":
                target_entry = layer_bank.get(spec.example_key)
                if not isinstance(target_entry, dict):
                    return hidden_states, {
                        "layer": int(layer_idx),
                        "status": "skipped",
                        "reason": "missing_target_example",
                        "operator": spec.operator,
                    }
                target_values = target_entry["values"]
                if max(spec.target_read_positions or (0,)) >= int(target_values.shape[0]):
                    return hidden_states, {
                        "layer": int(layer_idx),
                        "status": "skipped",
                        "reason": f"target_read_positions_out_of_bounds:{int(target_values.shape[0])}",
                        "operator": spec.operator,
                    }
                target_source_rows = target_values[list(spec.target_read_positions)].to(torch.float32)
            _, edge_stats = apply_residual_path_transport(
                hidden[list(query_positions)].to(torch.float32),
                query_positions=list(range(len(query_positions))),
                donor_rows=donor_rows,
                target_source_rows=target_source_rows,
                weight=float(weight),
                strength=float(spec.strength),
                transport=spec.transport,
            )
            if spec.transport == "replace":
                accumulated = accumulated + (float(weight) * float(spec.strength) * (donor_rows - hidden[list(query_positions)].to(torch.float32)))
            else:
                assert target_source_rows is not None
                accumulated = accumulated + (float(weight) * float(spec.strength) * (donor_rows - target_source_rows))
            path_stats.setdefault("edges", []).append(
                {
                    "source_layer": int(source_layer),
                    "write_layer": int(layer_idx),
                    "weight": float(weight),
                    "delta_norm_raw": float(edge_stats["delta_norm_raw"]),
                }
            )
        patched_section = hidden[list(query_positions)].to(torch.float32) + accumulated
        patched[list(query_positions)] = patched_section.to(dtype=hidden.dtype)
        path_stats.update(
            {
                "status": "ok",
                "token_count": int(len(query_positions)),
                "delta_norm_raw": float(torch.linalg.norm(accumulated).item()),
            }
        )
        stats = path_stats
    else:
        return hidden_states, {
            "layer": int(layer_idx),
            "status": "skipped",
            "reason": f"unsupported_operator:{spec.operator}",
            "operator": spec.operator,
        }
    if is_1d:
        patched = patched.squeeze(0)
    return patched.to(original_dtype), stats


def apply_layer_output_patching(
    *,
    owner_model: Any,
    layer_idx: int,
    custom_op: Any | None = None,
    output: Any,
) -> Any:
    hidden = extract_hidden_tensor(output)
    if hidden is None:
        return output
    force_custom_op_presence = bool(getattr(owner_model, "_v2_activation_patch_force_custom_op_presence", False))
    compiled_operator_hint = str(getattr(owner_model, "_v2_activation_patch_compiled_operator_hint", "") or "").strip()

    batch_specs = getattr(owner_model, "_v2_activation_patch_batch_specs", None)
    layer_operators = set()
    layer_project_out_contiguous = True
    if isinstance(batch_specs, list):
        for batch_spec in batch_specs:
            payload = batch_spec.get("patch_spec")
            if not isinstance(payload, dict):
                continue
            spec = spec_from_payload(dict(payload))
            if int(layer_idx) in spec.target_layers:
                layer_operators.add(spec.operator)
                if is_subspace_operator(spec.operator) and (
                    spec.rowwise or contiguous_token_span(spec.query_positions) is None
                ):
                    layer_project_out_contiguous = False

    if (
        force_custom_op_presence
        and custom_op is not None
        and compiled_operator_hint == "subspace"
        and layer_operators
        and layer_operators <= SUBSPACE_OPERATORS
    ):
        runtime_state = getattr(owner_model, "_v2_activation_patch_batch_runtime_state", None)
        layer_state = runtime_state.get(int(layer_idx)) if isinstance(runtime_state, dict) else None
        stats_state = getattr(owner_model, "_v2_activation_patch_batch_tensor_stats", {}).get(int(layer_idx))
        subspace = getattr(owner_model, "_v2_activation_patch_subspace", {})
        source_layer = int(layer_idx)
        if isinstance(batch_specs, list):
            for batch_spec in batch_specs:
                payload = batch_spec.get("patch_spec")
                if not isinstance(payload, dict):
                    continue
                spec = spec_from_payload(dict(payload))
                if int(layer_idx) in spec.target_layers and is_subspace_operator(spec.operator):
                    source_layer = spec.source_layer_for(int(layer_idx))
                    break
        layer_payload = subspace.get(int(source_layer)) if isinstance(subspace, dict) else None
        if (
            isinstance(layer_state, dict)
            and isinstance(stats_state, dict)
            and isinstance(layer_payload, dict)
        ):
            if debug_mode_enabled("log", "subspace_compiled_path"):
                debug_log(
                    "subspace_compiled_path",
                    layer_idx=int(layer_idx),
                    hidden_shape=tuple(hidden.shape),
                    active_sum=int(layer_state["active"].sum().item()),
                    row_counts=layer_state["row_counts"].detach().cpu().tolist(),
                    token_spans=layer_state["token_spans"].detach().cpu().tolist(),
                    mode_ids=layer_state["mode_ids"].detach().cpu().tolist(),
                )
            patched_hidden = run_custom_op(
                hidden=hidden,
                custom_op=custom_op,
                operator_id=PROJECT_OUT_MODE_ID,
                mean=layer_payload["mean"],
                scale=layer_payload["scale"],
                safe_scale=layer_payload["safe_scale"],
                query_positions=layer_state["query_positions"],
                token_counts=layer_state["token_counts"],
                donor_rows=layer_state["donor_rows"],
                selected_rows=layer_state["selected_rows"],
                row_counts=layer_state["row_counts"],
                strengths=layer_state["strengths"],
                active=layer_state["active"],
                stats_valid=stats_state["valid"],
                stats_scalars=stats_state["scalars"],
                stats_coeff_before=stats_state["coeff_before"],
                stats_coeff_after=stats_state["coeff_after"],
                token_spans=layer_state["token_spans"],
                mode_ids=layer_state["mode_ids"],
                direction_raw=layer_state["direction_raw"],
                direction_std=layer_state["direction_std"],
                donor_means=layer_state["donor_means"],
                random_rows=layer_state["random_rows"],
                rowwise=layer_state["rowwise"],
                match_projected_norm=layer_state["match_projected_norm"],
            )
            if debug_mode_enabled("panic", "subspace_postop_missing_stats"):
                debug_panic(
                    "subspace_postop_missing_stats",
                    layer_idx=int(layer_idx),
                    valid_sum=int(stats_state["valid"].sum().item()),
                    active_sum=int(layer_state["active"].sum().item()),
                    hidden_shape=tuple(hidden.shape),
                    token_spans=layer_state["token_spans"].detach().cpu().tolist(),
                )
            return replace_hidden_tensor(output, patched_hidden)

    if force_custom_op_presence and custom_op is not None and layer_operators <= {"interchange"}:
        runtime_state = getattr(owner_model, "_v2_activation_patch_batch_runtime_state", None)
        layer_state = runtime_state.get(int(layer_idx)) if isinstance(runtime_state, dict) else None
        stats_state = getattr(owner_model, "_v2_activation_patch_batch_tensor_stats", {}).get(int(layer_idx))
        if not isinstance(layer_state, dict) or not isinstance(stats_state, dict):
            return replace_hidden_tensor(output, noop_custom_op(hidden, custom_op=custom_op, operator_id=INTERCHANGE_MODE_ID))
        hidden_dim = int(hidden.shape[-1]) if hidden.ndim >= 2 else 1
        patched_hidden = run_custom_op(
            hidden=hidden,
            custom_op=custom_op,
            operator_id=INTERCHANGE_MODE_ID,
            mean=torch.zeros((hidden_dim,), device=hidden.device, dtype=torch.float32),
            scale=torch.ones((hidden_dim,), device=hidden.device, dtype=torch.float32),
            safe_scale=torch.ones((hidden_dim,), device=hidden.device, dtype=torch.float32),
            query_positions=layer_state["query_positions"],
            token_counts=layer_state["token_counts"],
            donor_rows=layer_state["donor_rows"],
            selected_rows=layer_state["selected_rows"],
            row_counts=layer_state["row_counts"],
            strengths=layer_state["strengths"],
            active=layer_state["active"],
            stats_valid=stats_state["valid"],
            stats_scalars=stats_state["scalars"],
            stats_coeff_before=stats_state["coeff_before"],
            stats_coeff_after=stats_state["coeff_after"],
        )
        return replace_hidden_tensor(output, patched_hidden)

    if force_custom_op_presence and custom_op is not None and layer_operators <= {"residual_path"}:
        runtime_state = getattr(owner_model, "_v2_activation_patch_batch_runtime_state", None)
        layer_state = runtime_state.get(int(layer_idx)) if isinstance(runtime_state, dict) else None
        stats_state = getattr(owner_model, "_v2_activation_patch_batch_tensor_stats", {}).get(int(layer_idx))
        if not isinstance(layer_state, dict) or not isinstance(stats_state, dict):
            return replace_hidden_tensor(output, noop_custom_op(hidden, custom_op=custom_op, operator_id=RESIDUAL_PATH_MODE_ID))
        flat_hidden, _ = flatten_hidden_for_patch(hidden)
        invalid_query_state = _first_invalid_active_query_position(
            query_positions=layer_state["query_positions"],
            token_counts=layer_state["token_counts"],
            active=layer_state["active"],
            hidden_token_count=int(flat_hidden.shape[0]),
        )
        if invalid_query_state is not None:
            raise RuntimeError(
                "activation-patch residual-path invalid query state "
                f"layer={int(layer_idx)} "
                + " ".join(f"{key}={value!r}" for key, value in sorted(invalid_query_state.items()))
            )
        hidden_dim = int(hidden.shape[-1]) if hidden.ndim >= 2 else 1
        patched_hidden = run_custom_op(
            hidden=hidden,
            custom_op=custom_op,
            operator_id=RESIDUAL_PATH_MODE_ID,
            mean=torch.zeros((hidden_dim,), device=hidden.device, dtype=torch.float32),
            scale=torch.ones((hidden_dim,), device=hidden.device, dtype=torch.float32),
            safe_scale=torch.ones((hidden_dim,), device=hidden.device, dtype=torch.float32),
            query_positions=layer_state["query_positions"],
            token_counts=layer_state["token_counts"],
            donor_rows=layer_state["donor_rows"],
            selected_rows=layer_state["selected_rows"],
            row_counts=layer_state["row_counts"],
            strengths=layer_state["strengths"],
            residual_path_transport_modes=layer_state["residual_path_transport_modes"],
            residual_path_replace_alphas=layer_state["residual_path_replace_alphas"],
            active=layer_state["active"],
            stats_valid=stats_state["valid"],
            stats_scalars=stats_state["scalars"],
            stats_coeff_before=stats_state["coeff_before"],
            stats_coeff_after=stats_state["coeff_after"],
        )
        return replace_hidden_tensor(output, patched_hidden)

    if force_custom_op_presence and custom_op is not None and layer_operators <= SUBSPACE_OPERATORS and layer_project_out_contiguous:
        runtime_state = getattr(owner_model, "_v2_activation_patch_batch_runtime_state", None)
        layer_state = runtime_state.get(int(layer_idx)) if isinstance(runtime_state, dict) else None
        stats_state = getattr(owner_model, "_v2_activation_patch_batch_tensor_stats", {}).get(int(layer_idx))
        if not isinstance(layer_state, dict) or not isinstance(stats_state, dict):
            return replace_hidden_tensor(output, noop_custom_op(hidden, custom_op=custom_op, operator_id=PROJECT_OUT_MODE_ID))
        subspace = getattr(owner_model, "_v2_activation_patch_subspace", {})
        source_layer = int(layer_idx)
        if isinstance(batch_specs, list):
            for batch_spec in batch_specs:
                payload = batch_spec.get("patch_spec")
                if not isinstance(payload, dict):
                    continue
                spec = spec_from_payload(dict(payload))
                if int(layer_idx) in spec.target_layers and is_subspace_operator(spec.operator):
                    source_layer = spec.source_layer_for(int(layer_idx))
                    break
        layer_payload = subspace.get(int(source_layer)) if isinstance(subspace, dict) else None
        if not isinstance(layer_payload, dict):
            return replace_hidden_tensor(output, noop_custom_op(hidden, custom_op=custom_op, operator_id=PROJECT_OUT_MODE_ID))
        patched_hidden = run_custom_op(
            hidden=hidden,
            custom_op=custom_op,
            operator_id=PROJECT_OUT_MODE_ID,
            mean=layer_payload["mean"],
            scale=layer_payload["scale"],
            safe_scale=layer_payload["safe_scale"],
            query_positions=layer_state["query_positions"],
            token_counts=layer_state["token_counts"],
            donor_rows=layer_state["donor_rows"],
            selected_rows=layer_state["selected_rows"],
            row_counts=layer_state["row_counts"],
            strengths=layer_state["strengths"],
            active=layer_state["active"],
            stats_valid=stats_state["valid"],
            stats_scalars=stats_state["scalars"],
            stats_coeff_before=stats_state["coeff_before"],
            stats_coeff_after=stats_state["coeff_after"],
            token_spans=layer_state["token_spans"],
            mode_ids=layer_state["mode_ids"],
            direction_raw=layer_state["direction_raw"],
            direction_std=layer_state["direction_std"],
            donor_means=layer_state["donor_means"],
            random_rows=layer_state["random_rows"],
            rowwise=layer_state["rowwise"],
            match_projected_norm=layer_state["match_projected_norm"],
        )
        return replace_hidden_tensor(output, patched_hidden)

    if not isinstance(batch_specs, list) or not batch_specs:
        return output
    patched_hidden = hidden
    for batch_spec in batch_specs:
        payload = batch_spec.get("patch_spec")
        if not isinstance(payload, dict):
            continue
        spec = spec_from_payload(payload)
        if int(layer_idx) not in spec.target_layers:
            continue
        patched_hidden, stats = patch_hidden_states_for_layer(
            patched_hidden,
            owner_model=owner_model,
            layer_idx=int(layer_idx),
            batch_spec=batch_spec,
        )
        if stats is not None:
            _record_patch_stats(
                owner_model,
                req_id=str(batch_spec.get("req_id")),
                layer_idx=int(layer_idx),
                stats=stats,
            )
    return replace_hidden_tensor(output, patched_hidden)


def build_activation_patched_layer_class() -> type[Any]:
    class _ActivationPatchedLayer(torch.nn.Module):
        def __init__(self, owner_model: Any, layer_idx: int, wrapped_layer: Any) -> None:
            super().__init__()
            self.layer_idx = int(layer_idx)
            self.wrapped_layer = wrapped_layer
            self._owner_ref = weakref.ref(owner_model)
            try:
                self.activation_patch_hidden_states_op = build_activation_patch_hidden_states_op()
            except Exception:
                self.activation_patch_hidden_states_op = None

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            owner_model = self._owner_ref()
            output = self.wrapped_layer(*args, **kwargs)
            if owner_model is None:
                return output
            return apply_layer_output_patching(
                owner_model=owner_model,
                layer_idx=self.layer_idx,
                custom_op=self.activation_patch_hidden_states_op,
                output=output,
            )

    return _ActivationPatchedLayer


try:
    ActivationPatchedLayerClass = build_activation_patched_layer_class()
except Exception:
    ActivationPatchedLayerClass = ActivationPatchedLayer


__all__ = [
    "ActivationPatchedLayerClass",
    "apply_layer_output_patching",
    "register_torch_library_interchange_batch_op",
    "register_torch_library_residual_path_batch_op",
    "register_torch_library_subspace_batch_op",
    "register_torch_library_subspace_op",
    "register_torch_library_project_out_batch_op",
    "register_torch_library_project_out_op",
]
