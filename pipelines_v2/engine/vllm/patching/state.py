"""Runtime state registration and harvest for activation patching."""

from __future__ import annotations

import math
from typing import Any

import torch

from ..activation_patch_math import operator_mode_id
from .base import (
    _DEFAULT_COMPILED_PATCH_MAX_COMPONENTS,
    _MAX_BATCH_PATCH_SLOTS,
    contiguous_token_span,
    debug_log,
    infer_model_device,
    spec_from_payload,
    unwrap_model,
)
from .subspace_family import (
    SUBSPACE_STATS_SCALAR_DIM,
    is_subspace_operator,
    resolve_subspace_inputs,
    summarize_harvested_subspace_stats,
)


def register_activation_patch_bank(model: Any, bank_payload: dict[int, dict[str, Any]]) -> dict[str, Any]:
    model = unwrap_model(model)
    if not getattr(model, "_v2_activation_patch_initialized", False):
        from .hooks import init_activation_patching

        init_activation_patching(model)
    device = getattr(model, "_v2_activation_patch_device", None) or infer_model_device(model)
    registered: dict[int, dict[str, Any]] = {}
    for raw_layer, layer_payload in bank_payload.items():
        layer = int(raw_layer)
        per_example: dict[str, Any] = {}
        for example_key, example_payload in dict(layer_payload).items():
            per_example[str(example_key)] = {
                "values": torch.as_tensor(example_payload["values"], device=device, dtype=torch.float32),
                "token_count": int(example_payload.get("token_count", 0)),
            }
        registered[layer] = per_example
    model._v2_activation_patch_bank = registered
    return {
        "layers": sorted(int(layer) for layer in registered),
        "examples": sum(len(items) for items in registered.values()),
    }


def register_activation_patch_subspace(model: Any, subspace_payload: dict[int, dict[str, Any]]) -> dict[str, Any]:
    model = unwrap_model(model)
    if not getattr(model, "_v2_activation_patch_initialized", False):
        from .hooks import init_activation_patching

        init_activation_patching(model)
    device = getattr(model, "_v2_activation_patch_device", None) or infer_model_device(model)
    registered = getattr(model, "_v2_activation_patch_subspace", None)
    if not isinstance(registered, dict):
        registered = {}
    for raw_layer, layer_payload in subspace_payload.items():
        layer = int(raw_layer)
        mean = torch.as_tensor(layer_payload["mean"], device=device, dtype=torch.float32)
        scale = torch.as_tensor(layer_payload["scale"], device=device, dtype=torch.float32)
        safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        components = torch.as_tensor(layer_payload["components"], device=device, dtype=torch.float32)
        if components.ndim == 1:
            components = components.unsqueeze(0)
        if components.numel():
            row_norms = torch.linalg.norm(components, dim=1, keepdim=True)
            components = components / torch.where(row_norms == 0, torch.ones_like(row_norms), row_norms)
        registered[layer] = {
            "mean": mean,
            "scale": scale,
            "safe_scale": safe_scale,
            "components": components,
            "named_components": {
                str(name): int(index)
                for name, index in dict(layer_payload.get("named_components", {})).items()
            },
        }
    model._v2_activation_patch_subspace = registered
    debug_log(
        "registered_subspace",
        layers=sorted(int(layer) for layer in registered),
        component_counts={
            int(layer): int(payload["components"].shape[0])
            for layer, payload in sorted(registered.items())
        },
    )
    return {
        "layers": sorted(int(layer) for layer in registered),
        "components": {
            str(int(layer)): int(payload["components"].shape[0])
            for layer, payload in registered.items()
        },
    }


def register_activation_patch_directions(model: Any, direction_payload: dict[int, dict[str, Any]]) -> dict[str, Any]:
    model = unwrap_model(model)
    if not getattr(model, "_v2_activation_patch_initialized", False):
        from .hooks import init_activation_patching

        init_activation_patching(model)
    device = getattr(model, "_v2_activation_patch_device", None) or infer_model_device(model)
    registered = getattr(model, "_v2_activation_patch_directions", None)
    if not isinstance(registered, dict):
        registered = {}
    for raw_layer, layer_payload in direction_payload.items():
        layer = int(raw_layer)
        entry: dict[str, Any] = {}
        raw_vector = layer_payload.get("raw_vector")
        if raw_vector is not None:
            entry["raw_vector"] = torch.as_tensor(raw_vector, device=device, dtype=torch.float32)
        subspace_weights = layer_payload.get("subspace_weights")
        if subspace_weights is not None:
            entry["subspace_weights"] = torch.as_tensor(subspace_weights, device=device, dtype=torch.float32)
        registered[layer] = entry
    model._v2_activation_patch_directions = registered
    return {"layers": sorted(int(layer) for layer in registered)}


def register_activation_patch_centroids(model: Any, centroid_payload: dict[int, dict[str, Any]]) -> dict[str, Any]:
    model = unwrap_model(model)
    if not getattr(model, "_v2_activation_patch_initialized", False):
        from .hooks import init_activation_patching

        init_activation_patching(model)
    device = getattr(model, "_v2_activation_patch_device", None) or infer_model_device(model)
    registered = getattr(model, "_v2_activation_patch_centroids", None)
    if not isinstance(registered, dict):
        registered = {}
    for raw_layer, layer_payload in centroid_payload.items():
        layer = int(raw_layer)
        centroids_payload = dict(layer_payload.get("centroids", {}))
        registered[layer] = {
            "centroids": {
                str(name): torch.as_tensor(value, device=device, dtype=torch.float32)
                for name, value in centroids_payload.items()
            }
        }
    model._v2_activation_patch_centroids = registered
    return {
        "layers": sorted(int(layer) for layer in registered),
        "centroid_counts": {
            str(int(layer)): len(payload.get("centroids", {}))
            for layer, payload in registered.items()
        },
    }


def set_batch_patch_specs(model: Any, batch_specs: list[dict[str, Any]]) -> None:
    model = unwrap_model(model)
    model._v2_activation_patch_batch_specs = [dict(spec) for spec in batch_specs]
    _load_batch_runtime_state(model, list(batch_specs))


def clear_batch_patch_specs(model: Any) -> None:
    model = unwrap_model(model)
    model._v2_activation_patch_batch_specs = []
    _reset_batch_runtime_state(model)
    _reset_batch_tensor_stats(model)


def collect_patch_stats(model: Any, req_id: str | None = None) -> dict[Any, dict[str, Any]]:
    model = unwrap_model(model)
    stats_by_req = getattr(model, "_v2_activation_patch_stats_by_req", {})
    if not isinstance(stats_by_req, dict):
        return {}
    if req_id is None:
        return {
            str(request_id): {
                int(layer): dict(payload)
                for layer, payload in layer_stats.items()
            }
            for request_id, layer_stats in stats_by_req.items()
        }
    lookup_id = str(req_id)
    payload = stats_by_req.get(lookup_id)
    if payload is None:
        short_id = lookup_id.split("-", 1)[0]
        for candidate_id, candidate_payload in stats_by_req.items():
            candidate_text = str(candidate_id)
            if candidate_text == short_id or candidate_text.startswith(f"{short_id}-"):
                payload = candidate_payload
                break
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        return {}
    return {int(layer): dict(layer_payload) for layer, layer_payload in payload.items()}


def harvest_batch_patch_stats(model: Any, batch_specs: list[dict[str, Any]]) -> None:
    model = unwrap_model(model)
    buffers = getattr(model, "_v2_activation_patch_batch_tensor_stats", None)
    if not isinstance(buffers, dict) or not batch_specs:
        return
    for slot_idx, batch_spec in enumerate(batch_specs[:_MAX_BATCH_PATCH_SLOTS]):
        payload = batch_spec.get("patch_spec")
        if not isinstance(payload, dict):
            continue
        spec = spec_from_payload(dict(payload))
        req_id = str(batch_spec.get("req_id") or "")
        for layer_idx in spec.target_layers:
            layer_buffers = buffers.get(int(layer_idx))
            if not isinstance(layer_buffers, dict):
                continue
            valid = int(layer_buffers["valid"][slot_idx].item())
            if valid <= 0:
                debug_log(
                    "missing_harvested_stats",
                    req_id=req_id or "<empty>",
                    layer=int(layer_idx),
                    operator=spec.operator,
                )
                continue
            scalars = layer_buffers["scalars"][slot_idx].detach().cpu().tolist()
            if spec.is_interchange():
                stats = {
                    "layer": int(layer_idx),
                    "status": "ok",
                    "operator": spec.operator,
                    "token_count": int(round(float(scalars[1]))),
                    "delta_norm_raw": float(scalars[0]),
                    "query_positions": [int(pos) for pos in spec.query_positions],
                    "donor_positions": [int(pos) for pos in spec.donor_positions],
                    "case_key": spec.case_key,
                    "control_name": spec.control_name,
                }
            elif spec.is_residual_path():
                unique_positions = sorted({int(pos) for pos in getattr(spec, "covered_abs_positions", ())})
                covered_abs_spans: list[list[int]] = []
                if unique_positions:
                    start = prev = unique_positions[0]
                    for pos in unique_positions[1:]:
                        if pos == prev + 1:
                            prev = pos
                            continue
                        covered_abs_spans.append([int(start), int(prev + 1)])
                        start = prev = pos
                    covered_abs_spans.append([int(start), int(prev + 1)])
                target_abs_tokens = len({int(pos) for pos in getattr(spec, "target_abs_positions", ())})
                layer_edges = [
                    {
                        "source_layer": int(source_layer),
                        "write_layer": int(write_layer),
                        "weight": float(weight),
                    }
                    for source_layer, write_layer, weight in spec.path_edges
                    if int(write_layer) == int(layer_idx)
                ]
                stats = {
                    "layer": int(layer_idx),
                    "source_layer": int(layer_edges[0]["source_layer"]) if layer_edges else int(layer_idx),
                    "status": "ok",
                    "operator": spec.operator,
                    "token_count": int(round(float(scalars[1]))),
                    "query_positions": [int(pos) for pos in spec.query_positions],
                    "case_key": spec.case_key,
                    "control_name": spec.control_name,
                    "covered_abs_spans": covered_abs_spans,
                    "covered_abs_tokens": int(len(unique_positions)),
                    "target_abs_tokens": int(target_abs_tokens),
                    "coverage_fraction": (
                        float(len(unique_positions)) / float(target_abs_tokens)
                        if int(target_abs_tokens) > 0
                        else 0.0
                    ),
                    "delta_norm_raw": float(scalars[0]),
                    "transport": spec.transport,
                    "path_edges": layer_edges,
                    "replace_alpha": float(scalars[2]),
                    "donor_example_key": spec.donor_example_key,
                }
            elif is_subspace_operator(spec.operator):
                coeff_dim = len(spec.selected_components_for(int(layer_idx)))
                if coeff_dim <= 0:
                    coeff_dim = int(layer_buffers["coeff_before"].shape[1])
                coeff_before = layer_buffers["coeff_before"][slot_idx, :coeff_dim].detach().cpu().tolist()
                coeff_after = layer_buffers["coeff_after"][slot_idx, :coeff_dim].detach().cpu().tolist()
                stats = summarize_harvested_subspace_stats(
                    spec=spec,
                    layer_idx=int(layer_idx),
                    scalars=[float(value) for value in scalars],
                    coeff_before=[float(v) for v in coeff_before],
                    coeff_after=[float(v) for v in coeff_after],
                    covered_abs_positions=list(getattr(spec, "covered_abs_positions", ())),
                )
            else:
                stats = {
                    "layer": int(layer_idx),
                    "status": "ok",
                    "operator": spec.operator,
                    "token_count": int(len(spec.query_positions)),
                    "query_positions": [int(pos) for pos in spec.query_positions],
                    "case_key": spec.case_key,
                    "control_name": spec.control_name,
                }
            _record_patch_stats(
                model,
                req_id=req_id,
                layer_idx=int(layer_idx),
                stats=stats,
            )
            debug_log(
                "harvested_stats",
                req_id=req_id or "<empty>",
                layer=int(layer_idx),
                operator=spec.operator,
                token_count=stats.get("token_count"),
                delta_norm_raw=stats.get("delta_norm_raw"),
            )


def _normalize_chunk_stats(stats: dict[str, Any]) -> list[dict[str, Any]]:
    chunk_stats = stats.get("chunk_stats")
    if isinstance(chunk_stats, list):
        normalized = [dict(item) for item in chunk_stats if isinstance(item, dict)]
        if normalized:
            return normalized
    base = dict(stats)
    base.pop("chunk_stats", None)
    base.pop("chunk_count", None)
    return [base]


def _merge_covered_abs_spans(*stats_payloads: dict[str, Any]) -> list[list[int]]:
    covered_spans = [
        [int(start), int(end)]
        for payload in stats_payloads
        for start, end in [
            (span[0], span[1])
            for span in payload.get("covered_abs_spans", ())
            if isinstance(span, (list, tuple)) and len(span) == 2
        ]
    ]
    if not covered_spans:
        return []
    covered_spans.sort()
    coalesced: list[list[int]] = []
    cur_start, cur_end = covered_spans[0]
    for start, end in covered_spans[1:]:
        if int(start) <= int(cur_end):
            cur_end = max(int(cur_end), int(end))
        else:
            coalesced.append([int(cur_start), int(cur_end)])
            cur_start, cur_end = int(start), int(end)
    coalesced.append([int(cur_start), int(cur_end)])
    return coalesced


def _maybe_stable_scalar(chunk_stats: list[dict[str, Any]], key: str) -> float | int | None:
    values = [item.get(key) for item in chunk_stats if item.get(key) is not None]
    if not values:
        return None
    first = values[0]
    if isinstance(first, bool):
        return int(first)
    if isinstance(first, int):
        if all(int(value) == int(first) for value in values if isinstance(value, (int, float))):
            return int(first)
        return None
    if isinstance(first, float):
        if all(math.isclose(float(value), float(first), rel_tol=1e-6, abs_tol=1e-6) for value in values if isinstance(value, (int, float))):
            return float(first)
        return None
    return None


def _aggregate_patch_stats(operator: str, chunk_stats: list[dict[str, Any]]) -> dict[str, Any]:
    aggregated: dict[str, Any] = {
        "chunk_count": int(len(chunk_stats)),
        "chunk_stats": [dict(item) for item in chunk_stats],
    }

    token_count = 0
    saw_token_count = False
    for item in chunk_stats:
        value = item.get("token_count")
        if isinstance(value, (int, float)):
            token_count += int(value)
            saw_token_count = True
    if saw_token_count:
        aggregated["token_count"] = int(token_count)

    covered_abs_spans = _merge_covered_abs_spans(*chunk_stats)
    if covered_abs_spans:
        aggregated["covered_abs_spans"] = covered_abs_spans
        aggregated["covered_abs_tokens"] = sum(max(0, int(end) - int(start)) for start, end in covered_abs_spans)
        target_abs_tokens = next(
            (
                int(value)
                for value in (
                    item.get("target_abs_tokens")
                    for item in reversed(chunk_stats)
                )
                if isinstance(value, (int, float)) and int(value) > 0
            ),
            0,
        )
        if target_abs_tokens > 0:
            aggregated["target_abs_tokens"] = int(target_abs_tokens)
            aggregated["coverage_fraction"] = float(aggregated["covered_abs_tokens"]) / float(target_abs_tokens)

    if operator in {"interchange", "residual_path"}:
        delta_norms = [
            float(item["delta_norm_raw"])
            for item in chunk_stats
            if isinstance(item.get("delta_norm_raw"), (int, float))
        ]
        if delta_norms:
            aggregated["delta_norm_raw"] = float(math.sqrt(sum(value * value for value in delta_norms)))

    if operator == "add_direction":
        direction_norm_raw = _maybe_stable_scalar(chunk_stats, "direction_norm_raw")
        if direction_norm_raw is not None:
            aggregated["direction_norm_raw"] = float(direction_norm_raw)

    stable_strength = _maybe_stable_scalar(chunk_stats, "strength")
    if stable_strength is not None:
        aggregated["strength"] = float(stable_strength)

    if operator == "residual_path":
        stable_replace_alpha = _maybe_stable_scalar(chunk_stats, "replace_alpha")
        if stable_replace_alpha is not None:
            aggregated["replace_alpha"] = float(stable_replace_alpha)

    return aggregated


def _record_patch_stats(owner_model: Any, *, req_id: str, layer_idx: int, stats: dict[str, Any]) -> None:
    stats_by_req = getattr(owner_model, "_v2_activation_patch_stats_by_req", None)
    if not isinstance(stats_by_req, dict):
        stats_by_req = {}
        owner_model._v2_activation_patch_stats_by_req = stats_by_req
    req_stats = stats_by_req.setdefault(str(req_id), {})
    existing = req_stats.get(int(layer_idx))
    if not isinstance(existing, dict):
        req_stats[int(layer_idx)] = dict(stats)
        return

    merged = dict(existing)
    merged.update(dict(stats))

    chunk_stats = _normalize_chunk_stats(existing)
    chunk_stats.append(dict(stats))
    operator = str(merged.get("operator") or existing.get("operator") or stats.get("operator") or "")
    aggregated = _aggregate_patch_stats(operator, chunk_stats)
    merged.update(aggregated)
    if int(aggregated.get("chunk_count", 1)) > 1:
        for field in (
            "query_positions",
            "donor_positions",
            "target_read_positions",
            "covered_abs_positions",
        ):
            merged.pop(field, None)
        if operator not in {"interchange", "residual_path"}:
            for field in (
                "delta_norm_raw",
                "delta_norm_std",
                "mean_norm_before",
                "mean_norm_after",
                "mean_std_norm_before",
                "mean_std_norm_after",
                "selected_proj_norm_before",
                "selected_coeff_before",
                "selected_coeff_after",
                "selected_component_count",
            ):
                if field not in aggregated:
                    merged.pop(field, None)

    req_stats[int(layer_idx)] = merged


def _ensure_batch_runtime_state_buffers(
    model: Any,
    *,
    layer_idx: int,
    max_tokens: int,
    max_rows: int,
    hidden_dim: int,
    device: Any,
) -> dict[str, Any]:
    buffers = getattr(model, "_v2_activation_patch_batch_runtime_state", None)
    if not isinstance(buffers, dict):
        buffers = {}
        model._v2_activation_patch_batch_runtime_state = buffers
    layer_buffers = buffers.get(int(layer_idx))
    if (
        not isinstance(layer_buffers, dict)
        or "mode_ids" not in layer_buffers
        or "match_projected_norm" not in layer_buffers
        or "direction_raw" not in layer_buffers
        or "direction_std" not in layer_buffers
        or "donor_means" not in layer_buffers
        or "random_rows" not in layer_buffers
        or "residual_path_transport_modes" not in layer_buffers
        or "residual_path_replace_alphas" not in layer_buffers
        or int(layer_buffers["query_positions"].shape[1]) != int(max_tokens)
        or int(layer_buffers["donor_rows"].shape[1]) != int(max_tokens)
        or int(layer_buffers["donor_rows"].shape[2]) != int(hidden_dim)
        or int(layer_buffers["selected_rows"].shape[1]) != int(max_rows)
        or int(layer_buffers["selected_rows"].shape[2]) != int(hidden_dim)
        or int(layer_buffers["random_rows"].shape[1]) != int(max_rows)
        or int(layer_buffers["random_rows"].shape[2]) != int(hidden_dim)
    ):
        layer_buffers = {
            "active": torch.zeros((_MAX_BATCH_PATCH_SLOTS,), device=device, dtype=torch.int32),
            "mode_ids": torch.zeros((_MAX_BATCH_PATCH_SLOTS,), device=device, dtype=torch.int32),
            "token_counts": torch.zeros((_MAX_BATCH_PATCH_SLOTS,), device=device, dtype=torch.int32),
            "row_counts": torch.zeros((_MAX_BATCH_PATCH_SLOTS,), device=device, dtype=torch.int32),
            "token_spans": torch.zeros((_MAX_BATCH_PATCH_SLOTS, 2), device=device, dtype=torch.int32),
            "strengths": torch.zeros((_MAX_BATCH_PATCH_SLOTS,), device=device, dtype=torch.float32),
            "match_projected_norm": torch.ones((_MAX_BATCH_PATCH_SLOTS,), device=device, dtype=torch.int32),
            "query_positions": torch.zeros((_MAX_BATCH_PATCH_SLOTS, int(max_tokens)), device=device, dtype=torch.int32),
            "donor_rows": torch.zeros(
                (_MAX_BATCH_PATCH_SLOTS, int(max_tokens), int(hidden_dim)),
                device=device,
                dtype=torch.float32,
            ),
            "selected_rows": torch.zeros(
                (_MAX_BATCH_PATCH_SLOTS, int(max_rows), int(hidden_dim)),
                device=device,
                dtype=torch.float32,
            ),
            "direction_raw": torch.zeros((_MAX_BATCH_PATCH_SLOTS, int(hidden_dim)), device=device, dtype=torch.float32),
            "direction_std": torch.zeros((_MAX_BATCH_PATCH_SLOTS, int(hidden_dim)), device=device, dtype=torch.float32),
            "donor_means": torch.zeros((_MAX_BATCH_PATCH_SLOTS, int(hidden_dim)), device=device, dtype=torch.float32),
            "random_rows": torch.zeros(
                (_MAX_BATCH_PATCH_SLOTS, int(max_rows), int(hidden_dim)),
                device=device,
                dtype=torch.float32,
            ),
            "residual_path_transport_modes": torch.zeros(
                (_MAX_BATCH_PATCH_SLOTS,),
                device=device,
                dtype=torch.int32,
            ),
            "residual_path_replace_alphas": torch.zeros(
                (_MAX_BATCH_PATCH_SLOTS,),
                device=device,
                dtype=torch.float32,
            ),
        }
        buffers[int(layer_idx)] = layer_buffers
    return layer_buffers


def _ensure_batch_tensor_stats_buffers(
    model: Any,
    *,
    layer_idx: int,
    coeff_dim: int,
    device: Any,
) -> dict[str, Any]:
    buffers = getattr(model, "_v2_activation_patch_batch_tensor_stats", None)
    if not isinstance(buffers, dict):
        buffers = {}
        model._v2_activation_patch_batch_tensor_stats = buffers
    layer_buffers = buffers.get(int(layer_idx))
    if (
        not isinstance(layer_buffers, dict)
        or int(layer_buffers["coeff_before"].shape[1]) != int(coeff_dim)
        or int(layer_buffers["coeff_after"].shape[1]) != int(coeff_dim)
    ):
        layer_buffers = {
            "valid": torch.zeros((_MAX_BATCH_PATCH_SLOTS,), device=device, dtype=torch.int32),
            "scalars": torch.zeros((_MAX_BATCH_PATCH_SLOTS, SUBSPACE_STATS_SCALAR_DIM), device=device, dtype=torch.float32),
            "coeff_before": torch.zeros((_MAX_BATCH_PATCH_SLOTS, int(coeff_dim)), device=device, dtype=torch.float32),
            "coeff_after": torch.zeros((_MAX_BATCH_PATCH_SLOTS, int(coeff_dim)), device=device, dtype=torch.float32),
        }
        buffers[int(layer_idx)] = layer_buffers
    return layer_buffers


def _reset_batch_runtime_state(model: Any) -> None:
    buffers = getattr(model, "_v2_activation_patch_batch_runtime_state", None)
    if not isinstance(buffers, dict):
        return
    for layer_buffers in buffers.values():
        if not isinstance(layer_buffers, dict):
            continue
        for tensor in layer_buffers.values():
            if hasattr(tensor, "zero_"):
                tensor.zero_()


def _reset_batch_tensor_stats(model: Any) -> None:
    buffers = getattr(model, "_v2_activation_patch_batch_tensor_stats", None)
    if not isinstance(buffers, dict):
        return
    for layer_buffers in buffers.values():
        if not isinstance(layer_buffers, dict):
            continue
        for tensor in layer_buffers.values():
            if hasattr(tensor, "zero_"):
                tensor.zero_()


def _load_batch_runtime_state(model: Any, batch_specs: list[dict[str, Any]]) -> None:
    bank = getattr(model, "_v2_activation_patch_bank", None)
    subspace = getattr(model, "_v2_activation_patch_subspace", None)
    _reset_batch_runtime_state(model)
    _reset_batch_tensor_stats(model)

    requirements: dict[int, tuple[int, int, int, Any]] = {}
    for batch_spec in batch_specs[:_MAX_BATCH_PATCH_SLOTS]:
        payload = batch_spec.get("patch_spec")
        if not isinstance(payload, dict):
            continue
        spec = spec_from_payload(dict(payload))
        token_count = len(spec.query_positions)
        if token_count <= 0:
            continue
        for layer_idx in spec.target_layers:
            hidden_dim = None
            max_rows = 1
            device = None
            if spec.is_interchange():
                if not isinstance(bank, dict):
                    continue
                layer_bank = bank.get(int(layer_idx))
                if not isinstance(layer_bank, dict):
                    continue
                donor_entry = layer_bank.get(spec.donor_example_key)
                if not isinstance(donor_entry, dict):
                    continue
                donor_values = donor_entry.get("values")
                hidden_dim = int(donor_values.shape[-1])
                device = donor_values.device
            elif spec.is_residual_path():
                if not isinstance(bank, dict):
                    continue
                layer_edges = [
                    (int(source_layer), float(weight))
                    for source_layer, write_layer, weight in spec.path_edges
                    if int(write_layer) == int(layer_idx)
                ]
                if not layer_edges:
                    continue
                source_layer = int(layer_edges[0][0])
                layer_bank = bank.get(source_layer)
                if not isinstance(layer_bank, dict):
                    continue
                donor_entry = layer_bank.get(spec.donor_example_key)
                if not isinstance(donor_entry, dict):
                    continue
                donor_values = donor_entry.get("values")
                hidden_dim = int(donor_values.shape[-1])
                device = donor_values.device
            elif is_subspace_operator(spec.operator):
                if not isinstance(subspace, dict):
                    continue
                source_layer = spec.source_layer_for(int(layer_idx))
                layer_payload = subspace.get(int(source_layer))
                if not isinstance(layer_payload, dict):
                    continue
                hidden_dim = int(layer_payload["mean"].shape[0])
                device = layer_payload["mean"].device
                subspace_inputs, _ = resolve_subspace_inputs(
                    owner_model=model,
                    spec=spec,
                    layer_idx=int(layer_idx),
                    hidden_dim=int(hidden_dim),
                    device=device,
                )
                if subspace_inputs is None:
                    continue
                max_rows = max(
                    _DEFAULT_COMPILED_PATCH_MAX_COMPONENTS,
                    int(subspace_inputs["selected_rows"].shape[0]),
                    int(subspace_inputs["random_rows"].shape[0]) if subspace_inputs["random_rows"] is not None else 0,
                    1,
                )
            else:
                continue
            if hidden_dim is None or device is None:
                continue
            current = requirements.get(int(layer_idx))
            if current is None:
                requirements[int(layer_idx)] = (token_count, max_rows, hidden_dim, device)
            else:
                requirements[int(layer_idx)] = (
                    max(int(current[0]), int(token_count)),
                    max(int(current[1]), int(max_rows)),
                    int(hidden_dim),
                    device,
                )

    for layer_idx, (max_tokens, max_rows, hidden_dim, device) in requirements.items():
        _ensure_batch_runtime_state_buffers(
            model,
            layer_idx=int(layer_idx),
            max_tokens=max(1, int(max_tokens)),
            max_rows=max(1, int(max_rows)),
            hidden_dim=int(hidden_dim),
            device=device,
        )
        _ensure_batch_tensor_stats_buffers(
            model,
            layer_idx=int(layer_idx),
            coeff_dim=max(1, int(max_rows)),
            device=device,
        )

    for slot_idx, batch_spec in enumerate(batch_specs[:_MAX_BATCH_PATCH_SLOTS]):
        payload = batch_spec.get("patch_spec")
        if not isinstance(payload, dict):
            continue
        spec = spec_from_payload(dict(payload))
        token_count = len(spec.query_positions)
        if token_count <= 0:
            continue
        for layer_idx in spec.target_layers:
            runtime_buffers = model._v2_activation_patch_batch_runtime_state.get(int(layer_idx))
            if not isinstance(runtime_buffers, dict):
                continue
            runtime_buffers["active"][slot_idx] = 0
            runtime_buffers["mode_ids"][slot_idx] = 0
            runtime_buffers["token_counts"][slot_idx] = 0
            runtime_buffers["row_counts"][slot_idx] = 0
            runtime_buffers["token_spans"][slot_idx].zero_()
            runtime_buffers["strengths"][slot_idx] = 0.0
            runtime_buffers["match_projected_norm"][slot_idx] = 1
            runtime_buffers["query_positions"][slot_idx].zero_()
            runtime_buffers["donor_rows"][slot_idx].zero_()
            runtime_buffers["selected_rows"][slot_idx].zero_()
            runtime_buffers["direction_raw"][slot_idx].zero_()
            runtime_buffers["direction_std"][slot_idx].zero_()
            runtime_buffers["donor_means"][slot_idx].zero_()
            runtime_buffers["random_rows"][slot_idx].zero_()
            runtime_buffers["residual_path_transport_modes"][slot_idx] = 0
            runtime_buffers["residual_path_replace_alphas"][slot_idx] = 0.0
            if spec.is_interchange():
                if int(runtime_buffers["query_positions"].shape[1]) < int(token_count):
                    continue
                if not isinstance(bank, dict):
                    continue
                layer_bank = bank.get(int(layer_idx))
                if not isinstance(layer_bank, dict):
                    continue
                donor_entry = layer_bank.get(spec.donor_example_key)
                if not isinstance(donor_entry, dict):
                    continue
                donor_values = donor_entry.get("values")
                if donor_values is None:
                    continue
                runtime_buffers["active"][slot_idx] = 1
                runtime_buffers["mode_ids"][slot_idx] = 0
                runtime_buffers["token_counts"][slot_idx] = int(token_count)
                runtime_buffers["query_positions"][slot_idx, :token_count] = donor_values.new_tensor(
                    list(spec.query_positions),
                    dtype=runtime_buffers["query_positions"].dtype,
                )
                runtime_buffers["donor_rows"][slot_idx, :token_count].copy_(
                    donor_values[list(spec.donor_positions)].to(torch.float32)
                )
            elif spec.is_residual_path():
                if not isinstance(bank, dict):
                    continue
                if int(runtime_buffers["query_positions"].shape[1]) < int(token_count):
                    continue
                layer_edges = [
                    (int(source_layer), float(weight))
                    for source_layer, write_layer, weight in spec.path_edges
                    if int(write_layer) == int(layer_idx)
                ]
                if not layer_edges:
                    continue
                payload_rows = None
                replace_alpha = 0.0
                for source_layer, weight in layer_edges:
                    layer_bank = bank.get(int(source_layer))
                    if not isinstance(layer_bank, dict):
                        payload_rows = None
                        break
                    donor_entry = layer_bank.get(spec.donor_example_key)
                    if not isinstance(donor_entry, dict):
                        payload_rows = None
                        break
                    donor_values = donor_entry.get("values")
                    if donor_values is None:
                        payload_rows = None
                        break
                    donor_positions = [int(pos) for pos in spec.donor_positions]
                    if any(pos < 0 or pos >= int(donor_values.shape[0]) for pos in donor_positions):
                        payload_rows = None
                        break
                    donor_rows = donor_values[donor_positions].to(torch.float32)
                    edge_scale = float(weight) * float(spec.strength)
                    if payload_rows is None:
                        payload_rows = torch.zeros_like(donor_rows)
                    if spec.transport == "replace":
                        payload_rows = payload_rows + (edge_scale * donor_rows)
                        replace_alpha += edge_scale
                    else:
                        target_entry = layer_bank.get(spec.example_key)
                        if not isinstance(target_entry, dict):
                            payload_rows = None
                            break
                        target_values = target_entry.get("values")
                        if target_values is None:
                            payload_rows = None
                            break
                        target_read_positions = [int(pos) for pos in spec.target_read_positions]
                        if any(pos < 0 or pos >= int(target_values.shape[0]) for pos in target_read_positions):
                            payload_rows = None
                            break
                        target_rows = target_values[target_read_positions].to(torch.float32)
                        payload_rows = payload_rows + (edge_scale * (donor_rows - target_rows))
                if payload_rows is None:
                    continue
                runtime_buffers["active"][slot_idx] = 1
                runtime_buffers["mode_ids"][slot_idx] = int(operator_mode_id(spec.operator))
                runtime_buffers["token_counts"][slot_idx] = int(token_count)
                runtime_buffers["residual_path_transport_modes"][slot_idx] = int(spec.transport == "replace")
                runtime_buffers["residual_path_replace_alphas"][slot_idx] = float(replace_alpha)
                runtime_buffers["query_positions"][slot_idx, :token_count] = payload_rows.new_tensor(
                    list(spec.query_positions),
                    dtype=runtime_buffers["query_positions"].dtype,
                )
                runtime_buffers["donor_rows"][slot_idx, :token_count].copy_(payload_rows)
            elif is_subspace_operator(spec.operator):
                token_span = contiguous_token_span(spec.query_positions)
                if token_span is None:
                    continue
                source_layer = spec.source_layer_for(int(layer_idx))
                layer_payload = subspace.get(int(source_layer)) if isinstance(subspace, dict) else None
                if not isinstance(layer_payload, dict):
                    continue
                subspace_inputs, _ = resolve_subspace_inputs(
                    owner_model=model,
                    spec=spec,
                    layer_idx=int(layer_idx),
                    hidden_dim=int(layer_payload["mean"].shape[0]),
                    device=layer_payload["mean"].device,
                )
                if subspace_inputs is None:
                    continue
                row_count = max(
                    int(subspace_inputs["selected_rows"].shape[0]),
                    int(subspace_inputs["random_rows"].shape[0]) if subspace_inputs["random_rows"] is not None else 0,
                )
                if int(runtime_buffers["selected_rows"].shape[1]) < row_count:
                    continue
                runtime_buffers["active"][slot_idx] = 1
                runtime_buffers["mode_ids"][slot_idx] = int(operator_mode_id(spec.operator))
                runtime_buffers["token_counts"][slot_idx] = int(token_count)
                runtime_buffers["row_counts"][slot_idx] = int(row_count)
                runtime_buffers["token_spans"][slot_idx, 0] = int(token_span[0])
                runtime_buffers["token_spans"][slot_idx, 1] = int(token_span[1])
                runtime_buffers["strengths"][slot_idx] = float(spec.strength)
                runtime_buffers["match_projected_norm"][slot_idx] = int(bool(spec.match_projected_norm))
                if int(subspace_inputs["selected_rows"].shape[0]) > 0:
                    runtime_buffers["selected_rows"][slot_idx, : int(subspace_inputs["selected_rows"].shape[0])].copy_(
                        subspace_inputs["selected_rows"]
                    )
                if subspace_inputs["direction_raw"] is not None:
                    runtime_buffers["direction_raw"][slot_idx].copy_(subspace_inputs["direction_raw"])
                if subspace_inputs["direction_std"] is not None:
                    runtime_buffers["direction_std"][slot_idx].copy_(subspace_inputs["direction_std"])
                if subspace_inputs["donor_mean"] is not None:
                    runtime_buffers["donor_means"][slot_idx].copy_(subspace_inputs["donor_mean"])
                if subspace_inputs["random_rows"] is not None and int(subspace_inputs["random_rows"].shape[0]) > 0:
                    runtime_buffers["random_rows"][slot_idx, : int(subspace_inputs["random_rows"].shape[0])].copy_(
                        subspace_inputs["random_rows"]
                    )


__all__ = [
    "clear_batch_patch_specs",
    "collect_patch_stats",
    "harvest_batch_patch_stats",
    "register_activation_patch_bank",
    "register_activation_patch_centroids",
    "register_activation_patch_directions",
    "register_activation_patch_subspace",
    "set_batch_patch_specs",
    "_record_patch_stats",
]
