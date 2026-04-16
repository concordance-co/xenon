"""Minimal eager-mode activation patching utilities for pipelines_v2 vLLM."""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RequestInterchangeSpec:
    target_layers: tuple[int, ...]
    donor_example_key: str
    donor_positions: tuple[int, ...]
    query_positions: tuple[int, ...]
    case_key: str = ""
    control_name: str = ""

    def __post_init__(self) -> None:
        self.target_layers = tuple(int(layer) for layer in self.target_layers)
        self.donor_example_key = str(self.donor_example_key)
        self.donor_positions = tuple(int(pos) for pos in self.donor_positions)
        self.query_positions = tuple(int(pos) for pos in self.query_positions)
        self.case_key = str(self.case_key)
        self.control_name = str(self.control_name)


def _spec_from_payload(payload: dict[str, Any]) -> RequestInterchangeSpec:
    query_positions = payload.get("query_positions")
    if query_positions is None:
        query_positions = payload.get("target_positions", ())
    return RequestInterchangeSpec(
        target_layers=tuple(int(layer) for layer in payload.get("target_layers", ())),
        donor_example_key=str(payload.get("donor_example_key") or ""),
        donor_positions=tuple(int(pos) for pos in payload.get("donor_positions", ())),
        query_positions=tuple(int(pos) for pos in query_positions),
        case_key=str(payload.get("case_key") or ""),
        control_name=str(payload.get("control_name") or ""),
    )


def find_decoder_layers(model: Any) -> dict[int, Any]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return {}
    return {int(idx): layer for idx, layer in enumerate(layers)}


class _ActivationPatchedLayer:
    pass


def _infer_model_device(model: Any) -> Any:
    layers = find_decoder_layers(model)
    for layer in layers.values():
        candidate = getattr(layer, "wrapped_layer", layer)
        for attr_name in ("input_layernorm", "post_attention_layernorm", "self_attn", "mlp"):
            module = getattr(candidate, attr_name, None)
            if module is None:
                continue
            weight = getattr(module, "weight", None)
            if weight is not None and hasattr(weight, "device"):
                return weight.device
            parameters = getattr(module, "parameters", None)
            if callable(parameters):
                try:
                    first = next(parameters())
                except StopIteration:
                    continue
                return first.device
    return None


def _extract_hidden_tensor(output: Any) -> Any | None:
    if hasattr(output, "hidden_states"):
        return output.hidden_states
    if isinstance(output, tuple) and output:
        first = output[0]
        if hasattr(first, "shape"):
            return first
    if hasattr(output, "shape"):
        return output
    return None


def _replace_hidden_tensor(output: Any, new_hidden: Any) -> Any:
    if hasattr(output, "hidden_states"):
        output.hidden_states = new_hidden
        return output
    if isinstance(output, tuple) and output:
        return (new_hidden, *output[1:])
    return new_hidden


def init_activation_patching(model: Any) -> None:
    import torch

    layers = find_decoder_layers(model)
    if not layers:
        raise RuntimeError("No decoder layers found on model")
    if getattr(model, "_v2_activation_patch_initialized", False):
        return

    container = getattr(getattr(model, "model", None), "layers", None)
    if container is None:
        raise RuntimeError("No decoder layer container found on model")

    for layer_idx, layer in layers.items():
        if isinstance(layer, _ActivationPatchedLayer):
            continue
        wrapped = _ActivationPatchedLayerClass(model, layer_idx, layer)
        wrapped._activation_patch_original_layer = layer
        container[layer_idx] = wrapped

    model._v2_activation_patch_initialized = True
    model._v2_activation_patch_bank = {}
    model._v2_activation_patch_batch_specs = []
    model._v2_activation_patch_stats_by_req = {}
    model._v2_activation_patch_device = _infer_model_device(model) or torch.device("cpu")


def register_activation_patch_bank(model: Any, bank_payload: dict[int, dict[str, Any]]) -> dict[str, Any]:
    import torch

    if not getattr(model, "_v2_activation_patch_initialized", False):
        init_activation_patching(model)
    device = getattr(model, "_v2_activation_patch_device", None) or _infer_model_device(model)
    registered: dict[int, dict[str, Any]] = {}
    for raw_layer, layer_payload in bank_payload.items():
        layer = int(raw_layer)
        per_example: dict[str, Any] = {}
        for example_key, example_payload in dict(layer_payload).items():
            per_example[str(example_key)] = {
                "values": torch.as_tensor(example_payload["values"], device=device),
                "token_count": int(example_payload.get("token_count", 0)),
            }
        registered[layer] = per_example
    model._v2_activation_patch_bank = registered
    return {
        "layers": sorted(int(layer) for layer in registered),
        "examples": sum(len(items) for items in registered.values()),
    }


def set_batch_patch_specs(model: Any, batch_specs: list[dict[str, Any]]) -> None:
    model._v2_activation_patch_batch_specs = [dict(spec) for spec in batch_specs]


def clear_batch_patch_specs(model: Any) -> None:
    model._v2_activation_patch_batch_specs = []


def collect_patch_stats(model: Any, req_id: str | None = None) -> dict[Any, dict[str, Any]]:
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


def _record_patch_stats(owner_model: Any, *, req_id: str, layer_idx: int, stats: dict[str, Any]) -> None:
    stats_by_req = getattr(owner_model, "_v2_activation_patch_stats_by_req", None)
    if not isinstance(stats_by_req, dict):
        stats_by_req = {}
        owner_model._v2_activation_patch_stats_by_req = stats_by_req
    req_stats = stats_by_req.setdefault(str(req_id), {})
    req_stats[int(layer_idx)] = dict(stats)


def _patch_hidden_states_for_layer(
    hidden_states: Any,
    *,
    owner_model: Any,
    layer_idx: int,
    batch_spec: dict[str, Any],
) -> tuple[Any, dict[str, Any] | None]:
    import torch

    spec = _spec_from_payload(dict(batch_spec["patch_spec"]))
    query_positions = list(spec.query_positions)
    donor_positions = list(spec.donor_positions)
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

    bank = getattr(owner_model, "_v2_activation_patch_bank", {})
    layer_bank = bank.get(int(layer_idx))
    if not isinstance(layer_bank, dict):
        return hidden_states, {
            "layer": int(layer_idx),
            "status": "skipped",
            "reason": "missing_layer_bank",
        }
    donor_entry = layer_bank.get(spec.donor_example_key)
    if not isinstance(donor_entry, dict):
        return hidden_states, {
            "layer": int(layer_idx),
            "status": "skipped",
            "reason": "missing_donor_example",
        }
    donor_values = donor_entry["values"]
    if max(query_positions) >= int(hidden.shape[0]):
        return hidden_states, {
            "layer": int(layer_idx),
            "status": "skipped",
            "reason": f"query_positions_out_of_bounds:{int(hidden.shape[0])}",
        }
    if max(donor_positions) >= int(donor_values.shape[0]):
        return hidden_states, {
            "layer": int(layer_idx),
            "status": "skipped",
            "reason": f"donor_positions_out_of_bounds:{int(donor_values.shape[0])}",
        }

    patched = hidden.clone()
    patched_rows = donor_values[donor_positions].to(dtype=hidden.dtype)
    before = hidden[query_positions].to(torch.float32)
    after = patched_rows.to(torch.float32)
    patched[query_positions] = patched_rows
    stats = {
        "layer": int(layer_idx),
        "status": "ok",
        "query_positions": list(query_positions),
        "donor_positions": list(donor_positions),
        "token_count": len(query_positions),
        "delta_norm_raw": float(torch.linalg.norm(after - before).item()),
        "control_name": spec.control_name,
        "case_key": spec.case_key,
    }
    if is_1d:
        patched = patched.squeeze(0)
    return patched.to(original_dtype), stats


def _apply_layer_output_patching(
    *,
    owner_model: Any,
    layer_idx: int,
    output: Any,
) -> Any:
    batch_specs = getattr(owner_model, "_v2_activation_patch_batch_specs", None)
    if not isinstance(batch_specs, list) or not batch_specs:
        return output
    hidden = _extract_hidden_tensor(output)
    if hidden is None:
        return output
    patched_hidden = hidden
    for batch_spec in batch_specs:
        payload = batch_spec.get("patch_spec")
        if not isinstance(payload, dict):
            continue
        spec = _spec_from_payload(payload)
        if int(layer_idx) not in spec.target_layers:
            continue
        patched_hidden, stats = _patch_hidden_states_for_layer(
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
    return _replace_hidden_tensor(output, patched_hidden)


def _build_activation_patched_layer_class() -> type[Any]:
    import torch

    class ActivationPatchedLayer(torch.nn.Module):
        def __init__(self, owner_model: Any, layer_idx: int, wrapped_layer: Any) -> None:
            super().__init__()
            self.layer_idx = int(layer_idx)
            self.wrapped_layer = wrapped_layer
            self._owner_ref = weakref.ref(owner_model)

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            owner_model = self._owner_ref()
            output = self.wrapped_layer(*args, **kwargs)
            if owner_model is None:
                return output
            return _apply_layer_output_patching(
                owner_model=owner_model,
                layer_idx=self.layer_idx,
                output=output,
            )

    return ActivationPatchedLayer


try:
    _ActivationPatchedLayerClass = _build_activation_patched_layer_class()
except Exception:
    _ActivationPatchedLayerClass = _ActivationPatchedLayer


__all__ = [
    "clear_batch_patch_specs",
    "collect_patch_stats",
    "init_activation_patching",
    "register_activation_patch_bank",
    "set_batch_patch_specs",
]
