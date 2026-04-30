"""Shared base helpers for activation patching."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch


_MAX_BATCH_PATCH_SLOTS = 64
_DEFAULT_COMPILED_PATCH_MAX_COMPONENTS = 8


def in_compiled_region() -> bool:
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    if callable(is_compiling):
        try:
            return bool(is_compiling())
        except Exception:
            return False
    dynamo = getattr(torch, "_dynamo", None)
    is_dynamo_compiling = getattr(dynamo, "is_compiling", None)
    if callable(is_dynamo_compiling):
        try:
            return bool(is_dynamo_compiling())
        except Exception:
            return False
    return False


def debug_mode_enabled(*labels: str) -> bool:
    if in_compiled_region():
        return False
    raw = str(os.getenv("XENON_ACTIVATION_PATCH_DEBUG", "") or "").strip()
    if not raw:
        return False
    enabled = {item.strip() for item in raw.split(",") if item.strip()}
    return any(label in enabled for label in labels)


def debug_log(label: str, **fields: Any) -> None:
    if not debug_mode_enabled("log", label):
        return
    payload = " ".join(f"{key}={value!r}" for key, value in sorted(fields.items()))
    print(f"[activation-patch-debug] {label} {payload}".rstrip())


def debug_panic(label: str, **fields: Any) -> None:
    if not debug_mode_enabled("panic", label):
        return
    payload = " ".join(f"{key}={value!r}" for key, value in sorted(fields.items()))
    raise RuntimeError(f"activation-patch panic [{label}] {payload}".rstrip())


@dataclass(slots=True)
class RequestPatchSpec:
    operator: str
    target_layers: tuple[int, ...]
    donor_example_key: str
    donor_positions: tuple[int, ...]
    target_read_positions: tuple[int, ...]
    query_positions: tuple[int, ...]
    query_span: tuple[int, ...] = ()
    target_abs_positions: tuple[int, ...] = ()
    covered_abs_positions: tuple[int, ...] = ()
    covered_abs_spans: tuple[tuple[int, int], ...] = ()
    phase_counts: tuple[tuple[str, int], ...] = ()
    target_policy: dict[str, Any] | None = None
    rowwise: bool = False
    example_key: str = ""
    case_key: str = ""
    control_name: str = ""
    centroid_name: str = ""
    transport: str = "replace"
    path_edges: tuple[tuple[int, int, float], ...] = ()
    source_layer_map: tuple[tuple[int, int], ...] = ()
    component_indices_by_layer: tuple[tuple[int, tuple[int, ...]], ...] = ()
    strength: float = 1.0
    random_seed: int = 0
    match_projected_norm: bool = True

    def __post_init__(self) -> None:
        self.operator = str(self.operator or "interchange")
        self.target_layers = tuple(int(layer) for layer in self.target_layers)
        self.donor_example_key = str(self.donor_example_key)
        self.donor_positions = tuple(int(pos) for pos in self.donor_positions)
        self.target_read_positions = tuple(int(pos) for pos in self.target_read_positions)
        self.query_positions = tuple(int(pos) for pos in self.query_positions)
        if self.query_span:
            if len(self.query_span) != 2:
                raise ValueError("query_span must be a [start, end] pair")
            self.query_span = (int(self.query_span[0]), int(self.query_span[1]))
        self.target_abs_positions = tuple(int(pos) for pos in self.target_abs_positions)
        self.covered_abs_positions = tuple(int(pos) for pos in self.covered_abs_positions)
        self.covered_abs_spans = tuple((int(start), int(end)) for start, end in self.covered_abs_spans)
        self.phase_counts = tuple((str(name), int(count)) for name, count in self.phase_counts)
        self.target_policy = dict(self.target_policy or {})
        self.rowwise = bool(self.rowwise)
        self.example_key = str(self.example_key)
        self.case_key = str(self.case_key)
        self.control_name = str(self.control_name)
        self.centroid_name = str(self.centroid_name)
        self.transport = str(self.transport or "replace")
        self.path_edges = tuple(
            (int(source_layer), int(write_layer), float(weight))
            for source_layer, write_layer, weight in self.path_edges
        )
        self.source_layer_map = tuple(
            (int(write_layer), int(source_layer))
            for write_layer, source_layer in self.source_layer_map
        )
        self.component_indices_by_layer = tuple(
            (int(layer), tuple(int(index) for index in indices))
            for layer, indices in self.component_indices_by_layer
        )
        self.strength = float(self.strength)
        self.random_seed = int(self.random_seed)
        self.match_projected_norm = bool(self.match_projected_norm)

    def is_interchange(self) -> bool:
        return self.operator == "interchange"

    def is_residual_path(self) -> bool:
        return self.operator == "residual_path"

    def uses_subspace(self) -> bool:
        return self.operator in {
            "project_out",
            "random_control",
            "add_direction",
            "swap_components",
        }

    def source_layer_for(self, write_layer: int) -> int:
        for candidate_write_layer, source_layer in self.source_layer_map:
            if int(candidate_write_layer) == int(write_layer):
                return int(source_layer)
        return int(write_layer)

    def token_count(self) -> int:
        if self.query_span:
            start, end = self.query_span
            return max(0, int(end) - int(start))
        return len(self.query_positions)

    def selected_components_for(self, write_layer: int) -> tuple[int, ...]:
        for candidate_write_layer, indices in self.component_indices_by_layer:
            if int(candidate_write_layer) == int(write_layer):
                return tuple(int(index) for index in indices)
        return ()


def spec_from_payload(payload: dict[str, Any]) -> RequestPatchSpec:
    query_positions = payload.get("query_positions")
    query_span = payload.get("query_span", ())
    if query_positions is None and not query_span:
        query_positions = payload.get("target_positions", ())
    component_indices_payload = dict(payload.get("component_indices_by_layer", {}))
    source_layer_payload = dict(payload.get("source_layer_map", {}))
    return RequestPatchSpec(
        operator=str(payload.get("operator") or "interchange"),
        target_layers=tuple(int(layer) for layer in payload.get("target_layers", ())),
        donor_example_key=str(payload.get("donor_example_key") or ""),
        donor_positions=tuple(int(pos) for pos in payload.get("donor_positions", ())),
        target_read_positions=tuple(int(pos) for pos in payload.get("target_read_positions", ())),
        query_positions=tuple(int(pos) for pos in (query_positions or ())),
        query_span=tuple(int(pos) for pos in query_span) if query_span else (),
        target_abs_positions=tuple(int(pos) for pos in payload.get("target_abs_positions", ())),
        covered_abs_positions=tuple(int(pos) for pos in payload.get("covered_abs_positions", ())),
        covered_abs_spans=tuple(
            (int(span[0]), int(span[1]))
            for span in payload.get("covered_abs_spans", ())
            if isinstance(span, (list, tuple)) and len(span) == 2
        ),
        phase_counts=tuple(
            (str(key), int(value))
            for key, value in dict(payload.get("phase_counts", {})).items()
        ),
        target_policy=dict(payload.get("target_policy") or {}),
        rowwise=bool(payload.get("rowwise", False)),
        example_key=str(payload.get("example_key") or ""),
        case_key=str(payload.get("case_key") or ""),
        control_name=str(payload.get("control_name") or ""),
        centroid_name=str(payload.get("centroid_name") or ""),
        transport=str(payload.get("transport") or "replace"),
        path_edges=tuple(
            (
                int(edge.get("source_layer")),
                int(edge.get("write_layer")),
                float(edge.get("weight", 1.0)),
            )
            for edge in payload.get("path_edges", ())
            if isinstance(edge, dict)
        ),
        source_layer_map=tuple(
            (int(write_layer), int(source_layer))
            for write_layer, source_layer in source_layer_payload.items()
        ),
        component_indices_by_layer=tuple(
            (int(layer), tuple(int(index) for index in indices))
            for layer, indices in component_indices_payload.items()
        ),
        strength=float(payload.get("strength", 1.0)),
        random_seed=int(payload.get("random_seed", 0)),
        match_projected_norm=bool(payload.get("match_projected_norm", True)),
    )


def find_decoder_layers(model: Any) -> dict[int, Any]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return {}
    return {int(idx): layer for idx, layer in enumerate(layers)}


class ActivationPatchedLayer:
    pass


def unwrap_model(model: Any) -> Any:
    current = model
    seen: set[int] = set()
    while True:
        unwrap = getattr(current, "unwrap", None)
        if not callable(unwrap):
            return current
        current_id = id(current)
        if current_id in seen:
            return current
        seen.add(current_id)
        try:
            next_model = unwrap()
        except Exception:
            return current
        if next_model is current:
            return current
        current = next_model


def infer_model_device(model: Any) -> Any:
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


def infer_layer_hidden_dim(layer: Any) -> int:
    candidate = getattr(layer, "wrapped_layer", layer)
    for attr_name in ("input_layernorm", "post_attention_layernorm", "self_attn", "mlp"):
        module = getattr(candidate, attr_name, None)
        if module is None:
            continue
        weight = getattr(module, "weight", None)
        if weight is not None and hasattr(weight, "shape") and len(weight.shape) >= 1:
            return int(weight.shape[0])
        parameters = getattr(module, "parameters", None)
        if callable(parameters):
            try:
                first = next(parameters())
            except StopIteration:
                continue
            if hasattr(first, "shape") and len(first.shape) >= 1:
                return int(first.shape[0])
    raise RuntimeError("Could not infer hidden dimension for decoder layer")


def contiguous_token_span(positions: tuple[int, ...] | list[int]) -> tuple[int, int] | None:
    if not positions:
        return None
    start = int(positions[0])
    prev = start
    for raw_pos in positions[1:]:
        pos = int(raw_pos)
        if pos != prev + 1:
            return None
        prev = pos
    return (start, prev + 1)


def extract_hidden_tensor(output: Any) -> Any | None:
    if hasattr(output, "hidden_states"):
        return output.hidden_states
    if isinstance(output, tuple) and output:
        first = output[0]
        if hasattr(first, "shape"):
            return first
    if hasattr(output, "shape"):
        return output
    return None


def replace_hidden_tensor(output: Any, new_hidden: Any) -> Any:
    if hasattr(output, "hidden_states"):
        output.hidden_states = new_hidden
        return output
    if isinstance(output, tuple) and output:
        return (new_hidden, *output[1:])
    return new_hidden


def selected_component_rows(
    *,
    layer_payload: dict[str, Any],
    spec: RequestPatchSpec,
    layer_idx: int,
) -> Any:
    components = layer_payload["components"]
    selected = list(spec.selected_components_for(int(layer_idx)))
    if not selected:
        return components
    valid = [int(index) for index in selected if 0 <= int(index) < int(components.shape[0])]
    if not valid:
        return components[:0]
    return components[valid]


def random_orthogonal_rows(
    *,
    target_rows: Any,
    num_rows: int,
    dim: int,
    seed: int,
    device: Any,
    dtype: Any,
) -> Any:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    attempts = 0
    while attempts < 8:
        rand = torch.randn((num_rows, dim), generator=generator, device=device, dtype=torch.float32)
        if target_rows.numel() > 0:
            rand = rand - (rand @ target_rows.T) @ target_rows
        row_norm = torch.linalg.norm(rand, dim=1)
        keep = row_norm > 1e-6
        rand = rand[keep]
        if rand.shape[0] < int(num_rows):
            attempts += 1
            continue
        q, _ = torch.linalg.qr(rand[:num_rows].T, mode="reduced")
        ortho = q.T.contiguous()
        if ortho.shape[0] == int(num_rows):
            return ortho.to(dtype=dtype)
        attempts += 1
    raise RuntimeError("Failed to sample random orthogonal control rows")


__all__ = [
    "ActivationPatchedLayer",
    "RequestPatchSpec",
    "_DEFAULT_COMPILED_PATCH_MAX_COMPONENTS",
    "_MAX_BATCH_PATCH_SLOTS",
    "contiguous_token_span",
    "debug_log",
    "debug_mode_enabled",
    "debug_panic",
    "extract_hidden_tensor",
    "find_decoder_layers",
    "infer_layer_hidden_dim",
    "infer_model_device",
    "in_compiled_region",
    "random_orthogonal_rows",
    "replace_hidden_tensor",
    "selected_component_rows",
    "spec_from_payload",
    "unwrap_model",
]
