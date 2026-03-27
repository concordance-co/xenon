"""Worker-side market-section patching utilities for vLLM Qwen models.

This module follows the same pattern as :mod:`pipelines.interp.vllm_qwen3_moe`:
it installs wrapped forwards inside vLLM workers in eager mode, but instead of
recording router logits it applies tokenwise interventions to a targeted market
token span.

The primary use case is to causally intervene on the discovered Phase 17
``market_mean`` subspace during prefill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


PATCH_MODE_PROJECT_OUT = "project_out"
PATCH_MODE_ADD_DIRECTION = "add_direction"
PATCH_MODE_SWAP_MEAN = "swap_mean"
PATCH_MODE_SWAP_COMPONENTS = "swap_components"
PATCH_MODE_RANDOM_CONTROL = "random_control"
ALLOWED_PATCH_MODES = (
    PATCH_MODE_PROJECT_OUT,
    PATCH_MODE_ADD_DIRECTION,
    PATCH_MODE_SWAP_MEAN,
    PATCH_MODE_SWAP_COMPONENTS,
    PATCH_MODE_RANDOM_CONTROL,
)


@dataclass(slots=True)
class MarketPatchSpec:
    mode: str
    target_layers: tuple[int, ...]
    token_span: tuple[int, int]
    strength: float = 1.0
    component_indices_by_layer: dict[int, tuple[int, ...]] = field(default_factory=dict)
    direction_weights_by_layer: dict[int, np.ndarray] = field(default_factory=dict)
    donor_mean_by_layer: dict[int, np.ndarray] = field(default_factory=dict)
    random_seed: int = 0
    match_projected_norm: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ALLOWED_PATCH_MODES:
            raise ValueError(f"Unsupported patch mode: {self.mode}")
        self.target_layers = tuple(int(layer) for layer in self.target_layers)
        if not self.target_layers:
            raise ValueError("target_layers must not be empty")
        start, end = int(self.token_span[0]), int(self.token_span[1])
        if start < 0 or end <= start:
            raise ValueError(f"Invalid token_span={self.token_span}")
        self.token_span = (start, end)
        self.strength = float(self.strength)
        self.component_indices_by_layer = {
            int(layer): tuple(int(index) for index in indices)
            for layer, indices in self.component_indices_by_layer.items()
        }
        self.direction_weights_by_layer = {
            int(layer): np.asarray(weights, dtype=np.float32).copy()
            for layer, weights in self.direction_weights_by_layer.items()
        }
        self.donor_mean_by_layer = {
            int(layer): np.asarray(mean, dtype=np.float32).copy()
            for layer, mean in self.donor_mean_by_layer.items()
        }
        self.random_seed = int(self.random_seed)
        self.match_projected_norm = bool(self.match_projected_norm)

        if self.mode == PATCH_MODE_ADD_DIRECTION and not self.direction_weights_by_layer:
            raise ValueError("add_direction requires direction_weights_by_layer")
        if self.mode in {PATCH_MODE_SWAP_MEAN, PATCH_MODE_SWAP_COMPONENTS} and not self.donor_mean_by_layer:
            raise ValueError(f"{self.mode} requires donor_mean_by_layer")

    def to_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "target_layers": [int(layer) for layer in self.target_layers],
            "token_span": [int(self.token_span[0]), int(self.token_span[1])],
            "strength": float(self.strength),
            "component_indices_by_layer": {
                str(layer): [int(index) for index in indices]
                for layer, indices in self.component_indices_by_layer.items()
            },
            "direction_weights_by_layer": {
                str(layer): np.asarray(weights, dtype=np.float32).tolist()
                for layer, weights in self.direction_weights_by_layer.items()
            },
            "donor_mean_by_layer": {
                str(layer): np.asarray(mean, dtype=np.float32).tolist()
                for layer, mean in self.donor_mean_by_layer.items()
            },
            "random_seed": int(self.random_seed),
            "match_projected_norm": bool(self.match_projected_norm),
        }


def find_decoder_layers(model: Any) -> dict[int, Any]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return {}
    return {int(idx): layer for idx, layer in enumerate(layers)}


def _infer_model_device(model: Any) -> Any:
    layers = find_decoder_layers(model)
    for layer in layers.values():
        for attr_name in ("input_layernorm", "post_attention_layernorm", "self_attn", "mlp"):
            module = getattr(layer, attr_name, None)
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


def _safe_scale(scale: Any) -> Any:
    import torch

    return torch.where(scale == 0, torch.ones_like(scale), scale)


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


def _orthonormalize_rows(rows: Any) -> Any:
    import torch

    if rows.ndim != 2:
        raise ValueError(f"Expected 2D rows, got {rows.shape}")
    if rows.shape[0] == 0:
        return rows
    q, _ = torch.linalg.qr(rows.T, mode="reduced")
    return q.T.contiguous()


def _random_orthogonal_rows(
    *,
    target_rows: Any,
    num_rows: int,
    dim: int,
    seed: int,
    device: Any,
    dtype: Any,
) -> Any:
    import torch

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
        if rand.shape[0] < num_rows:
            attempts += 1
            continue
        ortho = _orthonormalize_rows(rand[:num_rows])
        if ortho.shape[0] == num_rows:
            return ortho.to(dtype=torch.float32)
        attempts += 1
    raise RuntimeError("Failed to sample random orthogonal control rows")


def _selected_component_rows(basis_layer: dict[str, Any], spec: MarketPatchSpec, layer_idx: int) -> Any:
    components = basis_layer["components"]
    component_indices = spec.component_indices_by_layer.get(layer_idx)
    if not component_indices:
        return components
    valid = [index for index in component_indices if 0 <= index < components.shape[0]]
    if not valid:
        return components[:0]
    return components[valid]


def _patch_hidden_states_for_layer(
    hidden_states: Any,
    *,
    basis_layer: dict[str, Any],
    spec: MarketPatchSpec,
    layer_idx: int,
) -> tuple[Any, dict[str, Any] | None]:
    import torch

    original_dtype = hidden_states.dtype
    is_1d = hidden_states.ndim == 1
    if is_1d:
        hidden_states = hidden_states.unsqueeze(0)
    if hidden_states.ndim != 2:
        return (
            hidden_states.squeeze(0) if is_1d else hidden_states,
            {
                "layer": int(layer_idx),
                "mode": spec.mode,
                "token_span": [int(spec.token_span[0]), int(spec.token_span[1])],
                "status": "skipped",
                "reason": f"unsupported_hidden_ndim:{hidden_states.ndim}",
            },
        )

    start, end = spec.token_span
    if start >= hidden_states.shape[0] or end > hidden_states.shape[0]:
        return (
            hidden_states.squeeze(0) if is_1d else hidden_states,
            {
                "layer": int(layer_idx),
                "mode": spec.mode,
                "token_span": [int(start), int(end)],
                "status": "skipped",
                "reason": f"token_span_out_of_bounds:{hidden_states.shape[0]}",
            },
        )

    hidden_f32 = hidden_states.to(torch.float32)
    section = hidden_f32[start:end]
    if section.numel() == 0:
        return (
            hidden_states.squeeze(0) if is_1d else hidden_states,
            {
                "layer": int(layer_idx),
                "mode": spec.mode,
                "token_span": [int(start), int(end)],
                "status": "skipped",
                "reason": "empty_section",
            },
        )

    mean = basis_layer["mean"]
    scale = basis_layer["scale"]
    components = basis_layer["components"]
    safe_scale = basis_layer["safe_scale"]

    mu = section.mean(dim=0)
    centered_std = (mu - mean) / safe_scale

    delta_std = None
    selected_rows = _selected_component_rows(basis_layer, spec, layer_idx)
    selected_coeff_before = None
    selected_proj_norm = 0.0

    if selected_rows.numel() > 0:
        selected_coeff_before = centered_std @ selected_rows.T
        selected_projected_std = selected_coeff_before @ selected_rows
        selected_proj_norm = float(torch.linalg.norm(selected_projected_std).item())
    else:
        selected_projected_std = torch.zeros_like(centered_std)

    if spec.mode == PATCH_MODE_PROJECT_OUT:
        delta_std = -float(spec.strength) * selected_projected_std
    elif spec.mode == PATCH_MODE_ADD_DIRECTION:
        weights = spec.direction_weights_by_layer.get(layer_idx)
        if weights is None:
            return (
                hidden_states.squeeze(0) if is_1d else hidden_states,
                {
                    "layer": int(layer_idx),
                    "mode": spec.mode,
                    "token_span": [int(start), int(end)],
                    "status": "skipped",
                    "reason": "missing_direction_weights",
                },
            )
        weights_t = torch.as_tensor(weights, device=hidden_f32.device, dtype=torch.float32)
        if weights_t.ndim != 1:
            raise ValueError(f"direction weights must be 1D, got {weights_t.shape}")
        if weights_t.shape[0] > components.shape[0]:
            raise ValueError(
                f"direction weights length {weights_t.shape[0]} exceeds components "
                f"{components.shape[0]} for layer {layer_idx}"
            )
        direction_std = weights_t @ components[: weights_t.shape[0]]
        delta_std = float(spec.strength) * direction_std
    elif spec.mode == PATCH_MODE_SWAP_MEAN:
        donor_mean = spec.donor_mean_by_layer.get(layer_idx)
        if donor_mean is None:
            return (
                hidden_states.squeeze(0) if is_1d else hidden_states,
                {
                    "layer": int(layer_idx),
                    "mode": spec.mode,
                    "token_span": [int(start), int(end)],
                    "status": "skipped",
                    "reason": "missing_donor_mean",
                },
            )
        donor_mean_t = torch.as_tensor(donor_mean, device=hidden_f32.device, dtype=torch.float32)
        if donor_mean_t.shape != mu.shape:
            raise ValueError(
                f"donor_mean shape mismatch for layer {layer_idx}: "
                f"{donor_mean_t.shape} vs {mu.shape}"
            )
        delta_raw = float(spec.strength) * (donor_mean_t - mu)
        hidden_f32[start:end] = section + delta_raw
        patched_mu = hidden_f32[start:end].mean(dim=0)
        stats = {
            "layer": int(layer_idx),
            "mode": spec.mode,
            "token_span": [int(start), int(end)],
            "delta_norm_raw": float(torch.linalg.norm(delta_raw).item()),
            "mean_norm_before": float(torch.linalg.norm(mu).item()),
            "mean_norm_after": float(torch.linalg.norm(patched_mu).item()),
            "mean_std_norm_before": float(torch.linalg.norm(centered_std).item()),
            "mean_std_norm_after": float(torch.linalg.norm((patched_mu - mean) / safe_scale).item()),
            "strength": float(spec.strength),
        }
        patched = hidden_f32.to(original_dtype)
        return (patched.squeeze(0) if is_1d else patched), stats
    elif spec.mode == PATCH_MODE_SWAP_COMPONENTS:
        donor_mean = spec.donor_mean_by_layer.get(layer_idx)
        if donor_mean is None:
            return (
                hidden_states.squeeze(0) if is_1d else hidden_states,
                {
                    "layer": int(layer_idx),
                    "mode": spec.mode,
                    "token_span": [int(start), int(end)],
                    "status": "skipped",
                    "reason": "missing_donor_mean",
                },
            )
        donor_mean_t = torch.as_tensor(donor_mean, device=hidden_f32.device, dtype=torch.float32)
        if donor_mean_t.shape != mu.shape:
            raise ValueError(
                f"donor_mean shape mismatch for layer {layer_idx}: "
                f"{donor_mean_t.shape} vs {mu.shape}"
            )
        donor_centered_std = (donor_mean_t - mean) / safe_scale
        donor_selected_coeff = (
            donor_centered_std @ selected_rows.T
            if selected_rows.numel() > 0
            else torch.empty((0,), device=hidden_f32.device)
        )
        donor_selected_projected_std = (
            donor_selected_coeff @ selected_rows
            if selected_rows.numel() > 0
            else torch.zeros_like(centered_std)
        )
        delta_std = float(spec.strength) * (donor_selected_projected_std - selected_projected_std)
    elif spec.mode == PATCH_MODE_RANDOM_CONTROL:
        num_rows = max(1, int(selected_rows.shape[0]))
        random_rows = _random_orthogonal_rows(
            target_rows=selected_rows,
            num_rows=num_rows,
            dim=centered_std.shape[0],
            seed=spec.random_seed + layer_idx,
            device=hidden_f32.device,
            dtype=torch.float32,
        )
        random_coeff = centered_std @ random_rows.T
        random_projected_std = random_coeff @ random_rows
        if spec.match_projected_norm:
            random_norm = float(torch.linalg.norm(random_projected_std).item())
            if random_norm > 1e-8 and selected_proj_norm > 0.0:
                random_projected_std = random_projected_std * (selected_proj_norm / random_norm)
        delta_std = -float(spec.strength) * random_projected_std
    else:
        raise ValueError(f"Unsupported patch mode: {spec.mode}")

    delta_raw = delta_std * scale
    hidden_f32[start:end] = section + delta_raw

    patched_section = hidden_f32[start:end]
    patched_mu = patched_section.mean(dim=0)
    patched_centered_std = (patched_mu - mean) / safe_scale
    selected_coeff_after = (
        patched_centered_std @ selected_rows.T if selected_rows.numel() > 0 else torch.empty((0,), device=hidden_f32.device)
    )

    stats = {
        "layer": int(layer_idx),
        "mode": spec.mode,
        "token_span": [int(start), int(end)],
        "delta_norm_raw": float(torch.linalg.norm(delta_raw).item()),
        "delta_norm_std": float(torch.linalg.norm(delta_std).item()),
        "mean_norm_before": float(torch.linalg.norm(mu).item()),
        "mean_norm_after": float(torch.linalg.norm(patched_mu).item()),
        "mean_std_norm_before": float(torch.linalg.norm(centered_std).item()),
        "mean_std_norm_after": float(torch.linalg.norm(patched_centered_std).item()),
        "selected_proj_norm_before": float(selected_proj_norm),
        "selected_coeff_before": (
            selected_coeff_before.detach().cpu().tolist()
            if selected_coeff_before is not None
            else []
        ),
        "selected_coeff_after": selected_coeff_after.detach().cpu().tolist(),
        "strength": float(spec.strength),
    }

    patched = hidden_f32.to(original_dtype)
    return (patched.squeeze(0) if is_1d else patched), stats


def _make_patched_forward(model: Any, layer_idx: int, layer: Any) -> Any:
    original_forward = layer.forward

    def patched_forward(*args: Any, **kwargs: Any) -> Any:
        output = original_forward(*args, **kwargs)
        batch_specs = getattr(model, "_market_patch_batch_specs", None)
        if isinstance(batch_specs, list) and batch_specs:
            basis = getattr(model, "_market_patch_basis", None)
            if not isinstance(basis, dict) or layer_idx not in basis:
                return output
            hidden = _extract_hidden_tensor(output)
            if hidden is None:
                return output
            patched_hidden = hidden
            stats_by_req = getattr(model, "_market_patch_last_stats_by_req", None)
            if not isinstance(stats_by_req, dict):
                stats_by_req = {}
                model._market_patch_last_stats_by_req = stats_by_req
            coverage_by_req = getattr(model, "_market_patch_coverage_by_req", None)
            if not isinstance(coverage_by_req, dict):
                coverage_by_req = {}
                model._market_patch_coverage_by_req = coverage_by_req
            for batch_spec in batch_specs:
                req_id = str(batch_spec.get("req_id"))
                payload = batch_spec.get("patch_spec")
                if not isinstance(payload, dict):
                    continue
                spec = MarketPatchSpec(**payload)
                if layer_idx not in spec.target_layers:
                    continue
                patched_hidden, stats = _patch_hidden_states_for_layer(
                    patched_hidden,
                    basis_layer=basis[layer_idx],
                    spec=spec,
                    layer_idx=layer_idx,
                )
                if stats is not None:
                    stats["req_id"] = req_id
                    for meta_key in (
                        "target_span",
                        "chunk_abs_span",
                        "overlap_abs_span",
                        "query_span",
                        "prefill_chunk_len",
                    ):
                        if meta_key in batch_spec:
                            stats[meta_key] = batch_spec[meta_key]
                    layer_coverage = coverage_by_req.setdefault(req_id, {}).setdefault(int(layer_idx), [])
                    overlap_abs_span = batch_spec.get("overlap_abs_span")
                    if (
                        stats.get("status") != "skipped"
                        and isinstance(overlap_abs_span, (list, tuple))
                        and len(overlap_abs_span) == 2
                    ):
                        start = int(overlap_abs_span[0])
                        end = int(overlap_abs_span[1])
                        if end > start:
                            layer_coverage.append((start, end))
                            layer_coverage.sort()
                            merged: list[tuple[int, int]] = []
                            for span_start, span_end in layer_coverage:
                                if not merged or span_start > merged[-1][1]:
                                    merged.append((span_start, span_end))
                                else:
                                    merged[-1] = (merged[-1][0], max(merged[-1][1], span_end))
                            coverage_by_req[req_id][int(layer_idx)] = merged
                            layer_coverage = merged
                    existing_stats = stats_by_req.setdefault(req_id, {}).get(int(layer_idx))
                    if isinstance(existing_stats, dict):
                        merged_stats = dict(existing_stats)
                        merged_stats.update(stats)
                    else:
                        merged_stats = dict(stats)
                    if layer_coverage:
                        merged_stats["covered_abs_spans"] = [
                            [int(span_start), int(span_end)]
                            for span_start, span_end in layer_coverage
                        ]
                        target_span = merged_stats.get("target_span")
                        if isinstance(target_span, (list, tuple)) and len(target_span) == 2:
                            target_start = int(target_span[0])
                            target_end = int(target_span[1])
                            covered = 0
                            for span_start, span_end in layer_coverage:
                                covered += max(
                                    0,
                                    min(span_end, target_end) - max(span_start, target_start),
                                )
                            total = max(1, target_end - target_start)
                            merged_stats["covered_abs_tokens"] = int(covered)
                            merged_stats["target_abs_tokens"] = int(target_end - target_start)
                            merged_stats["coverage_fraction"] = float(covered / total)
                    stats_by_req[req_id][int(layer_idx)] = merged_stats
            return _replace_hidden_tensor(output, patched_hidden)

        spec = getattr(model, "_market_patch_spec", None)
        if spec is None or layer_idx not in spec.target_layers:
            return output
        applied_layers = getattr(model, "_market_patch_applied_layers", None)
        if isinstance(applied_layers, set) and layer_idx in applied_layers:
            return output
        basis = getattr(model, "_market_patch_basis", None)
        if not isinstance(basis, dict) or layer_idx not in basis:
            return output
        hidden = _extract_hidden_tensor(output)
        if hidden is None:
            patch_stats = getattr(model, "_market_patch_last_stats", None)
            if not isinstance(patch_stats, dict):
                patch_stats = {}
                model._market_patch_last_stats = patch_stats
            patch_stats[int(layer_idx)] = {
                "layer": int(layer_idx),
                "mode": spec.mode,
                "token_span": [int(spec.token_span[0]), int(spec.token_span[1])],
                "status": "skipped",
                "reason": f"no_hidden_tensor:{type(output).__name__}",
            }
            return output
        patched_hidden, stats = _patch_hidden_states_for_layer(
            hidden,
            basis_layer=basis[layer_idx],
            spec=spec,
            layer_idx=layer_idx,
        )
        if stats is not None:
            patch_stats = getattr(model, "_market_patch_last_stats", None)
            if not isinstance(patch_stats, dict):
                patch_stats = {}
                model._market_patch_last_stats = patch_stats
            patch_stats[int(layer_idx)] = stats
            if stats.get("status") != "skipped":
                applied_layers = getattr(model, "_market_patch_applied_layers", None)
                if not isinstance(applied_layers, set):
                    applied_layers = set()
                    model._market_patch_applied_layers = applied_layers
                applied_layers.add(int(layer_idx))
        return _replace_hidden_tensor(output, patched_hidden)

    return patched_forward


def init_market_patching(model: Any) -> None:
    layers = find_decoder_layers(model)
    if not layers:
        raise RuntimeError("No decoder layers found on model")
    if getattr(model, "_market_patch_initialized", False):
        return

    for layer_idx, layer in layers.items():
        layer._market_patch_original_forward = layer.forward
        layer.forward = _make_patched_forward(model, layer_idx, layer)

    model._market_patch_initialized = True
    model._market_patch_basis = {}
    model._market_patch_spec = None
    model._market_patch_last_stats = {}
    model._market_patch_applied_layers = set()
    model._market_patch_batch_specs = []
    model._market_patch_last_stats_by_req = {}
    model._market_patch_coverage_by_req = {}


def register_patch_basis(model: Any, basis_payload: dict[int, dict[str, Any]]) -> dict[str, Any]:
    import torch

    if not getattr(model, "_market_patch_initialized", False):
        init_market_patching(model)

    device = _infer_model_device(model)
    registered: dict[int, dict[str, Any]] = {}
    for layer_idx, payload in basis_payload.items():
        layer = int(layer_idx)
        mean = torch.as_tensor(payload["mean"], device=device, dtype=torch.float32)
        scale = torch.as_tensor(payload["scale"], device=device, dtype=torch.float32)
        components = torch.as_tensor(payload["components"], device=device, dtype=torch.float32)
        safe_scale = _safe_scale(scale)
        row_norms = torch.linalg.norm(components, dim=1, keepdim=True)
        components = components / torch.where(row_norms == 0, torch.ones_like(row_norms), row_norms)
        registered[layer] = {
            "mean": mean,
            "scale": scale,
            "safe_scale": safe_scale,
            "components": components,
            "named_components": {
                str(name): int(index)
                for name, index in dict(payload.get("named_components", {})).items()
            },
        }
    model._market_patch_basis = registered
    return {
        "num_layers": len(registered),
        "layers": sorted(int(layer) for layer in registered),
    }


def set_patch_spec(model: Any, patch_spec: MarketPatchSpec | dict[str, Any]) -> dict[str, Any]:
    spec = patch_spec if isinstance(patch_spec, MarketPatchSpec) else MarketPatchSpec(**patch_spec)
    model._market_patch_spec = spec
    model._market_patch_last_stats = {}
    model._market_patch_applied_layers = set()
    return {
        "mode": spec.mode,
        "target_layers": list(spec.target_layers),
        "token_span": list(spec.token_span),
    }


def clear_patch_spec(model: Any) -> None:
    model._market_patch_spec = None
    model._market_patch_applied_layers = set()


def set_batch_patch_specs(model: Any, batch_specs: list[dict[str, Any]]) -> None:
    model._market_patch_batch_specs = [dict(spec) for spec in batch_specs]
    if not isinstance(getattr(model, "_market_patch_last_stats_by_req", None), dict):
        model._market_patch_last_stats_by_req = {}
    if not isinstance(getattr(model, "_market_patch_coverage_by_req", None), dict):
        model._market_patch_coverage_by_req = {}


def clear_batch_patch_specs(model: Any) -> None:
    model._market_patch_batch_specs = []


def collect_patch_stats(model: Any, req_id: str | None = None) -> dict[Any, dict[str, Any]]:
    if req_id is not None:
        stats_by_req = getattr(model, "_market_patch_last_stats_by_req", {})
        req_key = str(req_id)
        req_stats = stats_by_req.get(req_key)
        if req_stats is None and "-" in req_key:
            req_stats = stats_by_req.get(req_key.rsplit("-", 1)[0])
        if req_stats is None:
            for candidate_key, candidate_stats in stats_by_req.items():
                if candidate_key.rsplit("-", 1)[0] == req_key:
                    req_stats = candidate_stats
                    break
        if req_stats is None:
            req_stats = {}
        return {
            int(layer): dict(payload)
            for layer, payload in req_stats.items()
        }
    stats = getattr(model, "_market_patch_last_stats", {})
    return {
        int(layer): dict(payload)
        for layer, payload in stats.items()
    }


def restore_original_forwards(model: Any) -> None:
    layers = find_decoder_layers(model)
    for layer in layers.values():
        original = getattr(layer, "_market_patch_original_forward", None)
        if original is not None:
            layer.forward = original
            del layer._market_patch_original_forward
    for attr in (
        "_market_patch_initialized",
        "_market_patch_basis",
        "_market_patch_spec",
        "_market_patch_last_stats",
        "_market_patch_applied_layers",
        "_market_patch_batch_specs",
        "_market_patch_last_stats_by_req",
        "_market_patch_coverage_by_req",
    ):
        if hasattr(model, attr):
            delattr(model, attr)
