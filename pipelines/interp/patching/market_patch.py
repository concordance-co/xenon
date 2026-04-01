"""Worker-side market-section patching utilities for vLLM Qwen models.

This module follows the same pattern as :mod:`pipelines.interp.vllm_qwen3_moe`:
it installs wrapped forwards inside vLLM workers in eager mode, but instead of
recording router logits it applies tokenwise interventions to a targeted market
token span.

The primary use case is to causally intervene on the discovered Phase 17
``market_mean`` subspace during prefill.
"""

from __future__ import annotations

import functools
import inspect
import weakref
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pipelines.interp.patching.custom_op import (
    apply_market_patch_tensor,
    build_market_patch_hidden_states_op,
    patch_mode_to_id,
    summarize_market_patch_tensor,
    summarize_market_patch_tensor_tensors,
)


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

_MAX_BATCH_PATCH_STATS_SLOTS = 64
_DEFAULT_COMPILED_PATCH_MAX_COMPONENTS = 8


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


class _MarketPatchedLayer:  # concrete base swapped at runtime when torch is available
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


def _infer_layer_hidden_dim(layer: Any) -> int:
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
    custom_op: Any | None = None,
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

    selected_rows = _selected_component_rows(basis_layer, spec, layer_idx)
    direction_std = None
    donor_mean_t = None
    random_rows = None

    if spec.mode == PATCH_MODE_ADD_DIRECTION:
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
    elif spec.mode == PATCH_MODE_RANDOM_CONTROL:
        num_rows = max(1, int(selected_rows.shape[0]))
        random_rows = _random_orthogonal_rows(
            target_rows=selected_rows,
            num_rows=num_rows,
            dim=int(mean.shape[0]),
            seed=spec.random_seed + layer_idx,
            device=hidden_f32.device,
            dtype=torch.float32,
        )
    elif spec.mode != PATCH_MODE_PROJECT_OUT:
        raise ValueError(f"Unsupported patch mode: {spec.mode}")

    patched = None
    if custom_op is not None:
        try:
            patched = custom_op(
                hidden_states,
                mean,
                scale,
                safe_scale,
                selected_rows,
                torch.tensor([start, end], device=hidden_f32.device, dtype=torch.int32),
                patch_mode_to_id(spec.mode),
                float(spec.strength),
                direction_std=direction_std,
                donor_mean=donor_mean_t,
                random_rows=random_rows,
            )
        except Exception:
            patched = None

    if patched is None:
        patched, stats = apply_market_patch_tensor(
            hidden_states,
            mean=mean,
            scale=scale,
            safe_scale=safe_scale,
            selected_rows=selected_rows,
            token_span=(start, end),
            mode=spec.mode,
            strength=float(spec.strength),
            direction_std=direction_std,
            donor_mean=donor_mean_t,
            random_rows=random_rows,
            collect_stats=True,
        )
    else:
        stats = summarize_market_patch_tensor(
            hidden_states.to(torch.float32),
            patched.to(torch.float32),
            mean=mean,
            safe_scale=safe_scale,
            selected_rows=selected_rows,
            token_span=(start, end),
            strength=float(spec.strength),
        )
        patched = patched.to(original_dtype)

    stats.update(
        {
            "layer": int(layer_idx),
            "mode": spec.mode,
            "token_span": [int(start), int(end)],
        }
    )
    return (patched.squeeze(0) if is_1d else patched), stats


def _apply_noop_custom_op(
    hidden_states: Any,
    *,
    custom_op: Any | None,
) -> Any:
    import torch

    if custom_op is None:
        return hidden_states

    original_dtype = hidden_states.dtype
    is_1d = hidden_states.ndim == 1
    if is_1d:
        hidden_states = hidden_states.unsqueeze(0)
    if hidden_states.ndim != 2 or hidden_states.shape[0] <= 0:
        return hidden_states.squeeze(0) if is_1d else hidden_states

    hidden_dim = int(hidden_states.shape[-1])
    device = hidden_states.device

    try:
        patched = custom_op(
            hidden_states,
            torch.zeros((hidden_dim,), device=device, dtype=torch.float32),
            torch.ones((hidden_dim,), device=device, dtype=torch.float32),
            torch.ones((hidden_dim,), device=device, dtype=torch.float32),
            torch.empty((0, hidden_dim), device=device, dtype=torch.float32),
            torch.tensor([0, 1], device=device, dtype=torch.int32),
            patch_mode_to_id(PATCH_MODE_PROJECT_OUT),
            1.0,
        )
    except Exception:
        return hidden_states.squeeze(0) if is_1d else hidden_states

    patched = patched.to(original_dtype)
    return patched.squeeze(0) if is_1d else patched


def _ensure_batch_tensor_stats_buffers(
    model: Any,
    *,
    layer_idx: int,
    coeff_dim: int,
    device: Any,
) -> dict[str, Any]:
    import torch

    buffers = getattr(model, "_market_patch_batch_tensor_stats", None)
    if not isinstance(buffers, dict):
        buffers = {}
        model._market_patch_batch_tensor_stats = buffers
    layer_buffers = buffers.get(int(layer_idx))
    if (
        not isinstance(layer_buffers, dict)
        or layer_buffers["valid"].shape[0] != _MAX_BATCH_PATCH_STATS_SLOTS
        or layer_buffers["coeff_before"].shape[1] != int(coeff_dim)
    ):
        layer_buffers = {
            "valid": torch.zeros((_MAX_BATCH_PATCH_STATS_SLOTS,), device=device, dtype=torch.int32),
            "scalars": torch.zeros((_MAX_BATCH_PATCH_STATS_SLOTS, 8), device=device, dtype=torch.float32),
            "coeff_before": torch.zeros(
                (_MAX_BATCH_PATCH_STATS_SLOTS, int(coeff_dim)),
                device=device,
                dtype=torch.float32,
            ),
            "coeff_after": torch.zeros(
                (_MAX_BATCH_PATCH_STATS_SLOTS, int(coeff_dim)),
                device=device,
                dtype=torch.float32,
            ),
        }
        buffers[int(layer_idx)] = layer_buffers
    return layer_buffers


def _ensure_batch_runtime_state_buffers(
    model: Any,
    *,
    layer_idx: int,
    coeff_dim: int,
    hidden_dim: int,
    device: Any,
) -> dict[str, Any]:
    import torch

    buffers = getattr(model, "_market_patch_batch_runtime_state", None)
    if not isinstance(buffers, dict):
        buffers = {}
        model._market_patch_batch_runtime_state = buffers
    layer_buffers = buffers.get(int(layer_idx))
    if (
        not isinstance(layer_buffers, dict)
        or layer_buffers["active"].shape[0] != _MAX_BATCH_PATCH_STATS_SLOTS
        or layer_buffers["selected_rows"].shape[1] != int(coeff_dim)
        or layer_buffers["selected_rows"].shape[2] != int(hidden_dim)
    ):
        layer_buffers = {
            "active": torch.zeros((_MAX_BATCH_PATCH_STATS_SLOTS,), device=device, dtype=torch.int32),
            "row_counts": torch.zeros((_MAX_BATCH_PATCH_STATS_SLOTS,), device=device, dtype=torch.int32),
            "token_spans": torch.zeros(
                (_MAX_BATCH_PATCH_STATS_SLOTS, 2),
                device=device,
                dtype=torch.int32,
            ),
            "strengths": torch.zeros(
                (_MAX_BATCH_PATCH_STATS_SLOTS,),
                device=device,
                dtype=torch.float32,
            ),
            "donor_means": torch.zeros(
                (_MAX_BATCH_PATCH_STATS_SLOTS, int(hidden_dim)),
                device=device,
                dtype=torch.float32,
            ),
            "selected_rows": torch.zeros(
                (_MAX_BATCH_PATCH_STATS_SLOTS, int(coeff_dim), int(hidden_dim)),
                device=device,
                dtype=torch.float32,
            ),
        }
        buffers[int(layer_idx)] = layer_buffers
    return layer_buffers


def _reset_batch_tensor_stats(model: Any) -> None:
    buffers = getattr(model, "_market_patch_batch_tensor_stats", None)
    if not isinstance(buffers, dict):
        return
    for layer_buffers in buffers.values():
        if not isinstance(layer_buffers, dict):
            continue
        for tensor in layer_buffers.values():
            if hasattr(tensor, "zero_"):
                tensor.zero_()


def _reset_batch_runtime_state(model: Any) -> None:
    buffers = getattr(model, "_market_patch_batch_runtime_state", None)
    if not isinstance(buffers, dict):
        return
    for layer_buffers in buffers.values():
        if not isinstance(layer_buffers, dict):
            continue
        for tensor in layer_buffers.values():
            if hasattr(tensor, "zero_"):
                tensor.zero_()


def _load_batch_runtime_state(model: Any, batch_specs: list[dict[str, Any]]) -> None:
    import torch

    basis = getattr(model, "_market_patch_basis", None)
    if not isinstance(basis, dict):
        return

    _reset_batch_runtime_state(model)

    for slot_idx, batch_spec in enumerate(batch_specs[:_MAX_BATCH_PATCH_STATS_SLOTS]):
        payload = batch_spec.get("patch_spec")
        if not isinstance(payload, dict):
            continue
        spec = MarketPatchSpec(**payload)
        if spec.mode not in {PATCH_MODE_PROJECT_OUT, PATCH_MODE_SWAP_COMPONENTS}:
            raise NotImplementedError(
                "Compiled non-eager batch patching currently supports only project_out and swap_components"
            )
        for layer_idx in spec.target_layers:
            basis_layer = basis.get(int(layer_idx))
            if not isinstance(basis_layer, dict):
                continue
            selected_rows = _selected_component_rows(basis_layer, spec, int(layer_idx))
            runtime_buffers = model._market_patch_batch_runtime_state.get(int(layer_idx))
            if not isinstance(runtime_buffers, dict):
                continue
            capacity = int(runtime_buffers["selected_rows"].shape[1])
            if int(selected_rows.shape[0]) > capacity:
                raise ValueError(
                    f"selected_rows exceeds compiled capacity for layer {layer_idx}: "
                    f"{selected_rows.shape[0]} > {capacity}"
                )
            runtime_buffers["active"][slot_idx] = 1
            runtime_buffers["row_counts"][slot_idx] = int(selected_rows.shape[0])
            runtime_buffers["token_spans"][slot_idx, 0] = int(spec.token_span[0])
            runtime_buffers["token_spans"][slot_idx, 1] = int(spec.token_span[1])
            runtime_buffers["strengths"][slot_idx] = float(spec.strength)
            runtime_buffers["donor_means"][slot_idx].zero_()
            runtime_buffers["selected_rows"][slot_idx].zero_()
            if selected_rows.numel() > 0:
                runtime_buffers["selected_rows"][slot_idx, : selected_rows.shape[0]].copy_(
                    selected_rows.to(torch.float32)
                )
            if spec.mode == PATCH_MODE_SWAP_COMPONENTS:
                donor_mean = spec.donor_mean_by_layer.get(int(layer_idx))
                if donor_mean is None:
                    raise ValueError(f"missing donor_mean for layer {layer_idx}")
                donor_mean_t = torch.as_tensor(donor_mean, device=runtime_buffers["donor_means"].device, dtype=torch.float32)
                if donor_mean_t.shape != runtime_buffers["donor_means"][slot_idx].shape:
                    raise ValueError(
                        f"donor_mean shape mismatch for layer {layer_idx}: "
                        f"{tuple(donor_mean_t.shape)} vs {tuple(runtime_buffers['donor_means'][slot_idx].shape)}"
                    )
                runtime_buffers["donor_means"][slot_idx].copy_(donor_mean_t)


def _record_batch_tensor_stats(
    model: Any,
    *,
    layer_idx: int,
    slot_idx: int,
    original_hidden: Any,
    patched_hidden: Any,
    basis_layer: dict[str, Any],
    spec: MarketPatchSpec,
) -> None:
    import torch

    if slot_idx < 0 or slot_idx >= _MAX_BATCH_PATCH_STATS_SLOTS:
        return
    selected_rows = _selected_component_rows(basis_layer, spec, layer_idx)
    tensors = summarize_market_patch_tensor_tensors(
        original_hidden.to(torch.float32),
        patched_hidden.to(torch.float32),
        mean=basis_layer["mean"],
        safe_scale=basis_layer["safe_scale"],
        selected_rows=selected_rows,
        token_span=spec.token_span,
        strength=float(spec.strength),
    )
    buffers = _ensure_batch_tensor_stats_buffers(
        model,
        layer_idx=layer_idx,
        coeff_dim=int(selected_rows.shape[0]),
        device=patched_hidden.device,
    )
    buffers["valid"][slot_idx] = 1
    buffers["scalars"][slot_idx].copy_(tensors["scalars"])
    coeff_before = tensors["selected_coeff_before"]
    coeff_after = tensors["selected_coeff_after"]
    if coeff_before.numel() > 0:
        buffers["coeff_before"][slot_idx, : coeff_before.shape[0]].copy_(coeff_before)
        buffers["coeff_after"][slot_idx, : coeff_after.shape[0]].copy_(coeff_after)


def harvest_batch_tensor_stats(model: Any, batch_specs: list[dict[str, Any]]) -> None:
    buffers = getattr(model, "_market_patch_batch_tensor_stats", None)
    if not isinstance(buffers, dict) or not batch_specs:
        return
    stats_by_req = getattr(model, "_market_patch_last_stats_by_req", None)
    if not isinstance(stats_by_req, dict):
        stats_by_req = {}
        model._market_patch_last_stats_by_req = stats_by_req
    coverage_by_req = getattr(model, "_market_patch_coverage_by_req", None)
    if not isinstance(coverage_by_req, dict):
        coverage_by_req = {}
        model._market_patch_coverage_by_req = coverage_by_req

    for slot_idx, batch_spec in enumerate(batch_specs):
        req_id = str(batch_spec.get("req_id"))
        payload = batch_spec.get("patch_spec")
        if not isinstance(payload, dict):
            continue
        spec = MarketPatchSpec(**payload)
        for layer_idx in spec.target_layers:
            layer_buffers = buffers.get(int(layer_idx))
            if not isinstance(layer_buffers, dict):
                continue
            valid = int(layer_buffers["valid"][slot_idx].item())
            if valid <= 0:
                continue
            scalars = layer_buffers["scalars"][slot_idx].detach().cpu().tolist()
            coeff_before = layer_buffers["coeff_before"][slot_idx].detach().cpu().tolist()
            coeff_after = layer_buffers["coeff_after"][slot_idx].detach().cpu().tolist()
            coeff_dim = len(spec.component_indices_by_layer.get(int(layer_idx), ()))
            if coeff_dim <= 0:
                basis = getattr(model, "_market_patch_basis", None)
                if isinstance(basis, dict) and int(layer_idx) in basis:
                    coeff_dim = int(_selected_component_rows(basis[int(layer_idx)], spec, int(layer_idx)).shape[0])
            if coeff_dim > 0:
                coeff_before = coeff_before[:coeff_dim]
                coeff_after = coeff_after[:coeff_dim]
            else:
                coeff_before = []
                coeff_after = []
            stats = {
                "layer": int(layer_idx),
                "mode": spec.mode,
                "token_span": [int(spec.token_span[0]), int(spec.token_span[1])],
                "delta_norm_raw": float(scalars[0]),
                "delta_norm_std": float(scalars[1]),
                "mean_norm_before": float(scalars[2]),
                "mean_norm_after": float(scalars[3]),
                "mean_std_norm_before": float(scalars[4]),
                "mean_std_norm_after": float(scalars[5]),
                "selected_proj_norm_before": float(scalars[6]),
                "strength": float(scalars[7]),
                "selected_coeff_before": [float(v) for v in coeff_before],
                "selected_coeff_after": [float(v) for v in coeff_after],
                "req_id": req_id,
            }
            _merge_batch_stats(model, req_id=req_id, layer_idx=int(layer_idx), batch_spec=batch_spec, stats=stats)


def _merge_batch_stats(
    model: Any,
    *,
    req_id: str,
    layer_idx: int,
    batch_spec: dict[str, Any],
    stats: dict[str, Any],
) -> None:
    stats_by_req = getattr(model, "_market_patch_last_stats_by_req", None)
    if not isinstance(stats_by_req, dict):
        stats_by_req = {}
        model._market_patch_last_stats_by_req = stats_by_req
    coverage_by_req = getattr(model, "_market_patch_coverage_by_req", None)
    if not isinstance(coverage_by_req, dict):
        coverage_by_req = {}
        model._market_patch_coverage_by_req = coverage_by_req

    merged_input = dict(stats)
    merged_input["req_id"] = str(req_id)
    for meta_key in (
        "target_span",
        "chunk_abs_span",
        "overlap_abs_span",
        "query_span",
        "prefill_chunk_len",
    ):
        if meta_key in batch_spec:
            merged_input[meta_key] = batch_spec[meta_key]

    layer_coverage = coverage_by_req.setdefault(str(req_id), {}).setdefault(int(layer_idx), [])
    overlap_abs_span = batch_spec.get("overlap_abs_span")
    if (
        merged_input.get("status") != "skipped"
        and isinstance(overlap_abs_span, (list, tuple))
        and len(overlap_abs_span) == 2
    ):
        start = int(overlap_abs_span[0])
        end = int(overlap_abs_span[1])
        if end > start:
            layer_coverage.append((start, end))
            layer_coverage.sort()
            merged_spans: list[tuple[int, int]] = []
            for span_start, span_end in layer_coverage:
                if not merged_spans or span_start > merged_spans[-1][1]:
                    merged_spans.append((span_start, span_end))
                else:
                    merged_spans[-1] = (merged_spans[-1][0], max(merged_spans[-1][1], span_end))
            coverage_by_req[str(req_id)][int(layer_idx)] = merged_spans
            layer_coverage = merged_spans

    existing_stats = stats_by_req.setdefault(str(req_id), {}).get(int(layer_idx))
    merged_stats = dict(existing_stats) if isinstance(existing_stats, dict) else {}
    merged_stats.update(merged_input)
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
                covered += max(0, min(span_end, target_end) - max(span_start, target_start))
            total = max(1, target_end - target_start)
            merged_stats["covered_abs_tokens"] = int(covered)
            merged_stats["target_abs_tokens"] = int(target_end - target_start)
            merged_stats["coverage_fraction"] = float(covered / total)
    stats_by_req[str(req_id)][int(layer_idx)] = merged_stats


def _apply_layer_output_patching(
    *,
    owner_model: Any,
    layer_idx: int,
    custom_op: Any | None,
    output: Any,
) -> Any:
    force_custom_op_presence = bool(
        getattr(owner_model, "_market_patch_force_custom_op_presence", False)
    )

    if force_custom_op_presence:
        basis = getattr(owner_model, "_market_patch_basis", None)
        hidden = _extract_hidden_tensor(output)
        if hidden is None:
            return output
        if not isinstance(basis, dict) or layer_idx not in basis:
            return output
        runtime_state = getattr(owner_model, "_market_patch_batch_runtime_state", None)
        layer_state = runtime_state.get(int(layer_idx)) if isinstance(runtime_state, dict) else None
        if not isinstance(layer_state, dict):
            return output
        layer_buffers = getattr(owner_model, "_market_patch_batch_tensor_stats", {}).get(int(layer_idx))
        if not isinstance(layer_buffers, dict):
            return output
        if custom_op is None:
            return output
        compiled_modes = getattr(owner_model, "_market_patch_compiled_batch_modes", {})
        batch_mode = compiled_modes.get(int(layer_idx), PATCH_MODE_PROJECT_OUT)
        custom_kwargs: dict[str, Any] = {
            "batch_selected_rows": layer_state["selected_rows"],
            "batch_row_counts": layer_state["row_counts"],
            "batch_token_spans": layer_state["token_spans"],
            "batch_strengths": layer_state["strengths"],
            "batch_active": layer_state["active"],
            "stats_valid": layer_buffers["valid"],
            "stats_scalars": layer_buffers["scalars"],
            "stats_coeff_before": layer_buffers["coeff_before"],
            "stats_coeff_after": layer_buffers["coeff_after"],
        }
        if batch_mode == PATCH_MODE_SWAP_COMPONENTS:
            custom_kwargs["batch_donor_means"] = layer_state["donor_means"]
        patched_hidden = custom_op(
            hidden,
            basis[layer_idx]["mean"],
            basis[layer_idx]["scale"],
            basis[layer_idx]["safe_scale"],
            layer_state["selected_rows"][0, :0],
            layer_state["token_spans"][0],
            patch_mode_to_id(batch_mode),
            1.0,
            **custom_kwargs,
        )
        return _replace_hidden_tensor(output, patched_hidden)

    batch_specs = getattr(owner_model, "_market_patch_batch_specs", None)
    if isinstance(batch_specs, list) and batch_specs:
        basis = getattr(owner_model, "_market_patch_basis", None)
        if not isinstance(basis, dict) or layer_idx not in basis:
            return output
        hidden = _extract_hidden_tensor(output)
        if hidden is None:
            return output
        patched_hidden = hidden
        for slot_idx, batch_spec in enumerate(batch_specs):
            payload = batch_spec.get("patch_spec")
            if not isinstance(payload, dict):
                continue
            spec = MarketPatchSpec(**payload)
            if layer_idx not in spec.target_layers:
                continue
            pre_patch_hidden = patched_hidden
            patched_hidden, eager_stats = _patch_hidden_states_for_layer(
                pre_patch_hidden,
                basis_layer=basis[layer_idx],
                spec=spec,
                layer_idx=layer_idx,
                custom_op=custom_op,
            )
            try:
                _record_batch_tensor_stats(
                    owner_model,
                    layer_idx=layer_idx,
                    slot_idx=slot_idx,
                    original_hidden=pre_patch_hidden,
                    patched_hidden=patched_hidden,
                    basis_layer=basis[layer_idx],
                    spec=spec,
                )
            except Exception:
                pass
            if eager_stats is not None:
                _merge_batch_stats(
                    owner_model,
                    req_id=str(batch_spec.get("req_id")),
                    layer_idx=layer_idx,
                    batch_spec=batch_spec,
                    stats=eager_stats,
                )
        return _replace_hidden_tensor(output, patched_hidden)

    spec = getattr(owner_model, "_market_patch_spec", None)
    if spec is None or layer_idx not in spec.target_layers:
        if force_custom_op_presence:
            hidden = _extract_hidden_tensor(output)
            if hidden is not None:
                return _replace_hidden_tensor(
                    output,
                    _apply_noop_custom_op(hidden, custom_op=custom_op),
                )
        return output
    applied_layers = getattr(owner_model, "_market_patch_applied_layers", None)
    if isinstance(applied_layers, set) and layer_idx in applied_layers:
        if force_custom_op_presence:
            hidden = _extract_hidden_tensor(output)
            if hidden is not None:
                return _replace_hidden_tensor(
                    output,
                    _apply_noop_custom_op(hidden, custom_op=custom_op),
                )
        return output
    basis = getattr(owner_model, "_market_patch_basis", None)
    if not isinstance(basis, dict) or layer_idx not in basis:
        if force_custom_op_presence:
            hidden = _extract_hidden_tensor(output)
            if hidden is not None:
                return _replace_hidden_tensor(
                    output,
                    _apply_noop_custom_op(hidden, custom_op=custom_op),
                )
        return output
    hidden = _extract_hidden_tensor(output)
    if hidden is None:
        patch_stats = getattr(owner_model, "_market_patch_last_stats", None)
        if not isinstance(patch_stats, dict):
            patch_stats = {}
            owner_model._market_patch_last_stats = patch_stats
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
        custom_op=custom_op,
    )
    if stats is not None:
        patch_stats = getattr(owner_model, "_market_patch_last_stats", None)
        if not isinstance(patch_stats, dict):
            patch_stats = {}
            owner_model._market_patch_last_stats = patch_stats
        patch_stats[int(layer_idx)] = stats
        if stats.get("status") != "skipped":
            applied_layers = getattr(owner_model, "_market_patch_applied_layers", None)
            if not isinstance(applied_layers, set):
                applied_layers = set()
                owner_model._market_patch_applied_layers = applied_layers
            applied_layers.add(int(layer_idx))
    return _replace_hidden_tensor(output, patched_hidden)


def _build_market_patched_layer_class() -> type[Any]:
    import torch

    class MarketPatchedLayer(torch.nn.Module):
        def __init__(self, owner_model: Any, layer_idx: int, wrapped_layer: Any) -> None:
            super().__init__()
            self.layer_idx = int(layer_idx)
            self.wrapped_layer = wrapped_layer
            self._owner_ref = weakref.ref(owner_model)
            try:
                self.market_patch_hidden_states_op = build_market_patch_hidden_states_op()
            except Exception:
                self.market_patch_hidden_states_op = None

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            owner_model = self._owner_ref()
            output = self.wrapped_layer(*args, **kwargs)
            if owner_model is None:
                return output
            return _apply_layer_output_patching(
                owner_model=owner_model,
                layer_idx=self.layer_idx,
                custom_op=self.market_patch_hidden_states_op,
                output=output,
            )

    return MarketPatchedLayer


try:
    _MarketPatchedLayer = _build_market_patched_layer_class()
except Exception:
    _MarketPatchedLayer = _MarketPatchedLayer


def install_qwen3_moe_market_patch_hooks() -> bool:
    try:
        from vllm.model_executor.models.qwen3_moe import (
            Qwen3MoeDecoderLayer,
            Qwen3MoeForCausalLM,
        )
    except Exception:
        return False

    if getattr(Qwen3MoeDecoderLayer, "_market_patch_forward_installed", False):
        return True

    original_model_init = Qwen3MoeForCausalLM.__init__
    original_layer_forward = Qwen3MoeDecoderLayer.forward

    @functools.wraps(original_model_init)
    def patched_model_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_model_init(self, *args, **kwargs)
        for layer_idx, layer in find_decoder_layers(self).items():
            if not hasattr(layer, "market_patch_hidden_states_op"):
                try:
                    layer.market_patch_hidden_states_op = build_market_patch_hidden_states_op()
                except Exception:
                    layer.market_patch_hidden_states_op = None
            layer.market_patch_layer_idx = int(layer_idx)
            layer._market_patch_owner_ref = weakref.ref(self)
            layer._market_patch_class_hooked = True
        self._market_patch_class_hooked = True

    def patched_layer_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        output = original_layer_forward(self, *args, **kwargs)
        owner_ref = getattr(self, "_market_patch_owner_ref", None)
        owner_model = owner_ref() if owner_ref is not None else None
        layer_idx = getattr(self, "market_patch_layer_idx", None)
        if owner_model is None or layer_idx is None:
            return output
        return _apply_layer_output_patching(
            owner_model=owner_model,
            layer_idx=int(layer_idx),
            custom_op=getattr(self, "market_patch_hidden_states_op", None),
            output=output,
        )

    Qwen3MoeForCausalLM.__init__ = patched_model_init
    Qwen3MoeDecoderLayer.forward = patched_layer_forward
    patched_model_init.__signature__ = inspect.signature(original_model_init)
    Qwen3MoeForCausalLM._market_patch_init_installed = True
    Qwen3MoeDecoderLayer._market_patch_forward_installed = True
    Qwen3MoeForCausalLM._market_patch_original_init = original_model_init
    Qwen3MoeDecoderLayer._market_patch_original_forward = original_layer_forward
    return True


def init_market_patching(model: Any) -> None:
    import torch

    layers = find_decoder_layers(model)
    if not layers:
        raise RuntimeError("No decoder layers found on model")
    if getattr(model, "_market_patch_initialized", False):
        return

    container = getattr(getattr(model, "model", None), "layers", None)
    if container is None:
        raise RuntimeError("No decoder layer container found on model")

    early_hooked = all(
        getattr(layer, "_market_patch_class_hooked", False) for layer in layers.values()
    )
    if not early_hooked:
        for layer_idx, layer in layers.items():
            if isinstance(layer, _MarketPatchedLayer):
                continue
            wrapped = _MarketPatchedLayer(model, layer_idx, layer)
            wrapped._market_patch_original_layer = layer
            container[layer_idx] = wrapped

    device = _infer_model_device(model)
    placeholder_basis: dict[int, dict[str, Any]] = {}
    for layer_idx, layer in find_decoder_layers(model).items():
        hidden_dim = _infer_layer_hidden_dim(layer)
        placeholder_basis[int(layer_idx)] = {
            "mean": torch.zeros((hidden_dim,), device=device, dtype=torch.float32),
            "scale": torch.ones((hidden_dim,), device=device, dtype=torch.float32),
            "safe_scale": torch.ones((hidden_dim,), device=device, dtype=torch.float32),
            "components": torch.empty((0, hidden_dim), device=device, dtype=torch.float32),
            "named_components": {},
        }

    model._market_patch_initialized = True
    model._market_patch_basis = placeholder_basis
    model._market_patch_spec = None
    model._market_patch_last_stats = {}
    model._market_patch_applied_layers = set()
    model._market_patch_batch_specs = []
    model._market_patch_last_stats_by_req = {}
    model._market_patch_coverage_by_req = {}
    model._market_patch_batch_tensor_stats = {}
    model._market_patch_batch_runtime_state = {}
    model._market_patch_compiled_batch_modes = {}
    model._market_patch_force_custom_op_presence = False
    for layer_idx, basis_layer in placeholder_basis.items():
        _ensure_batch_tensor_stats_buffers(
            model,
            layer_idx=int(layer_idx),
            coeff_dim=_DEFAULT_COMPILED_PATCH_MAX_COMPONENTS,
            device=device,
        )
        _ensure_batch_runtime_state_buffers(
            model,
            layer_idx=int(layer_idx),
            coeff_dim=_DEFAULT_COMPILED_PATCH_MAX_COMPONENTS,
            hidden_dim=int(basis_layer["mean"].shape[0]),
            device=device,
        )


def register_patch_basis(model: Any, basis_payload: dict[int, dict[str, Any]]) -> dict[str, Any]:
    import torch

    if not getattr(model, "_market_patch_initialized", False):
        init_market_patching(model)

    device = _infer_model_device(model)
    current_basis = getattr(model, "_market_patch_basis", None)
    registered: dict[int, dict[str, Any]] = (
        {
            int(layer): dict(payload)
            for layer, payload in current_basis.items()
        }
        if isinstance(current_basis, dict)
        else {}
    )
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
    for layer_idx, basis_layer in registered.items():
        _ensure_batch_tensor_stats_buffers(
            model,
            layer_idx=int(layer_idx),
            coeff_dim=_DEFAULT_COMPILED_PATCH_MAX_COMPONENTS,
            device=device,
        )
        _ensure_batch_runtime_state_buffers(
            model,
            layer_idx=int(layer_idx),
            coeff_dim=_DEFAULT_COMPILED_PATCH_MAX_COMPONENTS,
            hidden_dim=int(basis_layer["mean"].shape[0]),
            device=device,
        )
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
    compiled_modes: dict[int, str] = {}
    enforce_compiled_mode = bool(getattr(model, "_market_patch_force_custom_op_presence", False))
    for batch_spec in model._market_patch_batch_specs:
        payload = batch_spec.get("patch_spec")
        if not isinstance(payload, dict):
            continue
        spec = MarketPatchSpec(**payload)
        for layer_idx in spec.target_layers:
            existing = compiled_modes.get(int(layer_idx))
            if existing is None:
                compiled_modes[int(layer_idx)] = spec.mode
            elif enforce_compiled_mode and existing != spec.mode:
                raise NotImplementedError(
                    "Compiled non-eager batch patching does not support mixing patch modes within a layer"
                )
    model._market_patch_compiled_batch_modes = compiled_modes
    _reset_batch_tensor_stats(model)
    _reset_batch_runtime_state(model)
    if getattr(model, "_market_patch_force_custom_op_presence", False):
        _load_batch_runtime_state(model, model._market_patch_batch_specs)
    if not isinstance(getattr(model, "_market_patch_last_stats_by_req", None), dict):
        model._market_patch_last_stats_by_req = {}
    if not isinstance(getattr(model, "_market_patch_coverage_by_req", None), dict):
        model._market_patch_coverage_by_req = {}


def clear_batch_patch_specs(model: Any) -> None:
    model._market_patch_batch_specs = []
    model._market_patch_compiled_batch_modes = {}
    _reset_batch_runtime_state(model)


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
    container = getattr(getattr(model, "model", None), "layers", None)
    if container is not None:
        for idx, layer in list(enumerate(container)):
            original = getattr(layer, "_market_patch_original_layer", None)
            if original is not None:
                container[idx] = original
    for attr in (
        "_market_patch_initialized",
        "_market_patch_basis",
        "_market_patch_spec",
        "_market_patch_last_stats",
        "_market_patch_applied_layers",
        "_market_patch_batch_specs",
        "_market_patch_last_stats_by_req",
        "_market_patch_coverage_by_req",
        "_market_patch_batch_tensor_stats",
        "_market_patch_batch_runtime_state",
        "_market_patch_force_custom_op_presence",
    ):
        if hasattr(model, attr):
            delattr(model, attr)
