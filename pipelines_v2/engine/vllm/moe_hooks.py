"""vLLM-side MoE router hooks.

This module captures router outputs at the router-selection boundary rather
than by monkey-patching model block forwards. Modular eager, chunked, and
compiled MoE paths funnel through ``router.select_experts(...)`` in supported
vLLM versions. Monolithic kernels intentionally bypass that boundary, so they
are rejected during initialization rather than silently producing incomplete
capture data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_PATCHED = False
_ORIGINAL_SELECT_EXPERTS = None
_PATCH_TARGET = None


@dataclass(slots=True)
class _LayerCapture:
    layer_id: int
    num_experts: int
    top_k: int
    logits_buffer: Any
    topk_ids_buffer: Any
    topk_weights_buffer: Any
    count: int = 0

    def reset(self) -> None:
        self.logits_buffer.zero_()
        self.topk_ids_buffer.zero_()
        self.topk_weights_buffer.zero_()
        self.count = 0


class _RouterCaptureSession:
    def __init__(self, *, max_tokens: int, layers: dict[int, Any]) -> None:
        import torch

        self.max_tokens = int(max_tokens)
        self.layers: dict[int, _LayerCapture] = {}
        self.enabled = True

        for layer_id, layer in sorted(layers.items()):
            num_experts = _num_experts_for_layer(layer)
            top_k = _top_k_for_layer(layer)
            device = _device_for_layer(layer)
            self.layers[int(layer_id)] = _LayerCapture(
                layer_id=int(layer_id),
                num_experts=num_experts,
                top_k=top_k,
                logits_buffer=torch.zeros(
                    (self.max_tokens, num_experts),
                    dtype=torch.float32,
                    device=device,
                ),
                topk_ids_buffer=torch.zeros(
                    (self.max_tokens, top_k),
                    dtype=torch.int32,
                    device=device,
                ),
                topk_weights_buffer=torch.zeros(
                    (self.max_tokens, top_k),
                    dtype=torch.float32,
                    device=device,
                ),
            )

    def reset(self) -> None:
        for layer in self.layers.values():
            layer.reset()

    def capture(
        self,
        *,
        layer_id: int,
        router_logits: Any,
        topk_ids: Any,
        topk_weights: Any,
    ) -> None:
        capture = self.layers.get(int(layer_id))
        if capture is None or not self.enabled:
            return

        if len(router_logits.shape) != 2:
            raise RuntimeError(
                "MoE router capture expected two-dimensional router logits, "
                f"but layer {layer_id} produced shape {tuple(router_logits.shape)}"
            )
        if len(topk_ids.shape) != 2 or len(topk_weights.shape) != 2:
            raise RuntimeError(
                "MoE router capture expected two-dimensional top-k tensors, "
                f"but layer {layer_id} produced ids={tuple(topk_ids.shape)} "
                f"and weights={tuple(topk_weights.shape)}"
            )
        num_tokens = int(router_logits.shape[0])
        expected_logits_shape = (num_tokens, int(capture.num_experts))
        expected_topk_shape = (num_tokens, int(capture.top_k))
        if tuple(router_logits.shape) != expected_logits_shape:
            raise RuntimeError(
                "MoE router capture logits shape changed unexpectedly: "
                f"layer {layer_id} produced {tuple(router_logits.shape)}, "
                f"expected {expected_logits_shape}"
            )
        if tuple(topk_ids.shape) != expected_topk_shape:
            raise RuntimeError(
                "MoE router capture expert-id shape changed unexpectedly: "
                f"layer {layer_id} produced {tuple(topk_ids.shape)}, "
                f"expected {expected_topk_shape}"
            )
        if tuple(topk_weights.shape) != expected_topk_shape:
            raise RuntimeError(
                "MoE router capture expert-weight shape changed unexpectedly: "
                f"layer {layer_id} produced {tuple(topk_weights.shape)}, "
                f"expected {expected_topk_shape}"
            )
        start = int(capture.count)
        end = start + num_tokens
        if end > capture.logits_buffer.shape[0]:
            raise RuntimeError(
                "Router logits buffer overflow: "
                f"needed {end} rows for layer {layer_id}, allocated {capture.logits_buffer.shape[0]}. "
                "Increase the configured max token budget for router capture."
            )

        capture.logits_buffer[start:end].copy_(router_logits.float())
        capture.topk_ids_buffer[start:end].copy_(topk_ids.to(dtype=capture.topk_ids_buffer.dtype))
        capture.topk_weights_buffer[start:end].copy_(topk_weights.float())
        capture.count = end

    def collect(self) -> dict[int, dict[str, Any]]:
        missing_layers = [
            int(layer_id)
            for layer_id, capture in self.layers.items()
            if capture.count <= 0
        ]
        if self.enabled and missing_layers:
            raise RuntimeError(
                "MoE routing capture did not observe router outputs for initialized "
                f"layers {missing_layers}. This usually means vLLM selected a "
                "monolithic or otherwise unsupported execution path."
            )

        result: dict[int, dict[str, Any]] = {}
        for layer_id, capture in self.layers.items():
            result[int(layer_id)] = {
                "logits": capture.logits_buffer[: capture.count].detach().cpu().clone(),
                "topk_ids": capture.topk_ids_buffer[: capture.count].detach().cpu().clone(),
                "topk_weights": capture.topk_weights_buffer[: capture.count].detach().cpu().clone(),
            }
        return result


def _patch_base_router() -> None:
    global _PATCHED, _ORIGINAL_SELECT_EXPERTS, _PATCH_TARGET
    if _PATCHED:
        return

    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

    _ORIGINAL_SELECT_EXPERTS = BaseRouter.select_experts
    _PATCH_TARGET = BaseRouter

    def patched_select_experts(
        self: Any,
        hidden_states: Any,
        router_logits: Any,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        topk_weights, topk_ids = _ORIGINAL_SELECT_EXPERTS(
            self,
            hidden_states,
            router_logits,
            *args,
            **kwargs,
        )
        session = getattr(self, "_xenon_router_capture_session", None)
        layer_id = getattr(self, "_xenon_router_layer_id", None)
        if session is not None and layer_id is not None:
            session.capture(
                layer_id=int(layer_id),
                router_logits=router_logits,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            )
        return topk_weights, topk_ids

    BaseRouter.select_experts = patched_select_experts
    _PATCHED = True


def _iter_fused_moe_layers(model: Any) -> dict[int, Any]:
    try:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    except Exception:
        FusedMoE = None

    fused_moe_type = FusedMoE if isinstance(FusedMoE, type) else None
    moe_runner_type = None
    if fused_moe_type is None:
        # vLLM 0.25 made FusedMoE a factory returning MoERunner. Importing the
        # concrete runner is necessary because a function cannot be used as
        # the second argument to isinstance().
        try:
            from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
                MoERunner,
            )
        except Exception:
            MoERunner = None
        if isinstance(MoERunner, type):
            moe_runner_type = MoERunner

    known_types = tuple(
        layer_type
        for layer_type in (fused_moe_type, moe_runner_type)
        if layer_type is not None
    )
    layers: dict[int, Any] = {}
    if known_types and hasattr(model, "modules"):
        for module in model.modules():
            if isinstance(module, known_types):
                try:
                    layer_id = int(module.layer_id)
                except Exception:
                    continue
                layers[layer_id] = module
    if layers:
        return layers

    # Fallback for older or non-standard model surfaces.
    model_layers = getattr(getattr(model, "model", None), "layers", [])
    for idx, layer in enumerate(model_layers):
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None)
        router = getattr(experts, "router", None)
        if experts is not None and router is not None:
            layers[int(idx)] = experts
    return layers


def _num_experts_for_layer(layer: Any) -> int:
    for value in (
        getattr(layer, "logical_num_experts", None),
        getattr(getattr(layer, "moe_config", None), "num_logical_experts", None),
        getattr(layer, "global_num_experts", None),
        getattr(getattr(layer, "router", None), "global_num_experts", None),
        getattr(getattr(layer, "moe_config", None), "global_num_experts", None),
    ):
        if value is not None:
            return int(value)
    gate = getattr(layer, "gate", None) or getattr(layer, "_gate", None)
    weight = getattr(gate, "weight", None)
    if weight is None and hasattr(gate, "linear"):
        weight = getattr(gate.linear, "weight", None)
    if weight is not None:
        return int(weight.shape[0])
    raise RuntimeError(f"Could not determine MoE expert count for layer {layer!r}")


def _top_k_for_layer(layer: Any) -> int:
    for value in (
        getattr(layer, "top_k", None),
        getattr(getattr(layer, "router", None), "top_k", None),
        getattr(getattr(layer, "moe_config", None), "top_k", None),
    ):
        if value is not None:
            return int(value)
    raise RuntimeError(f"Could not determine MoE top-k for layer {layer!r}")


def _device_for_layer(layer: Any) -> Any:
    gate = getattr(layer, "gate", None)
    if gate is None:
        gate = getattr(layer, "_gate", None)
    weight = getattr(gate, "weight", None)
    if weight is None and hasattr(gate, "linear"):
        weight = getattr(gate.linear, "weight", None)
    if weight is not None:
        return weight.device
    router = getattr(layer, "router", None)
    if router is not None and hasattr(router, "parameters"):
        for param in router.parameters():
            return param.device
    if hasattr(layer, "get_expert_weights"):
        for param in layer.get_expert_weights():
            return param.device
    if hasattr(layer, "parameters"):
        for param in layer.parameters():
            return param.device
    raise RuntimeError(f"Could not determine MoE device for layer {layer!r}")


def _is_monolithic_layer(layer: Any) -> bool:
    """Return whether a vLLM MoE layer bypasses router.select_experts."""

    candidates = (
        getattr(layer, "is_monolithic", None),
        getattr(getattr(layer, "quant_method", None), "is_monolithic", None),
        getattr(
            getattr(getattr(layer, "routed_experts", None), "quant_method", None),
            "is_monolithic",
            None,
        ),
        getattr(
            getattr(getattr(layer, "runner", None), "quant_method", None),
            "is_monolithic",
            None,
        ),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        if callable(candidate):
            candidate = candidate()
        if bool(candidate):
            return True
    return False


def find_moe_blocks(model: Any) -> dict[int, Any]:
    """Return discovered MoE layers keyed by transformer layer index."""

    return _iter_fused_moe_layers(model)


def init_router_capture(model: Any, max_tokens: int) -> bool:
    """Initialize router capture for all discovered MoE layers on a model."""

    layers = _iter_fused_moe_layers(model)
    if not layers:
        return False

    monolithic_layers = [
        int(layer_id)
        for layer_id, layer in layers.items()
        if _is_monolithic_layer(layer)
    ]
    if monolithic_layers:
        raise RuntimeError(
            "MoE routing capture does not support vLLM monolithic MoE kernels "
            f"(layers {monolithic_layers}). Monolithic kernels route internally "
            "and do not expose the selected expert weights and ids at "
            "router.select_experts; use a modular kernel configuration."
        )

    missing_routers = [
        int(layer_id)
        for layer_id, layer in layers.items()
        if getattr(layer, "router", None) is None
    ]
    if missing_routers:
        raise RuntimeError(
            "MoE routing capture found layers without an exposed vLLM router: "
            f"{missing_routers}"
        )

    _patch_base_router()
    session = _RouterCaptureSession(max_tokens=max_tokens, layers=layers)
    setattr(model, "_xenon_router_capture_session", session)
    for layer_id, layer in layers.items():
        router = getattr(layer, "router", None)
        setattr(router, "_xenon_router_capture_session", session)
        setattr(router, "_xenon_router_layer_id", int(layer_id))
    return True


def enable_router_capture(model: Any) -> None:
    session = getattr(model, "_xenon_router_capture_session", None)
    if session is not None:
        session.enabled = True


def reset_router_buffers(model: Any) -> None:
    session = getattr(model, "_xenon_router_capture_session", None)
    if session is not None:
        session.reset()


def collect_router_capture(model: Any) -> dict[int, dict[str, Any]]:
    session = getattr(model, "_xenon_router_capture_session", None)
    if session is None:
        return {}
    return session.collect()


def collect_router_logits(model: Any) -> dict[int, Any]:
    """Compatibility helper returning only captured gate logits."""

    return {
        int(layer): payload["logits"]
        for layer, payload in collect_router_capture(model).items()
    }
