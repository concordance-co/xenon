"""Model hook installation for activation patching."""

from __future__ import annotations

import functools
import inspect
import weakref
from typing import Any

import torch

from .apply import (
    ActivationPatchedLayerClass,
    apply_layer_output_patching,
    register_torch_library_interchange_batch_op,
    register_torch_library_project_out_batch_op,
    register_torch_library_project_out_op,
    register_torch_library_residual_path_batch_op,
    register_torch_library_subspace_batch_op,
    register_torch_library_subspace_op,
)
from .base import (
    _DEFAULT_COMPILED_PATCH_MAX_COMPONENTS,
    ActivationPatchedLayer,
    debug_panic,
    find_decoder_layers,
    infer_layer_hidden_dim,
    infer_model_device,
    unwrap_model,
)
from .state import _ensure_batch_runtime_state_buffers, _ensure_batch_tensor_stats_buffers


def _bind_activation_patch_layer_instance(owner_model: Any, *, layer_idx: int, layer: Any) -> None:
    if not getattr(layer, "_v2_activation_patch_forward_installed", False):
        original_forward = layer.forward

        @functools.wraps(original_forward)
        def patched_layer_forward(*args: Any, **kwargs: Any) -> Any:
            debug_panic(
                "layer_forward_entry",
                layer_cls=type(layer).__name__,
                has_owner_ref=hasattr(layer, "_v2_activation_patch_owner_ref"),
                layer_idx=getattr(layer, "_v2_activation_patch_layer_idx", None),
            )
            output = original_forward(*args, **kwargs)
            owner_ref = getattr(layer, "_v2_activation_patch_owner_ref", None)
            resolved_owner_model = owner_ref() if owner_ref is not None else None
            resolved_layer_idx = getattr(layer, "_v2_activation_patch_layer_idx", None)
            if resolved_owner_model is None or resolved_layer_idx is None:
                return output
            return apply_layer_output_patching(
                owner_model=resolved_owner_model,
                layer_idx=int(resolved_layer_idx),
                custom_op=getattr(layer, "activation_patch_hidden_states_op", None),
                output=output,
            )

        patched_layer_forward.__signature__ = inspect.signature(original_forward)
        layer._v2_activation_patch_original_forward = original_forward
        layer.forward = patched_layer_forward
        layer._v2_activation_patch_forward_installed = True


def bind_activation_patch_layers(owner_model: Any) -> None:
    for layer_idx, layer in find_decoder_layers(owner_model).items():
        _bind_activation_patch_layer_instance(owner_model, layer_idx=int(layer_idx), layer=layer)
        if not hasattr(layer, "activation_patch_hidden_states_op"):
            try:
                from ..activation_patch_custom_op import build_activation_patch_hidden_states_op

                layer.activation_patch_hidden_states_op = build_activation_patch_hidden_states_op()
            except Exception:
                layer.activation_patch_hidden_states_op = None
        layer._v2_activation_patch_owner_ref = weakref.ref(owner_model)
        layer._v2_activation_patch_layer_idx = int(layer_idx)
        layer._v2_activation_patch_instance_hooked = True
    owner_model._v2_activation_patch_layers_bound = True


def init_activation_patching(model: Any) -> None:
    model = unwrap_model(model)
    layers = find_decoder_layers(model)
    if not layers:
        raise RuntimeError("No decoder layers found on model")
    if getattr(model, "_v2_activation_patch_initialized", False):
        return

    register_torch_library_interchange_batch_op()
    register_torch_library_residual_path_batch_op()
    register_torch_library_subspace_op()
    register_torch_library_subspace_batch_op()
    register_torch_library_project_out_op()
    register_torch_library_project_out_batch_op()

    container = getattr(getattr(model, "model", None), "layers", None)
    if container is None:
        raise RuntimeError("No decoder layer container found on model")

    early_hooked = all(getattr(layer, "_v2_activation_patch_instance_hooked", False) for layer in layers.values())
    if not early_hooked:
        for layer_idx, layer in layers.items():
            if isinstance(layer, ActivationPatchedLayer):
                continue
            wrapped = ActivationPatchedLayerClass(model, layer_idx, layer)
            wrapped._activation_patch_original_layer = layer
            container[layer_idx] = wrapped

    model._v2_activation_patch_initialized = True
    model._v2_activation_patch_bank = {}
    device = infer_model_device(model) or torch.device("cpu")
    placeholder_subspace: dict[int, dict[str, Any]] = {}
    for layer_idx, layer in find_decoder_layers(model).items():
        hidden_dim = infer_layer_hidden_dim(layer)
        placeholder_subspace[int(layer_idx)] = {
            "mean": torch.zeros((hidden_dim,), device=device, dtype=torch.float32),
            "scale": torch.ones((hidden_dim,), device=device, dtype=torch.float32),
            "safe_scale": torch.ones((hidden_dim,), device=device, dtype=torch.float32),
            "components": torch.empty((0, hidden_dim), device=device, dtype=torch.float32),
            "named_components": {},
        }
    model._v2_activation_patch_subspace = placeholder_subspace
    model._v2_activation_patch_directions = {}
    model._v2_activation_patch_centroids = {}
    model._v2_activation_patch_batch_specs = []
    model._v2_activation_patch_stats_by_req = {}
    model._v2_activation_patch_device = device
    model._v2_activation_patch_batch_runtime_state = {}
    model._v2_activation_patch_batch_tensor_stats = {}
    model._v2_activation_patch_compiled_operator_hint = ""
    for layer_idx, layer_payload in placeholder_subspace.items():
        _ensure_batch_tensor_stats_buffers(
            model,
            layer_idx=int(layer_idx),
            coeff_dim=_DEFAULT_COMPILED_PATCH_MAX_COMPONENTS,
            device=device,
        )
        _ensure_batch_runtime_state_buffers(
            model,
            layer_idx=int(layer_idx),
            max_tokens=1,
            max_rows=_DEFAULT_COMPILED_PATCH_MAX_COMPONENTS,
            hidden_dim=int(layer_payload["mean"].shape[0]),
            device=device,
        )


def install_activation_patch_model_init_hook(model_cls: type[Any]) -> bool:
    if getattr(model_cls, "_v2_activation_patch_init_installed", False):
        return True

    original_model_init = model_cls.__init__

    @functools.wraps(original_model_init)
    def patched_model_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_model_init(self, *args, **kwargs)
        bind_activation_patch_layers(self)
        self._v2_activation_patch_class_hooked = True

    model_cls.__init__ = patched_model_init
    patched_model_init.__signature__ = inspect.signature(original_model_init)
    model_cls._v2_activation_patch_init_installed = True
    model_cls._v2_activation_patch_original_init = original_model_init
    return True


def restore_activation_patch_model_init_hook(model_cls: type[Any]) -> bool:
    if not getattr(model_cls, "_v2_activation_patch_init_installed", False):
        return False
    original_model_init = getattr(model_cls, "_v2_activation_patch_original_init", None)
    if callable(original_model_init):
        model_cls.__init__ = original_model_init
    if hasattr(model_cls, "_v2_activation_patch_original_init"):
        delattr(model_cls, "_v2_activation_patch_original_init")
    model_cls._v2_activation_patch_init_installed = False
    return True


__all__ = [
    "init_activation_patching",
    "install_activation_patch_model_init_hook",
    "restore_activation_patch_model_init_hook",
]
