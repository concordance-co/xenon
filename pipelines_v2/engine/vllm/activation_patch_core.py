"""Compatibility façade for activation patch runtime helpers."""

from .patching import (
    clear_batch_patch_specs,
    collect_patch_stats,
    harvest_batch_patch_stats,
    init_activation_patching,
    install_activation_patch_model_init_hook,
    register_torch_library_project_out_batch_op,
    register_activation_patch_bank,
    register_activation_patch_centroids,
    register_activation_patch_directions,
    register_activation_patch_subspace,
    set_batch_patch_specs,
)

_register_torch_library_project_out_batch_op = register_torch_library_project_out_batch_op

__all__ = [
    "clear_batch_patch_specs",
    "collect_patch_stats",
    "harvest_batch_patch_stats",
    "init_activation_patching",
    "install_activation_patch_model_init_hook",
    "register_torch_library_project_out_batch_op",
    "_register_torch_library_project_out_batch_op",
    "register_activation_patch_bank",
    "register_activation_patch_centroids",
    "register_activation_patch_directions",
    "register_activation_patch_subspace",
    "set_batch_patch_specs",
]
