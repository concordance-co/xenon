"""Internal activation patching submodules."""

from .apply import register_torch_library_subspace_batch_op, register_torch_library_subspace_op
from .apply import register_torch_library_project_out_batch_op
from .hooks import (
    init_activation_patching,
    install_activation_patch_model_init_hook,
    restore_activation_patch_model_init_hook,
)
from .state import (
    clear_batch_patch_specs,
    collect_patch_stats,
    harvest_batch_patch_stats,
    register_activation_patch_bank,
    register_activation_patch_centroids,
    register_activation_patch_directions,
    register_activation_patch_subspace,
    set_batch_patch_specs,
)

__all__ = [
    "clear_batch_patch_specs",
    "collect_patch_stats",
    "harvest_batch_patch_stats",
    "init_activation_patching",
    "install_activation_patch_model_init_hook",
    "restore_activation_patch_model_init_hook",
    "register_torch_library_subspace_batch_op",
    "register_torch_library_subspace_op",
    "register_torch_library_project_out_batch_op",
    "register_activation_patch_bank",
    "register_activation_patch_centroids",
    "register_activation_patch_directions",
    "register_activation_patch_subspace",
    "set_batch_patch_specs",
]
