"""Legacy compatibility wrapper for activation patch custom ops."""

from pipelines.interp.patching.activation_patch_custom_op import *  # noqa: F401,F403
from pipelines.interp.patching.activation_patch_custom_op import (
    apply_activation_patch_tensor,
    build_activation_patch_hidden_states_op,
    summarize_activation_patch_tensor,
    summarize_activation_patch_tensor_tensors,
)

summarize_market_patch_tensor = summarize_activation_patch_tensor
summarize_market_patch_tensor_tensors = summarize_activation_patch_tensor_tensors
apply_market_patch_tensor = apply_activation_patch_tensor
build_market_patch_hidden_states_op = build_activation_patch_hidden_states_op
