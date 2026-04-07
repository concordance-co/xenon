"""Legacy compatibility wrapper for activation patching."""

from pipelines.interp.patching.activation_patch_core import *  # noqa: F401,F403
from pipelines.interp.patching.activation_patch_core import (
    ActivationPatchSpec,
    init_activation_patching,
    install_qwen3_moe_activation_patch_hooks,
    teardown_activation_patching,
)

MarketPatchSpec = ActivationPatchSpec
install_qwen3_moe_market_patch_hooks = install_qwen3_moe_activation_patch_hooks
init_market_patching = init_activation_patching
teardown_market_patching = teardown_activation_patching
