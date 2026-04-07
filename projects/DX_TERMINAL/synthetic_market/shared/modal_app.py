"""Non-shadowing Modal entrypoint wrapper for patching workflows.

`modal run <path>.py` executes the target file as a top-level module, so using a
file literally named `modal.py` causes `import modal` inside that file to
resolve to itself instead of the real Modal package. This wrapper keeps the
cleaned module layout while providing a stable CLI entrypoint.
"""

from projects.DX_TERMINAL.synthetic_market.shared.modal_patching import (  # noqa: F401
    SyntheticMarketBehaviorMatrixWorker,
    analyze_synthetic_market_behavior_modal,
    analyze_synthetic_market_patching_modal,
    app,
    benchmark_customop_vs_stock_vllm_modal,
    benchmark_direct_vllm_variant_modal,
    benchmark_standard_vllm_serve_modal,
    download_model,
    plan_synthetic_market_behavior_battery_modal,
    prepare_synthetic_market_behavior_donors_modal,
    run_synthetic_market_behavior_matrix_modal,
    run_synthetic_market_behavior_modal,
    run_synthetic_market_patching_modal,
)
