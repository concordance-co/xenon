# Infrastructure scripts

Use `python -m pipelines_v2.cli` for ordinary workflow operations.

- `check_research_hub_wheel.py`: packaged assets and optional-import checks.
- `pipelines_v2_orchestrator_smoke.py`: orchestration smoke check.
- `pipelines_v2_activation_patch_smoke.py`: patching smoke check.
- `pipelines_v2_router_layer_probe.py`: router probe helper.
- `pipelines_v2_vllm_upgrade_benchmark.py`: explicit runtime upgrade benchmark.

Study-specific Assistant Axis scripts were bundled outside this checkout.
Maintained workflow recipes live in [examples](../examples/README.md).
