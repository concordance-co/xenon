# Maintained examples

Runnable platform recipes and integration fixtures. Start with
[the experiment guide](../platform/LLM_RESEARCH.md) and
[capability map](../platform/RESEARCH_HUB.md) to choose one.

- `llm_research/`: configurable local CPU / Modal GPU workflow loader.
- `local_research/`, `native_neurons/`, `native_route_smoke/`: residual,
  native-site and paired-effect examples.
- `mattr/`, `mattr_ioi_smoke/`: configurable and fixed official attribution recipes.
- `qwen_crosscoder/`, `circuit_trace_smoke/`: paired-model dictionaries and tracing.
- `recognition/`, `sdxl_steering/`: vision capture and generation steering.
- `toy_pack/`: model-free workflow, catalog, pack-validation and report fixture.

These sources previously lived under `workspaces/`. Commands and imports now
use `examples/`. Source-bound recipe locks have been updated; older cached runs
may legitimately miss reuse when the source/import identity changes.

Create actual studies in a separate research repo that depends on Xenon.
Copy a recipe and use the study’s chosen record format.
[Research prompts](../methodology/README.md) can help develop the idea. The short phase notes here
describe fixture purpose and limits, not an ongoing research backlog.

Keep reusable example sources, small fixtures, configuration and pack metadata
here. Generated artifacts, scratch, exports, checkpoints, caches and reports
are ignored. Shared data and captures retain their Postgres/Modal homes; use
the catalog to discover runs. See [locality](../operations/LOCALITY.md).
