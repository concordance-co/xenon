# Experiment workspaces

Start with the [experiment guide](../platform/LLM_RESEARCH.md) and
[recipe map](../platform/RESEARCH_HUB.md). Copy a recipe into a new study so its
verification fixture remains reproducible.

Use `workspaces/<study>/<phase>/` (an optional research-thread directory may sit
between them). Each phase owns an executable `workflow.py` and `PHASE.md`; add `pack.json` when
using the experiment-pack validation/preset system. Follow the
[phase template](../methodology/templates/PHASE.md) for `PHASE.md`. Keep the existing research-thread
hypothesis catalog and lifecycle; a pack does not create a second catalog.

Commit source, pack metadata, evaluator definitions and report templates. Put
local smoke fixtures/artifacts under `artifacts/`, scratch under `scratch/`,
bounded audit exports under `exports/`, checkpoints under `checkpoints/`, and
rendered reports under `reports/report_<id>/`. These generated roots are ignored.
Real data and captures retain their assigned Postgres/Modal stores. Catalogs,
not directory listing, discover runs. The optional `.catalog/` is a FileCatalog
for isolated engineering fixtures; ordinary workflows keep team defaults.

See [experiment packs](../platform/EXPERIMENT_PACKS.md) and the [toy fixture](toy_pack/README.md). Actual studies must
state their own matched paper settings, controls, splits and claim boundaries.
