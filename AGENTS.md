# Project Overview

This repo is organized around two layers:

- `./pipelines`
  Reusable infrastructure code for ingest, dataset publication, capture,
  analysis, reporting, patching, and related runtime helpers.
- `./projects`
  Human-facing research workspaces. Each project gets its own folder for specs,
  notes, scripts, outputs, and reports.

The main design rule is:

- reusable infrastructure belongs in `pipelines/`
- project-specific work belongs in `projects/<project_name>/`

Archive folders are historical only. Treat them as effectively deleted unless
you are explicitly digging through old work.

## Repo Layout

- `./docs`
  Infrastructure-level docs for how to run and use the platform.
- `./pipelines`
  Raw infrastructure code for ingest, datasets, capture, analysis, patching,
  reporting, and workflow registry logic.
- `./projects`
  Each mech interp research project is associated with a specific folder.
- `./projects/{project_name}/specs/workflow.json`
  Checked-in snapshot of the canonical workflow spec for the project.
- `./projects/{project_name}/docs/`
  Project-specific methodology notes, pitfalls, and operating guidance.
- `./projects/{project_name}/shared/`
  Code needed across multiple phases of the project that does not belong in
  reusable `pipelines/`.
- `./projects/{project_name}/outputs/`
  Local outputs produced while running the project.
- `./projects/{project_name}/reports/report.typ`
  The final Typst report for the project. Only create this when the project is
  mature enough that the user explicitly wants a full project report.
- `./projects/{project_name}/reports/assets/`
  Assets needed for the project report.
- `./projects/{project_name}/reports/scripts/`
  Scripts used to generate project-report assets.
- `./projects/{project_name}/{phase_name}/phase_spec.json`
  Optional phase-level spec. This is a lightweight first-pass structure for a
  bounded sub-effort inside a project.
- `./projects/{project_name}/{phase_name}/scripts/`
  Scripts needed to run or analyze that phase.
- `./projects/{project_name}/{phase_name}/report/`
  Optional phase-level Typst report outputs.
- `./projects/{project_name}/{phase_name}/report/assets/`
  Assets for a phase report.
- `./projects/{project_name}/{phase_name}/report/scripts/`
  Scripts used to generate phase-report assets.

# Canonical Execution Model

The canonical workflow is:

`workflow spec -> published Neon dataset relation -> capture run -> analysis run -> report`

The canonical operator surface is:

```bash
uv run -m pipelines.cli ...
```

For real jobs:

- capture runs on Modal
- analysis runs on Modal
- reports are built locally from analysis outputs

## Source of Truth

There are two useful forms of a spec:

1. Neon-backed canonical workflow spec
   Stored in `workflow_specs` and used by the runtime.
2. Checked-in project snapshot
   Stored in `./projects/{project_name}/specs/workflow.json` for reviewability,
   reproducibility, and agent onboarding.

These are not competing systems. The checked-in JSON is the project-local copy
of the workflow definition; the Neon row is the runtime source of truth used by
the CLI and run registry.

The usual pattern is:

1. explore the data/problem with the agent
2. write `projects/{project_name}/specs/workflow.json`
3. register it with `uv run -m pipelines.cli spec create --file ...`
4. run dataset/capture/analysis/report from the CLI

# Starting a New Project

1. Create `./projects/{project_name}/`
2. Create `./projects/{project_name}/specs/workflow.json`
3. Add `docs/`, `shared/`, `outputs/`, and `reports/` only as needed
4. Register the workflow spec in Neon
5. Run the project through `pipelines.cli`
6. If custom glue is needed, keep it inside `./projects/{project_name}/`

Most new work should be:

- a new workflow spec
- maybe a small local helper script in the project folder

Not:

- ad hoc edits to reusable platform code

# Adding a New Phase to a Project

Use a phase only when the project clearly has a bounded sub-effort with its own
inputs, scripts, or deliverables.

1. Create `./projects/{project_name}/{phase_name}/`
2. Create `phase_spec.json` if the phase needs its own explicit config
3. Add phase-local `scripts/`, `report/`, and `assets/` only as needed
4. Keep the phase aligned with the parent project workflow spec

Do not create phases by default. If a project is simple, keep everything at the
project level.

# Project Spec Definition

The project-level checked-in spec should mirror the canonical workflow spec
stored in Neon.

Minimum shape:

```json
{
  "id": "project_id",
  "name": "Project Name",
  "description": "Short description",
  "version": 1,
  "dataset": {
    "source": {
      "mode": "table",
      "table": "interp_examples_v0"
    },
    "filters": {},
    "label": {
      "mode": "direct",
      "expression_sql": "decision_type"
    },
    "split": {
      "mode": "random_stratified",
      "train_pct": 70,
      "val_pct": 15,
      "test_pct": 15
    },
    "probe_defaults": {
      "data_source": "router",
      "pooling": "last_token",
      "n_folds": 5
    },
    "publish_mode": "view"
  },
  "capture": {
    "model": "Qwen/Qwen3-8B",
    "layers": [16, 24, 32],
    "pooling": "mean_pool",
    "router": true,
    "residual": true
  },
  "analysis": {
    "methods": ["probe"],
    "targets": ["workflow_label"],
    "data_source": "router",
    "pooling": "last_token"
  },
  "report": {
    "output_dir": "projects/project_id/reports"
  }
}
```

Required ideas:

- `dataset`
  Defines where rows come from, how they are filtered, how labels are created,
  and how the dataset is published.
- `capture`
  Defines model, layers, pooling, and whether router/residual data is captured.
- `analysis`
  Defines which analysis methods and targets to run.
- `report`
  Defines where local report outputs should be written.

Defaults:

- dataset publication should default to a named Neon view
- capture and analysis should default to Modal execution
- reports should be built locally

# Phase Spec Definition

There is not yet a rigid canonical `phase_spec.json` schema. For now, use a
simple best-effort shape that extends or narrows the parent project spec.

Recommended first-pass shape:

```json
{
  "id": "phase_id",
  "name": "Phase Name",
  "description": "Short description",
  "project_id": "project_id",
  "inherits": "projects/project_id/specs/workflow.json",
  "goal": "What this phase is trying to prove or produce",
  "overrides": {
    "dataset": {},
    "capture": {},
    "analysis": {},
    "report": {}
  },
  "outputs": {
    "output_dir": "projects/project_id/phase_name/report",
    "notes": "Optional notes about expected artifacts"
  }
}
```

Interpretation:

- `inherits`
  Points at the parent workflow snapshot the phase is based on.
- `goal`
  States the research purpose in plain language.
- `overrides`
  Contains only the parts of the parent workflow spec that change for this
  phase.
- `outputs`
  Documents where phase-local artifacts should go.

This is documentation and organization scaffolding first, not a strict runtime
contract. Keep it lightweight.

# Running Specs

The canonical execution path is through `pipelines.cli`.

## Register a project workflow spec

```bash
uv run -m pipelines.cli spec create --file projects/{project_name}/specs/workflow.json
```

## Inspect registered specs

```bash
uv run -m pipelines.cli spec list
uv run -m pipelines.cli spec show --id <spec_id>
```

## Publish the dataset relation

```bash
uv run -m pipelines.cli dataset build --spec <spec_id>
uv run -m pipelines.cli publication list --spec <spec_id>
```

## Run capture on Modal

```bash
uv run -m pipelines.cli capture run --spec <spec_id> --output-dir projects/{project_name}/outputs/<run_name>
```

## Run analysis on Modal

```bash
uv run -m pipelines.cli analysis run --capture-run <run_id> --output-dir projects/{project_name}/outputs/<analysis_name>
```

## Build a local report

```bash
uv run -m pipelines.cli report build --analysis-run <run_id>
```

## Practical Agent Workflow

When starting fresh on a new project:

1. read `README.md`, `docs/WORKFLOW.md`, and this file
2. inspect Neon/data sources directly
3. create or refine `projects/{project_name}/specs/workflow.json`
4. register the spec in Neon
5. run the workflow through `pipelines.cli`
6. keep project-specific scripts, notes, and deliverables inside the project folder

If something needs reusable infrastructure support, add it to `pipelines/`.
If it is project-specific, keep it in `projects/`.
