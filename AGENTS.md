# Project Overview

This repo is organized around two layers:

- `./pipelines`
  Reusable infrastructure code for ingest, dataset publication, capture,
  analysis, reporting, patching, and related runtime helpers.
- `./projects`
  Human-facing project workspaces. Each umbrella project gets its own folder,
  and each project may contain multiple subprojects, each with its own phase
  folders as needed.

The main design rule is:

- reusable infrastructure belongs in `pipelines/`
- project- and phase-specific work belongs in `projects/<project_name>/...`

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
- `./projects/{project_name}/project_spec.json`
  Organizational spec for the umbrella project.
- `./projects/{project_name}/docs/`
  Project-level methodology notes, pitfalls, and operating guidance.
- `./projects/{project_name}/shared/`
  Code needed across multiple phases of the project that does not belong in
  reusable `pipelines/`.
- `./projects/{project_name}/outputs/`
  Local outputs produced at the umbrella-project level.
- `./projects/{project_name}/reports/report.typ`
  The final Typst report for the project. Only create this when the project is
  mature enough that the user explicitly wants a full project report.
- `./projects/{project_name}/reports/assets/`
  Assets needed for the project report.
- `./projects/{project_name}/reports/scripts/`
  Scripts used to generate project-report assets.
- `./projects/{project_name}/{subproject}/{phase_name}/phase_spec.json`
  Optional phase-level spec. This is a lightweight first-pass structure for a
  bounded sub-effort inside a project.
- `./projects/{project_name}/{subproject}/{phase_name}/specs/workflow.json`
  Checked-in snapshot of the executable workflow spec for the phase.
- `./projects/{project_name}/{subproject}/{phase_name}/scripts/`
  Scripts needed to run or analyze that phase.
- `./projects/{project_name}/{subproject}/{phase_name}/reports/`
  Optional phase-level Typst report outputs.
- `./projects/{project_name}/{subproject}/{phase_name}/reports/assets/`
  Assets for a phase report.
- `./projects/{project_name}/{subproject}/{phase_name}/reports/scripts/`
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
2. Checked-in phase workflow snapshot
   Stored in `./projects/{project_name}/{subproject}/{phase_name}/specs/workflow.json`
   for reviewability, reproducibility, and agent onboarding.
3. Checked-in project and phase organizational specs
   Stored as `project_spec.json` and `phase_spec.json` to describe structure,
   goals, and deliverables, not to replace the executable workflow spec.

These are not competing systems.

- `workflow_specs` in Neon is the runtime source of truth.
- checked-in `specs/workflow.json` is the repo-local mirror of the executable spec.
- `project_spec.json` and `phase_spec.json` are organizational scaffolding.

The usual pattern is:

1. explore the data/problem with the agent
2. write `projects/{project_name}/{subproject}/{phase_name}/specs/workflow.json`
3. register it with `uv run -m pipelines.cli spec create --file ...`
4. run dataset/capture/analysis/report from the CLI

# Starting a New Project

1. Create `./projects/{project_name}/`
2. Create `./projects/{project_name}/project_spec.json`
3. Create `./projects/{project_name}/{subproject}/{phase_name}/`
4. Create `./projects/{project_name}/{subproject}/{phase_name}/phase_spec.json`
5. Create `./projects/{project_name}/{subproject}/{phase_name}/specs/workflow.json`
6. Register the workflow spec in Neon
7. Run the project through `pipelines.cli`
8. If custom glue is needed, keep it inside the project or phase folder

Shortcut:

```bash
uv run -m pipelines.cli project init --project {project_name}
```

This creates a default first phase at
`projects/{project_name}/{subproject}/phase_01/`. Pass `--phase {phase_name}` if you
want a different initial phase name.

Most new work should be:

- a new phase folder
- a new workflow spec inside that phase
- maybe a small local helper script in that phase

Not:

- ad hoc edits to reusable platform code

# Adding a New Phase to a Project

Use a phase only when the project clearly has a bounded sub-effort with its own
inputs, scripts, or deliverables.

1. Create `./projects/{project_name}/{subproject}/{phase_name}/`
2. Create `phase_spec.json` if the phase needs its own explicit config
3. Add phase-local `scripts/`, `reports/`, and `assets/` only as needed
4. Keep the phase aligned with the parent project workflow spec

Do not create phases by default. If a project is simple, keep everything at the
project level.

# Project Spec Definition

The project-level checked-in spec is organizational, not the executable runtime
contract.

Minimum shape:

```json
{
  "id": "project_id",
  "name": "Project Name",
  "description": "Short description",
  "version": 1,
  "goal": "Umbrella research goal",
  "subprojects": [
    "subproject_a",
    "subproject_b"
  ],
  "defaults": {
    "capture_execution": "modal",
    "analysis_execution": "modal",
    "report_execution": "local"
  }
}
```

Required ideas:

- `goal`
  Defines the umbrella objective of the project.
- `subprojects`
  Lists the subprojects under the umbrella project.
- `defaults`
  Documents high-level execution defaults shared across phases.

Defaults:

- project-level docs should not replace the phase workflow spec
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
  "inherits": "projects/project_id/project_spec.json",
  "goal": "What this phase is trying to prove or produce",
  "workflow_specs": [
    "projects/project_id/{subproject}/phase_id/specs/workflow.json"
  ],
  "overrides": {
    "dataset": {},
    "capture": {},
    "analysis": {},
    "report": {}
  },
  "outputs": {
    "output_dir": "projects/project_id/{subproject}/phase_id/reports",
    "notes": "Optional notes about expected artifacts"
  }
}
```

Interpretation:

- `inherits`
  Points at the parent project spec.
- `goal`
  States the research purpose in plain language.
- `workflow_specs`
  Lists the checked-in executable workflow specs associated with the phase.
- `overrides`
  Contains phase-level defaults or notes that narrow the project-level intent.
- `outputs`
  Documents where phase-local artifacts should go.

This is documentation and organization scaffolding first, not a strict runtime
contract. Keep it lightweight.

# Running Specs

The canonical execution path is through `pipelines.cli`.

## Register a project workflow spec

```bash
uv run -m pipelines.cli spec create --file projects/{project_name}/{subproject}/{phase_name}/specs/workflow.json
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
uv run -m pipelines.cli capture run --spec <spec_id> --output-dir projects/{project_name}/{subproject}/{phase_name}/outputs/<run_name>
```

## Run analysis on Modal

```bash
uv run -m pipelines.cli analysis run --capture-run <run_id> --output-dir projects/{project_name}/{subproject}/{phase_name}/outputs/<analysis_name>
```

## Build a local report

```bash
uv run -m pipelines.cli report build --analysis-run <run_id>
```

## Practical Agent Workflow

When starting fresh on a new project:

1. read `README.md`, `docs/WORKFLOW.md`, and this file
2. inspect Neon/data sources directly
3. create or refine `projects/{project_name}/{subproject}/{phase_name}/specs/workflow.json`
4. register the spec in Neon
5. run the workflow through `pipelines.cli`
6. keep project-specific scripts, notes, and deliverables inside the project or phase folder

If something needs reusable infrastructure support, add it to `pipelines/`.
If it is project-specific, keep it in `projects/`.
