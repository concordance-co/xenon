# Project Overview

This repo is organized around three top-level areas:

- `./pipelines_v2`
  Active reusable infrastructure for dataset access, capture, analysis,
  reporting, workflow orchestration, runtime integration, and artifact storage.
- `./pipelines`
  Legacy infrastructure for the older Neon/publication-driven runtime.
  Maintain it only when explicitly working on an unmigrated legacy workflow.
- `./projects`
  Human-facing project workspaces. Each umbrella project gets its own folder,
  and each project may contain multiple subprojects, each with its own phase
  folders as needed.

The main design rule is:

- reusable infrastructure for new work belongs in `pipelines_v2/`
- legacy-only infrastructure stays in `pipelines/`
- project- and phase-specific work belongs in `projects/<project_name>/...`

Archive folders are historical only. Treat them as effectively deleted unless
you are explicitly digging through old work.

## Repo Layout

- `./docs`
  Infrastructure-level docs for how to run and use the platform.
- `./pipelines_v2`
  Active workflow / artifact-oriented platform code.
- `./pipelines`
  Older runtime code for Neon/publication-driven workflows. Deprecated for new
  development.
- `./projects`
  Each mech interp research project is associated with a specific folder.
- `./projects/{project_name}/project_spec.json`
  Organizational spec for the umbrella project.
- `./projects/{project_name}/docs/`
  Project-level methodology notes, pitfalls, and operating guidance.
- `./projects/{project_name}/shared/`
  Code needed across multiple phases of the project that does not belong in
  reusable platform code.
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
- `./projects/{project_name}/{subproject}/{phase_name}/specs/workflow.py`
  Primary checked-in executable `pipelines_v2` workflow file for the phase.
- `./projects/{project_name}/{subproject}/{phase_name}/specs/workflow.json`
  Optional checked-in JSON snapshot of the workflow for reviewability or
  interchange. The Python workflow file remains the canonical authoring surface.
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

`workflow.py -> workflow plan/run -> capture + analysis artifacts -> local report`

The canonical operator surface is:

```bash
uv run python -m pipelines_v2.cli workflow ...
```

For real jobs:

- capture runs on Modal
- analysis runs on Modal
- reports are built locally from analysis outputs

For new development:

- use `pipelines_v2`
- treat `pipelines.cli` as a legacy surface unless the user explicitly asks to
  work on an older workflow that has not moved yet

## Source of Truth

There are three useful forms of a spec:

1. Checked-in Python workflow file
   Stored in `./projects/{project_name}/{subproject}/{phase_name}/specs/workflow.py`
   and used directly by `pipelines_v2.cli`.
2. Optional checked-in workflow snapshot
   Stored in `./projects/{project_name}/{subproject}/{phase_name}/specs/workflow.json`
   for reviewability, reproducibility, and agent onboarding.
3. Checked-in project and phase organizational specs
   Stored as `project_spec.json` and `phase_spec.json` to describe structure,
   goals, and deliverables, not to replace the executable workflow spec.

These are not competing systems.

- checked-in `specs/workflow.py` is the primary executable source of truth for
  `pipelines_v2`.
- checked-in `specs/workflow.json` is an optional snapshot or interchange form.
- `project_spec.json` and `phase_spec.json` are organizational scaffolding.

The usual pattern is:

1. explore the data/problem with the agent
2. write `projects/{project_name}/{subproject}/{phase_name}/specs/workflow.py`
3. optionally emit or update `specs/workflow.json` if the phase wants a checked-in snapshot
4. run `workflow plan` and `workflow run` from `pipelines_v2.cli`
5. use `workflow runs`, `workflow show`, `workflow resume`, `workflow rerun-step`, and `workflow rerun-from-step` for follow-up operations

For long-running jobs, prefer `workflow run --logging INFO` so the CLI prints
structured progress and remote app ids while the run is active.

# Starting a New Project

1. Create `./projects/{project_name}/`
2. Create `./projects/{project_name}/project_spec.json`
3. Create `./projects/{project_name}/{subproject}/{phase_name}/`
4. Create `./projects/{project_name}/{subproject}/{phase_name}/phase_spec.json`
5. Create `./projects/{project_name}/{subproject}/{phase_name}/specs/workflow.py`
6. Optionally create `./projects/{project_name}/{subproject}/{phase_name}/specs/workflow.json`
7. Run the project through `pipelines_v2.cli`
8. If custom glue is needed, keep it inside the project or phase folder

Most new work should be:

- a new phase folder
- a new `pipelines_v2` workflow inside that phase
- maybe a small local helper script in that phase

Not:

- starting from the legacy `pipelines.cli` path unless the task is explicitly
  about legacy infrastructure
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
    "projects/project_id/{subproject}/phase_id/specs/workflow.py"
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

The canonical execution path is through `pipelines_v2.cli`.

## Plan a workflow

```bash
uv run python -m pipelines_v2.cli workflow plan --file projects/{project_name}/{subproject}/{phase_name}/specs/workflow.py
```

## Run a workflow

```bash
uv run python -m pipelines_v2.cli workflow run --file projects/{project_name}/{subproject}/{phase_name}/specs/workflow.py --logging INFO
```

## Inspect workflow runs

```bash
uv run python -m pipelines_v2.cli workflow runs --file projects/{project_name}/{subproject}/{phase_name}/specs/workflow.py
uv run python -m pipelines_v2.cli workflow show --run-id <run_id>
```

`workflow show` now includes the latest locally persisted progress snapshot for
the run and its steps. The progress store lives under the local registry root in
`~/.xenon/pipelines_v2/catalog/workflow_progress/` by default.

## Resume a failed run

```bash
uv run python -m pipelines_v2.cli workflow resume --file projects/{project_name}/{subproject}/{phase_name}/specs/workflow.py --latest-failed
```

## Rerun one step

```bash
uv run python -m pipelines_v2.cli workflow rerun-step --file projects/{project_name}/{subproject}/{phase_name}/specs/workflow.py --run-id <run_id> --step report
```

## Rerun from a step through downstream dependents

```bash
uv run python -m pipelines_v2.cli workflow rerun-from-step --file projects/{project_name}/{subproject}/{phase_name}/specs/workflow.py --run-id <run_id> --step capture_prompt_eos_router
```

Workspace defaults for `pipelines_v2` can live in repo-root `xenon.toml`. Use
that for shared defaults such as the external catalog env var. The CLI still
writes the local run registry under `~/.xenon/pipelines_v2/catalog`.

## Workspace Modal Defaults

Shared Modal runner defaults for `pipelines_v2` live under
`[pipelines_v2.modal]` in repo-root `xenon.toml`.

Current keys:

- `model_volume`
  - Modal volume name to mount for model weights on Modal GPU runners.
- `model_volume_path`
  - Mount path for the model volume. Default is `/models`.
- `vllm_cache_volume`
  - Modal volume name to use for the vLLM torch.compile cache. If omitted and
    torch-compile caching is enabled, it defaults to `model_volume`.
- `vllm_cache_root`
  - Value to use for `VLLM_CACHE_ROOT`. If omitted and torch-compile caching is
    enabled, it defaults to `model_volume_path`.
- `use_vllm_torch_compile_cache`
  - When `true`, the CLI fills missing `VLLM_CACHE_ROOT` on Modal GPU runners
    and ensures the cache volume mount is created with
    `create_if_missing=true` and `commit_on_success=true`.

Current repo default:

```toml
[pipelines_v2.modal]
model_volume = "xenon-models"
model_volume_path = "/models"
vllm_cache_volume = "xenon-models"
vllm_cache_root = "/models"
use_vllm_torch_compile_cache = true
```

Operational rules:

- Prefer a shared cache root such as `/models`, not a workflow-specific cache
  prefix. vLLM already creates its own hashed subdirectories under
  `torch_compile_cache/` and `torch_aot_compile/`.
- These defaults only fill missing settings. Explicit runner-spec mounts or
  `env={"VLLM_CACHE_ROOT": ...}` still win.
- The defaults apply to Modal GPU runners. Local runners should set
  `LocalResources(env={"VLLM_CACHE_ROOT": ...})` explicitly when persistent
  local caching is desired.
- If the model volume and cache volume are the same and the cache root is under
  the model mount, use one shared `/models` mount and let the CLI upgrade it
  for cache persistence rather than adding a second overlapping mount.

## Practical Agent Workflow

When starting fresh on a new project:

1. read `README.md`, `docs/WORKFLOW.md`, and this file
2. inspect the relevant data sources directly
3. create or refine `projects/{project_name}/{subproject}/{phase_name}/specs/workflow.py`
4. optionally update `specs/workflow.json` if the phase keeps a checked-in snapshot
5. run the workflow through `pipelines_v2.cli`
6. use `resume` or `rerun-*` rather than manually reconstructing partial runs
7. keep project-specific scripts, notes, and deliverables inside the project or phase folder

When a workflow needs local helper code at remote runtime, keep
`local_python_sources` narrow and explicit. Do not mount `"."` into Modal
unless the whole workspace is intentionally required.

If something needs reusable infrastructure support for new work, add it to `pipelines_v2/`.
If the task is explicitly about the legacy runtime, add it to `pipelines/`.
If it is project-specific, keep it in `projects/`.
