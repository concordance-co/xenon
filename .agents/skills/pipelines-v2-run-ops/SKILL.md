---
name: pipelines-v2-run-ops
description: Use when operating existing `pipelines_v2` workflows in this repo. Covers planning, running, local run tracking, inspecting runs, resuming failed runs, rerunning a step, rerunning from a step, and regenerating local reports from existing artifacts.
---

# Pipelines v2 Run Operations

Use this skill when the workflow file already exists and the task is to operate
it rather than redesign it.

For workflow authoring or structural changes, use `constructing-workflows`.

## Start Here

1. Identify the workflow file under `projects/.../specs/workflow.py`.
2. Plan it first:

```bash
uv run python -m pipelines_v2.cli workflow plan --file path/to/workflow.py
```

3. Inspect recent runs before launching expensive work:

```bash
uv run python -m pipelines_v2.cli workflow runs --file path/to/workflow.py
```

## Core Commands

Run:

```bash
uv run python -m pipelines_v2.cli workflow run --file path/to/workflow.py
```

Inspect one run:

```bash
uv run python -m pipelines_v2.cli workflow show --run-id wr_...
```

Resume the latest failed run:

```bash
uv run python -m pipelines_v2.cli workflow resume --file path/to/workflow.py --latest-failed
```

Rerun one step only:

```bash
uv run python -m pipelines_v2.cli workflow rerun-step --file path/to/workflow.py --run-id wr_... --step report
```

Rerun from one step through downstream dependents:

```bash
uv run python -m pipelines_v2.cli workflow rerun-from-step --file path/to/workflow.py --run-id wr_... --step capture_prompt_eos_router
```

## Semantics

- `resume`
  - same run id
  - intended for failure recovery
  - completed sibling branches should be reused, not rerun
- `rerun-step`
  - new run id
  - reuses upstream artifacts from the source run
  - reruns only the named step
- `rerun-from-step`
  - new run id
  - reuses upstream artifacts from the source run
  - reruns the named step and downstream dependents

Use `resume` for interrupted or failed work. Use `rerun-*` for intentional branching.

## Local Run Registry

The CLI mirrors workflow state into:

- `~/.xenon/pipelines_v2/catalog`

Override if needed with:

```bash
--local-catalog-root /path/to/catalog
```

If an external catalog is also configured, the CLI still writes the local
registry. Use the local registry first for operator UX.

## Reports

For local report steps:

- `report.json`, `summary.json`, and `report.md` are published under the report output dir
- direct `OperationArtifact` inputs are copied into:
  - `report_<id>/results/{step_name}_results.json`
- direct capture inputs are summarized from manifests and are not localized just to build the report

If you only need a fresh report from existing artifacts, use:

- `rerun-step --step report`

Do not rebuild the whole workflow for that.

## Remote Runtime Notes

- `runtime_app_id` is persisted per step when the runner reports one
- use `workflow show --run-id ...` to inspect which remote app ran which step
- for Modal MoE router capture, remember the current constraints:
  - `enforce_eager=True`
  - `enable_prefix_caching=False`
  - `max_num_seqs=1`

## Failure Triage

When a run fails:

1. inspect the run record
2. inspect step statuses
3. identify whether the right recovery action is:
   - `resume`
   - `rerun-step`
   - `rerun-from-step`

Do not manually reconstruct prior artifacts unless the CLI surface is genuinely missing a needed operation.
