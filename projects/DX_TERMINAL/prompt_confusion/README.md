# Prompt Confusion

This effort directory holds the active conflict-probe redesign work that still
lives in `xenon`.

## Phase Guide

The active landmarks still kept in this repo are:

- [Phase 04](./phase_04/Index.md)
  - first `pipelines_v2` architecture target and readout/arbitration checkpoint
- [Phase 05](./phase_05/Index.md)
  - confound battery and family-level retraction
- [Phase 06](./phase_06/specs/design.md)
  - QA baselines plus combined capture/probe workflow
- [Phase 09](./phase_09/Index.md)
  - implemented benchmark and strongest empirical result so far

Older prompt-confusion phases were moved to the archive repo.

## Operator Stance

Prompt Confusion follows the `pipelines_v2` workflow model from the repo docs:

`workflow.py -> workflow run -> artifacts -> local report`

The default execution stance is:

- capture runs on Modal
- analysis runs on Modal
- reports are built locally from workflow outputs

## Workflow Spec

The active checked-in workflow files live at:

```text
projects/DX_TERMINAL/prompt_confusion/phase_04/specs/arch2_target.py
projects/DX_TERMINAL/prompt_confusion/phase_05/specs/workflow.py
projects/DX_TERMINAL/prompt_confusion/phase_06/specs/workflow.py
projects/DX_TERMINAL/prompt_confusion/phase_09/specs/workflow.py
```

JSON snapshots still exist for reviewability in some phases, but the Python
workflow files are the executable source of truth.

## Canonical Commands

```bash
uv run python -m pipelines_v2.cli workflow plan --file projects/DX_TERMINAL/prompt_confusion/phase_09/specs/workflow.py
uv run python -m pipelines_v2.cli workflow run --file projects/DX_TERMINAL/prompt_confusion/phase_09/specs/workflow.py
uv run python -m pipelines_v2.cli workflow runs --file projects/DX_TERMINAL/prompt_confusion/phase_09/specs/workflow.py
uv run python -m pipelines_v2.cli workflow show --run-id wr_...
```
