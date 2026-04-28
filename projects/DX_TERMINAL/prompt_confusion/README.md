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
- [Phase 10](./phase_10/)
  - strict risk-preference extension and trade-size/risk comparison work
- [Phase 11](./phase_11/)
  - additional prompt-confusion follow-up experiments
- [Phase 12](./phase_12/Index.md)
  - three-family geometry, strict family checkpoint, and real-transfer bridge work
- [Phase 13](./phase_13/Index.md)
  - coarse real-signal discovery over production prompt sections
- [Phase 14](./phase_14/PHASE.md)
  - mid-prompt synthetic probe/direction geometry before the next real-transfer pass
- [Phase 15](./phase_15/PHASE.md)
  - real-transfer comparison for Phase 14 section-local direction banks
- [Phase 16](./phase_16/PHASE.md)
  - Phase 13 split audit for cleaner real-data complaint/control buckets

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
projects/DX_TERMINAL/prompt_confusion/phase_13/specs/workflow.py
projects/DX_TERMINAL/prompt_confusion/phase_14/specs/workflow.py
projects/DX_TERMINAL/prompt_confusion/phase_15/specs/workflow.py
projects/DX_TERMINAL/prompt_confusion/phase_16/specs/workflow.py
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

## Operator Notes

Path handling for the newer DX Terminal scripts is centralized in:

- `projects/DX_TERMINAL/prompt_confusion/paths.py`
- `projects/DX_TERMINAL/prompt_confusion/neon.py`

Use those helpers instead of hardcoding:

- repo-local `projects/DX_TERMINAL/...` paths
- `~/.xenon/pipelines_v2/catalog`
- `~/.xenon/pipelines_v2/cache`

`dataset_exports_root(...)` accepts `XENON_DX_TERMINAL_DATASET_EXPORTS_ROOT`
when an explicit export location is needed. When called from a phase script, it
also recognizes a phase-local `dataset_exports/` directory before falling back
to `projects/DX_TERMINAL/dataset_exports`.

The real complaint export now has a repeatable uploader at:

- `projects/DX_TERMINAL/prompt_confusion/phase_12/scripts/upload_complaint_dataset_to_neon.py`

Default Neon destination table:

- `dx_terminal_complaint_dataset_enriched_v1`

Current status:

- the complaint export has been uploaded to Neon with `1090` rows

Typical upload command:

```bash
uv run python projects/DX_TERMINAL/prompt_confusion/phase_12/scripts/upload_complaint_dataset_to_neon.py \
  --input /tmp/complaint_dataset_enriched.parquet \
  --dest-table dx_terminal_complaint_dataset_enriched_v1 \
  --mode modal
```

Notes:

- `--mode local` expects `XENON_NEON_DATABASE_URL` in the local environment
- `--mode modal` uses the Modal secret `xenon-neon`
- the uploader will create or replace the destination table unless `--if-exists append` is passed
