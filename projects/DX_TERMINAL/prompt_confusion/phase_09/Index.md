# Prompt Confusion Phase 09 -- Index

Phase 09 is the first `pipelines_v2`-native scaffold for the prompt-confusion
rebuild after the family/arbitration pivot.

It is intentionally split into two layers:

1. a **new synthetic data generation scaffold** that encodes the updated
   prompt philosophy
2. a **`pipelines_v2` workflow skeleton** for conflict-only capture and
   analysis once the new dataset is published

This phase should be treated as a reset around:

- relational conflict
- descriptive market prompts
- binding settings semantics
- better behavioral sanity and probe calibration

## References

| File | What it covers |
|---|---|
| [../../notes/prompt_redesign_handoff_20260415.md](../notes/prompt_redesign_handoff_20260415.md) | Why we are rebuilding the prompt system at all |
| [design.md](specs/design.md) | Phase 09 scope, prompt philosophy, workflow plan |
| [workflow.py](specs/workflow.py) | `pipelines_v2` workflow skeleton |
| [build_phase_09_dataset.py](scripts/build_phase_09_dataset.py) | New synth-data builder scaffold |
| [upload_phase_09_dataset.py](scripts/upload_phase_09_dataset.py) | Neon uploader for the rebuilt dataset |

## Artifacts

- Neon destination table: `conflict_probe_examples_v5` (target — uploader has not yet run here)
- Report dir: `reports/pipelines_v2/`

## Intended execution model

1. iterate on the Phase 09 prompt/data generator locally
2. publish the rebuilt dataset table to Neon
3. run the text gate and behavior sanity slices
4. only then run prompt-only capture via `pipelines_v2`

## Current status

This folder is a scaffold, not a completed run:

- the dataset builder exists but has not yet been treated as final
- the Neon uploader exists but has not yet been run here
- the `pipelines_v2` workflow exists but expects the rebuilt dataset table to
  be published first
