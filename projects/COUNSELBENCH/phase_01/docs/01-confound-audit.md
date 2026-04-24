---
benchmark: counselbench
phase: 01
version: v1
frozen_date: 2026-04-23
input_artifacts:
  - projects/COUNSELBENCH/phase_01/docs/01-label-inventory.csv
---

# CounselBench Confound Audit

## Primary Confounds

- `adv_failure_mode` is likely recoverable from lexical trigger words.
- `medical_boundary_violation` is likely correlated with medication and diagnosis terms.
- `empathy_high` may be confounded by generic reassurance phrases.
- `specificity_high` may be confounded by response length.
- Eval labels are confounded by `responder`, `topic`, and `questionID`.

## Required Controls

- Always run text baselines before interpreting activation probes.
- Group Eval splits by `questionID`.
- Group Adv diagnostic splits by `source_row_id`.
- Track length buckets and lexical trigger flags for every prompt/response.
- Treat baseline-dominant results as `AUGMENTATION_NEEDED`, not as mechanism.

