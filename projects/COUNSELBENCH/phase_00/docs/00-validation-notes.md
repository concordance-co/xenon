---
benchmark: counselbench
phase: 00
version: v1
frozen_date: 2026-04-23
input_artifacts:
  - projects/COUNSELBENCH/phase_00/docs/00-validation-memo.md
---

# CounselBench Validation Notes

- `CounselBench-Adv` is stored in a wide shape: one row contains six different prompt columns. Project code melts this into one example per prompt; this is standard dataframe wide-to-long terminology.
- `CounselBench-Eval` rows are annotation rows, not independent response examples. Any analysis must aggregate by stable question-response identity.
- `medical_advice_score` is treated as a categorical expert boundary flag, not as a 1-5 score.
- The initial generated-response `medical_boundary_violation` label in the phase-03 workflow is a lexical heuristic for smoke/readout scaffolding only.
