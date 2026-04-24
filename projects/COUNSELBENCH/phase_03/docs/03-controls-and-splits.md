---
benchmark: counselbench
phase: 03
version: v1
frozen_date: 2026-04-23
input_artifacts:
  - projects/COUNSELBENCH/phase_01/docs/01-confound-audit.md
---

# CounselBench Controls And Splits

- Adv grouped split: `source_row_id`.
- Eval grouped split: `questionID`.
- Required baselines: `countvectorizer_logreg` over prompt or generated response text.
- Required nuisance tracking: topic, responder, length buckets, medication trigger, diagnosis trigger, crisis trigger, therapy trigger, boundary/ethics trigger.
- Baseline-dominant results route to `AUGMENTATION_NEEDED`.

