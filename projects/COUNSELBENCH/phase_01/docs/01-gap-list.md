---
benchmark: counselbench
phase: 01
version: v1
frozen_date: 2026-04-23
input_artifacts:
  - projects/COUNSELBENCH/phase_01/docs/01-latent-label-spec.md
---

# CounselBench Gap List

- Materialize public Eval rows and persist the actual aggregated row-level label set.
- Validate generated-response labels for the target model before treating Adv generations as response-side supervision.
- Add hard negatives that separate empathy from safety, especially supportive-but-unsafe and safe-but-cold responses.
- Add anti-shortcut prompt variants for medication, diagnosis, crisis, and therapy-boundary trigger families.

