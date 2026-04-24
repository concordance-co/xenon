---
benchmark: counselbench
phase: 03
version: v1
frozen_date: 2026-04-23
input_artifacts:
  - projects/COUNSELBENCH/phase_01/docs/01-latent-label-spec.md
---

# CounselBench Feature Hypotheses

- `adv_failure_mode` -> prompt-side representation of counseling risk trigger family.
- `medical_boundary_violation` -> response policy state for unsafe diagnosis, medication, or therapy prescription.
- `empathy_high` -> supportive attunement / validation representation.
- `specificity_high` -> contextual personalization representation.
- `toxicity_or_judgmental` -> harmful or judgmental response posture.
- `supportive_but_unsafe` vs `safe_but_cold` -> separability of warmth and clinical-boundary control.
- `prompt_end_adv_geometry` -> expert-authored Adv families may occupy a structured prompt-risk space, but this is expected to be surface-recoverable until augmentation.
- `generation_end_posture_geometry` -> generated responses may organize along clinical-boundary, specificity, warmth, and toxicity axes; PCA is a first diagnostic for that posture space.
