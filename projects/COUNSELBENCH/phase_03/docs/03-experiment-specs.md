---
benchmark: counselbench
phase: 03
version: v1
frozen_date: 2026-04-23
input_artifacts:
  - projects/COUNSELBENCH/phase_03/docs/03-analysis-plan.md
---

# CounselBench Experiment Specs

## E1 Prompt Failure Mode

- Data: balanced CounselBench-Adv prompt slice.
- Feature: prompt-end residual.
- Label: `adv_failure_mode`.
- Controls: prompt text baseline, grouped split by `source_row_id`.
- Success: activation readout beats text baseline and survives grouped split. Otherwise route to augmentation.

## E2 Generated Medical Boundary

- Data: generated responses from the Adv slice.
- Feature: generation-end residual.
- Label: provisional `medical_boundary_violation`.
- Controls: generated text baseline, grouped split by `source_row_id`, only after label-support gate passes.
- Gate: at least two non-missing label classes in both grouped train and test splits.
- Success: no strong claim until response-side labels are validated.
- Current first-smoke behavior: record label-support counts and defer trainable generated-boundary baselines/probes if the slice is one-class.

## E2.5 PCA Response-Posture Geometry

- Data: successful generated responses from the Adv slice.
- Features: prompt-end and generation-end residuals.
- Method: `GeometrySpec(method="pca", components=3)` over selected layers.
- Labels/colors: Adv failure mode, topic, prompt/response length bucket, and provisional medical-boundary label.
- Success: interpretable low-dimensional separation that is not trivially identical to topic or length coloring. This is a geometry diagnostic, not proof of mechanism.

## E3 Eval Empathy/Specificity

- Data: aggregated CounselBench-Eval question-response examples.
- Feature: response-context residual.
- Labels: `empathy_high`, `specificity_high`.
- Controls: responder/topic/length tracking, grouped split by `questionID`.

## E4 Warmth Safety Contrast

- Data: hard-negative response set.
- Labels: supportive-but-unsafe and safe-but-cold.
- Status: blocked until response labels and hard negatives exist.
