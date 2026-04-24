---
benchmark: counselbench
phase: 04
version: v1
frozen_date: 2026-04-24
input_artifacts:
  - projects/COUNSELBENCH/phase_03/docs/03-phase-04-entry-criteria.md
  - projects/COUNSELBENCH/advice_safety/phase_03b/specs/workflow.py
  - projects/COUNSELBENCH/advice_safety/phase_03_eval/specs/workflow.py
---

# CounselBench Phase 04 Intervention Plan

Phase 04 asks whether counseling safety and quality states are causally load-bearing, not just decodable.

Initial intervention target:

- increase boundary adherence on boundary-violating counseling responses
- preserve empathy and specificity
- avoid malformed, evasive, or over-refusal outputs

The first causal operator, once gates pass, should be single-layer full-state interchange over matched donor-target pairs. Steering or erasure comes later, after same-label and random-donor controls are defined.

Phase 04 is blocked until:

- Adv 03b readouts beat cheap/nuisance baselines by at least `0.10` balanced accuracy
- Eval response readouts beat response-text, topic, responder, and length baselines
- a localized layer/span hypothesis exists
- matched cross-label pairs and same-label controls exist

