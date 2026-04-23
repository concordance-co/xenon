---
benchmark: morebench
phase: 01
version: v1
frozen_date: 2026-04-22
input_artifacts:
  - projects/MECH_INTERP/morebench/phase_01/outputs/prompt_side_labels.json
  - projects/MECH_INTERP/morebench/phase_01/outputs/response_side_labels.json
---

# MoReBench 01 Derivability Report

## Labels Derivable Now

- `action_locus` as metadata surface, but not as a clean probe target
- `dilemma_structure`
- `domain_topic`
- `theory_identity` as metadata only

## Labels Not Yet Derivable Reliably

- `stakeholder_tradeoff_density`: needs a validated counting policy and gold slice
- all response-side labels: need fresh generations under the intended protocol
