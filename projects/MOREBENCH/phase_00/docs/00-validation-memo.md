---
benchmark: morebench
phase: 00
version: v1
frozen_date: 2026-04-22
input_artifacts:
  - projects/MOREBENCH/phase_00/outputs/benchmark_framing.json
---

# MoReBench Phase 00 Benchmark Validation

## Bottom Line

Phase 00 succeeded.
`MoReBench` is a strong benchmark-first substrate, but the current public split already tells us two things we should treat as hard contract facts before phase 02:

- `action_locus` is effectively not probeable on the current public split without augmentation
- `theory_identity` is promising, but it is not yet a clean prompt-side variable unless the runtime prompt exposes `THEORY`

## Key Counts

- public rows: `500`
- theory rows: `150`
- source-controlled mixed-role cells for `action_locus`: `0`
- source-and-type-controlled mixed-role cells for `action_locus`: `0`
- exact theory/public dilemma overlap: `18` of `30`

## High-Priority Confounds

- `source_role_aliasing_public`
- `source_type_aliasing_public`
- `domain_topic_imbalance`
- `action_locus_not_probeable_without_augmentation`
- `theory_not_automatically_prompt_side`

## Recommendation

Proceed to phase 01, but treat theory and action-locus as likely augmentation-bound from the start.

