---
benchmark: morebench
phase: 02
version: v1
frozen_date: 2026-04-22
input_artifacts:
  - projects/MECH_INTERP/morebench/phase_02/outputs/theory_prompt_augmentation_examples.jsonl
  - projects/MECH_INTERP/morebench/phase_02/outputs/theory_control_augmentation_examples.jsonl
  - projects/MECH_INTERP/morebench/phase_02/outputs/theory_wording_variant_examples.jsonl
  - projects/MECH_INTERP/morebench/phase_02/outputs/action_locus_rewrite_pairs.jsonl
---

# MoReBench 02 Augmentation Report

## What Was Materialized

- `150` direct theory-exposed prompt variants
- `30` structurally matched neutral wrapper controls
- `150` same-label wording variants for theory prompts
- `10` matched advisor/agent rewrite pairs

## What Improved

- theory is now explicit in prompt text with framework-specific anchors
- the neutral control family is fully substituted and structurally matched to the theory prompt skeleton
- placeholder templates have been removed from materialized output data
- action_locus now has a non-zero rewrite batch built from coherent agent-owned scenarios instead of prefix-only edits

## Behavioral Smoke

- not yet run

## Residual Confounds

- the action_locus repair is still only a starter batch, not a full source-balanced rewrite set
- structure, length, and person-grammar controls are still unmaterialized
- response-side labels still require fresh generations under the intended protocol
- no behavioral smoke run has been completed on the augmented slice
