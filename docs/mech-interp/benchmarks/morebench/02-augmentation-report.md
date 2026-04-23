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

- provisional smoke model: `/models/Qwen/Qwen3-30B-A3B`
- sampled prompts: `20`
- nonempty response rate: `1.0`
- recommendation-present rate: `1.0`
- manual review pass rate: `1.0`
- smoke decision: `pass`

## Residual Confounds

- the action_locus repair is still only a starter batch, not a full source-balanced rewrite set
- structure, length, and person-grammar controls are still unmaterialized
- response-side labels still require fresh generations under the intended protocol
- the smoke run used a provisional model/protocol and produced only a caution result, so this slice is not yet ready to green-light phase 03 on behavior grounds
