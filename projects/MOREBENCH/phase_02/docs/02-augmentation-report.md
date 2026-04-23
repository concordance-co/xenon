---
benchmark: morebench
phase: 02
version: v2
frozen_date: 2026-04-23
input_artifacts:
  - projects/MOREBENCH/phase_02/outputs/theory_prompt_augmentation_examples.jsonl
  - projects/MOREBENCH/phase_02/outputs/theory_prompt_repair_examples.jsonl
  - projects/MOREBENCH/phase_02/outputs/theory_shortcut_preflight.json
  - projects/MOREBENCH/phase_02/outputs/theory_control_augmentation_examples.jsonl
  - projects/MOREBENCH/phase_02/outputs/theory_wording_variant_examples.jsonl
  - projects/MOREBENCH/phase_02/outputs/action_locus_rewrite_pairs.jsonl
---

# MoReBench 02 Augmentation Report

## What Was Materialized

- `150` legacy direct theory-exposed prompt variants
- `30` legacy structurally matched neutral wrapper controls
- `150` legacy same-label wording variants for theory prompts
- `2250` shortcut-stress-test theory prompt rows across name, alias, description, and factorial variants
- `180` shortcut-stress-test theory controls and mismatch decoys
- `10` matched advisor/agent rewrite pairs

## What Improved

- the old explicit-theory family is no longer treated as clean by default; it is retained as known-broken for traceability
- theory prompt repair now includes factorial variants designed to break one-to-one recoverability from names or fixed anchors
- shortcut preflight is now materialized as a benchmark artifact before any prompt-side retry
- placeholder templates have been removed from materialized output data
- action_locus now has a non-zero rewrite batch built from coherent agent-owned scenarios instead of prefix-only edits

## Shortcut Preflight Snapshot

- legacy family cue-text bag-of-words balanced accuracy: `1.0`
- recommended prompt-side diagnostic family: `alias_only`
- strongest held-out alias baseline for the diagnostic family: `0.675`
- explicit alias-token rule score on raw alias rows: `1.0`
- recommended generation-time priming family: `description_only`
- strongest held-out description baseline for the priming family: `1.0`
- retry rule: Retry prompt-side theory work only on a family whose strongest held-out alias/description text baselines no longer solve the label cleanly.

## Behavioral Smoke

- not yet run

## Residual Confounds

- the action_locus repair is still only a starter batch, not a full source-balanced rewrite set
- even the new theory repair families should be treated as candidates until their cheap-baseline preflight is explicitly beaten in the chosen retry slice
- the description-only family remains semantically text-decodable and should be treated as a generation-time priming family rather than a clean prompt-side retry family
- structure, length, and person-grammar controls are still unmaterialized
- response-side labels still require fresh generations under the intended protocol
- no behavioral smoke run has been completed on the augmented slice
