---
benchmark: morebench
phase: 02
version: v3
frozen_date: 2026-04-24
input_artifacts:
  - projects/MOREBENCH/phase_02/outputs/theory_prompt_augmentation_examples.jsonl
  - projects/MOREBENCH/phase_02/outputs/theory_prompt_repair_examples.jsonl
  - projects/MOREBENCH/phase_02/outputs/theory_shortcut_preflight.json
  - projects/MOREBENCH/phase_02/outputs/theory_control_augmentation_examples.jsonl
  - projects/MOREBENCH/phase_02/outputs/theory_wording_variant_examples.jsonl
  - projects/MOREBENCH/phase_02/outputs/action_locus_rewrite_pairs.jsonl
  - projects/MOREBENCH/phase_02/outputs/dilemma_family_keyword_variant_labels.jsonl
  - projects/MOREBENCH/phase_02/reports/dilemma_family_keyword_variant_preflight/report.md
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

## Dilemma Family Keyword Variant Shortcut Stress Test (2026-04-24)

Phase-03 label lexical gate flagged four keyword-defined prompt-side dilemma-family labels as `augmentation_candidate`: `close_relationship_obligation`, `privacy_monitoring_conflict`, `disclosure_transparency_conflict`, `institutional_policy_constraint`. Under the `latent-label-data-augmentation` skill, these labels are tautologically defined (label = keyword set) and require a shortcut stress test before any phase-03 probing.

### What was materialized

- three keyword-variant banks per label (variant_a = original, variant_b = disjoint single-word, variant_c = paraphrastic multi-word), applied to all 500 public-test dilemmas
- `500` rows in `projects/MOREBENCH/phase_02/outputs/dilemma_family_keyword_variant_labels.jsonl`
- cross-variant char-TFIDF preflight with within-variant CV, cross-variant transfer for all 6 ordered pairs per label, and pairwise agreement matrix

### Preflight outcome

All four labels triaged to `iterate_variant_design` — cross-variant char-TFIDF balanced accuracy is in the 0.65-0.75 range for variant_a→variant_b pairs, above the pass threshold but below the shortcut-dominated threshold.

Honest interpretation: the benchmark dilemma vocabulary is dominated by core keyword sets. Extended vocabulary (variant_b: cousin, grandparent, statute, directive, etc.) and paraphrases (variant_c: multi-word patterns) rarely appear in dilemmas without the core vocabulary also appearing. So variant_b and variant_c fire rarely (3-19% True vs 33-46% for variant_a) and when they fire they mostly overlap with variant_a. Pairwise agreement between variants is 0.56-0.97 driven largely by majority-class base rate.

The structural implication: keyword-variant iteration alone cannot break shortcut recoverability for these four constructs on this benchmark. Further keyword engineering is unlikely to drop cross-variant transfer below 0.65. The recommended next repair is LLM-judge relabeling on all 500 dilemmas for each of the four constructs; the variant labels here should be retained as diagnostic artifacts but not promoted as phase-03 probe targets.

### What improved

- four prompt-side keyword labels now have explicit variant banks and cross-variant preflight data on record, rather than being phase-03-ready by default
- the skill's `known_bugs` discipline is satisfied: the manifest now records the structural limit rather than silently handing the labels downstream
- placeholder templates were kept under `specs/` (`dilemma_family_keyword_variants.json`); the materialized `outputs/` JSONL contains only fully-instantiated True/False per variant per row

### Residual confound for this track

- keyword-based definitions of these four dilemma-family labels are structurally unable to beat their own cheap char-TFIDF baseline via keyword iteration on this benchmark
- LLM-judge relabeling is the recommended repair; not materialized in this pass

## Residual Confounds

- the action_locus repair is still only a starter batch, not a full source-balanced rewrite set
- even the new theory repair families should be treated as candidates until their cheap-baseline preflight is explicitly beaten in the chosen retry slice
- the description-only family remains semantically text-decodable and should be treated as a generation-time priming family rather than a clean prompt-side retry family
- structure, length, and person-grammar controls are still unmaterialized
- response-side labels still require fresh generations under the intended protocol
- no behavioral smoke run has been completed on the augmented slice
- four keyword-defined dilemma-family labels remain phase-03-blocked pending LLM-judge relabeling (see `dilemma_family_keyword_variant_labels` entry in the manifest)
