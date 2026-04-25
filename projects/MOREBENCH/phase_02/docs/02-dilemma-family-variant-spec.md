---
benchmark: morebench
phase: 02
version: v1
frozen_date: 2026-04-24
input_artifacts:
  - projects/MOREBENCH/phase_02/specs/dilemma_family_keyword_variants.json
  - projects/MOREBENCH/phase_02/outputs/dilemma_family_keyword_variant_labels.jsonl
  - projects/MOREBENCH/phase_02/reports/dilemma_family_keyword_variant_preflight/report.md
  - projects/MOREBENCH/phase_03/reports/experiment_03_label_lexical_gate/report.md
---

# MoReBench 02 Dilemma Family Keyword Variant Spec

## Goal

Shortcut stress test for keyword-defined dilemma-family prompt labels. The
phase-03 label lexical gate identified four prompt-side keyword labels whose
construct is coherent but whose current keyword sets are the limiting factor:
`close_relationship_obligation`, `privacy_monitoring_conflict`,
`disclosure_transparency_conflict`, `institutional_policy_constraint`. Each
label is defined by a single keyword list (variant_a). A classifier trained on
these labels using char-TFIDF trivially recovers the keyword surface — the
label IS the surface feature.

This spec repairs that by constructing additional keyword-variant banks per
label. Variants preserve the semantic construct but minimize character-level
overlap with variant_a so that cross-variant char-TFIDF transfer becomes a
real anti-shortcut test. Phase 02 is complete for this track only when
cross-variant transfer drops below a pre-committed threshold.

## Why This Exists

The phase-03 label lexical gate (`experiment_03_label_lexical_gate`) placed
these four labels in the `augmentation_candidate` bucket: CV char-TFIDF in
the 0.66-0.84 range with meaningful CV-vs-source-family-holdout gaps. The
raw CV signal is driven by the keyword definitions themselves. The proper
repair per the `latent-label-data-augmentation` skill §Preferred repair
moves.5 is a shortcut stress test with a paraphrase bank, not more data.

Labels deliberately excluded from this track with reasons:

- `public_safety`: construct too broad; keywords are neutral moral vocabulary
- `uncertainty_incomplete_info`: measures uncertainty language not construct
- `authority_constraint`: domain-noun confound (doctor, teacher, therapist)
- `loyalty_relationship`: heavy overlap with `close_relationship_obligation`
- `fairness_access_conflict`: construct undercounted; needs LLM-judge not keyword variants
- `public_harm_resource_tradeoff`: broad, overlaps `public_safety`

## Design Principles

Each label gets three variant banks:

- `variant_a`: original keyword list from the label definition
- `variant_b`: single-word keywords capturing the same construct, disjoint from variant_a
- `variant_c`: multi-word paraphrastic patterns that express the construct without variant_a single-word tokens where possible

The design rule: maximize within-construct semantic coverage while minimizing
character-level overlap with variant_a. Perfect char-n-gram disjointness is
not achievable because English morphology shares suffixes, but specific
construct-anchoring vocabulary should not overlap.

## Materialized Counts

- total dilemmas scored: `500`
- labels: `4`
- variants per label: `3`
- total binary label columns produced: `12`

Per-label True rates:

- `close_relationship_obligation`: variant_a=190, variant_b=17, variant_c=5
- `privacy_monitoring_conflict`: variant_a=175, variant_b=23, variant_c=3
- `disclosure_transparency_conflict`: variant_a=168, variant_b=68, variant_c=14
- `institutional_policy_constraint`: variant_a=231, variant_b=97, variant_c=18

## Exit Gate

Cross-variant char-TFIDF transfer is the exit gate:

- `pass`: cross-variant BA ≤ 0.65 for every ordered variant pair AND pairwise agreement ≥ 0.70
- `iterate_variant_design`: cross-variant BA in (0.65, 0.75) — variants not disjoint enough
- `shortcut_dominated`: cross-variant BA ≥ 0.75 — mark in `known_bugs`
- `variants_incoherent`: pairwise agreement < 0.70 — variants don't measure the same construct

## Current Phase-02 Status

All four labels triaged to `iterate_variant_design` in the first preflight
pass. Honest interpretation: benchmark dilemmas mostly express these
constructs through their core vocabulary (variant_a), and extended
vocabulary (variant_b) or paraphrases (variant_c) rarely appear without the
core vocabulary also appearing in the same dilemma. The cross-variant
char-TFIDF AUROC of 0.70-0.78 reflects high within-dilemma keyword
co-occurrence rather than variants capturing genuinely disjoint lexical
surfaces.

The track is not shortcut-dominated (no cross-variant BA ≥ 0.75) but also
not phase-02-ready. A structural limitation of the benchmark prevents
iteration-to-pass from being productive: you cannot design lexically
disjoint variants when the benchmark authors consistently use core
vocabulary.

## Assets

Builder:

- `projects/MOREBENCH/phase_02/scripts/build_dilemma_family_keyword_variants.py`

Preflight:

- `projects/MOREBENCH/phase_02/scripts/analyze_dilemma_family_keyword_variant_preflight.py`

Materialized:

- `projects/MOREBENCH/phase_02/outputs/dilemma_family_keyword_variant_labels.jsonl`

Report:

- `projects/MOREBENCH/phase_02/reports/dilemma_family_keyword_variant_preflight/report.md`
