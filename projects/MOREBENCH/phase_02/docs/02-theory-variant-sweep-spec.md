---
benchmark: morebench
phase: 02
version: v1
frozen_date: 2026-04-24
input_artifacts:
  - projects/MOREBENCH/phase_02/outputs/theory_prompt_variant_sweep_examples.jsonl
  - projects/MOREBENCH/phase_02/outputs/theory_prompt_variant_sweep_controls.jsonl
  - projects/MOREBENCH/phase_02/outputs/theory_prompt_variant_sweep_summary.json
---

# MoReBench 02 Theory Variant Sweep Spec

## Goal

Materialize a real training-distribution-variation family for theory prompts.

The point of this asset is not to enlarge the dataset.
The point is to test whether held-out prompt variants can break lexical confounds while preserving theory stance.

## Why This Exists

Earlier theory prompt repair work used:

- `name_only`
- `alias_only`
- `description_only`
- `name_plus_description`

Those families were useful, but they did not amount to a systematic lexical-variation study.
The current asset is a stronger variant bank:

- `6` description-only banks per theory
- matched generic-control banks in the same styles
- explicit human-review gate before any run

## Design Principles

Each bank must satisfy all of these:

- preserve the intended theory stance
- avoid explicit theory-name leakage
- vary syntax, tone, and lexical surface substantially across banks
- remain short enough to function as a practical prompt-side family
- keep generic controls theory-neutral

The six banks are:

- `analytic`: compact abstract theory language
- `everyday`: plain-language decision advice
- `checklist`: explicit stepwise procedure
- `comparative`: side-by-side option comparison
- `stakeholder`: affected-party and standpoint foregrounding
- `policy`: standing-rule / policy framing

## Materialized Counts

- matched groups: `30`
- theories: `5`
- theory variant rows: `900`
- generic control rows: `180`
- total prompts: `1080`

## Intended Evaluation Shape

This asset is meant for prompt-side testing first.

Primary intended workflow:

- capture prompt-final or prompt-end residual states only
- train on `5` variant banks
- test on the held-out bank
- compare probe AUROC against lexical baselines with the same held-out-bank structure

Scaling study:

- `N = 1`
- `N = 3`
- `N = 6`

The real question is whether probe-over-text delta holds as variant count grows.

## Human Review Gate

No run should start until all prompt variants are manually reviewed for:

- theory fidelity
- generic neutrality
- bank distinctness
- absence of anchor reuse across all banks
- absence of accidental theory-name leakage

The review packet is:

- `projects/MOREBENCH/phase_02/reports/theory_prompt_variant_sweep_review.md`

## Current Status

- asset materialized
- ready for human review
- not yet run
