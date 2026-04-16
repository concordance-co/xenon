# Phase 11 Methodology Notes

## Purpose

Phase 11 is the first explicit multi-conflict benchmark.

The goal is to place the two currently robust prompt-local conflict
families in the same prompt:

- `trade_size`
- `risk_preference`

This phase is the intermediate step before returning to activation
patching. The main question is whether both conflict families remain
simultaneously readable from the same forward pass, and whether the
shared conflict-family geometry seen across separate prompts survives
composition into one prompt.

## Settled Design

The Phase 11 dataset keeps the individual Phase 09 / Phase 10 dynamics
intact rather than inventing a new label system.

- `Trade Size`
  - setting `1/5 -> small`
  - setting `5/5 -> large`
- `Asset Risk Preference`
  - setting `1/5 -> conservative -> ALPHA`
  - setting `5/5 -> aggressive -> BETA`

Each row includes:

- one size preference line in `STRATEGY`
- one risk preference line in `STRATEGY`
- one size setting in `ACTIVE SETTINGS`
- one risk setting in `ACTIVE SETTINGS`

Expected output is the product of the two resolved settings:

- `asset` from resolved risk
- `size` from resolved size

Primary labels:

- `size_conflict_present`
- `risk_conflict_present`
- `any_conflict_present`
- `double_conflict_present`
- `conflict_count`
- `conflict_band`

## Dataset Shape

Generated dataset:

- `1536` rows total
- `384` aligned
- `768` single-conflict
- `384` double-conflict
- balanced `lexical_split`
- balanced `size_conflict_present`
- balanced `risk_conflict_present`

## Behavioral Smoke

### Double-conflict slice

Report:

- `reports/behavior_smoke_double_conflict.json`

Headline:

- valid JSON: `1.0`
- exact expected: `0.6901`
- action match: `1.0`
- asset match: `0.6953`
- size match: `0.9948`

Interpretation:

- the joint prompt does not collapse
- `size` survives composition almost perfectly
- `risk` keeps the same `ALPHA`-default asymmetry seen in Phase 10

### Aligned slice

Report:

- `reports/behavior_smoke_aligned.json`

Headline:

- valid JSON: `1.0`
- exact expected: `0.8229`
- action match: `1.0`
- asset match: `0.8229`
- size match: `1.0`

Important implication:

- unlike single-axis Phase 10, aligned rows are not fully clean in the
  joint prompt
- the remaining aligned misses are almost entirely on aggressive-risk
  rows that still resolve to `ALPHA`

### Single-conflict slice

Report:

- `reports/behavior_smoke_single_conflict.json`

Headline:

- valid JSON: `1.0`
- exact expected: `0.6875`
- action match: `1.0`
- asset match: `0.6979`
- size match: `0.9896`

### Behavioral interpretation

Current Phase 11 behavior is best understood as:

- `action` is stable
- `size` is behaviorally grounded
- `risk` remains the noisy axis
- the mixed prompt inherits enough of the Phase 10 risk asymmetry that
  aggressive-risk rows are not behaviorally clean even when aligned

That means:

- Phase 11 is still usable for representational analysis
- but it is not a clean behavioral benchmark in the same sense as
  `trade_size`

## Capture / Probe Run

Settled workflow run:

- run id: `wr_d2332cf19a22_a06f6120`
- report:
  - `reports/pipelines_v2/report_6a8a0904734f_4ddbff0e/report.md`

### Text gates

Both text baselines are at exact chance:

- `size_conflict_present`: balanced accuracy `0.500`
- `risk_conflict_present`: balanced accuracy `0.500`

### Probe results

`size_conflict_present`

- best layer: `L36`
- balanced accuracy: `0.9414`
- AUROC: `0.9862`

`risk_conflict_present`

- best layer: `L32`
- balanced accuracy: `0.9388`
- AUROC: `0.9871`

`any_conflict_present`

- best layer: `L36`
- balanced accuracy: `0.9306`
- AUROC: `0.9503`

`double_conflict_present`

- best layer: `L36`
- balanced accuracy: `0.8898`
- AUROC: `0.9687`

Interpretation:

- both individual conflict labels remain strongly readable from the same
  forward pass
- the model also builds a strong readout for `any_conflict_present`
- `double_conflict_present` is slightly weaker but still very strong,
  which is consistent with a richer joint structure rather than total
  collapse

## Joint Geometry

Direction vectors were extracted for:

- `size_conflict_present`
- `risk_conflict_present`

Within the same joint-prompt capture, cosine similarity between the two
directions is:

- `L24`: `0.5199`
- `L28`: `0.4984`
- `L32`: `0.5142`
- `L36`: `0.6305`
- `L40`: `0.7623`
- `L44`: `0.9010`

Current interpretation:

- the shared size-risk conflict-family structure survives in joint
  prompts
- around the main probe peak (`L32-L36`), the cosine is already
  substantial and stronger than the separate-prompt `trade_size` vs
  `risk_preference` comparison
- very late-layer cosine becomes extremely high, but those layers should
  be interpreted cautiously because they may reflect stronger shared
  readout structure rather than the cleanest conflict-computation stage

The most important current geometry point is:

- Phase 11 supports simultaneous decodability
- and it supports nontrivial shared structure between the two conflict
  families inside the same prompt

## Current Read

Phase 11 succeeded at the main representational question:

- both conflict families remain readable when composed into one prompt

But it also surfaced an important caveat:

- the joint prompt is not behaviorally clean on aggressive-risk rows,
  including some aligned rows

So the right framing is:

- good compositional representation benchmark
- not a clean execution-behavior benchmark

## Next Intended Step

Use Phase 11 as the intermediate compositional readout phase, then
return to activation patching once the updated infra is ready.

The highest-value follow-on analyses from the current capture are:

1. compare Phase 11 joint-prompt size-risk geometry directly against the
   separate-prompt Phase 10 geometry
2. test whether the shared conflict-family direction is better described
   by:
   - a common `any_conflict` readout
   - or two partially separable size / risk directions
3. return to activation patching on the shared Phase 10 / Phase 11
   structure
