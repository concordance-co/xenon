---
benchmark: morebench
phase: 03
version: v1
frozen_date: 2026-04-23
input_artifacts:
  - projects/MOREBENCH/phase_01/docs/01-labeling-functions.md
  - projects/MOREBENCH/phase_01/docs/01-frozen-label-set.csv
  - projects/MOREBENCH/phase_02/docs/02-behavioral-smoke-report.md
  - projects/MOREBENCH/phase_03/docs/03-execution-targets.md
---

# MoReBench 03 Response Label Pilot

## Purpose

Phase-03 response-side experiments must not jump directly from fresh generations to probing.
This artifact defines the required gate:

1. generate
2. annotate
3. validate
4. freeze
5. probe

## Labels In Scope

- `tradeoff_engagement`
- `commitment_style`
- `refuses_or_hedges`
- `helpfulness_invoked`
- `harm_avoidance_invoked`
- `uncertainty_and_scope_calibration`

## Proposed Pilot Slice

Core pilot generation slice:

- `50` `theory_direct`
- `50` `theory_wording_variant`
- `50` `neutral_control`

Optional shadow slice:

- current `action_locus_rewrite` rows for qualitative comparison only
- do not mix this shadow slice into the main frozen response-side training set until the repair batch expands

## Annotation Procedure

- two-pass labeling on the core pilot slice
- first pass:
  independent label application using the canonical phase-01 labeling functions
- second pass:
  disagreement review plus adjudication notes

## Validation Standard

Before freezing:

- each label must have a documented agreement check on a shared subset
- any label that remains too unstable should be downgraded or deferred
- the freeze artifact should mark:
  - `usable_now`
  - `usable_with_caution`
  - `blocked`

## Freeze Outputs

Expected execution artifacts:

- `03-response-label-pilot.md`
  pilot protocol and validation summary
- `03-response-label-freeze.csv`
  frozen row-level response-side labels for the pilot batch
- `03-response-label-freeze-summary.json`
  compact counts, agreement, and per-label readiness status

## Probe Gate

Response-side probing may begin only when:

- the generation batch is complete
- the annotation pass is complete
- validation notes exist
- the frozen slice has explicit per-label readiness

If a label is still unstable after the pilot, it should not be included in the first response-side probe run.
