---
benchmark: morebench
phase: 03
version: v3
frozen_date: 2026-04-23
input_artifacts:
  - projects/MOREBENCH/phase_01/docs/01-confound-audit.md
  - projects/MOREBENCH/phase_02/docs/02-augmented-data-manifest.json
  - projects/MOREBENCH/phase_02/docs/02-gap-list-resolution.md
  - projects/MOREBENCH/phase_03/docs/03-response-label-pilot.md
---

# MoReBench 03 Controls And Splits

## Core Rule

Every phase-03 readout must be paired with a nuisance-aware split or control that could realistically kill a shortcut explanation.

Before probing begins, run the lexical-baseline gate on the exact planned evaluation split.
If cheap lexical baselines are already near ceiling on that split, probing is blocked until the target, split, or data design is changed.

For MoReBench, the next mainline full-public-split experiment should treat this as mandatory:

- `char-TFIDF + logistic` on prompt text
- prompt length baseline
- source-family-aware holdout on the exact intended label

## Theory-Identity Controls

- held-out alias-bank transfer is mandatory for any prompt-side theory diagnostic
- generic-ethics controls should be included wherever the comparison otherwise reduces to cue-family classification
- do not treat explicit alias recoverability as sufficient evidence
- before treating Experiment 1 as informative, require the readout to beat the strongest held-out alias text baseline
- localize across alias-bearing cue spans and later prompt states rather than reading only the first explicit cue token
- include a positional control if the alias clause always occurs in one fixed location

## Theory-Generation Controls

- compare the same dilemma under at least two theory primes plus a generic-ethics control
- compare at least one same-prime / different-dilemma slice so theory-persistence claims are not really dilemma-topic readouts
- probe generated tokens, not prompt tokens, in the main persistence readout
- mark and analyze explicit theory-name copying separately from the main signal
- operationalize theory-copying flags before probing:
  - direct theory-name mention in the generation
  - or repeated reuse of distinctive cue-text tokens from the prime
- require a generated-text lexical baseline before upgrading any persistence claim
- require an explicit response-length baseline for Experiment 2
- run a behavioral-divergence pre-check on the matched prime-swap batch before treating probe metrics as meaningful
- prefer same-dilemma prime swaps over cross-dilemma comparisons whenever possible

## Response-Side Controls

- evaluate `tradeoff_engagement`, `commitment_style`, `helpfulness_invoked`, and `harm_avoidance_invoked` on the same frozen generation set where possible
- compare conclusion-span readout to non-conclusion-span readout
- label-shuffled sanity checks are mandatory for every first-pass probe
- control for simple length effects and generic verbosity
- add an assertive-vs-cautious tone baseline for `helpfulness_invoked` and `harm_avoidance_invoked`
- do not begin probing until the response-label pilot has produced a validated, frozen slice

## Stakeholder-Tradeoff-Density Controls

- validate the gold slice before any serious readout
- report at least one split that reduces source-family leakage
- explicitly compare against prompt length and dilemma-structure baselines
- include a source/topic-aware baseline in the first serious pilot

## Objective-Pressure-Profile Controls

- freeze the rubric-derived label before probing
- run the lexical gate first on the exact planned split
- report source-family-aware holdout before any random split result
- add context-aware holdout where support permits
- explicitly compare against prompt length and source-family baselines
- do not promote the family if text is already near ceiling on the same holdout

## Action-Locus Controls

- treat the current rewrite set as pilot-only
- use matched advisor-vs-agent pairs only
- do not mix unrepaired public rows into the main readout
- do not make broad source-general claims until the rewrite batch is expanded and source-balanced

## Split Plan

### Preferred prompt-side split hierarchy

1. same family, same control set, different wording variants
2. theory-name-aware control split
3. source-family-aware split where feasible
4. dilemma-structure-aware split

For the full-public-split `objective_pressure_profile` experiment, override the generic ordering:

1. source-family-aware split
2. context-aware split
3. random stratified split only as a secondary convenience report

### Preferred response-side split hierarchy

1. mixed-family generation set with family labels retained
2. hold out one prompt family at a time
3. compare conclusion-span vs non-conclusion-span windows as an explicit analysis factor, not a hidden filter
4. compare simple length and assertive-vs-cautious baselines before upgrading any claim

## Cheap Baselines To Beat

- prompt length
- source family
- dilemma structure
- topic/domain
- simple conclusion token cues for response-side labels
- generic verbosity
- assertive-vs-cautious tone for helpfulness / harm readouts

If a fancy readout cannot beat those controls, it should not be promoted up the evidence ladder.

## What Counts As Good Phase-03 Evidence

- signal survives at least one nuisance-aware split
- same-label controls do not destroy the claimed effect
- localization narrows the likely signal site beyond "somewhere in the prompt" or "somewhere in the response"
- for response-side labels, the readout is built on a frozen labeled slice rather than on ad hoc post hoc impressions

## What Still Blocks Stronger Claims

- incomplete structure/length/person-grammar controls
- partial action-locus repair
- missing anti-shortcut control slice for strong `theory_identity` claims
- `stakeholder_tradeoff_density` gold slice not yet validated
