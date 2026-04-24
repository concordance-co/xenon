---
benchmark: morebench
phase: 03
version: v1
frozen_date: 2026-04-23
input_artifacts:
  - projects/MOREBENCH/phase_03/docs/03-analysis-plan.md
  - projects/MOREBENCH/phase_03/docs/03-experiment-specs.md
  - projects/MOREBENCH/phase_03/docs/03-feature-hypotheses.md
  - projects/MOREBENCH/phase_03/reports/experiment_02_behavior_recommendation_analysis/report.md
  - projects/MOREBENCH/phase_03/reports/experiment_02_pca_geometry/pca_geometry_summary.json
---

# MoReBench 03 Behavioral Contrast Set Spec

## Purpose

Freeze the behavioral and representational hypotheses for the broad theory-primed generation batch before the `150 x 6 = 900` behavior-only run is interpreted.

This artifact is not a claim of success.
It is a pre-registered target definition so that:

- the `900`-generation batch does not determine the theory grouping post hoc
- behavioral contrast sets are constructed against a fixed hypothesis
- later capture is restricted to cases that are behaviorally informative rather than all generated rows

## Why This Exists

The current phase-03 evidence does not support treating `6`-way prime identity as the main target.

What the current evidence suggests instead:

- full-sequence `description_only` theory identity is shortcut-dominated
- tail-window cross-family transfer reopens the target, but only weakly and with controls still pending
- recommendation differences exist, but on a minority of dilemmas rather than uniformly
- slot-centered PCA does not show five clean row-level prime clusters
- centroid geometry is low-rank and suggests `2-3` coarse stance directions rather than five independent framework directions

The key planning update is:

- keep probing as a method
- stop centering the main phase-03 target on `6`-way prime identity
- center the next pass on behavior-linked coarse stance structure

## Current Hypothesis Source

The hypothesis below is motivated by two completed analyses:

1. recommendation-level review on the initial `30`-dilemma slice
2. PCA geometry on the strict captured activations

The current suggestive grouping is:

- deontic / character-oriented:
  `deontology`, `virtue_ethics`
- welfarist / outcome-cooperation oriented:
  `utilitarian`, `contractarianism`, `generic_ethics_control`
- separate diagnostic case:
  `contractualism`

This grouping is suggestive, not confirmed.
The purpose of the `900`-generation batch is to test whether this grouping shows up behaviorally in a way that is stable enough to support a narrower capture run.

## Primary Hypothesis

### Binary Behavioral Hypothesis

There is a stable coarse behavioral split between:

- `deontic_cluster = {deontology, virtue_ethics}`
- `welfarist_cluster = {utilitarian, contractarianism, generic_ethics_control}`

with `contractualism` treated as an out-of-cluster diagnostic case rather than forced into either side during the first pass.

### Behavioral Expectation

On behaviorally informative dilemmas:

- members of the deontic cluster should recommend the same action more often than they agree with the welfarist cluster
- members of the welfarist cluster should recommend the same action more often than they agree with the deontic cluster
- `contractualism` may align with either side on some dilemmas but should not be assumed to do so consistently

### Representational Follow-On If Behavioral Signal Exists

If the behavioral contrast sets support the binary grouping, the first activation target should be:

- binary `deontic_cluster` vs `welfarist_cluster`

with leave-one-prime-out evaluation across in-scope primes.

## Secondary Hypothesis

### Trinary Behavioral Hypothesis

There are three stable stance groupings:

- `deontic = {deontology, virtue_ethics}`
- `welfarist = {utilitarian, contractarianism, generic_ethics_control}`
- `contractualist = {contractualism}`

This hypothesis should only become active if the broad behavior run shows that `contractualism` behaves as a persistent outlier rather than a noisy member of one of the two larger groups.

### Representational Follow-On If Behavioral Signal Exists

If the behavioral contrast sets support the trinary grouping, the next activation target should be:

- `deontic` vs `welfarist` vs `contractualist`

with a trinary probe and a one-vs-rest diagnostic for `contractualism`.

## Continuous-Axis Fallback

If discrete grouping is unstable but stance differences still appear graded, fall back to a continuous target:

- project class centroids onto the top centroid principal component at layer `8` and layer `44`
- treat prime conditions as occupying an ordinal stance axis rather than discrete classes
- evaluate whether held-out dilemmas show recommendation movement that correlates with that stance score

This is a fallback, not the preferred first claim.

## Behavioral Contrast Set Construction

The broad `900`-generation batch should be used to construct behavioral contrast sets as follows.

### Unit

Use each theory-native source case crossed with the six prime conditions as the base unit:

- `150` source cases
- `6` prime conditions each

### Recommendation Extraction

For each generated response:

- extract the final recommendation sentence or final explicit action recommendation span
- normalize obvious surface variants that preserve the same action
- preserve uncertainty markers separately rather than collapsing them into the action label

### Per-Case Behavioral Partition

For each source case:

- cluster the six recommendations by action-level equivalence
- identify whether there is:
  - unanimous behavior
  - a binary split
  - a higher-arity split

### Contrast Set Eligibility

A case is capture-eligible for the next phase only if:

- at least one prime differs behaviorally from another prime on the same source case
- the difference is action-level rather than wording-only
- the final recommendation can be normalized with high confidence

## Primary Success Thresholds

These thresholds are for deciding whether the broad behavior run supports the binary grouping strongly enough to justify a focused capture run.

### Behavioral Thresholds

- divergence floor:
  at least `20%` of source cases must show a non-unanimous recommendation pattern across the six prime conditions
- within-cluster stability:
  average within-cluster agreement for the deontic cluster and the welfarist cluster should each exceed average cross-cluster agreement by at least `0.15`
- nontrivial separation:
  at least one deontic-vs-welfarist pair should differ in agreement rate from at least one within-cluster pair by `>= 0.20`

If these thresholds are not met, the binary grouping does not earn a focused capture run as the primary target.

### Representational Thresholds For Later Follow-On

These are not for the current generation-only batch.
They are the thresholds that will govern any later activation follow-up if the behavioral gates pass.

- binary probe target:
  leave-one-prime-out probe AUROC must exceed the strongest text baseline AUROC by `>= 0.15` on at least `3` relevant folds
- trinary probe target:
  macro AUROC must exceed the strongest text baseline by `>= 0.10`, with `contractualism` one-vs-rest AUROC clearly above chance
- continuous target:
  held-out recommendation-cluster assignment or recommendation polarity should correlate with the stance score at `>= 0.20`

## Falsification Conditions

The current meta-cluster hypothesis should be treated as unsupported if any of the following happen on the broad behavior run:

- fewer than `20%` of source cases show behavioral divergence across prime conditions
- within-cluster agreement is not materially higher than cross-cluster agreement
- `contractualism` does not behave like a stable outlier and also does not stably join either larger cluster
- recommendation variation is dominated by idiosyncratic single-prime effects rather than cluster structure

If falsified, the correct update is:

- the model may still show theory-conditioned wording differences
- but the current coarse stance grouping is not behaviorally coherent enough to anchor a focused activation program

## Capture Routing Rule

Do not capture the full broad batch by default.

Instead:

1. run the full behavior-only batch
2. parse recommendations
3. construct the behavioral contrast-set slice
4. capture only the behaviorally informative slice plus a matched unanimous-control slice

The matched unanimous-control slice should contain cases that:

- are similar in domain and length to the contrast cases
- but remain behaviorally unanimous across primes

This is needed so later probes can distinguish:

- framework-linked action divergence
from
- generic dilemma difficulty or lexical variance

## Copy-Filter Note

The strict theory-copy filter used for the earlier generation-persistence work should not be mechanically reused as the main gating rule for behavioral contrast-set construction.

Reason:

- behavior identification depends on the final recommendation, not on making every row lexically sterile
- the filter was useful for anti-shortcut probing, but it is too aggressive as a universal behavioral admission rule
- rows that mention a theory name once while still making a substantive recommendation should not automatically be excluded from behavioral contrast-set discovery

Any later capture-stage anti-shortcut filter can be stricter than the behavior-stage inclusion rule.

## Reporting Requirements

When the `900`-generation batch completes, summarize at minimum:

- total cases
- number and fraction of behaviorally divergent cases
- pairwise prime agreement matrix
- within-cluster vs cross-cluster agreement under the primary binary hypothesis
- whether `contractualism` behaves as a stable outlier, a deontic member, a welfarist member, or a noisy miscellaneous case
- the list of capture-eligible behavioral contrast cases

## Claim Ceiling

If the broad behavior run passes the thresholds above, the correct claim is still limited:

- behavioral evidence for a low-rank theory-linked stance structure

It is not yet:

- proof of a specific internal mechanism
- proof of five framework-specific policies
- proof that the coarse grouping is representationally encoded beyond text

Those require the later focused capture and nuisance-aware activation analyses.
