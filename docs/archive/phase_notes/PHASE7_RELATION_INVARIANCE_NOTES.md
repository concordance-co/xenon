# Phase 7 Relation Invariance Notes

Date: 2026-03-21

## Goal

Execute the relation-first next step from the prior relational representation report:

- stop using row retrieval as the main object
- keep the latent anchor-pair relation fixed
- vary nuisance axes that matter for real market reading:
  - style
  - layout
  - roster composition / rank context
  - absolute magnitude scale

The main question is:

- does the model preserve `anchor_left - anchor_right` as a relation object when the same pair is moved across new roster and scale contexts?

## Dataset

Phase name:

- `phase7_relation_invariance_v1`

Design:

- `384` market-only prompts
- `4` scenario families
  - `momentum_edge_near_tie`
  - `flow_edge_near_tie`
  - `broad_participation_edge`
  - `concentration_penalty_edge`
- `2` surface styles
- `4` row-layout permutations
- `4` roster variants
- `3` global magnitude scales

The anchor pair is always the first two latent assets, but its roster rank moves across:

- `1v2`
- `2v3`
- `3v4`
- `1v3` (for the near-tie families)

## Capture / Pooling

Capture:

- smoke test passed first
- full capture completed at `384 / 384`
- average capture time: `1.21s`
- seq len range: `572-631`

Pooling:

- sharded synthetic structure pooling
- merged outputs written to:
  - `activations/synthetic_structure/phase7_relation_invariance_v1/metadata.parquet`
  - `activations/synthetic_structure/phase7_relation_invariance_v1/tick_labels.parquet`
  - `activations/synthetic_structure/phase7_relation_invariance_v1/asset_labels.parquet`

## Important correction

The first Phase 7 analysis pass produced `None` summaries for the new relation controls.

Cause:

- the runner was evaluating the rank/scale controls on scenario-filtered subsets
- that removed the negative comparison pool

Fix:

- keep the full example pool for comparison
- anchor the metric on one scenario while comparing against other-scenario matches

Regression coverage added in:

- `tests/test_synthetic_market_representation_analysis.py`

## Main results

Primitive factor decode remains effectively perfect:

- all primitive regressions peak at `row_mean @ layer 1`
- best `R²` values are all about `0.99997-0.99998`

Focal pairwise decode is trivial on this slice:

- `a_beats_b_on_attractiveness`: `AUROC 1.0`
- `a_beats_b_on_risk_adjusted`: `AUROC 1.0`
- both at `row_mean @ layer 0`

### Relation invariance

Best summary values are all at `row_mean @ layer 1`.

Per-scenario best margins:

- `momentum_edge_near_tie`
  - full: `0.2668`
  - style-only: `0.2366`
  - layout-only: `0.2263`
  - roster-only: `0.2679`
  - magnitude-only: `0.2558`
- `flow_edge_near_tie`
  - full: `0.2653`
  - style-only: `0.2439`
  - layout-only: `0.2183`
  - roster-only: `0.2669`
  - magnitude-only: `0.2211`
- `broad_participation_edge`
  - full: `0.2436`
  - style-only: `0.2334`
  - layout-only: `0.2466`
  - roster-only: `0.2444`
  - magnitude-only: `0.2426`
- `concentration_penalty_edge`
  - full: `0.2435`
  - style-only: `0.2236`
  - layout-only: `0.2148`
  - roster-only: `0.2442`
  - magnitude-only: `0.2274`

All of these best reads have:

- `nn_accuracy = 1.0`

### Relation-over-rank control

This is the more important new result.

It asks:

- does the relation vector prefer same relation identity over merely sharing the same anchor rank bucket?

Best margins:

- `momentum_edge_near_tie`: `0.2677`
- `flow_edge_near_tie`: `0.2669`
- `broad_participation_edge`: `0.2440`
- `concentration_penalty_edge`: `0.2439`

All best reads:

- `row_mean @ layer 1`
- `nn_accuracy = 1.0`

### Relation-over-scale control

This asks:

- does the relation vector prefer same relation identity over merely sharing the same absolute magnitude scale?

Best margins:

- `momentum_edge_near_tie`: `0.2312`
- `flow_edge_near_tie`: `0.2104`
- `broad_participation_edge`: `0.2137`
- `concentration_penalty_edge`: `0.2024`

Again:

- best at `row_mean @ layer 1`
- `nn_accuracy = 1.0`

## Interpretation

This is the strongest synthetic result yet for the relation-first track.

What Phase 7 now supports:

- the model carries primitive market factors explicitly
- relation vectors are more stable than row identity
- that stability survives explicit roster-rank shifts
- and it survives explicit global magnitude rescaling

What still tempers the result:

- everything peaks extremely early (`row_mean`, `layer 1`)
- all top reads hit `1.0`
- so this synthetic family is likely still too easy

That means the right reading is not:

- “we found the final market representation”

It is:

- “relation-first was the correct object shift, and this dataset confirms it cleanly enough that the next step should be harder relation confounds, not a return to row retrieval”

## Next step

Build Phase 8 around harder relation confounds:

- near-isometric relation pairs where raw factor deltas are matched more closely
- negative pairs that share the same pairwise factor ordering but differ in which factor actually drives the edge
- more severe display/layout disruptions with the same relation identity
- pairwise setups where `row_mean @ layer 1` should no longer solve the task trivially

The goal should now be:

- keep the relation-first object
- make the synthetic relation task hard enough that deeper structure has to matter
