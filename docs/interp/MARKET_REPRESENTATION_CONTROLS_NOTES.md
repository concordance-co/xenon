# Market Representation Controls

Date: 2026-03-21

This note records the current representation-focused checkpoint after the shift away from end-state behavior and back toward market-row understanding.

## Main Question

What survives when we ask about the model's *market representation* rather than its final action?

The working decomposition is:

- primitive market factors
- pairwise tradeoff relations
- profile-level abstractions that should survive row/symbol relabeling

## Phase 4: Harder Market Representation

Dataset:

- phase: `phase4_market_representation_v1`
- market-only prompts: `69`
- families:
  - `pairwise_tradeoff_hard`
  - `rank_context_tradeoff`

Main findings:

- Primitive factors are almost perfectly decodable from `row_mean`:
  - `pct_5m`: `R² 0.997`
  - `net_flow_5m`: `R² 0.994`
  - `unique_traders_5m`: `R² 0.998`
  - `top20_holder_pct`: `R² 0.999`
  - `attractiveness_score`: `R² 0.998`
  - `risk_adjusted_score`: `R² 0.998`
- Hard pairwise tradeoffs are linearly trivial on this synthetic slice:
  - `a_beats_b_on_attractiveness`: `AUROC 1.000`
- The first “invariance” read was not trustworthy:
  - `fixed_momentum_flow_pair`: `same_symbol_margin 0.174`
  - `fixed_participation_concentration_pair`: `same_symbol_margin 0.523`

Interpretation:

- The model clearly retains primitive market variables in the row states.
- But the phase-4 rank-context result was still confounded by row/symbol identity.
- The `AUROC 1.000` pairwise results should *not* be overinterpreted; those slices are intentionally easy and mostly validate the synthetic pipeline.

## Phase 5: Symbol Permutation Control

Dataset:

- phase: `phase5_symbol_permutation_v1`
- market-only prompts: `12`
- two scenarios:
  - `momentum_flow_permuted_market`
  - `participation_concentration_permuted_market`

Design:

- The same latent asset profiles are permuted across both:
  - display symbol (`A/B/C/D`)
  - row index
- New control metric:
  - `profile_control_nn_accuracy`
  - nearest-neighbor profile retrieval after excluding all candidates with the same row or same symbol
- Derived random baseline:
  - `0.188`

Main findings:

- Primitive factor decoding remains essentially perfect even under permutation.
- `momentum_flow_permuted_market`
  - `profile_control_nn_accuracy 0.667`
  - `profile_control_margin 0.0009`
  - `same_row_nn_accuracy 0.833`
  - interpretation: above-chance retrieval exists, but separation is fragile and almost entirely collapsed in cosine space
- `participation_concentration_permuted_market`
  - `profile_control_nn_accuracy 0.542`
  - `profile_control_margin 0.262`
  - `same_row_nn_accuracy 0.792`
  - interpretation: this profile survives the harder control meaningfully better

## Current Read

What looks supported:

- Primitive market variables are genuinely present in row states.
- The model is not just preserving generic “big number” salience; specific factors are separately recoverable.
- At least one profile family (`participation × concentration`) survives a hard symbol/row permutation control.

What does *not* look supported yet:

- A universal profile-level market abstraction that cleanly ignores row position and display symbol.
- A strong invariance result for the `momentum × flow` family.
- Any broad claim that synthetic pairwise perfection means rich semantic understanding.

## Best Current Hypothesis

The model likely has:

- strong early representations for primitive market factors
- some selective higher-order profile abstraction
- but not one uniformly robust profile manifold across all factor families

That is narrower and more defensible than the earlier “global market manifold” framing.

## Best Next Step

Keep the focus on representations, but make the next synthetic control harder:

1. Add row-text paraphrases while preserving numeric profiles.
2. Add profile families with near-tied momentum/flow but sharply different participation or concentration.
3. Add a profile-retrieval task where all lexical asset identifiers are neutralized further.
4. Only after that, bridge the strongest synthetic profile family back to real DX rows for validation.
