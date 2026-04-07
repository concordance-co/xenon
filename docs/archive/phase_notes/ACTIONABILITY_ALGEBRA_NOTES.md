# Actionability Algebra Notes

## Current synthetic phases

- `actionability_algebra_v1`
  - Compositional permission dataset, but the prompt header leaked the exact `permission_mode`.
- `actionability_algebra_v2`
  - Removed the header leak.
  - Result: market-best asset stayed perfectly stable in the row states, while `expected_action_type` and `permission_mode` became strongest late in `constraints_eos`.
- `actionability_algebra_v3`
  - Same latent permission rules as `v2`, but with paraphrased portfolio and constraint language and shuffled bullet order within sections.

## What changed across phases

### `v1 -> v2`

Removing the header shortcut materially changed the probe picture:

- `expected_action_type`
  - `v1`: best at `active_settings_eos @ layer 0`, balanced accuracy `1.00`
  - `v2`: best at `constraints_eos @ layer 25`, balanced accuracy `1.00`
- `permission_mode`
  - `v1`: best at `active_settings_eos @ layer 0`, balanced accuracy `1.00`
  - `v2`: best at `constraints_eos @ layer 27`, balanced accuracy `1.00`
- `market_best_asset`
  - stayed perfect at `row_mean @ layer 0`
- `permission_top_symbol_invariance`
  - stayed `1.00`

Interpretation:

- the market-preferred asset is early and stable
- after removing the explicit mode leak, actionability became a late-section phenomenon
- this supported the "early preference, late permission" hypothesis

### `v2 -> v3`

Paraphrasing the portfolio/constraint language and shuffling the section bullets destroyed the previously perfect late permission signal:

- `expected_action_type`
  - `v2`: `constraints_eos @ layer 25`, balanced accuracy `1.00`
  - `v3`: `active_strategies_eos @ layer 3`, balanced accuracy `0.694`
- `permission_mode`
  - `v2`: `constraints_eos @ layer 27`, balanced accuracy `1.00`
  - `v3`: `active_strategies_eos @ layer 2`, balanced accuracy `0.625`
- `policy_best_asset`
  - `v2`: `portfolio_eos @ layer 4`, balanced accuracy `0.700`
  - `v3`: `active_strategies_eos @ layer 12`, balanced accuracy `0.383`
- `market_best_asset`
  - remained perfect at `row_mean @ layer 0`
- `permission_top_symbol_invariance`
  - remained `1.00`

Interpretation:

- the stable market preference result is robust
- the late permission signal in `v2` was not robust to modest wording variation
- this means `v2` was informative, but not yet enough to claim a strong lexical-robust permission circuit

## Current read

The synthetic evidence now looks like this:

- strong evidence for a stable early market-preference representation
- moderate evidence that actionability can be expressed downstream
- weak evidence, so far, for a wording-robust late permission representation

## Best next synthetic move

Build `actionability_algebra_v4` with:

- paraphrase variation in both portfolio and constraints
- explicit distractor lines so threshold words are not the only numbers in those sections
- a larger scenario group count
- alternative section-local phrasing that keeps semantics fixed but reduces exact lexical alignment further

## Best real-data bridge

Use real DX ticks to validate only the strongest surviving claim:

- asset preference is early and stable
- permissions / affordances should be tested as a noisy downstream gating signal, not assumed to be crisply encoded from the current synthetic result
