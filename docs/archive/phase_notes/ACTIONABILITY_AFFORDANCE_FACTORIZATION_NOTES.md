# Actionability Affordance Factorization Notes

## Setup

`actionability_algebra_v4` kept the latent permission rules from `v3` while adding:

- paraphrased portfolio and constraint wording
- shuffled bullet order within sections
- distractor numeric lines in portfolio, strategies, and constraints

The original `v4` read was:

- `market_best_asset`: `row_mean @ L0`, AUROC `1.000`
- `permission_mode`: `portfolio_eos @ L13`, balanced accuracy `0.458`
- `expected_action_type`: `last_token @ L13`, balanced accuracy `0.500`
- `policy_best_asset`: `last_token @ L47`, balanced accuracy `0.417`

That looked like another collapse of the downstream permission story.

## New question

Maybe `permission_mode` is the wrong label.

It is a fused 4-way quantity:

- `buy_and_sell`
- `buy_only`
- `sell_only`
- `observe_only`

But the model may represent the primitive affordances more cleanly:

- `can_buy`
- `can_sell`
- `observe_vs_act`

This is exactly the kind of "wrong fused probe target" failure mode highlighted in the counting-manifolds paper.

## Result

The affordance-factorization pattern now appears in both hardened variants:

### `actionability_algebra_v3`

- `permission_mode`: `0.625`
- `can_buy`: `constraints_eos @ L40`, balanced accuracy `0.875`
- `can_sell`: `active_strategies_eos @ L3`, balanced accuracy `0.792`
- `observe_vs_act`: `last_token @ L40`, balanced accuracy `0.833`

### `actionability_algebra_v4`

- `permission_mode`: `0.458`
- `can_buy`: `portfolio_eos @ L18`, balanced accuracy `0.792`
- `can_sell`: `constraints_eos @ L41`, balanced accuracy `0.750`
- `observe_vs_act`: `active_strategies_eos @ L2`, balanced accuracy `0.667`

## Interpretation

This is the strongest current result in the actionability line.

- The 4-way fused permission label is weaker than the primitive affordance bits.
- The primitive affordance bits survive materially better in both `v3` and `v4`.
- The exact best section moves with prompt presentation, but the factorized affordance story itself survives.

That suggests the downstream computation is more factorized than the earlier report implied.

## What this changes

The new best hypothesis is no longer:

- "there is a clean downstream permission-mode representation"

It is now:

- "the model represents primitive affordance bits more cleanly than the fused permission mode"

That is a much better mechanistic target.

## Next step

Use this factorized target set going forward:

- `can_buy`
- `can_sell`
- `observe_vs_act`

Then test:

- whether these bits survive further prompt variation
- whether they transfer to real DX reruns
- whether they combine downstream into the final action choice
