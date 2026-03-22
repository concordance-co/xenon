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

On the exact same pooled `actionability_algebra_v4` activations:

- `can_buy`: `portfolio_eos @ L18`, balanced accuracy `0.792`
- `can_sell`: `constraints_eos @ L41`, balanced accuracy `0.750`
- `observe_vs_act`: `active_strategies_eos @ L2`, balanced accuracy `0.667`
- `permission_mode`: still only `0.458`

## Interpretation

This is the strongest current result in the actionability line.

- The 4-way fused permission label is weak.
- The primitive affordance bits survive materially better.
- The sections also split in a plausible way:
  - buyability is strongest in `portfolio_eos`
  - sellability is strongest in `constraints_eos`
  - act-vs-observe appears early in `active_strategies_eos`

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
