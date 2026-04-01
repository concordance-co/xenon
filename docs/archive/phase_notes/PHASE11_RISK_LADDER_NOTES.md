# Phase 11: Set Geometry Under a DX-Native Risk Ladder

## Question

Phase 10 showed that a simple low-risk / high-risk split did not overwrite the base 4-asset market frame. Phase 11 upgrades that test to the actual DX-terminal setting space:

- `market_only`
- `risk_1`
- `risk_2`
- `risk_3`
- `risk_4`
- `risk_5`

The goal is to test whether the same shared market geometry survives across the full ladder, and whether risk behaves like a smooth deformation of that geometry rather than a rewrite.

## Data

- phase: `phase11_set_geometry_risk_ladder_v1`
- prompts: `576`
- contexts: `6`
- set-level object: `4` assets per market
- geometry families:
  - `dominant_outlier`
  - `even_ladder`
  - `middle_gap`
  - `top_pair_cluster`

## Main results

### 1. The base market frame survives all risk levels

Market-only probes still recover the latent 4-asset coordinates in every risk-conditioned prompt:

- `latent_x`: `0.9951` to `0.9982` held-out `R²`
- `latent_y`: `0.9954` to `0.9987` held-out `R²`

All peaks are in early `row_mean` states. Risk settings do not erase the base coordinate frame.

### 2. Late states realign toward score geometry at nearly fixed depth

The best `score_over_base_margin` is positive for every context:

- `market_only`: `+0.0277`
- `risk_1`: `+0.0259`
- `risk_2`: `+0.0221`
- `risk_3`: `+0.0304`
- `risk_4`: `+0.0229`
- `risk_5`: `+0.0265`

Every peak is in `row_eos` at `L13–L14`.

That is the cleanest new Phase 11 result: the late compression toward score geometry is not a special case of one settings bucket. It is stable across the full DX-native ladder.

### 3. Risk steps induce structured local warps, but not one perfectly smooth global map

Adjacent risk steps have positive deformation Spearman in every case:

- `market_only -> risk_1`: `0.2708`
- `risk_1 -> risk_2`: `0.1762`
- `risk_2 -> risk_3`: `0.3161`
- `risk_3 -> risk_4`: `0.2673`
- `risk_4 -> risk_5`: `0.5199`

But the ladder is not a clean monotonic rotation:

- the strongest local step is `risk_4 -> risk_5`
- `risk_1 -> risk_2` is much weaker and partly mixed
- the long-span `market_only -> risk_5` comparison has positive rank alignment but negative cosine at its best layer

So the right picture is:

- shared base frame
- consistent late score-like compression
- uneven local warps across the ladder

not one single smooth global transform.

## Strongest claim after Phase 11

The best representational claim now is:

> The model carries a stable multi-asset market coordinate frame across the full DX risk ladder, and later row states push that shared geometry toward a risk-conditioned score geometry without replacing the underlying frame.

## What is not supported yet

- A single global “risk rotation” that stays clean from `market_only` all the way to `risk_5`
- A claim that the entire ladder is explained by one simple linear transform
- A real-DX validation of the same geometry-deformation story

## Best next steps

1. Fit explicit adjacent-step transforms (`market_only -> risk_1`, `risk_1 -> risk_2`, etc.) rather than only summary correlations.
2. Add a second context family, like portfolio or affordance overlays, to test whether those act as deformations of the same market frame or as qualitatively different overlays.
3. Bridge the coordinate/deformation lens back to real DX prompts with a small validation set.
