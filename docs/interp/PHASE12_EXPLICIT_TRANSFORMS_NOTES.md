# Phase 12: Explicit Risk-Step Transforms

## Question

Phase 11 showed three things:

1. the base 4-asset market frame survives the full DX-native risk ladder
2. late `row_eos` states become more score-like
3. adjacent risk steps create real geometry changes, but not one obviously smooth global transform

Phase 12 asks what those transforms actually look like.

Instead of summarizing them only with distance-Spearman and cosine, this phase:

- decodes each asset row into the shared 2D market coordinate frame
- fits explicit maps for each adjacent risk step
- compares transform families
- tests whether adjacent maps compose into the direct `market_only -> risk_5` shift

## State selection

States were chosen automatically from Phase 11:

- `early`: best average cross-context coordinate transfer
  - `row_mean @ L1`
- `late`: best average cross-context score realignment
  - `row_eos @ L13`

That means Phase 12 is explicitly testing the same “stable base frame” and “late deformed geometry” states identified in the prior phase, not a new cherry-picked slice.

## Transform families

For each risk-step pair we fit:

- `identity`
- `orthogonal`
- `similarity` = scale × rotation
- `diagonal`
- `linear`

All transforms are fit on centered decoded 2D asset coordinates, so the object is market shape, not global translation.

## Main results

### 1. Early state is almost rigid

At `row_mean @ L1`, every adjacent step is already almost perfectly explained by rigid or near-rigid maps:

- `market_only -> risk_1`: orthogonal best, `R² = 0.9954`
- `risk_1 -> risk_2`: all families near ceiling, linear only barely ahead, `R² = 0.9992`
- `risk_2 -> risk_3`: same story, `R² = 0.9992`
- `risk_3 -> risk_4`: similarity best, `R² = 0.9997`
- `risk_4 -> risk_5`: identity best, `R² = 0.9979`
- `market_only -> risk_5`: identity best, `R² = 0.9991`

So the early coordinate frame really is shared. Risk is not doing much geometry work there beyond tiny local nudges.

### 2. Late state mixes rigid and flexible local fits

At `row_eos @ L13`, the picture changes:

- `market_only -> risk_1`: orthogonal best, `R² = 0.685`
- `risk_1 -> risk_2`: linear best, `R² = 0.873`
- `risk_2 -> risk_3`: linear best, `R² = 0.842`
- `risk_3 -> risk_4`: orthogonal best, `R² = 0.788`
- `risk_4 -> risk_5`: linear best, `R² = 0.703`
- `market_only -> risk_5`: orthogonal best, `R² = 0.728`

That is the central Phase 12 pattern: locally, some risk steps want a more flexible transform, but the ladder does not become uniformly non-rigid.

### 3. The winning late linear maps are strongly anisotropic

The best late linear fits are not mild corrections. Some have very strong anisotropy:

- `risk_1 -> risk_2`: anisotropy `7.17`, angle `46°`
- `risk_2 -> risk_3`: anisotropy `25.86`, angle `-67.7°`
- `risk_4 -> risk_5`: anisotropy `1.65`, angle `-9.6°`

So when linear wins locally, it is not because it is just a slightly better version of a rotation. It is absorbing genuinely directional distortions.

### 4. But the global late shift composes best under orthogonal maps

This is the strongest result in the phase.

For the late state:

- composed `orthogonal` maps vs direct `market_only -> risk_5`
  - matrix cosine `0.99998`
  - composed `R² = 0.7293`
  - direct `R² = 0.7283`

By contrast:

- `linear`
  - matrix cosine `0.0865`
  - composed `R² = 0.6224`
  - direct `R² = 0.7190`

So the flexible local linear fits do not define a coherent end-to-end global map. The late ladder composes far more cleanly as a sequence of near-rigid local reorientations than as one large linear shear.

## Strongest claim after Phase 12

The best representational claim now is:

> The model’s late risk-conditioned market geometry is best understood as a shared base frame plus a chain of local risk-step transforms. Some local steps admit anisotropic distortions, but the end-to-end ladder is globally much closer to a composed near-rigid map than to one flexible linear transform.

## What is still missing

- an explicit real-DX validation of the same transform story
- a second context family to test whether portfolio / affordance overlays act on the same market frame
- a more mechanistic answer for why some specific local steps (`1->2`, `2->3`, `4->5`) need flexible local fits

## Best next steps

1. Run the same explicit-transform analysis on one second context family built on the same 4-asset markets.
2. Bridge the transform lens back to a small matched real-DX risk set.
3. If needed, inspect the `risk_1 -> risk_2` and `risk_2 -> risk_3` local distortions more closely with representative examples.
