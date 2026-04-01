# Phase 10 Geometry Deformation Notes

## Question

Keep the same 4-asset latent market from Phase 9, add stronger context variants, and test whether context:

- preserves the base market coordinate system
- deforms the geometry in a structured way
- realigns later states toward context-adjusted score geometry instead of replacing the market representation

## Dataset

- Phase: `phase10_set_geometry_context_v1`
- Family: `set_geometry_control`
- Context variants:
  - `market_only`
  - `low_risk`
  - `high_risk`
- Full run:
  - `288` prompts
  - `96` base market-only prompts expanded across `3` context variants

## Objects To Measure

1. Coordinate transfer
   - Train on `market_only`
   - Test on `market_only`, `low_risk`, `high_risk`
   - Targets: `latent_x`, `latent_y`

2. Context realignment
   - Compare activation geometry against:
     - raw latent market geometry
     - context-adjusted score geometry in `(attractiveness_score, risk_adjusted_score)`

3. Deformation tracking
   - For matched examples across contexts, test whether activation-space pair-distance changes track score-space pair-distance changes

## Main Read

Phase 10 supports the "preserve then deform" story.

- The base 4-asset market coordinate system survives context changes almost unchanged.
- Later `row_eos` states are slightly but consistently closer to context-adjusted score geometry than to the raw latent geometry.
- The cleanest deformation is `market_only -> low_risk`, where the activation-space change tracks score-space change with both positive rank correlation and positive cosine.
- `market_only -> high_risk` is more mixed: deformation rank structure is positive, but the best cosine is negative, so the exact direction of the warp is less stable.

## Key Numbers

- Best coordinate transfer:
  - `latent_x`: `R² = 0.99469` (`market_only -> low_risk`, `row_mean @ L2`)
  - `latent_y`: `R² = 0.99678` (`market_only -> low_risk`, `row_mean @ L2`)
- Best realignment margin:
  - `+0.02323` (`low_risk`, `row_eos @ L12`)
  - at that layer, score-space distance Spearman is `0.30179` vs raw latent distance Spearman `0.27855`
- Best deformation Spearman / cosine:
  - `market_only -> low_risk`: `Spearman = 0.27738`, `cosine = 0.69926` (`row_mean @ L2`)
  - `market_only -> high_risk`: `Spearman = 0.29405`, `cosine = -0.20079` (`row_eos @ L1`)
  - `low_risk -> high_risk`: `Spearman = 0.26131`, `cosine = 0.32667` (`row_eos @ L38`)

## Interpretation

The most important positive is the transfer result. A probe trained on `market_only` still recovers the latent coordinates in both risk-conditioned contexts with only a small drop from the in-context ceiling. That means settings are not replacing the market frame.

The realignment result is subtler. By `row_eos @ L12-L13`, all three contexts are about `+0.021` to `+0.023` closer to score geometry than to the raw latent coordinates. So even `market_only` later states already look slightly more like a score-compressed market geometry than like the original hand-designed latent layout.

The deformation metrics then show what the settings are adding on top:

- `low_risk` produces the cleanest early warp
- `high_risk` changes pair-distance ordering in a structured way, but the vector direction is less clean
- the `low_risk -> high_risk` difference reappears later, suggesting stronger context deltas are handled further downstream

So the current best interpretation is:

- early states preserve the common market coordinate frame
- later states compress that frame toward a score-like geometry
- settings deform that later geometry rather than rewriting it from scratch

## Follow-up

Phase 10 suggests three concrete next steps:

1. Replace the discrete `market_only / low_risk / high_risk` split with a graded risk ladder and test whether the geometry rotates smoothly.
2. Add portfolio and affordance overlays on top of the same 4-asset base market and test whether those produce new deformations or just act as gates.
3. Move from pair-distance summaries to explicit alignment transforms, for example Procrustes or learned linear maps between context-specific geometries.
