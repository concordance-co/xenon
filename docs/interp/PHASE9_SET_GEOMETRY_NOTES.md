# Phase 9 Set Geometry Notes

## Goal

Move from pairwise relation tests to a 4-asset market object:

- keep rank order fixed across scenarios
- change the latent shape of the 4-asset market
- ask whether pooled row states preserve:
  - latent market coordinates
  - within-snapshot distance structure
  - whole-geometry identity against same-rank negatives

## Dataset

Phase: `phase9_set_geometry_v1`

- 96 market-only prompts
- 4 scenarios
  - `even_ladder`
  - `top_pair_cluster`
  - `dominant_outlier`
  - `middle_gap`
- nuisance axes
  - 4 permutations / layouts
  - 2 surface variants
  - 3 global magnitude scales

Each scenario keeps the same rank order while changing the pair-distance signature over 4 assets.

## Capture

- H200 smoke: passed
- Full capture: `96 / 96`
- Average capture time: ~`1.21s`
- Full pooling completed
- Full representation analysis completed on Modal

## Main Result

Phase 9 is a split verdict:

- **very strong positive:** latent asset coordinates are almost perfectly linearly explicit in row states
- **moderate positive:** within-snapshot activation geometry partially tracks the latent 4-asset shape
- **negative:** full geometry-family identity against same-rank negatives is effectively absent

That means the model does not look like it stores a clean, invariant "shape template" for each market family. But it does appear to preserve a usable coordinate-like representation of the assets.

## Key Numbers

### Latent coordinate regression

- `latent_x`: `R² = 0.99967` at `row_mean @ L2`
- `latent_y`: `R² = 0.99977` at `row_mean @ L1`

### Geometry alignment

Best distance Spearman by scenario:

- `even_ladder`: `0.31887` at `row_eos @ L25`
- `top_pair_cluster`: `0.34524` at `row_eos @ L23`
- `dominant_outlier`: `0.15476` at `row_eos @ L12`
- `middle_gap`: `0.35238` at `row_eos @ L25`

Exact pair identification at the best alignment layer stays weak:

- closest-pair accuracy ranges from `0.0` to `0.5`
- farthest-pair accuracy is mostly around `0.5`

So the model preserves some metric ordering / spacing information without cleanly recovering the exact canonical pair structure.

### Geometry identity over same-rank negatives

Best margins are tiny:

- best `full`: `0.00062` (`dominant_outlier`)
- best `style_only`: `0.00097` (`dominant_outlier`)
- all `layout_only` margins are negative

This is the clearest failure mode in the phase.

## Interpretation

The good read is:

- the model seems able to place assets in a latent market coordinate system
- this is stronger than fixed-pair identity and more useful scientifically

The limit is:

- those coordinates do not collapse into a robust, nuisance-invariant family code for the whole 4-asset shape
- layout changes especially break family identity

So Phase 9 argues for:

- **coordinate / geometry preservation** as the right object
- not **whole-shape retrieval** as the main target

## Best Next Step

Use the 4-asset object again, but change the question:

1. Treat the latent coordinates or pair-distance matrix as the object to recover.
2. Test whether later context deforms that geometry.
3. Ask whether settings act like a transformation on an existing market geometry rather than rewriting the asset map.

That would be a cleaner follow-up than more geometry-family retrieval.
