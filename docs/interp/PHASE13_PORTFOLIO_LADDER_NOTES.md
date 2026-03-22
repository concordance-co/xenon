# Phase 13: Portfolio Ladder Notes

## Dataset

- Base market object: the same 4-asset set-geometry markets used in Phases 9-12.
- Context family: `market_only`, then `portfolio_1..portfolio_5`.
- Portfolio ladder semantics:
  - the prompt progressively reduces free ETH,
  - marks `geo_alpha` as an already-held position,
  - and increases an explicit concentration-avoidance instruction.
- Synthetic scoring:
  - `geo_alpha` receives a growing portfolio penalty,
  - the other assets receive a small diversification bonus,
  - so this context family should produce asset-relative reweighting, not just generic caution.

## Main result

- The shared market frame still survives the full portfolio ladder early.
  - `market_only -> portfolio_k` coordinate transfer stays above `0.989` `R²` for both latent axes.
  - Best transfer layers remain early `row_mean`, mostly around `L4`.
- Later states still realign toward context-adjusted score geometry.
  - `portfolio_1..portfolio_4` all peak at `row_eos`, `L12-L14`, with margins `0.0268-0.0337`.
  - `portfolio_5` shows a `layer 0` anomaly with very low NN accuracy and should not be over-read.

## Transform result

- Early transforms are basically trivial.
  - Best adjacent-step fits are all `R² ~= 0.997-0.9996`.
  - Winning families are mostly `similarity` or `linear`, but the gaps from identity/orthogonal are tiny.
  - This is another strong confirmation that the base market frame is preserved through context insertion.
- Late transforms are different from the risk ladder.
  - `market_only -> portfolio_1`: best `identity`, `R² = 0.377`
  - `portfolio_1 -> portfolio_2`: best `linear`, `R² = 0.920`
  - `portfolio_2 -> portfolio_3`: best `diagonal`, `R² = 0.940`
  - `portfolio_3 -> portfolio_4`: best `linear`, `R² = 0.890`
  - `portfolio_4 -> portfolio_5`: best `similarity`, `R² = 0.870`
  - `market_only -> portfolio_5`: best `identity`, `R² = 0.496`

## Interpretation

- The portfolio ladder is not well-described as one end-to-end global warp.
- Instead:
  - the early market frame survives strongly,
  - the middle ladder steps admit clean local remappings,
  - but entering the ladder and end-to-end `market_only -> portfolio_5` are much weaker.
- Relative to Phase 12:
  - risk behaved like a globally coherent near-rigid ladder with local late distortions,
  - portfolio looks more local and target-specific.
- Best current reading:
  - portfolio context acts like a local reallocation overlay on top of a stable shared market frame.

## Composition

- `identity` trivially composes perfectly and is not informative by itself.
- Among nontrivial families:
  - late `orthogonal` has matrix cosine `0.99961`, but direct and composed end-to-end `R²` stay only around `0.49`.
  - late `linear` keeps matrix cosine `0.99797`, but composed `R²` drops from `0.407` to `0.293`.
- So:
  - orientation across the ladder is still coherent,
  - but portfolio does not induce one strong global transform in the late state.

## Strongest claim after Phase 13

- The model continues to carry a reusable multi-asset market coordinate frame.
- Different context families deform that frame differently.
- Risk is closer to a coherent near-rigid ladder.
- Portfolio is more asset-relative and locally redistributive.

## Best next step

- Keep the same 4-asset geometry object and test one more context family with clearly different semantics.
- Best candidates:
  - affordance / actionability overlay
  - strategy override overlay
- Then do a small real-DX bridge using matched prompt variants and the same coordinate/deformation lens.
