# Phase 14: Affordance Ladder Notes

## Dataset

- Base market object: the same 4-asset set-geometry markets used in Phases 9-13.
- Context family: `market_only`, then `affordance_1..affordance_5`.
- Affordance ladder semantics:
  - the prompt progressively caps or blocks the strongest routes,
  - the strongest asset is capped first,
  - the second asset then becomes confirmation-only and later blocked,
  - the third asset becomes size-limited,
  - by `affordance_5`, only the weakest-ranked route remains fully open.
- Synthetic scoring:
  - `geo_alpha` receives the strongest affordance penalty,
  - `geo_beta` is penalized next,
  - `geo_gamma` is penalized only late,
  - `geo_delta` remains mostly open until the highest step.

## Main question

- Does the same shared 4-asset market frame survive a ladder of progressively stronger execution masking?
- Do later states move toward affordance-adjusted score geometry the way they did for risk and portfolio?
- Is the ladder best described as:
  - a coherent global warp,
  - a locally structured masking process,
  - or a sharper route-elimination overlay than either risk or portfolio?

## Expected comparison against earlier phases

- Relative to risk:
  - risk should look smoother and more globally coherent.
- Relative to portfolio:
  - affordance should be sharper and more discrete, with stronger local masking of specific assets.
- If the market frame still survives early, that would strengthen the overall claim that context deforms a shared market geometry rather than replacing it.
