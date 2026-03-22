// ── Page Setup ──────────────────────────────────────────────────
#set page(
  paper: "us-letter",
  margin: (top: 2.4cm, bottom: 2.4cm, left: 2.6cm, right: 2.6cm),
  numbering: "1",
  number-align: right,
)
#set text(font: "Georgia", size: 10.5pt)
#set par(justify: true, leading: 0.7em)
#set heading(numbering: none)

#show heading.where(level: 1): it => {
  set text(size: 13pt, weight: "bold")
  v(1.2em)
  it
  v(0.4em)
}

#show heading.where(level: 2): it => {
  set text(size: 11pt, weight: "bold")
  v(0.8em)
  it
  v(0.3em)
}

#show heading.where(level: 3): it => {
  set text(size: 10pt, weight: "bold")
  v(0.5em)
  it
  v(0.2em)
}

// ── Title Block ─────────────────────────────────────────────────
#align(left)[
  #text(size: 9pt, fill: rgb("#b33a2a"), tracking: 0.08em, weight: "medium")[XENON INTERPRETABILITY]
  #v(0.3em)
  #text(size: 22pt, weight: "bold")[Synthetic Market Phase 13]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    Portfolio-ladder geometry on the same 4-asset market frame used in Phases 9-12. This phase asks whether a second context family
    deforms the shared market coordinates in the same way as risk, or whether portfolio context behaves more locally and asset-relatively.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[22 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[BASE DATA]\ #text(size: 9pt)[Phase 13 portfolio ladder]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[STATES]\ #text(size: 9pt)[row_mean L4, row_eos L14]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[OBJECT]\ #text(size: 9pt)[adjacent portfolio-step transforms]],
  )
  #v(0.3em)
  #line(length: 100%, stroke: 0.5pt + rgb("#ccc"))
]

#v(1em)

// ── Verdict ─────────────────────────────────────────────────────
#block(
  width: 100%,
  inset: (left: 14pt, top: 12pt, bottom: 12pt, right: 12pt),
  stroke: (left: 3pt + rgb("#b33a2a"), top: none, right: none, bottom: none),
  fill: rgb("#faf5f3"),
)[
  #text(size: 7.5pt, fill: rgb("#b33a2a"), weight: "bold", tracking: 0.08em)[MAIN READ]
  #v(0.3em)
  #text(size: 12.5pt, weight: "medium")[Phase 13 shows that portfolio context is not just another copy of the risk story. The shared 4-asset market frame still survives almost intact early, but later portfolio effects are best described as local reallocations around that frame rather than as one strong end-to-end global warp.]
  #v(0.4em)
  #text(size: 9.5pt, fill: rgb("#555"))[
    Early `row_mean @ L4` coordinate transfer stays above `0.989` `R²` for both latent axes across `portfolio_1..portfolio_5`. Later `row_eos @ L14`
    states still realign toward portfolio-adjusted score geometry, but the transform structure is middle-heavy: `portfolio_1 -> portfolio_2`, `portfolio_2 -> portfolio_3`,
    and `portfolio_3 -> portfolio_4` admit strong local fits, while `market_only -> portfolio_1` and `market_only -> portfolio_5` remain much weaker.
  ]
]

#v(1.2em)

// ── Summary Metrics ─────────────────────────────────────────────
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[EARLY FRAME FLOOR]\ #text(size: 16pt, weight: "bold")[0.98921] #text(size: 8pt, fill: rgb("#888"))[\ `latent_x`, `market→portfolio_4`]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[LATE STRONGEST LOCAL FIT]\ #text(size: 16pt, weight: "bold")[0.94023] #text(size: 8pt, fill: rgb("#888"))[\ diagonal on `P2→P3`]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[LATE END-TO-END]\ #text(size: 16pt, weight: "bold")[0.49596] #text(size: 8pt, fill: rgb("#888"))[\ identity on `M→P5`]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[NONTRIVIAL COMPOSE]\ #text(size: 16pt, weight: "bold")[0.99961] #text(size: 8pt, fill: rgb("#888"))[\ orthogonal matrix cosine]],
)


= Why Phase 13 Exists

Phase 12 established a strong risk result: the same 4-asset market geometry survives the DX-native risk ladder, and the late end-to-end ladder still composes most coherently under near-rigid maps. The next question was whether that was just a “settings in general” story or something specific to risk.

Portfolio is a useful second family because it is qualitatively different. In the synthetic prompts:

- free ETH decreases across the ladder
- one asset is already held
- the prompt increasingly warns against concentration

That should create an asset-relative overlay. Unlike risk, this context is not supposed to apply the same penalty to every asset.


= Experimental Design

#align(center)[#image("../../data/report_assets/synthetic_market_phase13_portfolio_ladder/experiment_design.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Phase 13 is built from repeated variants of the same 4-asset markets. The figure shows the dataset recipe, what the portfolio ladder means, and the actual question being tested.
]

#v(0.4em)

The easiest way to read this phase is:

- start with a base 4-asset market
- render many nuisance variants of the same market
- place each variant into six contexts: `market_only`, then `portfolio_1..portfolio_5`
- ask whether the model keeps the same market coordinates and how those coordinates change across the ladder

Concretely, the dataset contains:

- `4` latent market scenarios
- `2` surface styles
- `4` row / symbol permutations
- `3` global scale variants
- `6` contexts per base market

That yields:

- `576` prompts total
- `2,304` asset rows
- `6,912` pairwise asset comparisons

In this report, the surface styles, row permutations, and global magnitude scales are all #emph[nuisance variants]. That means they change how the same market is written on the page without changing the underlying latent market shape we actually care about.


= What “Portfolio Ladder” Means

In this report, a “ladder” just means an ordered sequence of matched prompt variants. The market stays the same, but the contextual pressure changes step by step.

For Phase 13:

- `market_only` has no portfolio overlay
- `portfolio_1` introduces a small existing position and plenty of free ETH
- `portfolio_5` has the largest existing position, the least free ETH, and the strongest anti-concentration instruction

So the ladder is not a time series. It is a controlled context sweep over the same base market.

#table(
  columns: (1.1fr, 1.2fr, 1.3fr, 2.8fr),
  align: (left, left, left, left),
  table.hline(stroke: 1pt),
  table.header([*Context*], [*Available ETH*], [*Existing position*], [*Instructional effect*]),
  table.hline(stroke: 0.5pt),
  [`market_only`], [`2.80`], [`none`], [No portfolio-specific pressure.],
  [`portfolio_1`], [`2.40`], [`8%`], [Light pressure against adding more concentration.],
  [`portfolio_2`], [`2.05`], [`16%`], [Moderate diversification pressure begins.],
  [`portfolio_3`], [`1.70`], [`24%`], [The held asset becomes more expensive to add to.],
  [`portfolio_4`], [`1.35`], [`34%`], [Concentration avoidance becomes strong.],
  [`portfolio_5`], [`1.00`], [`45%`], [Strongest reallocation pressure toward alternatives.],
)


= What “4-Asset Market Geometry” Means

#align(center)[#image("../../data/report_assets/synthetic_market_phase13_portfolio_ladder/geometry_scenarios.png", width: 95%)]
#text(size: 8pt, fill: rgb("#888"))[
The four latent market-shape families used in the dataset. Each point is one asset. Dashed edges mark the pairwise distances that collectively define the whole-market geometry.
]

#v(0.4em)

The key object here is not a single asset row and not a single pairwise comparison. It is the #emph[shape of the 4-asset market].

That means:

- four assets are placed at known latent coordinates
- those coordinates define six pairwise distances
- the distances together determine the market’s overall shape

Why this matters:

- two markets can have the same winner but different shapes
- two markets can have the same rank order but different spacing or clustering
- a real set-level representation should preserve more than “which token looks best”

So when the report says “shared market frame,” it means a reusable coordinate system that organizes all four assets together.

#table(
  columns: (1.3fr, 3.7fr),
  align: (left, left),
  table.hline(stroke: 1pt),
  table.header([*Scenario name*], [*Plain-language meaning*]),
  table.hline(stroke: 0.5pt),
  [`even ladder`], [The four assets step down in fairly regular intervals. There is a clear ordering, but no special clustering.],
  [`top pair cluster`], [The top two assets sit close together while the lower two are farther away.],
  [`dominant outlier`], [One asset is clearly separated above the other three, which are comparatively compressed.],
  [`middle gap`], [There is a larger-than-usual split between the upper pair and lower pair of assets.],
)


= Plain-Language Terminology

This is the shortest way to decode the jargon used later in the report:

- #text(weight: "medium")[market frame]
  the shared coordinate system the model seems to use to organize all four assets
- #text(weight: "medium")[market coordinates]
  the x/y position of each individual asset inside that shared frame
- #text(weight: "medium")[market geometry]
  the whole shape formed by those four coordinates together
- #text(weight: "medium")[rigid map]
  a transform that mostly keeps distances and angles intact, like a rotation or reflection
- #text(weight: "medium")[context-adjusted score geometry]
  the asset layout implied after we rescore the same market under a specific context, such as portfolio pressure
- #text(weight: "medium")[nuisance variants]
  prompt differences that should not change the underlying market meaning, like row order, symbol aliases, or formatting style

The point of Phase 13 is to separate:

- real geometry changes caused by context
- from fake changes caused by nuisance presentation differences


= The Shared Market Frame Still Survives the Full Ladder

#align(center)[#image("../../data/report_assets/synthetic_market_phase13_portfolio_ladder/coordinate_transfer.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Held-out coordinate transfer from `market_only` into each portfolio context. Both latent axes stay near ceiling, with the best readout always in early `row_mean`.
]

#v(0.4em)

This is the conservative base result, and it is strong:

- `market_only -> portfolio_1`: `0.9936 / 0.9942` for `latent_x / latent_y`
- `market_only -> portfolio_2`: `0.9941 / 0.9950`
- `market_only -> portfolio_3`: `0.9941 / 0.9947`
- `market_only -> portfolio_4`: `0.9892 / 0.9912`
- `market_only -> portfolio_5`: `0.9943 / 0.9946`

So portfolio context does #emph[not] erase the model’s underlying market coordinates. The same 4-asset frame is still present and linearly recoverable very early.


= Late States Still Realign Toward Portfolio-Adjusted Score Geometry

#align(center)[#image("../../data/report_assets/synthetic_market_phase13_portfolio_ladder/context_realignment.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Realignment margin between recovered activation-space geometry and portfolio-adjusted score geometry for each context.
]

#v(0.4em)

As in the risk ladder, the later state moves toward the context-adjusted score geometry:

- `portfolio_1`: margin `0.0268` at `row_eos @ L12`
- `portfolio_2`: margin `0.0327` at `row_eos @ L13`
- `portfolio_3`: margin `0.0304` at `row_eos @ L13`
- `portfolio_4`: margin `0.0337` at `row_eos @ L14`

That is a stable late-depth pattern. The one exception is `portfolio_5`, which numerically peaks at `layer 0` but with low NN accuracy (`0.0506`). That looks unstable and should not be treated as meaningful late structure.

The useful reading is therefore:

- early `row_mean` preserves the shared market frame
- later `row_eos` moves that frame toward portfolio-adjusted scores


= Early Portfolio Transforms Are Nearly Trivial

#align(center)[#image("../../data/report_assets/synthetic_market_phase13_portfolio_ladder/family_heatmap.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Coordinate reconstruction `R²` for transform families fit on each adjacent portfolio step. Left: early `row_mean @ L4`. Right: late `row_eos @ L14`.
]

#v(0.4em)

The early panel is almost flat in the same way that the early risk panel was. All adjacent-step fits are already near ceiling:

- `market_only -> portfolio_1`: similarity best, `R² = 0.9969`
- `portfolio_1 -> portfolio_2`: linear best, `R² = 0.9996`
- `portfolio_2 -> portfolio_3`: linear best, `R² = 0.9996`
- `portfolio_3 -> portfolio_4`: similarity best, `R² = 0.9996`
- `portfolio_4 -> portfolio_5`: similarity best, `R² = 0.9994`
- `market_only -> portfolio_5`: similarity best, `R² = 0.9970`

This is the same structural point as before: context insertion is not destroying the market coordinates in the early state.


= The Late Ladder Is Local, Mixed, and Middle-Heavy

#align(center)[#image("../../data/report_assets/synthetic_market_phase13_portfolio_ladder/context_deformation.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Best deformation margins for adjacent and end-to-end portfolio comparisons. Higher margins indicate stronger evidence that the geometry is shifting toward the context-adjusted arrangement.
]

#v(0.4em)

The late transform structure is where portfolio diverges from risk. At `row_eos @ L14`, the strongest local fits sit in the middle of the ladder:

- `market_only -> portfolio_1`: identity best, `R² = 0.377`
- `portfolio_1 -> portfolio_2`: linear best, `R² = 0.920`
- `portfolio_2 -> portfolio_3`: diagonal best, `R² = 0.940`
- `portfolio_3 -> portfolio_4`: linear best, `R² = 0.890`
- `portfolio_4 -> portfolio_5`: similarity best, `R² = 0.870`
- `market_only -> portfolio_5`: identity best, `R² = 0.496`

That pattern matters. The ladder has clean local structure once the portfolio overlay is active, but entering the ladder and the end-to-end `market_only -> portfolio_5` jump do not admit one comparably strong global map.

This is exactly what we would expect if portfolio context mostly changes #emph[relative allocation pressure] inside a preserved market frame. The overlay is not “global caution.” It is: “you already hold this one, you have less free ETH, and concentration is now more expensive.”


= Composition Keeps Orientation Coherent, Not a Strong Global Warp

#align(center)[#image("../../data/report_assets/synthetic_market_phase13_portfolio_ladder/composition.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[
For `market_only -> portfolio_5`, compare direct end-to-end fits against the composition of all adjacent portfolio-step maps. Gold marks matrix cosine between the composed and direct transforms.
]

#v(0.4em)

The composition plot needs one careful note: `identity` always composes perfectly by construction, so it is not the informative family here. The more useful comparisons are the nontrivial ones:

- `orthogonal`
  - direct `R² = 0.494`
  - composed `R² = 0.489`
  - matrix cosine `0.99961`
- `linear`
  - direct `R² = 0.407`
  - composed `R² = 0.293`
  - matrix cosine `0.99797`

So the end-to-end orientation is still highly coherent, but the portfolio ladder does #emph[not] behave like one strong late global warp. Relative to the risk ladder, this is a more local and more heterogeneous deformation story.


= Interpretation

#block(
  width: 100%,
  inset: (left: 14pt, top: 10pt, bottom: 10pt, right: 10pt),
  stroke: (left: 3pt + rgb("#2e7d32"), top: none, right: none, bottom: none),
  fill: rgb("#e8f5e9"),
)[
  #text(size: 7.5pt, fill: rgb("#2e7d32"), weight: "bold", tracking: 0.08em)[SUPPORTED NOW]
  #v(0.2em)
  The model still carries a shared multi-asset market frame through the full portfolio ladder. Later states then apply context-sensitive local reallocations inside that frame, especially in the middle portfolio steps.
]

#v(0.5em)

#block(
  width: 100%,
  inset: (left: 14pt, top: 10pt, bottom: 10pt, right: 10pt),
  stroke: (left: 3pt + rgb("#f57f17"), top: none, right: none, bottom: none),
  fill: rgb("#fff8e1"),
)[
  #text(size: 7.5pt, fill: rgb("#f57f17"), weight: "bold", tracking: 0.08em)[NOT SUPPORTED]
  #v(0.2em)
  A single strong end-to-end portfolio warp is not supported. The `portfolio_5` layer-0 realignment anomaly is also not strong enough to treat as a genuine late-stage effect.
]

#v(0.8em)

The cleanest synthesis after Phase 13 is:

- #text(weight: "medium")[risk and portfolio both preserve a shared market frame early]
- #text(weight: "medium")[risk looks globally more coherent as a ladder]
- #text(weight: "medium")[portfolio acts more like an asset-relative redistribution overlay]

That is a genuinely interesting result. It says the model is not applying one generic “settings transform” to the market. Different context families appear to deform the same underlying coordinates in different ways.


= What To Do Next

The next step should keep the same 4-asset geometry object and test one more context family with clearly different semantics:

1. `affordance` / actionability overlays
2. `strategy override` overlays

Then bring the same coordinate-and-deformation lens back to a small real-DX matched-variant set. The goal is no longer “does a latent space exist?” The goal is to test whether real contexts also look like structured deformations of a shared market frame.


= Appendix A: Representative Portfolio Geometry

#align(center)[#image("../../data/report_assets/synthetic_market_phase13_portfolio_ladder/score_geometry_example.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Representative `top_pair_cluster` market from the portfolio ladder. The figure shows the same latent base market with portfolio-adjusted score geometries layered on top of it.
]

#v(0.4em)

This figure anchors what “portfolio deformation” means in the report. The transform analyses above are not abstract regression exercises. They are attempts to recover how the same underlying 4-asset market is reweighted once an existing position, shrinking free ETH, and anti-concentration pressure are added.


= Appendix B: Glossary

#table(
  columns: (1.4fr, 3.6fr),
  align: (left, left),
  table.hline(stroke: 1pt),
  table.header([*Term*], [*Meaning*]),
  table.hline(stroke: 0.5pt),
  [`shared market frame`], [The recovered 2D coordinate system that organizes all four assets before context-specific reweighting.],
  [`market coordinates`], [The individual x/y positions of the four assets inside the shared market frame.],
  [`market geometry`], [The whole-market arrangement of those coordinates, including spacing, clustering, and pairwise distances.],
  [`4-asset geometry`], [The full relative placement of all four assets at once. In practice this means the coordinate layout or, equivalently, the pattern of six pairwise distances between assets.],
  [`portfolio ladder`], [A matched sequence of context variants where the base market is fixed and the portfolio pressure is stepped from `market_only` to `portfolio_5`.],
  [`nuisance variants`], [Prompt changes that alter presentation but should not alter the underlying market meaning, such as row order, formatting style, symbol aliases, or global scale formatting.],
  [`coordinate transfer`], [How well a probe trained in one context can recover the same latent coordinate in another context. High `R²` means the base frame survived.],
  [`realignment margin`], [How much closer the recovered geometry is to the context-adjusted score geometry than to the raw latent layout. Positive margins mean the context is actively reweighting the market.],
  [`context-adjusted score geometry`], [The geometry implied by rescoring the same four assets under the current context. In Phase 13, that means incorporating the existing position, shrinking free ETH, and diversification pressure.],
  [`identity`], [No transform at all. If identity already works, the two contexts share almost the same geometry in the decoded frame.],
  [`orthogonal`], [A pure rotation or reflection. Distances and angles are preserved; only orientation changes.],
  [`rigid map`], [A transform that preserves shape up to orientation. In practice here, the closest approximation is an orthogonal map or a near-rigid similarity map.],
  [`similarity`], [Uniform scaling plus rotation. This captures isotropic compression or expansion.],
  [`diagonal`], [Axis-aligned rescaling in the shared coordinate frame. This captures simple non-uniform compression without rotation.],
  [`linear`], [Any 2×2 linear map. This is the most flexible family here and can absorb rotation, anisotropic scaling, and shear-like behavior.],
  [`matrix cosine`], [Cosine similarity between two flattened transform matrices. Here it is used to compare direct end-to-end maps against the composition of adjacent ladder steps.],
)
