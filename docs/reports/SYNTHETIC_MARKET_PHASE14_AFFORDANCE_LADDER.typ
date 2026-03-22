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
  #text(size: 22pt, weight: "bold")[Synthetic Market Phase 14]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    Affordance-ladder geometry on the same 4-asset market frame used in Phases 9-13. This phase asks whether progressively stronger
    execution constraints deform the shared market coordinates like risk and portfolio, or whether affordance behaves more like route masking.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[22 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[BASE DATA]\ #text(size: 9pt)[Phase 14 affordance ladder]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[STATES]\ #text(size: 9pt)[`row_mean @ L1`, `row_eos @ L25`]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[OBJECT]\ #text(size: 9pt)[adjacent affordance-step transforms]],
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
  #text(size: 12.5pt, weight: "medium")[The shared 4-asset market frame survives the full affordance ladder almost perfectly early, but late states look sharper and more route-masking than either risk or portfolio.]
  #v(0.4em)
  #text(size: 9.5pt, fill: rgb("#555"))[
    Early `row_mean` states keep both latent axes above `0.990 R²` all the way to `affordance_5`, so the base market coordinates are still present. Late
    `row_eos` states do move toward affordance-adjusted score geometry, but the best adjacent fits favor flexible local maps and the full
    `market_only -> affordance_5` warp is only moderate. The strongest interpretation is preserved shared geometry plus increasingly strong route masking.
  ]
]

#v(1.2em)

// ── Summary Metrics ─────────────────────────────────────────────
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[EARLY FRAME FLOOR]\ #text(size: 16pt, weight: "bold")[`0.990 R²`] #text(size: 8pt, fill: rgb("#888"))[\ `market_only -> affordance_5`, latent_y]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[LATE STRONGEST LOCAL FIT]\ #text(size: 16pt, weight: "bold")[`0.881 R²`] #text(size: 8pt, fill: rgb("#888"))[\ linear fit on `affordance_2 -> affordance_3`]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[LATE END-TO-END]\ #text(size: 16pt, weight: "bold")[`0.426 R²`] #text(size: 8pt, fill: rgb("#888"))[\ best direct family on `market_only -> affordance_5`]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[NONTRIVIAL COMPOSE]\ #text(size: 16pt, weight: "bold")[`0.924 cos`] #text(size: 8pt, fill: rgb("#888"))[\ late linear composed-vs-direct matrix cosine]],
)


= Why Phase 14 Exists

Phases 12 and 13 established two complementary context stories on the same 4-asset object:

- risk preserved a shared market frame and behaved like a globally coherent near-rigid ladder
- portfolio preserved the same frame, but looked more local and redistributive

Affordance is the next useful family because it changes #emph[execution availability] rather than preference weight. In the synthetic prompts:

- the strongest asset is capped first
- the second asset becomes partially restricted and then blocked
- the third asset becomes size-limited late
- by `affordance_5`, only the weakest route remains fully open

So this phase asks whether the model treats execution constraints as:

- another smooth deformation of the shared market frame
- or a sharper route-masking process layered on top of that frame


= Experimental Design

#align(center)[#image("../../data/report_assets/synthetic_market_phase14_affordance_ladder/experiment_design.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Phase 14 is built from repeated variants of the same 4-asset markets. The figure shows the dataset recipe, what the affordance ladder means, and the actual question being tested.
]

#v(0.4em)

The easiest way to read this phase is:

- start with a base 4-asset market
- render many nuisance variants of the same market
- place each variant into six contexts: `market_only`, then `affordance_1..affordance_5`
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


= What “Affordance Ladder” Means

In this report, a “ladder” just means an ordered sequence of matched prompt variants. The market stays the same, but the contextual pressure changes step by step.

For Phase 14:

- `market_only` has no hard execution constraints
- `affordance_1` lightly caps the current leader
- `affordance_3` blocks the top two routes
- `affordance_5` leaves only the weakest-ranked route fully unrestricted

So the ladder is not a time series. It is a controlled constraint sweep over the same base market.

#table(
  columns: (1.1fr, 1.2fr, 2.7fr),
  align: (left, left, left),
  table.hline(stroke: 1pt),
  table.header([*Context*], [*Constraint severity*], [*Instructional effect*]),
  table.hline(stroke: 0.5pt),
  [`market_only`], [`none`], [No hard execution constraints are supplied.],
  [`affordance_1`], [`light`], [The strongest route is capped for new adds.],
  [`affordance_2`], [`light-medium`], [The strongest route is capped and the next route becomes confirmation-only.],
  [`affordance_3`], [`medium`], [The top two routes face hard execution limits.],
  [`affordance_4`], [`strong`], [Top two blocked; third route can only be added in small size.],
  [`affordance_5`], [`strongest`], [Only the weakest-ranked route remains broadly open.],
)


= What “4-Asset Market Geometry” Means

#align(center)[#image("../../data/report_assets/synthetic_market_phase14_affordance_ladder/geometry_scenarios.png", width: 95%)]
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
  the asset layout implied after we rescore the same market under a specific context, such as progressively stronger execution constraints
- #text(weight: "medium")[nuisance variants]
  prompt differences that should not change the underlying market meaning, like row order, symbol aliases, or formatting style

The point of Phase 14 is to separate:

- real geometry changes caused by context
- from fake changes caused by nuisance presentation differences


= How To Read The Layerwise Charts

Several figures in this report use the same general format: some metric is plotted #emph[by layer]. The point of those charts is to show
#emph[where in the model] a representation or deformation is strongest.

The basic reading rules are:

- #text(weight: "medium")[x-axis = layer]
  moving left to right means moving from earlier model layers to later ones
- #text(weight: "medium")[y-axis = average metric value]
  higher usually means the tested representation is clearer or the tested context effect is stronger
- #text(weight: "medium")[a peak]
  the layer where that metric is strongest under the current comparison
- #text(weight: "medium")[`row_mean` vs `row_eos`]
  `row_mean` averages all tokens in a market row, while `row_eos` uses the row-ending token only

In this report, the two most common y-axis metrics are:

- #text(weight: "medium")[mean Spearman]
  a rank-correlation measure. Use this when the chart asks whether distances or relative orderings line up. `1.0` means the activation-space
  ordering matches the target ordering almost perfectly, `0` means little relationship, and negative values mean the ordering is inverted.
- #text(weight: "medium")[margin]
  a difference score between two candidate explanations. Here it usually means “how much closer is the recovered geometry to the
  context-adjusted score geometry than to the raw latent geometry?” Positive is evidence for realignment or deformation in the expected
  direction; values near `0` mean little separation; negative means the comparison is going the wrong way.

So a good mental model is:

- #emph[high early peak] = the representation is present very early and may be close to raw market parsing
- #emph[high late peak] = the effect emerges after more integration with context
- #emph[broad plateau] = the effect is distributed across many layers
- #emph[sharp isolated spike] = the effect is localized to a narrower part of the network

The figures below use those curves to answer two different questions:

- #text(weight: "medium")[coordinate transfer] asks whether the same base market frame survives across contexts
- #text(weight: "medium")[realignment or deformation] asks whether later layers actively move that frame toward a new context-specific geometry


= The Shared Market Frame Still Survives the Full Ladder

#align(center)[#image("../../data/report_assets/synthetic_market_phase14_affordance_ladder/coordinate_transfer.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Held-out coordinate transfer from `market_only` into each affordance context. Both latent axes stay high if the shared frame survives.
]

#v(0.4em)

This is the strongest stability result in the report. A probe trained on `market_only` coordinates still recovers the same latent axes in the
affordance contexts with very little loss:

- `market_only -> affordance_1`: `0.998` / `0.998` `R²` on `x` / `y`
- `market_only -> affordance_3`: `0.999` / `0.999` `R²`
- `market_only -> affordance_5`: `0.993` / `0.990` `R²`

The best layers remain extremely early: `row_mean @ L1` for most transfers, with only the `y` axis in `affordance_4` and `affordance_5`
shifting slightly to `L4`.

So affordance does #emph[not] erase the base market organization. Even when the strongest routes are progressively capped or blocked, the model still
keeps a reusable 4-asset coordinate system that looks almost identical to the original market frame.


= Late States Still Realign Toward Affordance-Adjusted Score Geometry

#align(center)[#image("../../data/report_assets/synthetic_market_phase14_affordance_ladder/context_realignment.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Realignment margin between recovered activation-space geometry and affordance-adjusted score geometry for each context.
]

#v(0.4em)

Late states do not stay purely latent. The best realignment margins all occur on `row_eos`, which means the end-of-row state is moving
toward the affordance-adjusted score layout rather than staying at the raw latent market geometry.

The pattern is uneven:

- `market_only`: margin `+0.024` at `L14`
- `affordance_1`: `+0.029` at `L31`
- `affordance_2`: only `+0.012` at `L42`
- `affordance_4`: strongest margin, `+0.072` at `L36`
- `affordance_5`: still strong, `+0.049` at `L25`

So the affordance ladder does reshape the late geometry, but it does so most clearly once the restrictions become genuinely severe. Mild
execution caps are comparatively weak.


= Early Affordance Transforms Are Nearly Trivial

#align(center)[#image("../../data/report_assets/synthetic_market_phase14_affordance_ladder/family_heatmap.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Coordinate reconstruction `R²` for transform families fit on each adjacent affordance step. Left: early state. Right: late state.
]

#v(0.4em)

The early state is almost too clean. Every adjacent affordance step is already extremely well described by a trivial family:

- `market_only -> affordance_1`: best `diagonal`, `0.9993 R²`
- `affordance_1 -> affordance_2`: best `linear`, `0.9989 R²`
- `affordance_2 -> affordance_3`: best `linear`, `0.9994 R²`
- `affordance_3 -> affordance_4`: best `similarity`, `0.9984 R²`
- `affordance_4 -> affordance_5`: best `linear`, `0.9998 R²`
- full `market_only -> affordance_5`: best `identity`, `0.9976 R²`

That means the early recovered frame barely changes at all. The model is still carrying an almost unchanged market layout before the later
context-sensitive reshaping begins.


= The Late Ladder Is Local, Mixed, or Mask-Like

#align(center)[#image("../../data/report_assets/synthetic_market_phase14_affordance_ladder/context_deformation.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Best deformation margins for adjacent and end-to-end affordance comparisons. Higher margins indicate stronger evidence that the geometry is shifting toward the context-adjusted arrangement.
]

#v(0.4em)

This is where affordance starts to separate from risk and portfolio.

On the late state, the strongest adjacent-step fits are all local and flexible:

- `affordance_1 -> affordance_2`: best `linear`, `0.789 R²`
- `affordance_2 -> affordance_3`: best `linear`, `0.881 R²`
- `affordance_3 -> affordance_4`: best `linear`, `0.802 R²`
- `affordance_4 -> affordance_5`: best `linear`, `0.808 R²`

The deformation margins tell the same story. The largest positive shifts are not end-to-end:

- `affordance_3 -> affordance_4`: `+0.365`
- `affordance_4 -> affordance_5`: `+0.340`
- `affordance_1 -> affordance_2`: `+0.325`

By contrast, the direct end-to-end `market_only -> affordance_5` deformation is only `+0.208`, and its best direct transform fit is
just `0.426 R²`.

So late affordance behavior looks less like one smooth global warp and more like a sequence of sharper local reallocations as routes get
partially blocked and then removed.


= Composition Keeps Orientation Coherent, or Breaks It

#align(center)[#image("../../data/report_assets/synthetic_market_phase14_affordance_ladder/composition.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[
For `market_only -> affordance_5`, compare direct end-to-end fits against the composition of all adjacent affordance-step maps. Gold marks matrix cosine between the composed and direct transforms.
]

#v(0.4em)

The composition result is mixed in a revealing way.

For the late `market_only -> affordance_5` comparison:

- direct `identity`: `0.236 R²`
- direct `orthogonal`: `0.261 R²`
- direct `similarity`: `0.424 R²`
- direct `diagonal`: `0.425 R²`
- direct `linear`: `0.426 R²`

If we compose the adjacent-step maps instead of fitting the end-to-end step directly, the flexible families improve slightly:

- composed `similarity`: `0.443 R²`
- composed `diagonal`: `0.442 R²`
- composed `linear`: `0.451 R²`

But the matrix-cosine story is still constrained:

- `orthogonal`: `0.9994`
- `similarity`: `0.9994`
- `diagonal`: `0.9999`
- `linear`: `0.9242`

So the ladder keeps a broadly coherent orientation, but the late end-to-end geometry is not captured by a single clean rigid map. The best
affordance story is a preserved base frame with increasingly strong local route-masking deformations layered on top.


= Interpretation

#block(
  width: 100%,
  inset: (left: 14pt, top: 10pt, bottom: 10pt, right: 10pt),
  stroke: (left: 3pt + rgb("#2e7d32"), top: none, right: none, bottom: none),
  fill: rgb("#e8f5e9"),
)[
  #text(size: 7.5pt, fill: rgb("#2e7d32"), weight: "bold", tracking: 0.08em)[SUPPORTED NOW]
  #v(0.2em)
  Affordance preserves the same shared 4-asset market frame very strongly in early states. Late states do move toward context-adjusted score
  geometry, especially once the strongest routes are heavily constrained. The best late adjacent-step fits prefer flexible `linear`
  transforms, which is consistent with local route closures or selective masking rather than simple preference reweighting.
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
  The full affordance ladder is not a single smooth global rigid map. Unlike risk, the late `market_only -> affordance_5` step is only
  moderately recoverable under the best family, and unlike the cleanest risk result, a single end-to-end orthogonal or identity map is not
  enough to explain the late geometry.
]

#v(0.8em)

Putting Phases 11-14 together now gives a cleaner taxonomy:

- #text(weight: "medium")[risk] behaves like a globally coherent near-rigid ladder
- #text(weight: "medium")[portfolio] behaves like local redistributive movement inside the same frame
- #text(weight: "medium")[affordance] behaves like preserved frame plus sharper local route masking

That is a stronger and more interpretable result than a generic “context matters” claim. Different context families appear to deform the same
underlying market coordinates in qualitatively different ways.


= What To Do Next

If Phase 14 keeps the shared frame but looks sharper and less globally coherent than both risk and portfolio, the next step should be a small real-DX bridge using matched prompt variants and the same coordinate-and-deformation lens.

The key question is no longer “does a latent space exist?” The key question is:

- do real execution overlays also look like structured deformations of a shared market frame
- and if so, which context families are smooth, local, or mask-like


= Appendix A: Representative Affordance Geometry

#align(center)[#image("../../data/report_assets/synthetic_market_phase14_affordance_ladder/score_geometry_example.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Representative `top_pair_cluster` market from the affordance ladder. The figure shows the same latent base market with affordance-adjusted score geometries layered on top of it.
]

#v(0.4em)

This figure anchors what “affordance deformation” means in the report. The transform analyses above are attempts to recover how the same underlying 4-asset market is reweighted once the strongest execution paths are progressively capped or blocked.


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
  [`affordance ladder`], [A matched sequence of context variants where the base market is fixed and execution availability is stepped from `market_only` to `affordance_5`.],
  [`nuisance variants`], [Prompt changes that alter presentation but should not alter the underlying market meaning, such as row order, formatting style, symbol aliases, or global scale formatting.],
  [`coordinate transfer`], [How well a probe trained in one context can recover the same latent coordinate in another context. High `R²` means the base frame survived.],
  [`realignment margin`], [How much closer the recovered geometry is to the context-adjusted score geometry than to the raw latent layout. Positive margins mean the context is actively reweighting the market.],
  [`context-adjusted score geometry`], [The geometry implied by rescoring the same four assets under the current context. In Phase 14, that means incorporating progressively stronger execution caps and blocks.],
  [`identity`], [No transform at all. If identity already works, the two contexts share almost the same geometry in the decoded frame.],
  [`orthogonal`], [A pure rotation or reflection. Distances and angles are preserved; only orientation changes.],
  [`rigid map`], [A transform that preserves shape up to orientation. In practice here, the closest approximation is an orthogonal map or a near-rigid similarity map.],
  [`similarity`], [Uniform scaling plus rotation. This captures isotropic compression or expansion.],
  [`diagonal`], [Axis-aligned rescaling in the shared coordinate frame. This captures simple non-uniform compression without rotation.],
  [`linear`], [Any 2×2 linear map. This is the most flexible family here and can absorb rotation, anisotropic scaling, and shear-like behavior.],
  [`matrix cosine`], [Cosine similarity between two flattened transform matrices. Here it is used to compare direct end-to-end maps against the composition of adjacent ladder steps.],
)
