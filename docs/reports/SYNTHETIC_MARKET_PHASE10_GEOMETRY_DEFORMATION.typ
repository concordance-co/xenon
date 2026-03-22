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
  #text(size: 22pt, weight: "bold")[Synthetic Market Phase 10]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    Context-conditioned geometry deformation. This phase keeps the same 4-asset latent market and asks whether stronger settings/context
    preserve the base coordinate system while deforming later set geometry in a structured way.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[22 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[PROMPTS]\ #text(size: 9pt)[288 total]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[CONTEXTS]\ #text(size: 9pt)[market-only, low-risk, high-risk]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[OBJECT]\ #text(size: 9pt)[4-asset geometry under settings deformation]],
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
  #text(size: 12.5pt, weight: "medium")[Phase 10 supports a preserve-then-deform story: the model keeps a common 4-asset market coordinate frame across settings, then later states bend that geometry toward score-like structure rather than replacing it.]
  #v(0.4em)
  #text(size: 9.5pt, fill: rgb("#555"))[
    The strongest clean positive is cross-context transfer: market-only probes still recover the latent axes in both risk-conditioned prompts with R² around 0.995 to 0.997. By #text(weight: "medium")[row_eos] around layers 12 to 13, the internal pair-distance geometry becomes slightly closer to the context-adjusted score space than to the raw latent layout. The cleanest warp is market-only to low-risk; the high-risk shift is real, but less directionally stable.
  ]
]

#v(1.2em)

// ── Summary Metrics ─────────────────────────────────────────────
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST X TRANSFER]\ #text(size: 16pt, weight: "bold")[0.99469] #text(size: 8pt, fill: rgb("#888"))[\ market -> low, row_mean L2]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST Y TRANSFER]\ #text(size: 16pt, weight: "bold")[0.99678] #text(size: 8pt, fill: rgb("#888"))[\ market -> low, row_mean L2]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST REALIGNMENT]\ #text(size: 16pt, weight: "bold")[+0.02323] #text(size: 8pt, fill: rgb("#888"))[\ low-risk, row_eos L12]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[CLEANEST WARP]\ #text(size: 16pt, weight: "bold")[0.27738] #text(size: 8pt, fill: rgb("#888"))[\ market -> low, cosine 0.699]],
)


= Why Phase 10 Exists

Phase 9 established a better object than row retrieval: the model clearly preserves per-asset coordinates and some within-snapshot geometry. The next question is whether later context:

- leaves that geometry alone
- destroys it
- or deforms it in a structured way

Phase 10 keeps the same 4-asset latent markets and adds stronger settings variants so we can separate #emph[preservation] from #emph[deformation].


= Coordinate Transfer Across Contexts

#align(center)[#image("../../data/report_assets/synthetic_market_phase10_geometry_deformation/coordinate_transfer.png", width: 88%)]
#text(size: 8pt, fill: rgb("#888"))[
Market-only probes test whether the same latent axes remain recoverable after low-risk and high-risk settings are added.
]

#v(0.4em)

The transfer result is the most important anchor in the report. A probe trained only on #text(weight: "medium")[market-only] rows still recovers the latent coordinate system in the risk-conditioned prompts with very little degradation:

- `latent_x`: `0.99469` on #text(weight: "medium")[market -> low-risk] and `0.99422` on #text(weight: "medium")[market -> high-risk]
- `latent_y`: `0.99678` on #text(weight: "medium")[market -> low-risk] and `0.99491` on #text(weight: "medium")[market -> high-risk]

All of those peaks occur in early #text(weight: "medium")[row_mean] states. So the settings variants do not erase or replace the base market frame. The same underlying 4-asset coordinate system is still present and linearly recoverable after context is added.


= Context Can Realign Later Geometry

#align(center)[#image("../../data/report_assets/synthetic_market_phase10_geometry_deformation/context_realignment.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
For each context, later states are compared against both the raw latent geometry and the context-adjusted score geometry.
]

#v(0.4em)

The realignment effect is smaller, but it is the real Phase 10 gain over Phase 9. By #text(weight: "medium")[row_eos] around layers 12 to 13, the activation-space pair distances become modestly closer to the context-adjusted score geometry than to the raw latent layout in all three contexts:

- `market_only`: `+0.02088`
- `low_risk`: `+0.02323`
- `high_risk`: `+0.02296`

At the low-risk peak, the score-space distance Spearman is `0.30179` versus `0.27855` for the base latent geometry. That means later states are not just preserving the original hand-designed coordinates. They are compressing or rotating the market into a more score-like geometry. The fact that #text(weight: "medium")[market_only] shows the same qualitative shift matters: settings do not invent this transformation from nothing, they operate on top of a broader late-stage move toward score-space structure.


= Geometry Deformation Is Structured

#align(center)[#image("../../data/report_assets/synthetic_market_phase10_geometry_deformation/context_deformation.png", width: 94%)]
#text(size: 8pt, fill: rgb("#888"))[
Matched prompt pairs isolate how the same market moves under changed settings, then compare activation-space movement to score-space movement.
]

#v(0.4em)

The deformation metrics show whether matched prompts move through activation space in the same direction that their score-space geometry moves.

The cleanest case is #text(weight: "medium")[market_only -> low_risk]:

- deformation Spearman `0.27738`
- deformation cosine `0.69926`
- best state `row_mean @ L2`

That is a coherent early warp: the same market shifts in nearly the same direction that the synthetic score geometry predicts.

The #text(weight: "medium")[market_only -> high_risk] case is more mixed:

- deformation Spearman `0.29405`
- deformation cosine `-0.20079`
- best state `row_eos @ L1`

So the ordering of pair-distance changes still tracks the score-space change, but the vector direction itself is not cleanly aligned.

Finally, #text(weight: "medium")[low_risk -> high_risk] peaks later:

- deformation Spearman `0.26131`
- deformation cosine `0.32667`
- best state `row_eos @ L38`

That suggests stronger setting deltas are handled further downstream than the softer market-to-low-risk adjustment.


= Interpretation

#block(
  width: 100%,
  inset: (left: 14pt, top: 10pt, bottom: 10pt, right: 10pt),
  stroke: (left: 3pt + rgb("#2e7d32"), top: none, right: none, bottom: none),
  fill: rgb("#e8f5e9"),
)[
  #text(size: 7.5pt, fill: rgb("#2e7d32"), weight: "bold", tracking: 0.08em)[SUPPORTED NOW]
  #v(0.2em)
  The model retains a common market coordinate frame across settings, and later context deforms that geometry in a measurable, structured way rather than wiping it out.
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
  A full geometry rewrite is not supported. Nor is a uniformly clean deformation story across all context pairs: the high-risk branch still looks partly mixed, with positive rank-like deformation but negative cosine at its best layer.
]

#v(0.8em)

The best synthesis is:

- #text(weight: "medium")[early row states preserve the base market axes]
- #text(weight: "medium")[later row_eos states compress those axes toward score geometry]
- #text(weight: "medium")[settings add structured warps on top of that compressed market space]

That is a much stronger representational result than another pairwise or family-retrieval win. It suggests the model has a reusable multi-asset market frame, and context acts more like a deformation operator than a replacement representation.


= What To Do Next

The next phase should stay set-level and make the deformation hypothesis more geometric and less probe-summary-driven:

1. Replace the discrete `market_only / low_risk / high_risk` split with a graded risk ladder and test whether the geometry rotates smoothly.
2. Add portfolio and affordance overlays to the same 4-asset base market and measure whether those produce new deformations or mostly gate the existing geometry.
3. Move from pair-distance summaries to explicit alignment transforms, for example Procrustes maps or learned linear maps between context-specific geometries.


= Appendix A: One Market In Pictures

#align(center)[#image("../../data/report_assets/synthetic_market_phase10_geometry_deformation/geometry_example.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Representative `top_pair_cluster` market. Top-left is the hand-designed latent layout, top-right is the context-adjusted score geometry, bottom-left is an early activation-space PCA (`row_mean @ L2`), and bottom-right is a later activation-space PCA (`row_eos @ L12`).
]

#v(0.4em)

This figure is illustrative rather than statistical, but it helps explain the Phase 10 logic:

- the same four assets occupy a stable shared frame early
- the context variants do not destroy that frame
- later states start to separate those same assets in a way that looks more like the context-adjusted score geometry than like the original latent layout

The PCA axes in the two 3D panels are fitted separately, so the exact axis values are not comparable across panels. The point of the figure is structural: early overlap versus later context-conditioned warping.


= Appendix B: How To Read The Metrics

#table(
  columns: (1.4fr, 3.6fr),
  align: (left, left),
  table.hline(stroke: 1pt),
  table.header([*Term*], [*Meaning*]),
  table.hline(stroke: 0.5pt),
  [`latent coordinate frame`], [The hand-designed 2D geometry used to place the four synthetic assets before any settings/context are applied.],
  [`score geometry`], [A context-adjusted geometry built from `(attractiveness_score, risk_adjusted_score)`. This is the target Phase 10 uses when asking whether later states move toward a settings-sensitive market representation.],
  [`row_mean`], [The mean-pooled residual state across all tokens belonging to one asset row. This tends to preserve primitive market factors very early.],
  [`row_eos`], [The residual state at the end of one asset row. This often behaves more like a summarized row representation and is where late geometry effects frequently appear.],
  [`coordinate transfer`], [Train a probe on `market_only`, then test it on another context. High R² means the same base market axes still exist after context is added.],
  [`realignment margin`], [Within one context, compare activation-space pair distances against raw latent geometry and against score geometry. A positive margin means the state is closer to the score-like market than to the original latent layout.],
  [`deformation Spearman`], [Across matched prompt pairs, compare how pair distances change in activation space versus score space. Positive values mean the ordering of changes tracks the predicted deformation.],
  [`deformation cosine`], [The cosine between the activation-space deformation vector and the score-space deformation vector. Positive is good directional alignment; negative means the warp is mixed even if rank order still tracks.],
  table.hline(stroke: 1pt),
)

#v(0.6em)

The short reading guide for Phase 10 is:

1. Look at #text(weight: "medium")[coordinate transfer] first.
   If it stays high, the base market frame survived.
2. Then look at #text(weight: "medium")[realignment margin].
   If it is positive later in the model, the geometry is bending toward a score-like space.
3. Then look at #text(weight: "medium")[deformation Spearman] and #text(weight: "medium")[deformation cosine].
   If both are positive, the context change is producing a coherent warp rather than noise.
