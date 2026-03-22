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
  #text(size: 22pt, weight: "bold")[Synthetic Market Phase 11]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    DX-native risk ladder deformation. This phase keeps the same 4-asset latent market from Phase 10 and upgrades the context axis to the
    actual DX-terminal risk levels `1–5`, testing whether the shared market frame survives and how later geometry changes across the full ladder.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[22 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[PROMPTS]\ #text(size: 9pt)[576 total]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[CONTEXTS]\ #text(size: 9pt)[market-only + risk 1–5]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[OBJECT]\ #text(size: 9pt)[4-asset geometry across the full risk ladder]],
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
  #text(size: 12.5pt, weight: "medium")[Phase 11 strengthens the geometry result: the model keeps the same reusable 4-asset market frame across the full DX-native risk ladder, then late row states bend that shared geometry toward risk-conditioned score structure without cleanly collapsing into one smooth global transform.]
  #v(0.4em)
  #text(size: 9.5pt, fill: rgb("#555"))[
    Every market-only to risk-level transfer stays above `0.995` held-out `R²`, so the base coordinate system survives intact. The late `row_eos`
    realignment toward score geometry is positive for every context and peaks at almost the same layer (`L13–L14`) across the entire ladder. What is not yet supported is a single smooth “risk rotation”: the local step deformations are real, but uneven, and the long-span `market_only -> risk_5` warp is still mixed.
  ]
]

#v(1.2em)

// ── Summary Metrics ─────────────────────────────────────────────
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[X TRANSFER FLOOR]\ #text(size: 16pt, weight: "bold")[0.99512] #text(size: 8pt, fill: rgb("#888"))[\ market -> risk_1]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[Y TRANSFER FLOOR]\ #text(size: 16pt, weight: "bold")[0.99543] #text(size: 8pt, fill: rgb("#888"))[\ market -> risk_1]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST REALIGNMENT]\ #text(size: 16pt, weight: "bold")[+0.03038] #text(size: 8pt, fill: rgb("#888"))[\ risk_3, row_eos L13]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[STRONGEST LOCAL WARP]\ #text(size: 16pt, weight: "bold")[0.51993] #text(size: 8pt, fill: rgb("#888"))[\ risk_4 -> risk_5, row_mean L1]],
)


= Why Phase 11 Exists

Phase 10 used only three contexts: `market_only`, `low_risk`, and `high_risk`. That was enough to show preserve-then-deform, but not enough to test the real product setting. DX-terminal does not expose a continuous slider in the prompt. It exposes integer risk levels `1–5`.

Phase 11 asks whether the same geometry story still holds under that real context family:

- does the base market frame survive all five settings?
- does late geometry still move toward score-like structure?
- do adjacent risk steps behave like a smooth transform, or only like uneven local warps?


= Base Coordinates Survive the Full Risk Ladder

#align(center)[#image("../../data/report_assets/synthetic_market_phase11_risk_ladder/coordinate_transfer.png", width: 88%)]
#text(size: 8pt, fill: rgb("#888"))[
Market-only probes test whether the same latent 4-asset coordinate system remains linearly recoverable after each DX-native risk setting is added.
]

#v(0.4em)

This is the cleanest Phase 11 anchor. The shared market frame survives every risk level with almost no degradation:

- `latent_x`: `0.99512` to `0.99825`
- `latent_y`: `0.99543` to `0.99867`

Every peak occurs in early `row_mean` states at layers `1–6`. That means the risk ladder does not rewrite the basic market coordinates. The same base 4-asset frame is still present after the settings are added.

The exact values are not monotonic in risk level, which matters. The model is not simply losing more information as risk increases. Instead, the base geometry appears to survive essentially intact everywhere on the ladder.


= Late Realignment Is Consistent Across All Six Contexts

#align(center)[#image("../../data/report_assets/synthetic_market_phase11_risk_ladder/context_realignment.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
For each context, `row_eos` pair distances are compared against both the raw latent geometry and the context-adjusted score geometry.
]

#v(0.4em)

The strongest new representational result is that late realignment is not a one-off artifact of one settings split. It shows up across the whole ladder with positive margins in every context:

- `market_only`: `+0.02771`
- `risk_1`: `+0.02591`
- `risk_2`: `+0.02208`
- `risk_3`: `+0.03038`
- `risk_4`: `+0.02293`
- `risk_5`: `+0.02649`

Every peak occurs in `row_eos` around `L13–L14`. That consistency matters more than the exact ordering of the margins. It suggests that later row states are doing the same kind of operation across the whole ladder: compressing the market toward a score-like geometry while preserving the underlying coordinate frame.

So the Phase 10 result was not a narrow `low/high` artifact. The same late geometric realignment survives the actual DX-native risk axis.


= The Ladder Produces Structured Local Warps, Not One Clean Global Rotation

#align(center)[#image("../../data/report_assets/synthetic_market_phase11_risk_ladder/context_deformation.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[
Matched prompt pairs isolate how the same market moves under adjacent risk changes and under the long-span `market_only -> risk_5` comparison.
]

#v(0.4em)

All adjacent steps show positive deformation Spearman, so the geometry changes are clearly not random:

- `market_only -> risk_1`: `0.27083`
- `risk_1 -> risk_2`: `0.17619`
- `risk_2 -> risk_3`: `0.31607`
- `risk_3 -> risk_4`: `0.26726`
- `risk_4 -> risk_5`: `0.51993`

But Phase 11 does #emph[not] support a simple “smooth global risk rotation” story. The strongest local step is `risk_4 -> risk_5`, while `risk_1 -> risk_2` is much weaker. The long-span `market_only -> risk_5` comparison is especially revealing: its best deformation Spearman is still positive, but the cosine is negative.

That combination means the ordering of pair-distance changes still tracks the score-space target, but the overall vector direction is mixed. The right description is:

- local risk steps induce structured warps
- those warps are not uniform across the ladder
- one end-to-end global transform is not yet supported


= Interpretation

#block(
  width: 100%,
  inset: (left: 14pt, top: 10pt, bottom: 10pt, right: 10pt),
  stroke: (left: 3pt + rgb("#2e7d32"), top: none, right: none, bottom: none),
  fill: rgb("#e8f5e9"),
)[
  #text(size: 7.5pt, fill: rgb("#2e7d32"), weight: "bold", tracking: 0.08em)[SUPPORTED NOW]
  #v(0.2em)
  The model carries a stable multi-asset market coordinate frame across `market_only` and all DX risk levels `1–5`, and later row states consistently shift that geometry toward risk-conditioned score structure.
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
  A single smooth risk transform over the whole ladder is not supported yet. The local step deformations are real, but uneven, and the `market_only -> risk_5` long-span change is still geometrically mixed.
]

#v(0.8em)

The best synthesis after Phase 11 is:

- #text(weight: "medium")[early row_mean states preserve a common market coordinate system]
- #text(weight: "medium")[late row_eos states reliably compress that system toward score geometry]
- #text(weight: "medium")[risk settings act as structured local deformations of that shared frame]

That is a materially stronger representational result than Phase 10 because it survives the actual product setting space rather than only a toy low/high split.


= What To Do Next

The next phase should stay set-level and sharpen the transform story instead of opening new fronts:

1. Fit explicit adjacent-step maps such as `market_only -> risk_1`, `risk_1 -> risk_2`, and so on.
   This should tell us whether the local risk warps are mostly rotations, low-rank shears, or simple compressions.
2. Add one second context family on top of the same 4-asset base market, such as portfolio or affordance overlays.
   That will tell us whether those contexts deform the same market frame or create a qualitatively different overlay.
3. Bridge this coordinate/deformation lens back to a small real DX set once the synthetic transform is characterized more explicitly.


= Appendix A: One Market Across the Risk Ladder

#align(center)[#image("../../data/report_assets/synthetic_market_phase11_risk_ladder/score_geometry_example.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Representative `top_pair_cluster` example. The top-left panel shows the hand-designed latent layout. The remaining panels show the score geometry for the same market under `market_only` and risk levels `1–5`.
]

#v(0.4em)

This appendix figure is not an activation-space figure. It shows the #emph[target deformation] that the later activation geometry is partially following. The important intuition is:

- the same four assets occupy a stable base market layout
- increasing risk does not reorder that market from scratch
- instead, it changes the spacing and compression of the assets in score space

Phase 11 then shows that later model states partially mirror that risk-conditioned reshaping.


= Appendix B: How To Read The Metrics

#table(
  columns: (1.4fr, 3.6fr),
  align: (left, left),
  table.hline(stroke: 1pt),
  table.header([*Term*], [*Meaning*]),
  table.hline(stroke: 0.5pt),
  [`latent coordinate frame`], [The hand-designed 2D geometry used to place the four synthetic assets before any settings are applied.],
  [`score geometry`], [A context-adjusted geometry built from `(attractiveness_score, risk_adjusted_score)`. This is the target Phase 11 uses when asking whether late states become more risk-conditioned.],
  [`row_mean`], [The mean-pooled residual state across all tokens in one asset row. This tends to preserve primitive market factors and base coordinates very early.],
  [`row_eos`], [The residual state at the end of one asset row. This often behaves more like a summarized row representation and is where the strongest late realignment appears.],
  [`coordinate transfer`], [Train a probe on `market_only`, then test it on a risk-conditioned prompt. High `R²` means the same base market axes still exist after the setting is changed.],
  [`realignment margin`], [Within one context, compare activation-space pair distances against raw latent geometry and against score geometry. A positive value means the state is closer to the score-like market than to the original latent layout.],
  [`deformation Spearman`], [Across matched context pairs, compare how pair distances change in activation space versus score space. Positive values mean the ordering of changes tracks the predicted deformation.],
  [`deformation cosine`], [The cosine between the activation-space deformation vector and the score-space deformation vector. Positive is good directional agreement; negative means the warp is mixed even when rank-order changes still track.],
  table.hline(stroke: 1pt),
)

#v(0.6em)

The short reading guide for Phase 11 is:

1. Look at #text(weight: "medium")[coordinate transfer] first.
   If it stays very high, the base market frame survived.
2. Then look at #text(weight: "medium")[realignment margin].
   If it is positive across the ladder, later states are becoming more score-like rather than only preserving the original latent layout.
3. Then look at #text(weight: "medium")[deformation].
   If adjacent steps are positive but uneven, the right picture is local context warps, not one clean global transform.
