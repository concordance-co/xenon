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
  #text(size: 22pt, weight: "bold")[Synthetic Market Phase 12]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    Explicit transform recovery over the DX-native risk ladder. This phase reuses Phase 11’s 4-asset markets and fits concrete step-to-step maps
    inside the shared market coordinate frame, asking what kind of geometry change risk actually induces.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[22 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[BASE DATA]\ #text(size: 9pt)[Phase 11 risk ladder]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[STATES]\ #text(size: 9pt)[row_mean L1, row_eos L13]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[OBJECT]\ #text(size: 9pt)[adjacent risk-step transforms]],
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
  #text(size: 12.5pt, weight: "medium")[Phase 12 clarifies the deformation story: early risk changes are almost rigid in the shared market frame, while late risk steps sometimes need locally anisotropic fits, but the end-to-end late ladder still composes best as a chain of near-rigid maps rather than as one big flexible linear warp.]
  #v(0.4em)
  #text(size: 9.5pt, fill: rgb("#555"))[
    At `row_mean @ L1`, identity or orthogonal maps already explain the ladder almost perfectly. At `row_eos @ L13`, some adjacent steps prefer linear fits with strong anisotropy, especially `risk_1 -> risk_2` and `risk_2 -> risk_3`. But those flexible local maps do not compose: the late `orthogonal` chain matches the direct `market_only -> risk_5` transform with matrix cosine `0.99998`, whereas the late `linear` chain collapses to matrix cosine `0.0865`.
  ]
]

#v(1.2em)

// ── Summary Metrics ─────────────────────────────────────────────
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[EARLY BEST STEP]\ #text(size: 16pt, weight: "bold")[0.99970] #text(size: 8pt, fill: rgb("#888"))[\ similarity on `3→4`]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[LATE STRONGEST LOCAL FIT]\ #text(size: 16pt, weight: "bold")[0.87296] #text(size: 8pt, fill: rgb("#888"))[\ linear on `1→2`]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[LATE ORTHO COMPOSE]\ #text(size: 16pt, weight: "bold")[0.99998] #text(size: 8pt, fill: rgb("#888"))[\ matrix cosine to direct]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[LATE LINEAR COMPOSE]\ #text(size: 16pt, weight: "bold")[0.08651] #text(size: 8pt, fill: rgb("#888"))[\ matrix cosine to direct]],
)


= Why Phase 12 Exists

Phase 11 established that the model carries a shared 4-asset market frame across the full DX risk ladder and that later states realign that geometry toward score structure. The missing piece was transform clarity. “Positive Spearman” alone does not tell us whether risk is behaving like:

- a rigid reorientation
- a simple compression
- a shear
- or a more arbitrary local distortion

Phase 12 keeps the same synthetic markets and directly fits adjacent risk-step maps in the shared 2D coordinate frame recovered from Phase 11.


= Early State Is Almost Rigid

#align(center)[#image("../../data/report_assets/synthetic_market_phase12_transforms/family_heatmap.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Coordinate reconstruction `R²` for transform families fit on each adjacent ladder step. Left panel: early `row_mean @ L1`. Right panel: late `row_eos @ L13`.
]

#v(0.4em)

The early panel is almost degenerate in a good way. Every step is already near ceiling:

- `market_only -> risk_1`: orthogonal best, `R² = 0.9954`
- `risk_1 -> risk_2`: `R² = 0.9992`
- `risk_2 -> risk_3`: `R² = 0.9992`
- `risk_3 -> risk_4`: similarity best, `R² = 0.9997`
- `risk_4 -> risk_5`: identity best, `R² = 0.9979`
- `market_only -> risk_5`: identity best, `R² = 0.9991`

That confirms the strongest conservative reading of Phase 11. In the early state, the risk ladder barely disturbs the shared market coordinate frame at all. The model is effectively carrying the same market geometry through those settings prompts.


= Late State Is Locally Mixed

The late panel in the same figure is the main new object. At `row_eos @ L13`, the best family now depends on the specific step:

- `market_only -> risk_1`: orthogonal best, `R² = 0.685`
- `risk_1 -> risk_2`: linear best, `R² = 0.873`
- `risk_2 -> risk_3`: linear best, `R² = 0.842`
- `risk_3 -> risk_4`: orthogonal best, `R² = 0.788`
- `risk_4 -> risk_5`: linear best, `R² = 0.703`
- `market_only -> risk_5`: orthogonal best, `R² = 0.728`

So the late risk-conditioned geometry is not uniformly rigid, but it is also not uniformly flexible. Some steps still behave like small near-rigid reorientations. Others admit a better local fit with a more expressive linear map.


= Winning Late Linear Fits Are Often Strongly Anisotropic

#align(center)[#image("../../data/report_assets/synthetic_market_phase12_transforms/late_winner_stats.png", width: 94%)]
#text(size: 8pt, fill: rgb("#888"))[
For each late ladder step, the chart shows the winning family’s anisotropy ratio and rotation angle.
]

#v(0.4em)

This is where the local-linear wins become interpretable. When linear wins in the late state, it is not just a marginally better rotation:

- `risk_1 -> risk_2`: anisotropy `7.17`, angle `46°`
- `risk_2 -> risk_3`: anisotropy `25.86`, angle `-67.7°`
- `risk_4 -> risk_5`: anisotropy `1.65`, angle `-9.6°`

Those first two steps are especially important. They suggest that in the late state, the model is not merely rotating the market frame. It is locally stretching and compressing it in highly directional ways.

So if we stopped here, the temptation would be to say: the late ladder is fundamentally non-rigid.


= But the Global Late Ladder Still Composes Best as Near-Rigid

#align(center)[#image("../../data/report_assets/synthetic_market_phase12_transforms/composition.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[
For `market_only -> risk_5`, compare direct end-to-end fits against the composition of all adjacent maps. The gold line shows matrix cosine between composed and direct transforms.
]

#v(0.4em)

This is the strongest result in the phase. In the late state:

- `orthogonal`
  - direct `R² = 0.7283`
  - composed `R² = 0.7293`
  - matrix cosine `0.99998`
- `linear`
  - direct `R² = 0.7190`
  - composed `R² = 0.6224`
  - matrix cosine `0.0865`

So the flexible local linear fits do #emph[not] add up to a coherent end-to-end risk transform. The near-rigid orthogonal maps do.

That is the key structural point:

- local late steps can look anisotropic
- but the ladder as a whole behaves more like a chain of small reorientations than like one large global shear

This is exactly the kind of clarification that the Phase 11 summary metrics alone could not provide.


= Interpretation

#block(
  width: 100%,
  inset: (left: 14pt, top: 10pt, bottom: 10pt, right: 10pt),
  stroke: (left: 3pt + rgb("#2e7d32"), top: none, right: none, bottom: none),
  fill: rgb("#e8f5e9"),
)[
  #text(size: 7.5pt, fill: rgb("#2e7d32"), weight: "bold", tracking: 0.08em)[SUPPORTED NOW]
  #v(0.2em)
  The late risk-conditioned market geometry is best described as a shared base frame plus a sequence of local risk-step transforms. Some adjacent steps admit anisotropic local fits, but the end-to-end ladder composes most cleanly under near-rigid maps.
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
  A single global flexible linear warp is not supported. Nor is a pure rigid story for every local step: some adjacent late steps still prefer strongly anisotropic local maps.
]

#v(0.8em)

The best synthesis after Phase 12 is:

- #text(weight: "medium")[early market geometry is almost unchanged by risk]
- #text(weight: "medium")[late geometry develops local distortions at some risk steps]
- #text(weight: "medium")[but those distortions sit on top of a globally coherent near-rigid ladder]

That is stronger than both the earlier “one market manifold” framing and the weaker “some deformation exists” framing. It gives a concrete shape to how settings act on the model’s internal market frame.


= What To Do Next

The next experiments should stay within this geometry-transform program:

1. Run the same explicit-transform analysis on one second context family built on the same 4-asset market.
   Portfolio or affordance overlays are the most useful next candidates.
2. Bring the transform lens back to a small real DX matched-risk set.
   The minimum viable bridge is to test whether a comparable near-rigid end-to-end transform exists between real `risk_1` and `risk_5` contexts.
3. If needed, inspect the two anomalous local late steps (`1→2` and `2→3`) with representative examples and targeted geometry plots.


= Appendix A: The Risk Ladder Target Geometry

#align(center)[#image("../../data/report_assets/synthetic_market_phase11_risk_ladder/score_geometry_example.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Representative `top_pair_cluster` example from Phase 11. This is the risk-conditioned score geometry that the late activation-space frame is partially tracking.
]

#v(0.4em)

This appendix figure is included to anchor the Phase 12 transform story. The fitted maps in this report are not arbitrary regressions over labels. They are attempts to recover how the same four-asset market moves through an internal coordinate frame as the risk setting changes.


= Appendix B: How To Read The Transform Families

#table(
  columns: (1.4fr, 3.6fr),
  align: (left, left),
  table.hline(stroke: 1pt),
  table.header([*Term*], [*Meaning*]),
  table.hline(stroke: 0.5pt),
  [`identity`], [No transform at all. If identity already works, the two contexts share almost the same geometry in the decoded frame.],
  [`orthogonal`], [A pure rotation or reflection. Norms and angles are preserved; only orientation changes.],
  [`similarity`], [Uniform scaling plus rotation. This captures isotropic compression or expansion.],
  [`diagonal`], [Axis-aligned rescaling in the shared coordinate frame. This captures simple non-uniform compression without rotation.],
  [`linear`], [Any 2×2 linear map. This is the most flexible family here and can absorb rotation, anisotropic scaling, and shear-like behavior.],
  [`anisotropy ratio`], [Largest singular value divided by smallest singular value. `1.0` means isotropic; larger values mean directional stretching/compression.],
  [`matrix cosine`], [Cosine similarity between two flattened transform matrices. Here it is used to compare the direct `market_only -> risk_5` transform to the composition of adjacent transforms.],
  [`coordinate R²`], [How well a transform predicts target 2D asset coordinates in the shared decoded market frame.],
  table.hline(stroke: 1pt),
)

#v(0.6em)

The short reading guide for Phase 12 is:

1. Look at the #text(weight: "medium")[early heatmap] first.
   If identity and orthogonal are near ceiling, the base market frame is genuinely shared.
2. Then look at the #text(weight: "medium")[late winner stats].
   If anisotropy spikes, some local context steps are doing real geometry work.
3. Then look at #text(weight: "medium")[composition].
   If orthogonal composes and linear does not, the right story is local distortions on top of a globally coherent frame, not one giant linear warp.
