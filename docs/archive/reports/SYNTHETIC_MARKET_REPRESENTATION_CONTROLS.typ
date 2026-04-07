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
  #text(size: 22pt, weight: "bold")[Synthetic Market Representation Controls]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    Representation-focused checkpoint after Phase 4 rank-context probing and the Phase 5 symbol-permutation control. Research anchors:
    `RANKED_RESEARCH_ROADMAP.md`, `MARKET_COUNTING_MANIFOLDS_PLAN.md`, and `MARKET_REPRESENTATION_CONTROLS_NOTES.md`.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[21 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[PHASE 4]\ #text(size: 9pt)[69 market-only prompts]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[PHASE 5]\ #text(size: 9pt)[12 permuted prompts]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[LAYERS]\ #text(size: 9pt)[48]],
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
  #text(size: 12.5pt, weight: "medium")[Primitive market variables are real; profile-level abstraction is selective.]
  #v(0.4em)
  #text(size: 9.5pt, fill: rgb("#555"))[
    The row states preserve momentum, flow, participation, concentration, and synthetic latent scores with near-perfect linear fidelity.
    But once row position and display symbol shortcuts are explicitly removed, only some profile families remain clearly separable.
    Participation/concentration survives the hard control; momentum/flow mostly collapses.
  ]
]

#v(1.2em)

// ── Summary Metrics ─────────────────────────────────────────────
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST PRIMITIVE R²]\ #text(size: 16pt, weight: "bold")[0.99999] #text(size: 8pt, fill: rgb("#888"))[\ concentration, Phase 5]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[PHASE 4 CONFOUND]\ #text(size: 16pt, weight: "bold")[0.523] #text(size: 8pt, fill: rgb("#888"))[\ same-symbol margin]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST CONTROL MARGIN]\ #text(size: 16pt, weight: "bold")[0.262] #text(size: 8pt, fill: rgb("#888"))[\ participation/concentration]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[CONTROL BASELINE]\ #text(size: 16pt, weight: "bold")[0.188] #text(size: 8pt, fill: rgb("#888"))[\ random profile retrieval]],
)


= Scope

This report narrows the question to market representation itself.

- *Not action labels.* The target is how the model encodes the market rows, not what final tool call it emits.
- *Primitive variables first.* The first test asks whether the model linearly preserves market factors like momentum, flow, participation, and concentration.
- *Confound controls next.* The second test asks whether those representations survive symbol and row permutations.
- *Synthetic-first.* Everything here is backed by synthetic datasets before any return to real DX validation.

#align(center)[#image("../../data/report_assets/synthetic_market_representation/dataset_counts.png", width: 82%)]
#text(size: 8pt, fill: rgb("#888"))[
Phase 4 and Phase 5 prompt counts. Phase 4 is the broader market-representation slice; Phase 5 is the tighter symbol-permutation control.
]


= Primitive Factor Results

The stable result across both phases is that primitive market variables are extremely decodable from the row states.

#align(center)[#image("../../data/report_assets/synthetic_market_representation/primitive_regression.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[
Best held-out regression performance for primitive market variables. Both phases show near-perfect linear recovery, which means the row states preserve these variables very explicitly.
]

#v(0.5em)

#table(
  columns: (1.3fr, 1.2fr, auto, auto),
  align: (left, left, right, right),
  table.hline(stroke: 1pt),
  table.header(
    [*Question*], [*Best State*], [*Metric*], [*Layer*],
  ),
  table.hline(stroke: 0.5pt),
  [`pct_5m`], [`row_mean`], [R² 0.997 / 1.000], [L6 / L1],
  [`net_flow_5m`], [`row_mean`], [R² 0.994 / 1.000], [L6 / L1],
  [`unique_traders_5m`], [`row_mean`], [R² 0.998 / 1.000], [L14 / L1],
  [`top20_holder_pct`], [`row_mean`], [R² 0.999 / 1.000], [L6 / L1],
  [`attractiveness_score`], [`row_mean`], [R² 0.998 / 1.000], [L6 / L1],
  [`risk_adjusted_score`], [`row_mean`], [R² 0.998 / 1.000], [L6 / L1],
  table.hline(stroke: 1pt),
)

#v(0.5em)

This is the strongest supported claim in the current program: the model really is carrying specific market factors in the row states. That is a representation result, not just a behavior result.


= Why Phase 4 Was Not Enough

Phase 4 also included a rank-context retrieval idea: keep the focal asset pair numerically fixed while changing the background roster, then ask whether the same latent asset remains nearest to itself across variants.

#align(center)[#image("../../data/report_assets/synthetic_market_representation/rank_context_confound.png", width: 86%)]
#text(size: 8pt, fill: rgb("#888"))[
Phase 4 looked strong if read naively, but these margins were still heavily driven by symbol identity and row position.
]

- `fixed_momentum_flow_pair`: same-symbol cosine margin `0.174`
- `fixed_participation_concentration_pair`: same-symbol cosine margin `0.523`
- Both scenarios had same-symbol nearest-neighbor accuracy `1.0`

That is exactly why Phase 5 was necessary. Phase 4 showed persistence, but not whether persistence came from latent market profile or from lexical/positional shortcuts.


= Phase 5 Symbol-Permutation Control

Phase 5 permutes the same latent profiles across both display symbol and row index. The main metric is stricter than the earlier nearest-neighbor read:

- `profile_control_nn_accuracy`
- `profile_control_margin`

These metrics exclude every candidate that shares the same display symbol or same row position, so the remaining retrieval problem is closer to the latent question we actually care about.

#align(center)[#image("../../data/report_assets/synthetic_market_representation/symbol_permutation_control.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[
Layerwise control metrics after removing row and symbol shortcuts. The dashed line is the random baseline induced by the phase-5 permutation layouts (`0.188`).
]


= What Survives The Hard Control

#align(center)[#image("../../data/report_assets/synthetic_market_representation/best_layer_similarity.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[
Best-layer similarity breakdown for each Phase 5 scenario. Participation/concentration remains profile-dominant; momentum/flow becomes nearly tied once row and symbol are removed.
]

#table(
  columns: (1.5fr, 1.1fr, auto, auto, auto),
  align: (left, left, right, right, right),
  table.hline(stroke: 1pt),
  table.header(
    [*Scenario*], [*Best State*], [*Control Margin*], [*Control NN Acc.*], [*Read*],
  ),
  table.hline(stroke: 0.5pt),
  [`momentum_flow_permuted_market`], [`row_mean` L9], [0.001], [0.667], [Above-chance retrieval exists, but cosine separation is almost fully collapsed.],
  [`participation_concentration_permuted_market`], [`row_eos` L1], [0.262], [0.542], [Profile identity survives the hard control meaningfully better.],
  table.hline(stroke: 1pt),
)

#v(0.5em)

At the best participation/concentration layer:

- same-profile cosine mean: `0.999`
- same-symbol cosine mean: `0.753`
- same-row cosine mean: `0.631`
- best non-profile cosine after row/symbol removal: `0.738`

At the best momentum/flow layer:

- same-profile cosine mean: `0.989`
- same-symbol cosine mean: `0.988`
- same-row cosine mean: `0.998`
- best non-profile cosine after row/symbol removal: `0.994`

So the participation/concentration family looks like a genuine profile-level abstraction candidate. The momentum/flow family does not, at least not yet.


= Why The AUROC 1.00 Results Are Not The Headline

#block(
  width: 100%,
  inset: (left: 14pt, top: 10pt, bottom: 10pt, right: 10pt),
  stroke: (left: 3pt + rgb("#f57f17"), top: none, right: none, bottom: none),
  fill: rgb("#fff8e1"),
)[
  #text(size: 7.5pt, fill: rgb("#f57f17"), weight: "bold", tracking: 0.08em)[CAUTION]
  #v(0.2em)
  `AUROC 1.000` on the hard pairwise synthetic slices does not mean we found a rich market semantics circuit. It mainly shows that the primitive synthetic tradeoff labels are linearly recoverable in an intentionally clean setting.
]

#v(0.6em)

This is why the report now emphasizes control-based profile retrieval rather than pairwise classification:

- pairwise AUROC can be perfect while still reflecting a very easy synthetic decision boundary
- confound controls tell us more about abstraction
- the real question is not whether the model can recover synthetic labels, but whether the same latent asset profile remains stable after superficial prompt identity changes


= Interpretation

#block(
  width: 100%,
  inset: (left: 14pt, top: 10pt, bottom: 10pt, right: 10pt),
  stroke: (left: 3pt + rgb("#2e7d32"), top: none, right: none, bottom: none),
  fill: rgb("#e8f5e9"),
)[
  #text(size: 7.5pt, fill: rgb("#2e7d32"), weight: "bold", tracking: 0.08em)[SUPPORTED NOW]
  #v(0.2em)
  Primitive market variables are explicitly present in row states, and at least one profile family survives a hard row/symbol permutation control.
]

#v(0.5em)

#block(
  width: 100%,
  inset: (left: 14pt, top: 10pt, bottom: 10pt, right: 10pt),
  stroke: (left: 3pt + rgb("#f57f17"), top: none, right: none, bottom: none),
  fill: rgb("#fff8e1"),
)[
  #text(size: 7.5pt, fill: rgb("#f57f17"), weight: "bold", tracking: 0.08em)[NOT SUPPORTED YET]
  #v(0.2em)
  We do not yet have evidence for a single, uniformly robust profile-level market representation across all factor families.
]

#v(0.6em)

Best current interpretation:

- The model has strong early access to primitive market factors.
- Some higher-order abstractions exist, but they are selective.
- Participation/concentration looks substantially more stable under symbolic relabeling than momentum/flow.
- The right next target is not a global market manifold claim; it is controlled, family-specific abstraction testing.


= Next Step

The most defensible next experiment is a harder synthetic profile-invariance pass:

+ preserve the Phase 5 permutation control
+ add row-text paraphrases and order perturbations
+ create near-tied momentum/flow cases with sharper participation/concentration differences
+ rerun the same profile-control metrics before bringing the strongest surviving family back to real DX rows

If participation/concentration remains stable under those stronger wording controls, that becomes the best candidate for real-data validation.

#v(2em)
#line(length: 100%, stroke: 0.5pt + rgb("#ccc"))
#v(0.3em)
#text(size: 8pt, fill: rgb("#999"))[
Synthetic Market Representation Controls — 21 March 2026. Phase 4: 69 market-only prompts. Phase 5: 12 market-only symbol-permutation prompts. Dedicated synthetic Modal volume, 48-layer surrogate capture stack.
]
