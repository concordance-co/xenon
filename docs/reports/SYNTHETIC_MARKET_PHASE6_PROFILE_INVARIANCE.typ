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
  #text(size: 22pt, weight: "bold")[Synthetic Market Phase 6]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    Profile-invariance checkpoint for the representation-first synthetic track. Research anchors:
    `PHASE6_PROFILE_INVARIANCE_NOTES.md`, `SYNTHETIC_MARKET_REPRESENTATION_CONTROLS.typ`, and `RANKED_RESEARCH_ROADMAP.md`.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[21 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[PROMPTS]\ #text(size: 9pt)[48 market-only]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DESIGN]\ #text(size: 9pt)[2 scenarios × 4 styles × 6 layouts]],
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
  #text(size: 12.5pt, weight: "medium")[Phase 6 says the abstraction bottleneck is layout, not wording.]
  #v(0.4em)
  #text(size: 9.5pt, fill: rgb("#555"))[
    Primitive market factors remain almost perfectly explicit in row states. Under the harder profile-invariance control,
    `participation × concentration` still survives better than `momentum × flow`. But the decomposition shows the crucial point:
    changing surface style barely breaks retrieval, while moving profiles across row layouts does.
  ]
]

#v(1.2em)

// ── Summary Metrics ─────────────────────────────────────────────
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST PRIMITIVE R²]\ #text(size: 16pt, weight: "bold")[0.9998] #text(size: 8pt, fill: rgb("#888"))[\ attractiveness, row_mean L1]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST FULL CONTROL]\ #text(size: 16pt, weight: "bold")[0.032] #text(size: 8pt, fill: rgb("#888"))[\ part./conc., row_eos L16]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST STYLE-ONLY]\ #text(size: 16pt, weight: "bold")[0.121] #text(size: 8pt, fill: rgb("#888"))[\ part./conc., row_eos L15]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST LAYOUT-ONLY]\ #text(size: 16pt, weight: "bold")[0.011] #text(size: 8pt, fill: rgb("#888"))[\ part./conc., row_eos L21]],
)


= Scope

Phase 6 is not another generic synthetic probe run. It isolates one very specific question:

- if profile-level market abstractions exist, what kinds of nuisance changes actually break them?

The dataset keeps the latent profiles fixed while changing:

- surface style
- display symbols
- row layout

This phase covers two scenario families:

- `momentum_flow_tiebreak`
- `participation_concentration_tiebreak`

The important methodological shift is that Phase 6 does *not* treat all nuisance factors as one undifferentiated stress test. The report decomposes the failure into *style-only* and *layout-only* retrieval.


= Primitive Factors Still Hold

The harder invariance slice does not challenge primitive-factor representation.

#align(center)[#image("../../data/report_assets/synthetic_market_phase6_profile_invariance/primitive_regression_phase6.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[
Best held-out primitive regression performance from Phase 6. Primitive market variables remain almost perfectly recoverable from `row_mean`, usually by layer 1.
]

#v(0.4em)

#table(
  columns: (1.3fr, 1.2fr, auto, auto),
  align: (left, left, right, right),
  table.hline(stroke: 1pt),
  table.header(
    [*Variable*], [*Best State*], [*Metric*], [*Layer*],
  ),
  table.hline(stroke: 0.5pt),
  [`pct_5m`], [`row_mean`], [R² 0.9997], [L1],
  [`net_flow_5m`], [`row_mean`], [R² 0.9996], [L1],
  [`unique_traders_5m`], [`row_mean`], [R² 0.9991], [L1],
  [`top20_holder_pct`], [`row_mean`], [R² 0.9998], [L1],
  [`attractiveness_score`], [`row_mean`], [R² 0.9998], [L1],
  [`risk_adjusted_score`], [`row_mean`], [R² 0.9998], [L1],
  table.hline(stroke: 1pt),
)

#v(0.5em)

This remains the most stable representation claim in the project: the row states explicitly carry primitive market factors. The interesting uncertainty is higher-order abstraction, not raw factor storage.


= Full Control

Compared to Phase 5, the all-nuisance Phase 6 control is much harsher.

#align(center)[#image("../../data/report_assets/synthetic_market_phase6_profile_invariance/full_control_comparison.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[
Phase 5 versus Phase 6 on the strict full profile-control metric. The harder combined nuisance stack compresses both families, but participation/concentration still holds up better.
]

#v(0.4em)

#table(
  columns: (1.7fr, 1.2fr, auto, auto, auto),
  align: (left, left, right, right, right),
  table.hline(stroke: 1pt),
  table.header(
    [*Scenario*], [*Best State*], [*Margin*], [*NN Acc.*], [*Read*],
  ),
  table.hline(stroke: 0.5pt),
  [`momentum × flow`], [`row_eos` L43], [0.020], [0.625], [Still above zero, but only weakly separated under the full nuisance stack.],
  [`participation × concentration`], [`row_eos` L16], [0.032], [0.771], [Remains the stronger profile-level candidate under the hardest read.],
  table.hline(stroke: 1pt),
)

#v(0.4em)

This is enough to keep `participation × concentration` as the lead abstraction family. It is *not* enough to claim a broadly nuisance-invariant market profile representation.


= What Actually Breaks

The decomposition is the main new result.

#align(center)[#image("../../data/report_assets/synthetic_market_phase6_profile_invariance/decomposition_best.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[
Best decomposition metrics by scenario. Style-only retrieval is strong for both families. Layout-only retrieval is the real bottleneck, and participation/concentration remains better on that harder axis.
]

#align(center)[#image("../../data/report_assets/synthetic_market_phase6_profile_invariance/decomposition_curves.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[
Layerwise margins for full control, style-only retrieval, and layout-only retrieval. The gap between style-only and layout-only is the core Phase 6 finding.
]

#v(0.4em)

#table(
  columns: (1.5fr, 1.2fr, auto, auto),
  align: (left, left, right, right),
  table.hline(stroke: 1pt),
  table.header(
    [*Scenario*], [*Mode*], [*Best Margin*], [*Best NN Acc.*],
  ),
  table.hline(stroke: 0.5pt),
  [`momentum × flow`], [`style-only`], [0.118], [1.000],
  [`momentum × flow`], [`layout-only`], [0.006], [0.667],
  [`participation × concentration`], [`style-only`], [0.121], [1.000],
  [`participation × concentration`], [`layout-only`], [0.011], [0.719],
  table.hline(stroke: 1pt),
)

#v(0.5em)

Interpretation:

- surface wording / formatting / alias changes are *not* the main problem
- layout movement is the main problem
- `participation × concentration` still has the better layout-sensitive signal
- the next useful experiment should therefore target *row-layout invariance*, not more paraphrase-only variation


= Why This Matters

#block(
  width: 100%,
  inset: (left: 14pt, top: 10pt, bottom: 10pt, right: 10pt),
  stroke: (left: 3pt + rgb("#2e7d32"), top: none, right: none, bottom: none),
  fill: rgb("#e8f5e9"),
)[
  #text(size: 7.5pt, fill: rgb("#2e7d32"), weight: "bold", tracking: 0.08em)[SUPPORTED NOW]
  #v(0.2em)
  Primitive market factors are explicit in row states, and profile retrieval is substantially more robust to style changes than to row-layout changes.
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
  We still do *not* have a strongly row-layout-invariant profile abstraction. The best full-control and layout-only margins are positive, but still small.
]

#v(0.8em)

This changes the next step. The project should not spend its next iteration on more generic manifold plots or more paraphrase variants. The right next move is a layout-targeted representation study:

- keep surface style fixed
- vary roster position and distractor composition harder
- test whether pairwise-relative or difference-vector representations recover profile identity better than raw row retrieval

That is the cleanest way to keep the work upstream and representation-focused without repeating the same test again.
