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
  #text(size: 22pt, weight: "bold")[Synthetic Market Relational Representation]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    Follow-up on Phase 6 profile invariance. This report replaces row retrieval as the main object and asks a more representation-faithful question:
    are pairwise market relations and whole-snapshot geometry more stable than single-row identity under nuisance variation?
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[21 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[PROMPTS]\ #text(size: 9pt)[48 market-only]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[SCENARIOS]\ #text(size: 9pt)[2 tied-tradeoff families]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[ANALYSIS]\ #text(size: 9pt)[relational + set-level]],
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
  #text(size: 12.5pt, weight: "medium")[Row retrieval was the wrong object. Pairwise market relations survive nuisance variation far better than single-row identity.]
  #v(0.4em)
  #text(size: 9.5pt, fill: rgb("#555"))[
    Under the hardest layout-only control, single-row profile identity is nearly gone. But pairwise relation invariance remains strong, with margins roughly 15–20× larger. Whole-snapshot geometry survives too, but more weakly. The best current reading is that the model's market understanding is more relational than row-identitarian.
  ]
]

#v(1.2em)

// ── Summary Metrics ─────────────────────────────────────────────
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[ROW IDENTITY]\ #text(size: 16pt, weight: "bold")[0.011] #text(size: 8pt, fill: rgb("#888"))[\ best layout-only margin]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[PAIRWISE RELATION]\ #text(size: 16pt, weight: "bold")[0.162] #text(size: 8pt, fill: rgb("#888"))[\ best layout-only margin]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[SNAPSHOT GEOMETRY]\ #text(size: 16pt, weight: "bold")[0.00093] #text(size: 8pt, fill: rgb("#888"))[\ best layout-only margin]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST READ]\ #text(size: 16pt, weight: "bold")[Relational] #text(size: 8pt, fill: rgb("#888"))[\ not row retrieval]],
)


= Why This Report Exists

Phase 6 already established two things:

- primitive market factors are explicit in row states
- single-row profile retrieval is fragile under layout changes

That left an obvious methodological question: was the model failing to abstract the market, or were we probing the wrong object?

This report answers that by shifting from *row identity* to two more faithful representation objects:

- *pairwise relations*: how one asset is represented relative to another
- *whole-snapshot geometry*: whether the market as a set retains the same latent shape


= The Comparison That Matters

#align(center)[#image("../../data/report_assets/synthetic_market_relational/object_comparison.png", width: 95%)]
#text(size: 8pt, fill: rgb("#888"))[
Layout-only margins across three candidate representation objects. Row identity is weak, pairwise relations are much stronger, and whole-snapshot geometry is present but secondary.
]

#v(0.4em)

#table(
  columns: (1.7fr, 1.1fr, auto, auto, auto),
  align: (left, left, right, right, left),
  table.hline(stroke: 1pt),
  table.header(
    [*Scenario*], [*Object*], [*Margin*], [*NN Acc.*], [*Read*],
  ),
  table.hline(stroke: 0.5pt),
  [`momentum × flow`], [`row identity`], [0.006], [0.667], [Barely stable under layout changes.],
  [`momentum × flow`], [`pairwise relation`], [0.127], [0.854], [Much stronger comparative structure.],
  [`momentum × flow`], [`snapshot geometry`], [0.00085], [0.583], [Weak but above pure collapse.],
  [`participation × concentration`], [`row identity`], [0.011], [0.719], [Still weak as a single-row object.],
  [`participation × concentration`], [`pairwise relation`], [0.162], [0.882], [Strongest current abstraction read.],
  [`participation × concentration`], [`snapshot geometry`], [0.00093], [0.667], [Set-level structure survives somewhat.],
  table.hline(stroke: 1pt),
)

#v(0.5em)

This is the key update. The earlier row-retrieval stress test was useful, but it was not the right scientific object. Once the same latent market is examined in pairwise-relative form, the signal becomes much more robust.


= Pairwise Relations Are The Strongest Current Object

#align(center)[#image("../../data/report_assets/synthetic_market_relational/relation_modes.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[
Pairwise relation margins and nearest-neighbor accuracy across nuisance modes. Style changes barely matter. Layout changes reduce the signal, but far less than they did for row identity.
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
  [`momentum × flow`], [`full`], [0.182], [0.965],
  [`momentum × flow`], [`style-only`], [0.344], [1.000],
  [`momentum × flow`], [`layout-only`], [0.127], [0.854],
  [`participation × concentration`], [`full`], [0.241], [0.924],
  [`participation × concentration`], [`style-only`], [0.348], [0.951],
  [`participation × concentration`], [`layout-only`], [0.162], [0.882],
  table.hline(stroke: 1pt),
)

#v(0.4em)

Interpretation:

- style changes are not the bottleneck
- layout changes still matter, but relations remain strongly recoverable
- `participation × concentration` remains the stronger family
- the comparative market representation is more stable than the single-row one


= Whole-Snapshot Geometry Is Secondary But Real

#align(center)[#image("../../data/report_assets/synthetic_market_relational/geometry_modes.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[
Whole-snapshot geometry survives nuisance variation more weakly than pairwise relations. Style-only alignment remains respectable; layout-only alignment is modest.
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
  [`momentum × flow`], [`full`], [0.00072], [0.792],
  [`momentum × flow`], [`style-only`], [0.00267], [0.792],
  [`momentum × flow`], [`layout-only`], [0.00085], [0.583],
  [`participation × concentration`], [`full`], [0.00044], [0.792],
  [`participation × concentration`], [`style-only`], [0.00256], [0.958],
  [`participation × concentration`], [`layout-only`], [0.00093], [0.667],
  table.hline(stroke: 1pt),
)

#v(0.4em)

This is not yet a strong global-manifold result. But it does suggest that some whole-market set structure survives nuisance variation even after row identity itself becomes unreliable.


= Interpretation

#block(
  width: 100%,
  inset: (left: 14pt, top: 10pt, bottom: 10pt, right: 10pt),
  stroke: (left: 3pt + rgb("#2e7d32"), top: none, right: none, bottom: none),
  fill: rgb("#e8f5e9"),
)[
  #text(size: 7.5pt, fill: rgb("#2e7d32"), weight: "bold", tracking: 0.08em)[SUPPORTED NOW]
  #v(0.2em)
  The model's market representation is better described as comparative structure over assets than as stable identity of isolated row embeddings.
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
  We still do not have evidence for a strongly layout-invariant row-profile representation or a clean global market manifold.
]

#v(0.8em)

Best current reading:

- primitive factors are explicit in row states
- row retrieval is too brittle to be the main object
- pairwise relations survive nuisance variation well enough to be the best current upstream representation target
- whole-snapshot geometry is worth tracking, but is not yet the strongest claim


= Next Step

The right next phase is not more row retrieval.

- Build a harder *relation-first* synthetic dataset with near ties and rank-vs-magnitude controls.
- Hold pairwise latent relations fixed while varying roster composition and distractors.
- Measure whether relation vectors, not row identity, remain stable.
- Then validate the strongest relation family on real DX rows.

#v(2em)
#line(length: 100%, stroke: 0.5pt + rgb("#ccc"))
#v(0.3em)
#text(size: 8pt, fill: rgb("#999"))[
Synthetic Market Relational Representation — 21 March 2026. Analysis derived from the completed `phase6_profile_invariance_v1` synthetic capture set. The key result is a representation-object shift: pairwise relations beat row identity under nuisance variation.
]
