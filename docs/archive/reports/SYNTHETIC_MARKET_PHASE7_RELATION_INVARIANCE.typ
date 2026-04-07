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
  #text(size: 22pt, weight: "bold")[Synthetic Market Phase 7]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    Relation-invariance follow-through on the relational representation report. This phase keeps the anchor-pair relation fixed while changing
    style, layout, roster-rank context, and global magnitude scale.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[21 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[PROMPTS]\ #text(size: 9pt)[384 market-only]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DESIGN]\ #text(size: 9pt)[4 scenarios × 2 styles × 4 layouts × 4 rosters × 3 scales]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[OBJECT]\ #text(size: 9pt)[anchor-pair relation]],
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
  #text(size: 12.5pt, weight: "medium")[Phase 7 confirms the relation-first shift: anchor-pair market relations survive layout, roster-rank, and magnitude changes.]
  #v(0.4em)
  #text(size: 9.5pt, fill: rgb("#555"))[
    This is the first synthetic phase that cleanly answers the “what next after row retrieval?” question. The corrected analysis shows that relation identity beats both
    rank-bucket matching and scale-bucket matching, with uniformly strong margins. The caveat is that the whole task still peaks at `row_mean @ layer 1`, so the
    dataset is likely too easy to reveal deeper structure.
  ]
]

#v(1.2em)

// ── Summary Metrics ─────────────────────────────────────────────
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST PRIMITIVE R²]\ #text(size: 16pt, weight: "bold")[0.99998] #text(size: 8pt, fill: rgb("#888"))[\ row_mean L1]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST RELATION]\ #text(size: 16pt, weight: "bold")[0.2679] #text(size: 8pt, fill: rgb("#888"))[\ roster-only, momentum edge]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST RANK CTRL]\ #text(size: 16pt, weight: "bold")[0.2677] #text(size: 8pt, fill: rgb("#888"))[\ row_mean L1]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST SCALE CTRL]\ #text(size: 16pt, weight: "bold")[0.2312] #text(size: 8pt, fill: rgb("#888"))[\ row_mean L1]],
)


= Why Phase 7 Exists

The previous relational report established the right object shift:

- row retrieval is too brittle
- pairwise relations are a better upstream target

Phase 7 executes the next logical step from that report:

- keep the same anchor-pair relation
- move that relation through different roster-rank contexts
- rescale the whole market magnitude
- and ask whether the relation vector still groups by relation identity rather than by rank or scale alone

This is a stronger test than Phase 6 because it adds two explicit nuisance axes:

- *roster-only*: same style/layout/scale, different rank context
- *magnitude-only*: same style/layout/roster, different absolute scale


= Primitive Factors Are Still Explicit

Primitive-factor decode remains effectively perfect in this family.

#table(
  columns: (1.3fr, 1.2fr, auto, auto),
  align: (left, left, right, right),
  table.hline(stroke: 1pt),
  table.header(
    [*Variable*], [*Best State*], [*Metric*], [*Layer*],
  ),
  table.hline(stroke: 0.5pt),
  [`pct_5m`], [`row_mean`], [R² 0.99998], [L1],
  [`net_flow_5m`], [`row_mean`], [R² 0.99998], [L1],
  [`unique_traders_5m`], [`row_mean`], [R² 0.99998], [L1],
  [`top20_holder_pct`], [`row_mean`], [R² 0.99997], [L1],
  [`attractiveness_score`], [`row_mean`], [R² 0.99998], [L1],
  [`risk_adjusted_score`], [`row_mean`], [R² 0.99998], [L1],
  table.hline(stroke: 1pt),
)

#v(0.5em)

That matters for interpretation: Phase 7 is not discovering that the row holds market information. It is testing whether the *comparative object* built from those rows stays stable across stronger nuisance changes.


= The Rank Context Is Real

#align(center)[#image("../../data/report_assets/synthetic_market_phase7_relation_invariance/rank_buckets.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[
The anchor pair is intentionally moved across different roster-rank contexts. This is not a cosmetic reformatting test; the same latent relation is evaluated under new relative position in the market.
]


= Relation Identity Survives The New Axes

#align(center)[#image("../../data/report_assets/synthetic_market_phase7_relation_invariance/relation_modes.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[
Best relation margins and nearest-neighbor accuracy across control modes. The signal stays strong not only under style and layout changes, but also under roster-only and magnitude-only controls.
]

#v(0.4em)

#table(
  columns: (1.55fr, 1.15fr, auto, auto),
  align: (left, left, right, right),
  table.hline(stroke: 1pt),
  table.header(
    [*Scenario*], [*Best Mode*], [*Best Margin*], [*NN Acc.*],
  ),
  table.hline(stroke: 0.5pt),
  [`momentum edge`], [`roster-only`], [0.2679], [1.000],
  [`flow edge`], [`roster-only`], [0.2669], [1.000],
  [`broad participation`], [`layout-only`], [0.2466], [1.000],
  [`concentration penalty`], [`roster-only`], [0.2442], [1.000],
  table.hline(stroke: 1pt),
)

#v(0.4em)

Two details are most important:

- the best results all occur at `row_mean @ layer 1`
- every control mode still hits `1.0` nearest-neighbor accuracy in the best read

This is much stronger than the earlier row-retrieval story. The same relation survives layout, roster, and scale changes well enough that the nuisance axis no longer dominates the grouping.


= Relation Beats Rank Bucket And Scale Bucket

#align(center)[#image("../../data/report_assets/synthetic_market_phase7_relation_invariance/relation_controls.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[
The real question is not just whether relation retrieval works, but whether the vector prefers the same relation over confounds that share the same rank bucket or magnitude scale.
]

#v(0.4em)

#table(
  columns: (1.55fr, auto, auto, auto),
  align: (left, right, right, right),
  table.hline(stroke: 1pt),
  table.header(
    [*Scenario*], [*Rank Margin*], [*Scale Margin*], [*Best State*],
  ),
  table.hline(stroke: 0.5pt),
  [`momentum edge`], [0.2677], [0.2312], [`row_mean` L1],
  [`flow edge`], [0.2669], [0.2104], [`row_mean` L1],
  [`broad participation`], [0.2440], [0.2137], [`row_mean` L1],
  [`concentration penalty`], [0.2439], [0.2024], [`row_mean` L1],
  table.hline(stroke: 1pt),
)

#v(0.5em)

This is the headline result of Phase 7. Relation identity does not merely survive nuisance variation in the abstract; it specifically beats:

- matching by the same anchor rank bucket
- matching by the same global magnitude bucket

That is exactly the control the previous report called for.


= Interpretation

#block(
  width: 100%,
  inset: (left: 14pt, top: 10pt, bottom: 10pt, right: 10pt),
  stroke: (left: 3pt + rgb("#2e7d32"), top: none, right: none, bottom: none),
  fill: rgb("#e8f5e9"),
)[
  #text(size: 7.5pt, fill: rgb("#2e7d32"), weight: "bold", tracking: 0.08em)[SUPPORTED NOW]
  #v(0.2em)
  Relation-first was the right methodological move. In this synthetic family, the model's market representation groups strongly by anchor-pair relation identity even when rank context and magnitude scale are changed.
]

#v(0.5em)

#block(
  width: 100%,
  inset: (left: 14pt, top: 10pt, bottom: 10pt, right: 10pt),
  stroke: (left: 3pt + rgb("#f57f17"), top: none, right: none, bottom: none),
  fill: rgb("#fff8e1"),
)[
  #text(size: 7.5pt, fill: rgb("#f57f17"), weight: "bold", tracking: 0.08em)[LIMITATION]
  #v(0.2em)
  The task is probably still too easy. Everything peaks at `row_mean @ layer 1`, which suggests Phase 7 is confirming the object shift more than revealing a deep later-stage abstraction.
]

#v(0.8em)

So the correct reading is:

- this is a real win for the relation-first program
- but not yet the final form of the market-representation experiment

The next dataset should preserve the relation-first target while making the negatives much harder:

- relation pairs with more closely matched raw factor deltas
- confounds that share pairwise ordering but differ in which factor actually drives the edge
- harder surface/layout disturbance without letting `row_mean @ L1` solve the task trivially


= Conclusion

Phase 7 does exactly what the previous report asked for:

- it replaces row identity with relation identity
- it adds explicit rank and magnitude controls
- and it shows that relation identity wins those comparisons cleanly

That means the project should keep moving *upstream* and *relational*, not slide back to row retrieval or end-state labels.

#v(2em)
#line(length: 100%, stroke: 0.5pt + rgb("#ccc"))
#v(0.3em)
#text(size: 8pt, fill: rgb("#999"))[
Synthetic Market Phase 7 Relation Invariance — 21 March 2026. Corrected Phase 7 rerun after fixing the scenario-anchored comparison pool for the new relation-over-rank and relation-over-scale controls.
]
