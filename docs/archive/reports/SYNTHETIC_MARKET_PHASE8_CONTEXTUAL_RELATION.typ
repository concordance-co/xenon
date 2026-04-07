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
  #text(size: 22pt, weight: "bold")[Synthetic Market Phase 8]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    Contextual relation follow-through on the Phase 7 relation-invariance result. This phase keeps the anchor pair numerically fixed and asks whether
    the pair's relation survives harder contextual roster pressure.
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
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[OBJECT]\ #text(size: 9pt)[contextual anchor-pair relation]],
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
  #text(size: 12.5pt, weight: "medium")[Phase 8 breaks most of the easy Phase 7 relation win: primitive factors stay explicit, but contextual relation identity is only weakly preserved.]
  #v(0.4em)
  #text(size: 9.5pt, fill: rgb("#555"))[
    This is a useful negative result. Once the anchor pair is held numerically fixed and the surrounding roster changes what that pair means,
    relation retrieval mostly collapses. The one family that still shows a modest surviving signal is `paired_cluster_context`.
  ]
]

#v(1.2em)

// ── Summary Metrics ─────────────────────────────────────────────
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST PRIMITIVE R²]\ #text(size: 16pt, weight: "bold")[0.99993] #text(size: 8pt, fill: rgb("#888"))[\ row_mean L1]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST RELATION]\ #text(size: 16pt, weight: "bold")[0.03273] #text(size: 8pt, fill: rgb("#888"))[\ paired cluster, style-only]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST RANK CTRL]\ #text(size: 16pt, weight: "bold")[0.00459] #text(size: 8pt, fill: rgb("#888"))[\ paired cluster]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST SCALE CTRL]\ #text(size: 16pt, weight: "bold")[0.01397] #text(size: 8pt, fill: rgb("#888"))[\ paired cluster]],
)


= Why Phase 8 Exists

Phase 7 showed that relation identity beat rank-bucket and scale-bucket confounds. That was encouraging, but still likely too easy.

Phase 8 tightens the question:

- keep the anchor pair numerically fixed
- move it through harder contextual rosters
- keep the same nuisance axes from Phase 7

The new target is not “can the model preserve a simple pair?” but “does the meaning of that pair stay stable when the surrounding market changes?”


= Primitive Factors Still Dominate The Rows

#table(
  columns: (1.3fr, 1.2fr, auto, auto),
  align: (left, left, right, right),
  table.hline(stroke: 1pt),
  table.header(
    [*Variable*], [*Best State*], [*Metric*], [*Layer*],
  ),
  table.hline(stroke: 0.5pt),
  [`pct_5m`], [`row_mean`], [R² 0.99992], [L1],
  [`net_flow_5m`], [`row_mean`], [R² 0.99991], [L1],
  [`unique_traders_5m`], [`row_mean`], [R² 0.99991], [L1],
  [`top20_holder_pct`], [`row_mean`], [R² 0.99990], [L1],
  [`attractiveness_score`], [`row_mean`], [R² 0.99993], [L1],
  [`risk_adjusted_score`], [`row_mean`], [R² 0.99993], [L1],
  table.hline(stroke: 1pt),
)

#v(0.5em)

Primitive market factors remain almost perfectly linearly explicit. So Phase 8 is not showing a loss of market information; it is showing a failure of the *relation object* under harder contextual pressure.


= The Contextual Rank Regimes Are Real

#align(center)[#image("../../data/report_assets/synthetic_market_phase8_contextual_relation/rank_buckets.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[
The anchor pair is numerically fixed, but the surrounding roster moves it across different rank regimes. This is a contextual market test, not a simple reformatting exercise.
]


= Relation Identity Mostly Collapses

#align(center)[#image("../../data/report_assets/synthetic_market_phase8_contextual_relation/relation_modes.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[
Phase 8 margins are much smaller than Phase 7. The remaining signal is modest and scenario-specific rather than broad and invariant.
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
  [`generic duel`], [`style-only`], [0.02874], [0.4167],
  [`momentum shadow`], [`style-only`], [0.02126], [0.5521],
  [`flow shadow`], [`style-only`], [0.01065], [0.5208],
  [`paired cluster`], [`style-only`], [0.03273], [0.6146],
  table.hline(stroke: 1pt),
)

#v(0.5em)

The large Phase 7 margins are gone. What remains is weak and uneven:

- `paired_cluster_context` is the only scenario that still looks somewhat coherent
- `generic_duel_context` and `momentum_shadow_context` retain only modest signal
- `flow_shadow_context` is nearly gone


= Hard Controls Expose The Failure Mode

#align(center)[#image("../../data/report_assets/synthetic_market_phase8_contextual_relation/relation_controls.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[
The hard question is whether relation identity beats explicit contextual confounds. In Phase 8, those margins are weak almost everywhere.
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
  [`generic duel`], [0.00134], [0.00782], [`row_mean` L1 / `row_eos` L36],
  [`momentum shadow`], [0.00034], [-0.00080], [`row_eos` L4],
  [`flow shadow`], [0.00013], [-0.00027], [`row_mean` L1 / `row_eos` L36],
  [`paired cluster`], [0.00459], [0.01397], [`row_eos` L35 / L36],
  table.hline(stroke: 1pt),
)

#v(0.5em)

The failure is most obvious in the harder axes:

- `layout_only` is weak or negative in three of four scenarios
- `roster_only` is near zero except for `paired_cluster_context`
- `rank_ctrl` is almost zero everywhere
- `scale_ctrl` is only meaningfully positive for `paired_cluster_context`

This is the real Phase 8 result: the broader contextual meaning of a fixed pair is not a broadly stable invariant representation.


= Interpretation

#block(
  width: 100%,
  inset: (left: 14pt, top: 10pt, bottom: 10pt, right: 10pt),
  stroke: (left: 3pt + rgb("#2e7d32"), top: none, right: none, bottom: none),
  fill: rgb("#e8f5e9"),
)[
  #text(size: 7.5pt, fill: rgb("#2e7d32"), weight: "bold", tracking: 0.08em)[SUPPORTED NOW]
  #v(0.2em)
  The model still carries primitive market factors very clearly and can still solve direct pairwise comparisons with ease.
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
  A fixed anchor-pair relation is not a broadly robust market representation object once contextual roster pressure is introduced. Phase 7 overstated the strength of that object.
]

#v(0.8em)

The strongest honest reading is:

- primitive local factors are real
- direct pairwise preference is real
- but contextual relation identity is not generally stable

This is useful because it tells us what *not* to overclaim.


= What To Do Next

Phase 8 suggests the next object should not be “the identity of a fixed pair.”

Better next directions:

1. set-level market geometry
2. factor-difference representations like `delta momentum`, `delta participation`, and `delta concentration`
3. context-conditioned relation families rather than one frozen anchor pair
4. explicit follow-up on `paired_cluster_context`, the only scenario that still shows a meaningful surviving signal


= Conclusion

Phase 8 is a negative result in the right place.

It keeps the good part of the relation-first program:

- focus on upstream comparative structure, not end actions

But it shows the limit of the current object:

- fixed anchor-pair relation is too brittle under real contextual roster pressure

That is progress. It means the next market-representation experiment should move from pair identity to richer context-sensitive relational structure.
