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

#let prompt-line(line) = {
  if line == "" {
    linebreak()
  } else if line.starts-with("   - ") {
    h(1.6em)
    [- #line.slice(5)]
    linebreak()
  } else if line.starts-with("  - ") {
    h(1.2em)
    [- #line.slice(4)]
    linebreak()
  } else {
    [#line]
    linebreak()
  }
}

#let prompt-block(path) = {
  set text(font: "Menlo", size: 7.0pt)
  set par(justify: false, leading: 0.48em)
  for line in read(path).split("\n") {
    prompt-line(line)
  }
}

#let summary = json("../../data/report_assets/synthetic_market_phase17_axis_decomposition/summary.json")
#let leader = summary.at("leader")
#let dispersion = summary.at("dispersion")
#let subspace = summary.at("subspace")
#let mm = subspace.at("market_mean")
#let me = subspace.at("market_eos")

#let fmt3(x) = {
  let y = calc.round(x * 1000) / 1000
  str(y)
}

#let cum(state, layer, idx) = state.at("selected_layers").at(str(layer)).at("cumulative").at(idx - 1)
#let pr(state, layer) = state.at("selected_layers").at(str(layer)).at("participation_ratio")

#let mono(text_value, size: 12pt, weight: "bold", fill: black) = {
  text(font: "Menlo", size: size, weight: weight, fill: fill)[#text_value]
}

#let summary-card(title, body, detail) = block(
  width: 100%,
  inset: (left: 6pt, right: 6pt, top: 6pt, bottom: 6pt),
)[
  #text(size: 7pt, fill: rgb("#888"), weight: "bold")[#title]
  #v(0.25em)
  #body
  #v(0.25em)
  #text(size: 8pt, fill: rgb("#666"))[#detail]
]

// ── Title Block ─────────────────────────────────────────────────
#align(left)[
  #text(size: 9pt, fill: rgb("#b33a2a"), tracking: 0.08em, weight: "medium")[XENON INTERPRETABILITY]
  #v(0.3em)
  #text(size: 22pt, weight: "bold")[Synthetic Market Phase 17]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    Prompt-derived decomposition of the strongest discovered market axes from the Phase 15 residualized discovery basis.
    This phase does not add new captures. It asks what the leader-like and dispersion-like axes are actually made of,
    using only visible market rows and aggregates computed from those rows.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[24 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[CAPTURES]\ #text(size: 9pt)[No new capture]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[PROMPTS]\ #text(size: 9pt)[`184` market-only prompts]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[METHOD]\ #text(size: 9pt)[prompt-derived feature decomposition + subspace summary]],
  )
  #v(0.3em)
  #line(length: 100%, stroke: 0.5pt + rgb("#ccc"))
]

#v(1em)

#block(
  width: 100%,
  inset: (left: 14pt, top: 12pt, bottom: 12pt, right: 12pt),
  stroke: (left: 3pt + rgb("#b33a2a"), top: none, right: none, bottom: none),
  fill: rgb("#faf5f3"),
)[
  #text(size: 7.5pt, fill: rgb("#b33a2a"), weight: "bold", tracking: 0.08em)[MAIN READ]
  #v(0.3em)
  #text(size: 12.5pt, weight: "medium")[
    Phase 17 sharpens the Phase 15 discovery story. The early leader axis is not a pure top-return read; it behaves more like a
    prominent leader with strong volume support. The late dispersion axis is not best described by literal standard deviation;
    `pct_1h_mad` is the strongest single visible proxy, and the aggregate-family results say `mad` beats `std`. At the same time,
    the underlying market subspace is broader than one or two PCs: within the stored top `5` PCs, `market_mean` is roughly `~4D`
    while `market_eos` is somewhat more compressed.
  ]
]

#v(1.2em)

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [
    #summary-card(
      [LEADER SINGLE],
      [#mono("vol_1h_max", size: 12.5pt)],
      [CV R² = #fmt3(leader.at("best_single_feature").at("cv_r2"))],
    )
  ],
  [
    #summary-card(
      [LEADER BEST PAIR],
      [
        #mono("pct_1h_max", size: 11.5pt)
        #linebreak()
        #mono("+ vol_5m_max", size: 11.5pt)
      ],
      [quadratic CV R² = #fmt3(leader.at("best_pair_quadratic").at("cv_r2"))],
    )
  ],
  [
    #summary-card(
      [DISPERSION SINGLE],
      [#mono("pct_1h_mad", size: 12.5pt)],
      [CV R² = #fmt3(dispersion.at("best_single_feature").at("cv_r2"))],
    )
  ],
  [
    #summary-card(
      [DISPERSION BEST PAIR],
      [
        #mono("vol_5m_mean", size: 11.5pt)
        #linebreak()
        #mono("+ vol_1h_median", size: 11.5pt)
      ],
      [quadratic CV R² = #fmt3(dispersion.at("best_pair_quadratic").at("cv_r2"))],
    )
  ],
)


= What Phase 17 Does

This phase is a decomposition pass built on top of the existing Phase 15 discovery captures and residualized PCA basis.

The target axes are:

- `leader_axis`: `market_mean`, `L4`, `PC1`
- `dispersion_axis`: `market_mean`, `L35`, `PC1`

The candidate bank contains only:

- prompt-visible market metrics
- and prompt-derived aggregates computed from those metrics

Visible metric families:

- `pct_5m`
- `pct_1h`
- `net_flow_5m`
- `vol_5m`
- `vol_1h`
- `unique_traders_5m`
- `top20_holder_pct`

Derived aggregates per family:

- `mean`, `std`, `max`, `min`, `range`, `gap`, `mad`, `median`
- `top2_mean`, `max_minus_rest_mean`, `top1_minus_median`
- `leader_zscore`, `cv_abs`

Hidden synthetic labels are excluded. That means:

- no `attractiveness_*`
- no `risk_adjusted_*`
- no `edge_after_fee_*`


= Sanity Checks

The decomposition passes the two checks that matter here:

- the target PC scores are nearly uncorrelated with nuisance variables after residualization
- shuffled controls collapse near zero

#table(
  columns: (1.2fr, 1fr, 1fr, 1fr, 1.1fr, 1.1fr, 1.1fr),
  align: (left, center, center, center, center, center, center),
  table.hline(stroke: 1pt),
  table.header([*Axis*], [*seq_len*], [*user_chars*], [*n_rows*], [*shuffle single*], [*shuffle linear pair*], [*shuffle quadratic pair*]),
  table.hline(stroke: 0.5pt),
  [leader],
  [#fmt3(leader.at("target_nuisance_correlations").at(0).at("abs_spearman"))],
  [#fmt3(leader.at("target_nuisance_correlations").at(1).at("abs_spearman"))],
  [0.000],
  [#fmt3(leader.at("shuffle_sanity").at("best_single_feature").at("cv_r2"))],
  [#fmt3(leader.at("shuffle_sanity").at("best_pair_linear").at("cv_r2"))],
  [#fmt3(leader.at("shuffle_sanity").at("best_pair_quadratic").at("cv_r2"))],
  [dispersion],
  [#fmt3(dispersion.at("target_nuisance_correlations").at(0).at("abs_spearman"))],
  [#fmt3(dispersion.at("target_nuisance_correlations").at(1).at("abs_spearman"))],
  [0.000],
  [#fmt3(dispersion.at("shuffle_sanity").at("best_single_feature").at("cv_r2"))],
  [#fmt3(dispersion.at("shuffle_sanity").at("best_pair_linear").at("cv_r2"))],
  [#fmt3(dispersion.at("shuffle_sanity").at("best_pair_quadratic").at("cv_r2"))],
)

So the readout below is not being driven by prompt length, prompt size, or a probe that still works on shuffled targets.


= What The Two Axes Actually Track

#align(center)[#image("../../data/report_assets/synthetic_market_phase17_axis_decomposition/phase17_axis_panels.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Top single-feature fits, metric-family fits, and aggregate-type fits for the two target axes. The main point is not just
which feature wins, but what kind of information is consistently useful across the whole prompt-derived feature bank.
]

== Leader Axis

The strongest single visible proxy is:

- `vol_1h_max` with CV R² = #fmt3(leader.at("best_single_feature").at("cv_r2"))

But the family-level readout makes the interpretation clearer than the single-feature ranking:

- `vol_1h` family fit: #fmt3(leader.at("metric_family_group_ridge_cv_r2").at(0).at("cv_r2"))
- `pct_1h` family fit: #fmt3(leader.at("metric_family_group_ridge_cv_r2").at(1).at("cv_r2"))

Those two families dominate everything else. The best nonlinear pair then combines:

- `pct_1h_max`
- `vol_5m_max`

So the cleanest interpretation is:

- the leader axis is a #emph[prominent leader with strong throughput]
- not just the asset with the highest `1h` return

This is also stable across folds:

- the single-feature winner is `vol_1h_max` in `5/5` folds

== Dispersion Axis

The strongest single visible proxy is:

- `pct_1h_mad` with CV R² = #fmt3(dispersion.at("best_single_feature").at("cv_r2"))

That matters because the obvious candidate was `pct_1h_std`, but:

- `pct_1h_mad`: #fmt3(dispersion.at("best_single_feature").at("cv_r2"))
- `pct_1h_std`: `0.347`

The aggregate-type results make the same point in a stronger way:

- `mad` group fit: #fmt3(dispersion.at("aggregate_group_ridge_cv_r2").at(0).at("cv_r2"))
- `std` group fit: #fmt3(dispersion.at("aggregate_group_ridge_cv_r2").at(2).at("cv_r2"))

So the best read is:

- this axis tracks #emph[unevenness / spread / leader-versus-rest structure]
- `std` is useful, but not the best formula

At the same time, the family-level fits show this is still multivariate:

- `vol_1h`: #fmt3(dispersion.at("metric_family_group_ridge_cv_r2").at(0).at("cv_r2"))
- `pct_1h`: #fmt3(dispersion.at("metric_family_group_ridge_cv_r2").at(1).at("cv_r2"))
- `vol_5m`: #fmt3(dispersion.at("metric_family_group_ridge_cv_r2").at(2).at("cv_r2"))

So the axis is not “pure `1h` dispersion” either. It is a broader spread-like market summary with strong co-moving volume structure.


= Layerwise Subspace Summary

This phase also expands the Phase 15 PCA readout beyond two named PCs.

Important caveat:

- the stored Phase 15 basis only keeps the top `5` PCs per layer and state

So the subspace summary below is a summary of the stored top `5`, not of the full `2048`-dimensional layer.

#align(center)[#image("../../data/report_assets/synthetic_market_phase17_axis_decomposition/phase17_subspace_summary.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Within the stored top `5` PCs, `market_mean` stays fairly broad while `market_eos` is more compressed in early-to-mid layers.
Neither state collapses into a tiny one- or two-PC summary across all layers.
]

Across all layers:

- `market_mean`
  - top-5 cumulative variance mean: #fmt3(mm.at("top5_mean"))
  - top-5 cumulative variance max: #fmt3(mm.at("top5_max"))
  - participation ratio mean: #fmt3(mm.at("participation_mean"))
- `market_eos`
  - top-5 cumulative variance mean: #fmt3(me.at("top5_mean"))
  - top-5 cumulative variance max: #fmt3(me.at("top5_max"))
  - participation ratio mean: #fmt3(me.at("participation_mean"))

The right conclusion is:

- `market_mean` is a broad multi-dimensional market summary, roughly `~4D` in its stored top-PC readout
- `market_eos` is more compressed, especially in early-to-mid layers, but it still does not collapse into one or two PCs

To make that concrete, the tables below show cumulative variance within the stored top `5` PCs at a few key layers.

== `market_mean` Selected Layers

#table(
  columns: (0.8fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr, 1fr),
  align: (center, center, center, center, center, center, center),
  table.hline(stroke: 1pt),
  table.header([*Layer*], [*PC1*], [*PC1-2*], [*PC1-3*], [*PC1-4*], [*PC1-5*], [*PR*]),
  table.hline(stroke: 0.5pt),
  [L1], [#fmt3(cum(mm, 1, 1))], [#fmt3(cum(mm, 1, 2))], [#fmt3(cum(mm, 1, 3))], [#fmt3(cum(mm, 1, 4))], [#fmt3(cum(mm, 1, 5))], [#fmt3(pr(mm, 1))],
  [L4], [#fmt3(cum(mm, 4, 1))], [#fmt3(cum(mm, 4, 2))], [#fmt3(cum(mm, 4, 3))], [#fmt3(cum(mm, 4, 4))], [#fmt3(cum(mm, 4, 5))], [#fmt3(pr(mm, 4))],
  [L35], [#fmt3(cum(mm, 35, 1))], [#fmt3(cum(mm, 35, 2))], [#fmt3(cum(mm, 35, 3))], [#fmt3(cum(mm, 35, 4))], [#fmt3(cum(mm, 35, 5))], [#fmt3(pr(mm, 35))],
  [L40], [#fmt3(cum(mm, 40, 1))], [#fmt3(cum(mm, 40, 2))], [#fmt3(cum(mm, 40, 3))], [#fmt3(cum(mm, 40, 4))], [#fmt3(cum(mm, 40, 5))], [#fmt3(pr(mm, 40))],
  [L42], [#fmt3(cum(mm, 42, 1))], [#fmt3(cum(mm, 42, 2))], [#fmt3(cum(mm, 42, 3))], [#fmt3(cum(mm, 42, 4))], [#fmt3(cum(mm, 42, 5))], [#fmt3(pr(mm, 42))],
)

== `market_eos` Selected Layers

#table(
  columns: (0.8fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr, 1fr),
  align: (center, center, center, center, center, center, center),
  table.hline(stroke: 1pt),
  table.header([*Layer*], [*PC1*], [*PC1-2*], [*PC1-3*], [*PC1-4*], [*PC1-5*], [*PR*]),
  table.hline(stroke: 0.5pt),
  [L1], [#fmt3(cum(me, 1, 1))], [#fmt3(cum(me, 1, 2))], [#fmt3(cum(me, 1, 3))], [#fmt3(cum(me, 1, 4))], [#fmt3(cum(me, 1, 5))], [#fmt3(pr(me, 1))],
  [L4], [#fmt3(cum(me, 4, 1))], [#fmt3(cum(me, 4, 2))], [#fmt3(cum(me, 4, 3))], [#fmt3(cum(me, 4, 4))], [#fmt3(cum(me, 4, 5))], [#fmt3(pr(me, 4))],
  [L35], [#fmt3(cum(me, 35, 1))], [#fmt3(cum(me, 35, 2))], [#fmt3(cum(me, 35, 3))], [#fmt3(cum(me, 35, 4))], [#fmt3(cum(me, 35, 5))], [#fmt3(pr(me, 35))],
  [L40], [#fmt3(cum(me, 40, 1))], [#fmt3(cum(me, 40, 2))], [#fmt3(cum(me, 40, 3))], [#fmt3(cum(me, 40, 4))], [#fmt3(cum(me, 40, 5))], [#fmt3(pr(me, 40))],
  [L42], [#fmt3(cum(me, 42, 1))], [#fmt3(cum(me, 42, 2))], [#fmt3(cum(me, 42, 3))], [#fmt3(cum(me, 42, 4))], [#fmt3(cum(me, 42, 5))], [#fmt3(pr(me, 42))],
)


= Raw Prompt Appendix

The point of Phase 17 is that every candidate formula is computed from visible market rows like the ones below.

Examples:

- `pct_1h_mad` = mean absolute deviation of the visible `1h` changes across the roster
- `vol_1h_max` = maximum visible `1h` volume in the roster
- `pct_1h_gap` = top visible `1h` change minus second-best visible `1h` change

So the decomposition is asking what summary statistics of the written market rows best explain the discovered PCs.

#text(size: 8pt, fill: rgb("#888"))[
Verbatim user prompt from the Phase 15 discovery cohort.
]
#v(0.3em)
#prompt-block("raw_prompts/phase15_market_basis_discovery_scalar_user.txt")
