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

#let summary = json("../../data/report_assets/synthetic_market_phase16_17_combined/summary.json")
#let ctx = summary.at("context_order")
#let leader = summary.at("leader")
#let dispersion = summary.at("dispersion")
#let subspace = summary.at("subspace")

#let mono(text_value, size: 12pt, weight: "bold", fill: black) = {
  text(font: "Menlo", size: size, weight: weight, fill: fill)[#text_value]
}

// ── Title Block ─────────────────────────────────────────────────
#align(left)[
  #text(size: 9pt, fill: rgb("#b33a2a"), tracking: 0.08em, weight: "medium")[XENON INTERPRETABILITY]
  #v(0.3em)
  #text(size: 22pt, weight: "bold")[Market Geometry: Structure and Sensitivity]
  #v(0.15em)
  #text(size: 14pt, fill: rgb("#4a4a4a"), weight: "medium")[Combined Phase 16 + 17 Report]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    This report combines two complementary investigations built on the Phase 15 residualized discovery basis.
    Phase 17 decomposes the discovered market axes into prompt-visible features, answering #emph[what] the model
    tracks. Phase 16 tests how context placement warps those axes, answering #emph[how stable] the market encoding
    is under realistic prompt variation.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[24 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DISCOVERY PROMPTS]\ #text(size: 9pt)[`184` market-only]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[CONTEXT-ORDER PROMPTS]\ #text(size: 9pt)[`920` (A/B/C variants)]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[METHOD]\ #text(size: 9pt)[residualized PCA + feature decomposition + order test]],
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
    The model encodes two clean market summaries: a #emph[leader axis] that points toward prominent assets with
    strong throughput, and a #emph[dispersion axis] that points toward something conceptually like roster unevenness
    --- better approximated by MAD than standard deviation, though neither fully explains the axis.
    These encodings are stable when context follows the market, but context placed #emph[before] the market
    warps the market-boundary state by up to `0.070` cosine distance --- and that warp projects
    onto the discovered market axes. Downstream integration absorbs most of the difference, so the
    strongest order effect is on #emph[perception], not final output.
  ]
]

#v(1.2em)

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[LEADER AXIS]\ #mono(leader.at("best_single_feature"), size: 11pt) #text(size: 8pt, fill: rgb("#888"))[\ CV R² = #str(leader.at("best_single_cv_r2"))]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[DISPERSION AXIS]\ #mono(dispersion.at("best_single_feature"), size: 11pt) #text(size: 8pt, fill: rgb("#888"))[\ CV R² = #str(dispersion.at("best_single_cv_r2"))]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[RISK PERCEPTION WARP]\ #text(size: 16pt, weight: "bold")[#str(ctx.at("risk_gap"))] #text(size: 8pt, fill: rgb("#888"))[\ cosine gap at L#str(ctx.at("risk_best_layer"))]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[AFFORDANCE WARP]\ #text(size: 16pt, weight: "bold")[#str(ctx.at("aff_gap"))] #text(size: 8pt, fill: rgb("#888"))[\ cosine gap at L#str(ctx.at("aff_best_layer"))]],
)


= How This Report Fits Together

This report synthesizes two phases of work that share the same foundation: the Phase 15 residualized discovery basis.

#align(center)[#image("../../data/report_assets/synthetic_market_phase16_17_combined/combined_methodology_overview.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
The three-step pipeline: (1) discover clean market-linked directions via residualized PCA on 184 market-only prompts, (2) decompose the strongest axes into prompt-visible features, (3) test whether context placement warps those axes.
]

The three steps are:

+ *Discovery* (Phase 15 + rerun). Run PCA on residualized activations from `184` market-only prompts. Regress out nuisance variables (`seq_len`, `user_chars`, `n_rows`) before PCA so the top components track market content, not prompt shape. This produces a clean basis with two standout directions: an early leader axis (`market_mean`, `L4`, `PC1`) and a late dispersion axis (`market_mean`, `L35`, `PC1`).

+ *Decomposition* (Phase 17). Take those two discovered axes and ask what prompt-visible market statistics best predict the PC scores. The candidate bank contains only features computable from the rendered market rows --- no hidden synthetic labels.

+ *Context-order test* (Phase 16). Take the same `184` base markets and wrap them in `920` prompt variants where risk or affordance context appears either before or after the market block. Measure whether context placement changes the market encoding at two boundary states: `market_mean` (section average) and corrected `market_eos` (last meaningful market token).


= Part I: What The Market Axes Track

Phase 17 decomposes the two strongest discovered axes using only features a reader could compute from the visible market rows.

== Leader Axis: `market_mean`, `L4`, `PC1`

#align(center)[#image("../../data/report_assets/synthetic_market_phase16_17_combined/combined_axis_decomposition.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Top single-feature fits, metric-family fits, and aggregate-type fits for the leader (top row) and dispersion (bottom row) axes.
]

The strongest single visible proxy for the leader axis is:

- `vol_1h_max` with CV R² = #str(leader.at("best_single_cv_r2"))

But the family-level readout makes the interpretation richer. The `vol_1h` and `pct_1h` families dominate all others, and the best nonlinear pair combines `pct_1h_max` with `vol_5m_max` (quadratic CV R² = #str(leader.at("best_pair_cv_r2"))).

The best current interpretation: the leader axis points toward something like a #emph[prominent asset with strong throughput] --- not just the highest-return asset, but one that also has volume support. The exact nature of this axis is not fully resolved.

== Dispersion Axis: `market_mean`, `L35`, `PC1`

The strongest single visible proxy is:

- `pct_1h_mad` with CV R² = #str(dispersion.at("best_single_cv_r2"))

This is suggestive because the obvious candidate was `pct_1h_std`, which scores lower. The aggregate-type comparison points in the same direction: `mad` outperforms `std` as a proxy. But neither fully explains the axis --- the best single-feature R² is still moderate, so this is a partial decomposition, not a solved readout. Conceptually, this axis appears to track something like #emph[unevenness or leader-versus-rest structure] in the roster, though the precise formula the model uses remains open.

The best nonlinear pair is `vol_5m_mean` + `vol_1h_median` (quadratic CV R² = #str(dispersion.at("best_pair_cv_r2"))), which suggests the axis is multivariate: a broader spread-like market summary with co-moving volume structure, not reducible to a single statistic.


== Sanity Checks

Both decompositions pass the checks that matter:

- target PC scores are nearly uncorrelated with nuisance variables after residualization
- shuffled controls collapse near zero

#table(
  columns: (1.2fr, 1.8fr, 1.2fr, 1.2fr),
  align: (left, left, center, center),
  table.hline(stroke: 1pt),
  table.header([*Axis*], [*Best single feature*], [*Real CV R²*], [*Shuffled CV R²*]),
  table.hline(stroke: 0.5pt),
  [Leader], [`vol_1h_max`], [#str(leader.at("best_single_cv_r2"))], [< 0.01],
  [Dispersion], [`pct_1h_mad`], [#str(dispersion.at("best_single_cv_r2"))], [< 0.01],
)

So the decomposition results are not being driven by prompt length, prompt size, or a probe that still works on shuffled targets.


= Part II: How Context Placement Warps The Market Encoding

Phase 16 tests whether the discovered market axes are stable under realistic prompt variation. Each of the `184` base markets appears in five prompt variants:

#table(
  columns: (0.8fr, 1.6fr, 3.6fr),
  align: (left, left, left),
  table.hline(stroke: 1pt),
  table.header([*Label*], [*Prompt variant*], [*Meaning*]),
  table.hline(stroke: 0.5pt),
  [`A`], [`market_only`], [Baseline: market block first, later sections neutral.],
  [`B`], [`context_after_market`], [Risk or affordance context placed #emph[after] the market block.],
  [`C`], [`context_before_market`], [Same context moved #emph[before] the market block.],
)

The three comparisons are:

- *Sanity check*: `A vs B` should be identical at `market_mean` and `market_eos`, because later context cannot retroactively change the market encoding.
- *Perception test*: `A vs C` measures how much earlier context warps the market read.
- *Integration test*: `B vs C` at downstream states measures whether the warp persists to the final output.


== The Perception Warp

#align(center)[#image("../../data/report_assets/synthetic_market_phase16_17_combined/combined_perception_curves.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
`A vs B` stays flat because later context cannot change the market encoding. `A vs C` pulls away once context is moved before the market, with the strongest divergence at corrected `market_eos`.
]

The sanity check passes exactly: `A vs B` is identical at both `market_mean` and corrected `market_eos`. But `A vs C` diverges sharply once the same context is moved before the market.

#table(
  columns: (1.2fr, 1.1fr, 0.9fr, 0.9fr, 0.9fr),
  align: (left, center, center, center, center),
  table.hline(stroke: 1pt),
  table.header([*Group*], [*State / layer*], [*A vs B*], [*A vs C*], [*Gap*]),
  table.hline(stroke: 0.5pt),
  [Risk], [`market_eos @ L42`], [`1.000`], [`0.939`], [`0.061`],
  [Affordance], [`market_eos @ L40`], [`1.000`], [`0.930`], [`0.070`],
  [Risk], [`market_mean @ L39`], [`1.000`], [`0.996`], [`0.004`],
  [Affordance], [`market_mean @ L39`], [`1.000`], [`0.994`], [`0.006`],
)

The section-average market summary is fairly robust. The precise state at the end of the market block is context-sensitive. Moving risk or affordance information before the market changes where the model #emph[finishes] reading the market.


== Where The Warp Lives In The Discovered Basis

The Phase 15 discovery basis lets us ask not just #emph[how much] the warp is, but #emph[where] it points.

#align(center)[#image("../../data/report_assets/synthetic_market_phase16_17_combined/combined_basis_shift.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
The same corrected `market_eos` warp read in two Phase 15 bases. On the native `market_eos` basis (left), the shift is more local. On the `market_mean` basis (right), the same warp is re-expressed in broader roster-summary directions.
]

At the strongest corrected `market_eos` layers:

- *Risk at `L42`*: the `A -> C` shift lands on PCs tied to `pct_1h_std`, `pct_5m_max`, and `net_flow_5m_max`
- *Affordance at `L40`*: the `A -> C` shift lands on PCs tied to `pct_5m_max`, `pct_1h_std`, and `pct_5m_std`

When the same activations are projected onto the Phase 15 `market_mean` basis instead, the warp is still strong but the dominant axis labels change. For risk, the largest shift moves from `pct_1h_std` to `pct_1h_gap`. For affordance, it moves to `pct_1h_gap` with `pct_1h_max` also large.

The robust conclusions:

- `A -> C` is much larger than `A -> B` in both bases
- the warp is real and projects onto market-linked directions

The non-robust part:

- the exact #emph[name] of the dominant axis depends on which Phase 15 basis is used
- the basis labels are descriptive tools, not proof that the model uses one unique semantic coordinate system


== Downstream Integration

#align(center)[#image("../../data/report_assets/synthetic_market_phase16_17_combined/combined_integration_curves.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Once both variants have read the full prompt, `B vs C` is still not identical, but much closer than the `market_eos` perception split.
]

The downstream picture is forgiving:

- `last_token`, `active_settings_eos`, `portfolio_eos`, and `constraints_eos` all remain high-cosine (`0.98--1.00`)
- the order effect is still present but much smaller than the corrected `market_eos` split

That means context order clearly changes the market read when context is seen first, but once the model reads the whole prompt, a large part of that difference gets absorbed. The sharpest order effect is on #emph[perception], not final integration.


= Part III: Subspace Dimensionality

The discovered market subspace is broader than one or two PCs. Within the stored top `5` PCs from Phase 15:

#align(center)[#image("../../data/report_assets/synthetic_market_phase16_17_combined/combined_subspace_summary.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
`market_mean` stays broadly distributed across 5 PCs (participation ratio ~4) while `market_eos` is more compressed, especially in early-to-mid layers.
]

#table(
  columns: (1.4fr, 1fr, 1fr, 1fr),
  align: (left, center, center, center),
  table.hline(stroke: 1pt),
  table.header([*State*], [*Top-5 variance (mean)*], [*Top-5 variance (max)*], [*Participation ratio (mean)*]),
  table.hline(stroke: 0.5pt),
  [`market_mean`], [#str(subspace.at("market_mean").at("top5_mean"))], [#str(subspace.at("market_mean").at("top5_max"))], [#str(subspace.at("market_mean").at("participation_mean"))],
  [`market_eos`], [#str(subspace.at("market_eos").at("top5_mean"))], [#str(subspace.at("market_eos").at("top5_max"))], [#str(subspace.at("market_eos").at("participation_mean"))],
)

`market_mean` is a broad multi-dimensional market summary, roughly `~4D` in its stored top-PC readout. `market_eos` is more compressed but still does not collapse into one or two PCs.


= Synthesis

The two phases tell a coherent story about how this model reads markets:

+ *The model builds clean market summaries.* The discovery basis contains a leader axis (pointing toward prominent assets with volume support) and a dispersion axis (pointing toward something like roster unevenness, conceptually approximated by MAD-like measures but not fully explained by any single statistic). Both survive nuisance residualization and shuffled controls.

+ *The summary is stable when context comes after the market.* The `A vs B` sanity check is exact: later context does not retroactively change the market encoding at either `market_mean` or corrected `market_eos`.

+ *Context before the market warps the market-boundary state.* The `A vs C` perception gap reaches `0.061` (risk) and `0.070` (affordance) at corrected `market_eos`. The warp projects onto the discovered market axes, not into orthogonal noise directions.

+ *The warp mostly washes out downstream.* By the time the model finishes the full prompt, `B vs C` reconverges to `0.98--1.00` cosine. The strongest order effect is on market #emph[perception], not the final integrated state.

+ *The subspace is broader than two axes.* The market encoding uses at least `~4` effective dimensions at `market_mean`. The two named axes (leader, dispersion) are the most interpretable, not the only important directions.

The practical implication: if the goal is to understand or steer the model's market read, `market_mean` is the better target state (cleaner, broader, more robust to context variation). The `market_eos` boundary is more sensitive and more informative about order effects, but also more fragile.
