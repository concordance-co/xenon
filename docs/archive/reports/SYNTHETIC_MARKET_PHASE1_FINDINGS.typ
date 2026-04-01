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
  #text(size: 22pt, weight: "bold")[Synthetic Market Phase 1]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[First-pass manifold findings for the controlled market-only synthetic dataset. Research anchors: `MARKET_COUNTING_MANIFOLDS_PLAN.md`, `MARKET_MANIFOLD_RESEARCH_PLAN.md`, and `MARKET_MANIFOLD_IMPLEMENTATION_PLAN.md`.]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[20 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[PROMPTS]\ #text(size: 9pt)[510 total, 170 market-only]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[LAYERS]\ #text(size: 9pt)[48]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[VOLUME]\ #text(size: 9pt)[Dedicated synthetic]],
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
  #text(size: 12.5pt, weight: "medium")[Phase 1 succeeded as controlled variable isolation — but not yet as manifold discovery.]
  #v(0.4em)
  #text(size: 9.5pt, fill: rgb("#555"))[The synthetic row states preserve clean latent variables extremely well (R² up to 0.990). But the scalar sweeps do _not_ yet look like a crisp counting-manifolds-style one-dimensional geometry. The manifold story is only partial at this stage.]
]

#v(1.2em)

// ── Summary Metrics ─────────────────────────────────────────────
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[TOTAL PROMPTS]\ #text(size: 16pt, weight: "bold")[510] #text(size: 8pt, fill: rgb("#888"))[\ 170 market-only]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[CAPTURED SLICE]\ #text(size: 16pt, weight: "bold")[170] #text(size: 8pt, fill: rgb("#888"))[\ 15 structure keys]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST LATENT R²]\ #text(size: 16pt, weight: "bold")[0.990] #text(size: 8pt, fill: rgb("#888"))[\ edge_after_fee L10]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST SCALAR GEOM.]\ #text(size: 16pt, weight: "bold")[0.659] #text(size: 8pt, fill: rgb("#888"))[\ net_flow Spearman L28]],
)


// ── Scope ───────────────────────────────────────────────────────
= Scope

This phase intentionally strips the problem down. The goal was _not_ to prove a rich market manifold in one step. The goal was to isolate clean latent variables, capture the synthetic prompts end to end, and ask whether those variables are linearly present in the market-row states before scaling into denser sweeps and context ladders.

- *Neutral assets.* The prompts use synthetic assets `A/B/C/D` to remove token-identity priors from real DX data.
- *Clean latent variables.* Labels are constructed directly for `attractiveness`, `risk_adjusted`, `edge_after_fee`, and pairwise preference.
- *Market-only first.* Phase 1 focuses only on the market slice before settings or portfolio context are layered back in.
- *Full activation path.* Prompts were captured on a dedicated Modal volume, pooled into row and section states, and analyzed on held-out splits.


// ── Dataset ─────────────────────────────────────────────────────
= Dataset

The full synthetic phase-1 dataset contains 510 prompts, but this report covers the 170 market-only captures that isolate the market representation before policy overlays. Prompt families: `scalar_sweep`, `pairwise_tradeoff`, and `archetype_family`. Four asset rows per prompt, yielding 680 unique market-only row states. Each captured prompt was reduced into `row_mean_i`, `row_eos_i`, section states, and `last_token`.

#align(center)[#image("../../data/report_assets/synthetic_phase1/dataset_composition.png", width: 95%)]
#text(size: 8pt, fill: rgb("#888"))[Phase-1 prompt counts by family. The report slice uses market-only prompts; the full dataset already includes low-risk and high-risk context-ladder variants for the next phase.]


// ── Main Quantitative Findings ──────────────────────────────────
#pagebreak()

= Main Quantitative Findings

#set table(stroke: none)
#table(
  columns: (1.3fr, 1.5fr, auto, 1.6fr),
  align: (left, left, right, left),
  table.hline(stroke: 1pt),
  table.header(
    [*Question*], [*Best State*], [*Metric*], [*Interpretation*],
  ),
  table.hline(stroke: 0.5pt),
  [`attractiveness_score`], [`row_mean` L10], [R² 0.989], [Clean latent score almost perfectly recoverable from row states.],
  [`risk_adjusted_score`], [`row_mean` L10], [R² 0.990], [Risk-adjusted valuation preserved without needing late context.],
  [`edge_after_fee_score`], [`row_mean` L10], [R² 0.990], [The model retains a highly usable notion of fee-adjusted edge.],
  [`is_best_asset`], [`row_mean` L1], [AUROC 1.000], [Best-asset choice is trivial on this controlled slice.],
  [`A beats B attractiveness`], [`diff:row_mean` L0], [AUROC 1.000], [Pairwise preference immediately separable from row-difference vectors.],
  [`A beats B risk-adjusted`], [`diff:row_mean` L3], [AUROC 1.000], [Risk-adjusted pairwise preference also perfectly separable.],
  table.hline(stroke: 1pt),
)

#v(0.5em)

This is the most important split in the report: probe results are _extremely_ strong, but scalar geometry results are only _moderately_ strong. The model preserves the variables, but whether it organizes them as a simple low-dimensional manifold is not yet clear.

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  [
    #align(center)[#image("../../data/report_assets/synthetic_phase1/latent_regression.png", width: 100%)]
    #text(size: 8pt, fill: rgb("#888"))[Held-out regression probes across layers. `row_mean` dominates and peaks around layer 10 for all three latent scores.]
  ],
  [
    #align(center)[#image("../../data/report_assets/synthetic_phase1/preference_probe_curves.png", width: 100%)]
    #text(size: 8pt, fill: rgb("#888"))[Choice-oriented probes are nearly perfect on this controlled slice — useful for pipeline validation, but confirms Phase 1 is an easy synthetic setting.]
  ],
)


// ── Geometry Read ───────────────────────────────────────────────
= Geometry Read

The scalar sweep analysis is where the manifold question starts to become nontrivial.

#table(
  columns: (1.3fr, 1.2fr, auto, auto, auto),
  align: (left, left, right, right, right),
  table.hline(stroke: 1pt),
  table.header(
    [*Scalar Family*], [*Best State*], [*Layer*], [*Dist/Val Spearman*], [*Participation Ratio*],
  ),
  table.hline(stroke: 0.5pt),
  [`pct_5m`], [`row_mean`], [26], [0.637], [3.09],
  [`net_flow_5m`], [`row_mean`], [28], [0.659], [3.70],
  [`top20_holder_pct`], [`row_mean`], [26], [0.452], [3.86],
  table.hline(stroke: 1pt),
)

#v(0.3em)

- The scalar sweeps are _ordered enough_ that activation distances track scalar differences.
- But the best correlations are well below 1.0.
- Participation ratios around 3--4 are higher than the clean intrinsic-dimension-1 story needed for a counting-manifolds-style result.
- `pct_5m` and `net_flow_5m` look more coherent than `top20_holder_pct`.

#align(center)[#image("../../data/report_assets/synthetic_phase1/scalar_geometry.png", width: 95%)]
#text(size: 8pt, fill: rgb("#888"))[Scalar sweep geometry is only partially organized. `pct_5m` and `net_flow_5m` show moderate ordering; `top20_holder_pct` is weaker. The overall structure is substantially higher-dimensional than a clean 1D curve.]


// ── Interpretation ──────────────────────────────────────────────
#pagebreak()

= Interpretation

#block(
  width: 100%,
  inset: (left: 14pt, top: 10pt, bottom: 10pt, right: 10pt),
  stroke: (left: 3pt + rgb("#2e7d32"), top: none, right: none, bottom: none),
  fill: rgb("#e8f5e9"),
)[
  #text(size: 7.5pt, fill: rgb("#2e7d32"), weight: "bold", tracking: 0.08em)[SUPPORTED NOW]
  #v(0.2em)
  The synthetic prompts successfully isolate clean latent variables, and those variables are strongly present in the row states.
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
  We do not yet have a compelling one-dimensional market-counting manifold analogous to the linebreak paper.
]

#v(0.8em)

Best current interpretation:

- Phase 1 proves that the capture stack, pooled synthetic structure path, and analysis stack are all working end to end.
- The model linearly preserves `attractiveness`, `risk_adjusted`, and `edge_after_fee` with very high fidelity.
- Pairwise preference is easy enough in this slice that the relevant differences are already perfectly separable.
- The geometry result is the limiting one: there is _some_ scalar ordering, but not yet the crisp low-dimensional scalar manifold we would want for a stronger counting-manifolds analogue.


= What Phase 1 Accomplished

Relative to the counting-manifolds-inspired plan, Phase 1 completed the necessary groundwork:

- Synthetic prompt generator with neutral assets and clean labels.
- Neon tables and views for synthetic examples, asset rows, and pairwise rows.
- Dedicated Modal volume for synthetic captures, separate from the crowded `xenon-data` volume.
- Smoke-tested and full market-only capture run.
- Synthetic structure pooling over the full captured slice.
- First-pass manifold analysis with held-out probes and scalar geometry metrics.


= What Is Still Missing

Phase 1 does _not_ yet close the main research question.

- The scalar sweeps are still too coarse: 21 values per family with effectively one market-only prompt per value.
- Backgrounds are too fixed to cleanly distinguish invariant scalar geometry from prompt-template artifacts.
- The current phase is market-only; it does not yet show how settings or portfolio context twist the synthetic manifold.
- The current phase does not include causal interventions or row-level patching.


= Next Phase

The most defensible next move is to expand the synthetic market-only geometry phase before writing strong manifold claims:

+ Increase scalar density substantially and repeat each scalar value across multiple background rosters.
+ Add a stripped-down ultra-minimal scalar sweep where only one asset row changes and everything else is flat.
+ Rerun the same geometry analysis and look for stronger monotonicity, lower participation ratio, and more stable layer-local organization.
+ Only then add the context ladder back in to see whether settings twist or reweight the synthetic market geometry.

If those denser sweeps still fail to produce cleaner scalar geometry, that would be the first real challenge to the current hypothesis.

#v(2em)
#line(length: 100%, stroke: 0.5pt + rgb("#ccc"))
#v(0.3em)
#text(size: 8pt, fill: rgb("#999"))[Synthetic Market Phase 1 Report — 20 March 2026. 170 market-only captures from 510 total prompts. Qwen3-30B-A3B surrogate, dedicated synthetic Modal volume, 48 layers.]
