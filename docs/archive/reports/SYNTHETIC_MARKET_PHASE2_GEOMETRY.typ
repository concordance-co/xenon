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
  #text(size: 22pt, weight: "bold")[Synthetic Market Phase 2 Geometry]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[Dense and minimal scalar sweeps for the counting-manifolds-inspired market-state program. This report compares Phase 2 against the earlier Phase 1 pilot and focuses on whether denser synthetic sweeps make the scalar geometry more manifold-like.]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[20 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[PROMPTS]\ #text(size: 9pt)[984 market-only]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[FAMILIES]\ #text(size: 9pt)[738 dense, 246 minimal]],
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
  #text(size: 12.5pt, weight: "medium")[Phase 2 improves the scalar-geometry story, but only partially.]
  #v(0.4em)
  #text(size: 9.5pt, fill: rgb("#555"))[The denser synthetic sweeps materially improved `pct_5m` ordering and noticeably improved concentration geometry in the minimal setting. But the market representation still does not collapse into a clean, universal one-dimensional manifold across all scalar families. The hypothesis survives; it is not yet closed.]
]

#v(1.2em)

// ── Summary Metrics ─────────────────────────────────────────────
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[PHASE 2 SCALE]\ #text(size: 16pt, weight: "bold")[984] #text(size: 8pt, fill: rgb("#888"))[\ market-only prompts]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST REGRESSION R²]\ #text(size: 16pt, weight: "bold")[0.997] #text(size: 8pt, fill: rgb("#888"))[\ dense, layer 2]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST SCALAR GEOM.]\ #text(size: 16pt, weight: "bold")[0.741] #text(size: 8pt, fill: rgb("#888"))[\ minimal `pct_5m`, layer 28]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[PHASE 1 `pct_5m`]\ #text(size: 16pt, weight: "bold")[0.637] #text(size: 8pt, fill: rgb("#888"))[\ baseline for comparison]],
)


= Scope

Phase 2 was designed to answer the specific objection left open by Phase 1: perhaps the pilot dataset was simply too coarse to reveal a cleaner scalar geometry. To test that, we expanded the synthetic market-only slice in two directions:

- *Dense sweep.* Repeat scalar values across varied background rosters to test whether the geometry survives more prompt diversity.
- *Minimal sweep.* Strip the prompt down so only the target scalar varies against a flatter background, testing whether a cleaner manifold appears when context clutter is removed.
- *Same pooled states.* Capture and analyze the same `row_mean_i` and `row_eos_i` states as before so the comparison is apples-to-apples.

The result is a stronger geometric read, not just a bigger dataset.


= Dataset

Phase 2 contains 984 market-only prompts on the dedicated synthetic Modal volume: 738 dense-sweep prompts and 246 minimal-sweep prompts. This should be read alongside the smaller 170-prompt Phase 1 pilot rather than in isolation.

#align(center)[#image("../../data/report_assets/synthetic_phase2/dataset_counts.png", width: 88%)]
#text(size: 8pt, fill: rgb("#888"))[Phase 2 is substantially larger than the Phase 1 pilot, with most of the new scale allocated to denser scalar sweeps rather than more context variants.]


= Main Quantitative Findings

#set table(stroke: none)
#table(
  columns: (1.6fr, auto, auto, auto),
  align: (left, right, right, right),
  table.hline(stroke: 1pt),
  table.header(
    [*Metric*], [*Phase 1*], [*Dense Sweep*], [*Minimal Sweep*],
  ),
  table.hline(stroke: 0.5pt),
  [`Best attractiveness R²`], [0.989], [0.997], [0.958],
  [`Best risk-adjusted R²`], [0.990], [0.997], [0.952],
  [`Best edge-after-fee R²`], [0.990], [0.997], [0.952],
  [`Best-asset AUROC`], [1.000], [0.992], [1.000],
  [`Best `pct_5m` geometry`], [0.637], [0.730], [0.741],
  [`Best `net_flow_5m` geometry`], [0.659], [0.565], [0.573],
  [`Best `top20_holder_pct` geometry`], [0.452], [0.518], [0.602],
  table.hline(stroke: 1pt),
)

#v(0.5em)

The broad pattern is clear:

- The latent variables remain very strongly linearly recoverable.
- `pct_5m` geometry improved substantially in both Phase 2 variants.
- Concentration geometry improved most in the minimal slice.
- `net_flow_5m` did *not* improve relative to Phase 1, suggesting that not every scalar family wants the same geometric treatment.

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  [
    #align(center)[#image("../../data/report_assets/synthetic_phase2/regression_comparison.png", width: 100%)]
    #text(size: 8pt, fill: rgb("#888"))[Latent-score regression remains near ceiling. Dense-sweep prompts slightly improve decodability; minimal prompts are a bit harsher but still very strong.]
  ],
  [
    #align(center)[#image("../../data/report_assets/synthetic_phase2/best_asset.png", width: 100%)]
    #text(size: 8pt, fill: rgb("#888"))[Best-asset probes remain near trivial across all phases, so the real action in Phase 2 is the scalar-geometry comparison, not the choice probes.]
  ],
)


= Geometry Read

The strongest Phase 2 evidence is not that *everything* got cleaner. It is that some scalar families got meaningfully cleaner under the right synthetic pressure.

#align(center)[#image("../../data/report_assets/synthetic_phase2/scalar_geometry_comparison.png", width: 95%)]
#text(size: 8pt, fill: rgb("#888"))[The largest Phase 2 win is `pct_5m`. Concentration also improves, especially in the minimal slice. Net-flow geometry remains more ambiguous and does not improve over Phase 1.]

#align(center)[#image("../../data/report_assets/synthetic_phase2/pct5m_layerwise.png", width: 95%)]
#text(size: 8pt, fill: rgb("#888"))[`pct_5m` becomes substantially more ordered in Phase 2 and peaks in the late 20s. This is the clearest sign that denser scalar sweeps are exposing a more stable geometric organization than the pilot did.]


= Interpretation

#block(
  width: 100%,
  inset: (left: 14pt, top: 10pt, bottom: 10pt, right: 10pt),
  stroke: (left: 3pt + rgb("#2e7d32"), top: none, right: none, bottom: none),
  fill: rgb("#e8f5e9"),
)[
  #text(size: 7.5pt, fill: rgb("#2e7d32"), weight: "bold", tracking: 0.08em)[SUPPORTED NOW]
  #v(0.2em)
  Denser synthetic sweeps do improve the scalar-geometry picture. The pilot’s weak geometry result was at least partly a data-resolution problem rather than a pure hypothesis failure.
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
  We still do not have a crisp counting-manifolds-style one-dimensional market scalar manifold that appears uniformly across variable families.
]

#v(0.8em)

Best current interpretation:

- `pct_5m` is the cleanest candidate for a counting-manifolds-style synthetic market variable.
- `top20_holder_pct` may also admit cleaner geometry, but appears more sensitive to prompt simplification than to dense varied backgrounds.
- `net_flow_5m` is either represented in a less purely scalar way or is more entangled with other latent structure than the current synthetic sweeps capture.
- The model is preserving the right variables very early, but the geometry differs by variable family. That is closer to a *family of manifold-like structures* than to one universal scalar manifold.


= Methodological Notes

The dense run emitted many sklearn `LinAlgWarning` messages during ridge regression. These were numerical-conditioning warnings, not capture or analysis failures:

- the job completed successfully and wrote the final JSON
- the warnings arose because the dense synthetic sweeps create highly collinear feature directions
- after this run, the ridge probe path was patched to use the more numerically stable `svd` solver for future runs

That means the current Phase 2 findings are usable, and future reruns should be quieter.


= What This Means For The Hypothesis

Phase 2 does *not* fundamentally challenge the hypothesis. It sharpens it.

The cleaner conclusion is:

- there is meaningful scalar organization in the market-row representation
- the clarity of that organization depends on *which scalar* we study and *how much nuisance structure* we leave in the prompt
- the next step should focus less on “one scalar manifold for everything” and more on identifying which market variables admit the cleanest geometric treatment


= Next Moves

The highest-value next moves are now more specific:

+ Treat `pct_5m` as the lead variable for the counting-manifolds analogue and deepen the synthetic sweep around it.
+ For `top20_holder_pct`, test whether its cleaner minimal-sweep behavior survives a modest reintroduction of background variation.
+ For `net_flow_5m`, construct more targeted synthetic families rather than assuming it should behave like momentum.
+ Start causal tests on the strongest variable-specific candidates before scaling context ladders again.

This is enough progress to justify a Phase 2 report. It is not yet enough to claim the broader market-manifold story is finished.

#v(2em)
#line(length: 100%, stroke: 0.5pt + rgb("#ccc"))
#v(0.3em)
#text(size: 8pt, fill: rgb("#999"))[Synthetic Market Phase 2 Geometry Report — 20 March 2026. 984 market-only captures on the dedicated synthetic Modal volume. Comparison baseline: Phase 1 pilot with 170 market-only captures.]
