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
  #text(size: 22pt, weight: "bold")[Synthetic Market Phase 3 Coupled Geometry]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[Coupled-factor geometry results for the counting-manifolds-inspired market-state program. This report moves past the single-factor search and asks whether small interacting factor sets form cleaner low-dimensional structure than the scalar sweeps from Phases 1 and 2.]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[20 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[PROMPTS]\ #text(size: 9pt)[1,089 market-only]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[FAMILIES]\ #text(size: 9pt)[726 dense, 363 minimal]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[STAGE]\ #text(size: 9pt)[Coupled geometry]],
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
  #text(size: 12.5pt, weight: "medium")[Phase 3 supports a coupled-space view of market representation, but it does not replace the scalar story.]
  #v(0.4em)
  #text(size: 9.5pt, fill: rgb("#555"))[The strongest pair, momentum × flow, forms a reproducible late-layer geometry that survives within-template checks almost unchanged (dense 0.708, minimal 0.693). But it still does not beat the best isolated scalar baseline from Phase 2 (`pct_5m` at 0.741). The result favors a layered picture: strong single-factor candidates plus a smaller number of modestly ordered coupled spaces, not one dominant universal manifold.]
]

#v(1.2em)

// ── Summary Metrics ─────────────────────────────────────────────
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[PHASE 3 SCALE]\ #text(size: 16pt, weight: "bold")[1,089] #text(size: 8pt, fill: rgb("#888"))[\ market-only prompts]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST COUPLED GEOM.]\ #text(size: 16pt, weight: "bold")[0.708] #text(size: 8pt, fill: rgb("#888"))[\ momentum × flow, dense L28]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST REGRESSION R²]\ #text(size: 16pt, weight: "bold")[0.998] #text(size: 8pt, fill: rgb("#888"))[\ dense risk/edge, L4]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[PHASE 2 `pct_5m`]\ #text(size: 16pt, weight: "bold")[0.741] #text(size: 8pt, fill: rgb("#888"))[\ best scalar baseline]],
)


= Scope

Phase 3 is the first explicit Stage 2 experiment from the updated counting-manifolds plan: hold onto the scalar search, but stop expecting the whole market to collapse into one universal one-dimensional variable.

- *Stage 1 recap.* Phases 1 and 2 showed that some synthetic scalar variables, especially `pct_5m`, have meaningful low-dimensional order, but not a universal 1D manifold.
- *Stage 2 question.* Do naturally interacting factor pairs produce cleaner low-dimensional geometry than the isolated scalar sweeps?
- *Controlled synthetic setting.* The prompts remain market-only, with neutral assets `A/B/C/D`, clean by-construction labels, and the same pooled row-state analysis path as earlier phases.

The coupled families in this phase are:

- `pct_5m × unique_traders_5m` as a momentum × participation candidate.
- `pct_5m × top20_holder_pct` as a momentum × concentration candidate.
- `pct_5m × net_flow_5m` as a momentum × flow candidate.


= Dataset

Phase 3 contains 1,089 market-only prompts on the dedicated synthetic Modal volume. The dataset is split into 726 dense prompts and 363 minimal prompts so the report can compare broad coupled variation against a cleaner stripped-down version of the same pairwise structure.

#align(center)[#image("../../data/report_assets/synthetic_phase3/dataset_counts.png", width: 88%)]
#text(size: 8pt, fill: rgb("#888"))[Phase 3 allocates almost all of its scale to coupled-factor sweeps. The minimal slice removes more nuisance structure; the dense slice stress-tests whether the geometry survives richer market backgrounds.]


= Main Quantitative Findings

#set table(stroke: none)
#table(
  columns: (1.8fr, auto, auto, auto),
  align: (left, right, right, right),
  table.hline(stroke: 1pt),
  table.header(
    [*Metric*], [*Phase 2 scalar best*], [*Dense coupled*], [*Minimal coupled*],
  ),
  table.hline(stroke: 0.5pt),
  [`Best attractiveness R²`], [0.997], [0.998], [0.990],
  [`Best risk-adjusted R²`], [0.997], [0.998], [0.989],
  [`Best edge-after-fee R²`], [0.997], [0.998], [0.989],
  [`Best scalar / coupled geometry`], [0.741], [0.708], [0.693],
  [`Best pairwise AUROC`], [0.997], [1.000], [0.997],
  table.hline(stroke: 1pt),
)

#v(0.5em)

The probe story remains easy: the synthetic latent variables are still almost perfectly recoverable. The important change in this phase is the geometry read, not whether a linear probe can decode the labels.

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  [
    #align(center)[#image("../../data/report_assets/synthetic_phase3/regression_comparison.png", width: 100%)]
    #text(size: 8pt, fill: rgb("#888"))[Latent-score regression remains near ceiling in both coupled families, confirming that the synthetic labels are cleanly preserved in row states.]
  ],
  [
    #align(center)[#image("../../data/report_assets/synthetic_phase3/axis_fidelity.png", width: 100%)]
    #text(size: 8pt, fill: rgb("#888"))[Top-2 PC axis fidelity gives a cleaner read on whether the coupled spaces are behaving like coherent low-dimensional factor systems rather than arbitrary high-dimensional clusters.]
  ],
)


= Coupled Geometry Read

This is the central stage-transition result. Phase 3 asks whether the right object was never a universal scalar manifold, but a family of small coupled manifolds instead.

#align(center)[#image("../../data/report_assets/synthetic_phase3/coupled_geometry.png", width: 95%)]
#text(size: 8pt, fill: rgb("#888"))[Overall versus within-template coupled geometry. Within-template scores test whether the ordering survives after removing the easiest prompt-template differences and reading only the latent geometry within repeated backgrounds.]

#align(center)[#image("../../data/report_assets/synthetic_phase3/coupled_layerwise.png", width: 95%)]
#text(size: 8pt, fill: rgb("#888"))[Layerwise coupled ordering for the three candidate factor pairs. The strongest pair should sustain a broad late-middle layer band rather than a single fragile spike.]

#table(
  columns: (1.7fr, 1fr, auto, auto),
  align: (left, left, right, right),
  table.hline(stroke: 1pt),
  table.header(
    [*Coupled family*], [*Best state*], [*Dense best*], [*Minimal best*],
  ),
  table.hline(stroke: 0.5pt),
  [`Momentum × Flow`], [`row_mean` L28], [0.708], [0.693],
  [`Momentum × Participation`], [`row_mean` L28], [0.504], [0.488],
  [`Momentum × Concentration`], [`row_eos` L36], [0.393], [0.379],
  table.hline(stroke: 1pt),
)

#v(0.5em)

- `pct_5m × net_flow_5m` is the clear lead coupled pair and remains almost unchanged under within-template filtering in the dense slice (0.708 overall vs 0.704 within).
- `pct_5m × unique_traders_5m` is real but materially weaker, suggesting participation behaves more like a secondary modulator than a dominant joint manifold axis.
- `pct_5m × top20_holder_pct` remains weak in both dense and minimal settings, so concentration still does not behave like a clean coupled factor in this synthetic form.
- The strongest coupled layers cluster in the late 20s to mid 30s, which is later than the near-trivial latent decodability and consistent with geometry sharpening after early preservation.


= Interpretation

#block(
  width: 100%,
  inset: (left: 14pt, top: 10pt, bottom: 10pt, right: 10pt),
  stroke: (left: 3pt + rgb("#2e7d32"), top: none, right: none, bottom: none),
  fill: rgb("#e8f5e9"),
)[
  #text(size: 7.5pt, fill: rgb("#2e7d32"), weight: "bold", tracking: 0.08em)[SUPPORTED NOW]
  #v(0.2em)
  At least one coupled market space — momentum × flow — is represented with stable low-dimensional order that survives prompt-template controls rather than disappearing when the synthetic backgrounds vary.
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
  Coupled geometry is not universally cleaner than the strongest scalar candidate, and we still do not have a dominant joint market manifold that subsumes the Stage 1 scalar search.
]

#v(0.8em)

Best current interpretation:

- Stage 1 remains necessary: `pct_5m` is still the single cleanest variable we have.
- Stage 2 is now justified: momentum × flow is too stable across dense, minimal, and within-template reads to dismiss as a template artifact.
- Participation does shape the market representation jointly with momentum, but more weakly than flow.
- Concentration is still better treated as a difficult secondary factor than as a clean geometric partner.


= What Phase 3 Accomplished

Relative to the staged plan in `MARKET_COUNTING_MANIFOLDS_PLAN.md`, Phase 3 closes the first pass on Stage 2:

- it constructed and captured the first dedicated coupled-factor synthetic dataset
- it evaluated the same pooled row states with coupled-geometry metrics rather than scalar-only metrics
- it compared dense versus minimal prompt families rather than relying on one synthetic setting
- it established whether the coupled story is stronger than the single-factor story from Phase 2


= What Is Still Missing

Phase 3 is not yet the full market-manifold result.

- It still studies only three handpicked factor pairs rather than a broader interaction graph.
- It does not yet test settings or portfolio context as twisting operators on the coupled spaces.
- It does not yet include causal interventions on the strongest pairwise geometry candidate.
- It does not yet characterize the higher-dimensional joint market-state space that would correspond to Stage 3 of the plan.


= Next Moves

The next step should be targeted rather than broad:

+ deepen the strongest coupled pair with denser local sweeps and repeated backgrounds
+ run causal patching or ablation on the best coupled geometry candidate
+ only then reintroduce settings and portfolio context to test whether those later sections rotate or reweight the coupled space
+ expand to Stage 3 only after the strongest Stage 2 candidate is causally validated

If the strongest coupled candidate fails causal tests, that would be the first real challenge to the new staged hypothesis.

#v(2em)
#line(length: 100%, stroke: 0.5pt + rgb("#ccc"))
#v(0.3em)
#text(size: 8pt, fill: rgb("#999"))[Synthetic Market Phase 3 Coupled Geometry Report — 20 March 2026. 1,089 market-only captures on the dedicated synthetic Modal volume. Reference baseline: Phase 2 scalar geometry report.]
