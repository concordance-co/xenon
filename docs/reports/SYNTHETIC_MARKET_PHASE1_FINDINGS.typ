#set page(
  paper: "us-letter",
  margin: (x: 0.62in, y: 0.68in),
  footer: context [
    #align(center)[
      #text(size: 8pt, fill: rgb("#6A7A89"))[
        Synthetic Market Phase 1 Report - March 20, 2026
      ]
    ]
  ],
)

#set par(justify: false, leading: 0.58em)
#set text(font: "Libertinus Serif", size: 10pt, fill: rgb("#16202A"))

#let ink = rgb("#16202A")
#let muted = rgb("#5E6F82")
#let navy = rgb("#16324F")
#let teal = rgb("#2E6A69")
#let gold = rgb("#CA9440")
#let rose = rgb("#B56662")
#let cream = rgb("#F6EFE3")
#let mist = rgb("#EAF2F2")
#let line = rgb("#D6DEE3")
#let softline = rgb("#E7ECEF")
#let pillnavy = rgb("#35506A")

#show heading.where(level: 1): it => block(
  above: 1.15em,
  below: 0.35em,
  text(16pt, weight: "bold", fill: navy)[#it.body],
)

#show heading.where(level: 2): it => block(
  above: 0.85em,
  below: 0.28em,
  text(12pt, weight: "bold", fill: teal)[#it.body],
)

#show figure.caption: set text(size: 8.5pt, fill: muted)

#let pill(content, fill-color: rgb("#FFFFFF"), text-color: ink) = box(
  fill: fill-color,
  stroke: (paint: softline, thickness: 0.6pt),
  radius: 999pt,
  inset: (x: 8pt, y: 4pt),
)[
  #text(size: 8pt, fill: text-color)[#content]
]

#let stat(label, value, note, tone: white) = block(
  fill: tone,
  stroke: (paint: softline, thickness: 0.6pt),
  radius: 12pt,
  inset: 12pt,
  width: 100%,
)[
  #text(size: 8pt, fill: muted, weight: "bold")[#label]
  #v(5pt)
  #text(size: 20pt, fill: navy, weight: "bold")[#value]
  #v(4pt)
  #text(size: 8.8pt, fill: muted)[#note]
]

#let signal(title, body, tone: white) = block(
  fill: tone,
  stroke: (paint: softline, thickness: 0.6pt),
  radius: 12pt,
  inset: 12pt,
  width: 100%,
)[
  #text(size: 8pt, fill: muted, weight: "bold")[#title]
  #v(6pt)
  #text(size: 11pt, fill: ink)[#body]
]

#let step(letter, title, body) = block(
  fill: white,
  stroke: (paint: softline, thickness: 0.6pt),
  radius: 12pt,
  inset: 12pt,
  width: 100%,
)[
  #box(
    fill: navy,
    radius: 999pt,
    inset: (x: 7pt, y: 4pt),
  )[
    #text(size: 8pt, fill: white, weight: "bold")[#letter]
  ]
  #v(7pt)
  #text(size: 11pt, fill: navy, weight: "bold")[#title]
  #v(5pt)
  #text(size: 9pt, fill: muted)[#body]
]

#let quote(body) = block(
  fill: rgb("#FBF7EF"),
  stroke: (paint: softline, thickness: 0.6pt),
  radius: 12pt,
  inset: 14pt,
  width: 100%,
)[
  #text(size: 10.5pt, fill: ink)[#body]
]

#block(
  fill: navy,
  inset: 18pt,
  radius: 16pt,
  width: 100%,
)[
  #text(size: 21pt, weight: "bold", fill: white)[Synthetic Market Phase 1]
  #v(5pt)
  #text(size: 11pt, fill: luma(245))[
    First-pass manifold findings for the controlled market-only synthetic dataset.
  ]
  #v(10pt)
  #text(size: 9pt, fill: luma(235))[
    Research anchors: `docs/MARKET_COUNTING_MANIFOLDS_PLAN.md`, `docs/MARKET_MANIFOLD_RESEARCH_PLAN.md`, and `docs/MARKET_MANIFOLD_IMPLEMENTATION_PLAN.md`
  ]
  #v(10pt)
  #grid(
    columns: (auto, auto, auto, auto),
    gutter: 6pt,
    pill([March 20, 2026], fill-color: pillnavy, text-color: white),
    pill([170 market-only captures], fill-color: pillnavy, text-color: white),
    pill([48 layers], fill-color: pillnavy, text-color: white),
    pill([Dedicated synthetic volume], fill-color: pillnavy, text-color: white),
  )
]

#v(12pt)

This report closes the first synthetic phase motivated by the counting-manifolds paper. The goal was *not* to prove a rich market manifold in one step. The goal was to isolate clean latent variables, capture the synthetic prompts end to end, and ask whether those variables are linearly present in the market-row states before we scale into denser sweeps and context ladders.

= Executive Read

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 8pt,
  stat([Total prompts], [510], [170 market-only, plus low-risk and high-risk ladder variants.]),
  stat([Captured slice], [170], [All market-only prompts captured and pooled into 15 synthetic structure keys.]),
  stat([Best latent R²], [0.990], [`edge_after_fee_score` from `row_mean` at layer 10.], tone: mist),
  stat([Best scalar geometry], [0.659], [`net_flow_5m` distance/value Spearman from `row_mean` at layer 28.], tone: cream),
)

#v(8pt)

#quote[
  *Main read:* Phase 1 succeeded as a controlled variable-isolation and pipeline-validation phase. The synthetic row states preserve the clean latent variables extremely well. But the scalar sweeps do *not* yet look like a crisp counting-manifolds-style one-dimensional geometry. The manifold story is only partial at this stage.
]

= Scope

This phase intentionally strips the problem down:

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 8pt,
  step([01], [Neutral assets], [The prompts use synthetic assets `A/B/C/D` to remove token-identity priors from real DX data.]),
  step([02], [Clean latent variables], [Labels are constructed directly for `attractiveness`, `risk_adjusted`, `edge_after_fee`, and pairwise preference.]),
  step([03], [Market-only first], [Phase 1 focuses only on the market slice before settings or portfolio context are layered back in.]),
  step([04], [Full activation path], [Prompts were captured on a dedicated Modal volume, pooled into row and section states, and analyzed on held-out splits.]),
)

= Dataset

The full synthetic phase-1 dataset contains `510` prompts, but this report is about the `170` market-only captures that isolate the market representation before policy overlays.

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 8pt,
  signal([Families], [`scalar_sweep`, `pairwise_tradeoff`, and `archetype_family`.]),
  signal([Rows per prompt], [Four asset rows per prompt, yielding `680` unique market-only row states.]),
  signal([Pooling output], [Each captured prompt was reduced into `row_mean_i`, `row_eos_i`, section states, and `last_token`.]),
)

#figure(
  image("../../data/report_assets/synthetic_phase1/dataset_composition.png", width: 100%),
  caption: [Phase-1 prompt counts by family. The report slice uses the market-only prompts, while the full dataset already includes low-risk and high-risk context-ladder variants for the next phase.]
)

= Main Quantitative Findings

#table(
  columns: (1.35fr, 1.7fr, auto, 1.6fr),
  align: (left, left, center, left),
  inset: 6pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#DDEBF0") } else if calc.odd(y) { rgb("#F8FBFC") } else { white },

  [*Question*], [*Best state*], [*Metric*], [*Interpretation*],

  [`attractiveness_score`], [`row_mean` @ layer 10], [R² `0.989`], [The clean latent score is almost perfectly linearly recoverable from row states.],
  [`risk_adjusted_score`], [`row_mean` @ layer 10], [R² `0.990`], [Risk-adjusted valuation is also preserved without needing late context.],
  [`edge_after_fee_score`], [`row_mean` @ layer 10], [R² `0.990`], [The model retains a highly usable notion of fee-adjusted edge in the row representation.],
  [`is_best_asset`], [`row_mean` @ layer 1], [AUROC `1.000`], [Best-asset choice is trivial on this controlled slice; the phase is easy by design.],
  [`A beats B on attractiveness`], [`diff:row_mean` @ layer 0], [AUROC `1.000`], [Pairwise preference is immediately linearly separable from row-difference vectors.],
  [`A beats B on risk-adjusted score`], [`diff:row_mean` @ layer 3], [AUROC `1.000`], [Risk-adjusted pairwise preference is also perfectly separable very early.],
)

This is the most important split in the report:

- probe results are *extremely* strong
- scalar geometry results are only *moderately* strong

That means "the model preserves the variables" is already clear, but "the model organizes them as a simple low-dimensional manifold" is not yet clear.

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,

  figure(
    image("../../data/report_assets/synthetic_phase1/latent_regression.png", width: 100%),
    caption: [Held-out regression probes across layers. `row_mean` dominates and peaks around layer 10 for all three latent scores.]
  ),
  figure(
    image("../../data/report_assets/synthetic_phase1/preference_probe_curves.png", width: 100%),
    caption: [Choice-oriented probes are nearly perfect on this controlled slice. That is useful for pipeline validation, but it also shows that Phase 1 is an easy synthetic setting.]
  ),
)

= Geometry Read

The scalar sweep analysis is where the manifold question starts to become nontrivial.

#table(
  columns: (1.3fr, 1.5fr, auto, auto, auto),
  align: (left, left, center, center, center),
  inset: 6pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#DDEBF0") } else if calc.odd(y) { rgb("#F8FBFC") } else { white },

  [*Scalar family*], [*Best state*], [*Layer*], [*Distance/value Spearman*], [*Participation ratio*],

  [`pct_5m`], [`row_mean`], [26], [0.637], [3.09],
  [`net_flow_5m`], [`row_mean`], [28], [0.659], [3.70],
  [`top20_holder_pct`], [`row_mean`], [26], [0.452], [3.86],
)

Interpretation:

- the scalar sweeps are *ordered enough* that activation distances track scalar differences
- but the best correlations are well below `1.0`
- participation ratios around `3-4` are higher than the clean intrinsic-dimension-1 story we would hope for in a counting-manifolds-style result
- `pct_5m` and `net_flow_5m` look more coherent than `top20_holder_pct`

#figure(
  image("../../data/report_assets/synthetic_phase1/scalar_geometry.png", width: 100%),
  caption: [Scalar sweep geometry is only partially organized. `pct_5m` and `net_flow_5m` show moderate ordering in activation space, while `top20_holder_pct` is weaker and the overall structure is still substantially higher-dimensional than a clean 1D curve.]
)

= Interpretation

#grid(
  columns: (1fr, 1fr),
  gutter: 8pt,
  signal([Supported now], [The synthetic prompts successfully isolate clean latent variables, and those variables are strongly present in the row states.]),
  signal([Not supported yet], [We do not yet have a compelling one-dimensional market-counting manifold analogous to the linebreak paper.], tone: mist),
)

#v(8pt)

Best current interpretation:

- Phase 1 proves that the capture stack, pooled synthetic structure path, and analysis stack are all working end to end
- the model linearly preserves `attractiveness`, `risk_adjusted`, and `edge_after_fee` with very high fidelity
- pairwise preference is easy enough in this slice that the relevant differences are already perfectly separable
- the geometry result is the limiting one: there is *some* scalar ordering, but not yet the crisp low-dimensional scalar manifold we would want for a stronger counting-manifolds analogue

#quote[
  *Best current interpretation:* Phase 1 is a successful controlled isolation phase, but not yet a decisive manifold-discovery phase. The next gains are more likely to come from denser scalar sweeps, repeated backgrounds, and cleaner family-specific datasets than from more probing on the current coarse grid.
]

= What Phase 1 Accomplished

Relative to the counting-manifolds-inspired plan, Phase 1 completed the necessary groundwork:

- synthetic prompt generator with neutral assets and clean labels
- Neon tables and views for synthetic examples, asset rows, and pairwise rows
- dedicated Modal volume for synthetic captures, separate from the crowded `xenon-data` volume
- smoke-tested and full market-only capture run
- synthetic structure pooling over the full captured slice
- first-pass manifold analysis with held-out probes and scalar geometry metrics

= What Is Still Missing

Phase 1 does *not* yet close the main research question.

- the scalar sweeps are still too coarse: `21` values per family with effectively one market-only prompt per value
- backgrounds are too fixed to cleanly distinguish invariant scalar geometry from prompt-template artifacts
- the current phase is market-only; it does not yet show how settings or portfolio context twist the synthetic manifold
- the current phase does not include causal interventions or row-level patching

= Next Phase

The most defensible next move is to expand the synthetic market-only geometry phase before writing strong manifold claims:

1. increase scalar density substantially and repeat each scalar value across multiple background rosters
2. add a stripped-down ultra-minimal scalar sweep where only one asset row changes and everything else is flat
3. rerun the same geometry analysis and look for stronger monotonicity, lower participation ratio, and more stable layer-local organization
4. only then add the context ladder back in to see whether settings twist or reweight the synthetic market geometry

If those denser sweeps still fail to produce cleaner scalar geometry, that would be the first real challenge to the current hypothesis.
