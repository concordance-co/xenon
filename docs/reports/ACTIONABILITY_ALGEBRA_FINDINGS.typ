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
  #text(size: 22pt, weight: "bold")[Actionability Algebra Findings]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    Synthetic policy results across `actionability_algebra_v1`, `v2`, and `v3`.
    Focus: whether the model keeps market preference early and computes permission or actionability later in a wording-robust way.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[21 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[PHASES]\ #text(size: 9pt)[3 synthetic runs]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[PROMPTS]\ #text(size: 9pt)[96 per phase]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[HELD-OUT GROUPS]\ #text(size: 9pt)[24 scenario groups]],
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
  #text(size: 12.5pt, weight: "medium")[The robust result is early market preference. The clean late permission result is not yet robust.]
  #v(0.4em)
  #text(size: 9.5pt, fill: rgb("#555"))[
    `v2` supported an "early preference, late permission" story after removing an explicit mode leak.
    But `v3` kept the latent rules fixed while paraphrasing and reordering section text, and the previously perfect late permission signal collapsed.
    The surviving claim is narrower: market-best asset is stable and early; downstream permission decoding is still fragile.
  ]
]

#v(1.2em)

// ── Summary Metrics ─────────────────────────────────────────────
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[MARKET-BEST ASSET]\ #text(size: 16pt, weight: "bold")[1.000] #text(size: 8pt, fill: rgb("#888"))[\ AUROC, all phases]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[V2 PERMISSION MODE]\ #text(size: 16pt, weight: "bold")[1.000] #text(size: 8pt, fill: rgb("#888"))[\ `constraints_eos` L27]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[V3 PERMISSION MODE]\ #text(size: 16pt, weight: "bold")[0.625] #text(size: 8pt, fill: rgb("#888"))[\ `active_strategies_eos` L2]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[TOP-SYMBOL INVARIANCE]\ #text(size: 16pt, weight: "bold")[1.000] #text(size: 8pt, fill: rgb("#888"))[\ across permission groups]],
)


= Scope

This report compares three synthetic policy phases:

- `v1`: compositional permission dataset with a prompt-header shortcut that leaked the exact `permission_mode`.
- `v2`: removed the shortcut, forcing the model to rely on the actual section content.
- `v3`: kept the same latent permission rules as `v2`, but paraphrased portfolio and constraint language and shuffled bullet order inside the sections.

The point of the sequence is not to maximize an easy classifier. The point is to see which claims survive progressively less convenient prompt wording.


= Experimental Design

#align(center)[#image("assets/actionability_algebra/phase_design_matrix.png", width: 95%)]
#text(size: 8pt, fill: rgb("#888"))[
  Phase design summary. `v1` is useful only as a leakage baseline. `v2` is the first clean read. `v3` is the first robustness check against wording and local ordering changes.
]


= Main Quantitative Findings

#set table(stroke: none)
#table(
  columns: (1.1fr, 1.8fr, 1.6fr, 1.2fr),
  align: (left, left, left, right),
  table.hline(stroke: 1pt),
  table.header(
    [*Question*], [*v1 Best*], [*v2 Best*], [*v3 Best*],
  ),
  table.hline(stroke: 0.5pt),
  [`market_best_asset`], [`row_mean` L0, AUROC 1.000], [`row_mean` L0, AUROC 1.000], [`row_mean` L0, AUROC 1.000],
  [`expected_action_type`], [`active_settings_eos` L0, bal acc 1.000], [`constraints_eos` L25, bal acc 1.000], [`active_strategies_eos` L3, bal acc 0.694],
  [`permission_mode`], [`active_settings_eos` L0, bal acc 1.000], [`constraints_eos` L27, bal acc 1.000], [`active_strategies_eos` L2, bal acc 0.625],
  [`policy_best_asset`], [`portfolio_eos` L5, bal acc 0.867], [`portfolio_eos` L4, bal acc 0.700], [`active_strategies_eos` L12, bal acc 0.383],
  [`permission_top_symbol_invariance`], [1.000], [1.000], [1.000],
  table.hline(stroke: 1pt),
)

#v(0.5em)

The critical comparison is `v2 -> v3`.
If the latent permission signal were strongly section-semantic and wording-robust, modest paraphrase and bullet shuffling should have left the downstream read mostly intact.
It did not.

#align(center)[#image("assets/actionability_algebra/metric_shift.png", width: 95%)]
#text(size: 8pt, fill: rgb("#888"))[
  Summary metric shift across phases. Early market preference stays pinned at ceiling. The permission-side classifiers move from perfect in `v2` to moderate or weak in `v3`.
]


= Where The Signal Lives

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  [
    #align(center)[#image("assets/actionability_algebra/section_heatmaps.png", width: 100%)]
    #text(size: 8pt, fill: rgb("#888"))[
      Best-layer heatmaps by section. `v2` concentrates the permission result late in `constraints_eos`; `v3` no longer does.
    ]
  ],
  [
    #align(center)[#image("assets/actionability_algebra/layerwise_comparison.png", width: 100%)]
    #text(size: 8pt, fill: rgb("#888"))[
      Layerwise comparison for the permission-side tasks. The `v2` curve is sharp and late. The `v3` curve is weaker and flatter.
    ]
  ],
)

#v(0.4em)

#align(center)[#image("assets/actionability_algebra/invariance.png", width: 76%)]
#text(size: 8pt, fill: rgb("#888"))[
  Invariance stayed perfect across all three phases. The preferred top symbol under permission compositions did not move even when the permission-side classifiers weakened.
]


= Interpretation

#block(
  width: 100%,
  inset: (left: 14pt, top: 10pt, bottom: 10pt, right: 10pt),
  stroke: (left: 3pt + rgb("#2e7d32"), top: none, right: none, bottom: none),
  fill: rgb("#e8f5e9"),
)[
  #text(size: 7.5pt, fill: rgb("#2e7d32"), weight: "bold", tracking: 0.08em)[SUPPORTED NOW]
  #v(0.2em)
  The model's market-best asset signal is early, clean, and robust across all three prompt variants.
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
  A crisp, wording-robust late permission circuit is not established by the current synthetic sequence.
]

#v(0.8em)

Best current read:

- `v1` should be treated as a leakage baseline, not an interpretable result.
- `v2` remains meaningful because removing the header leak moved actionability from a trivial early shortcut to a late section-local read.
- `v3` is the decisive robustness check. It shows that the `v2` permission result depended materially on surface presentation.
- The stable claim that survives every phase is narrower and cleaner: the model forms a market preference very early, while permission-related decoding is weaker and more presentation-sensitive.


= What This Means For Direction

This result is useful because it trims the research tree:

- It is worth keeping the "early preference" line as a main hypothesis.
- It is not worth over-claiming a clean downstream permission representation from the current synthetic setup.
- The right next synthetic move is not more of `v2`. It is a harder `v4` with paraphrases, distractor numerics, and less lexical shortcutting.
- The right real-data bridge is also narrower: validate early preference first, then treat permission as a noisy downstream gating signal.


= Immediate Next Step

`actionability_algebra_v4` should do four things simultaneously:

- keep the `v3` paraphrase and bullet-order variation
- add distractor numeric lines inside portfolio and constraint sections
- enlarge the scenario-group set so each held-out split is less templated
- vary local wording around permission cues without changing the underlying rules

If a downstream permission read survives that stress test, it becomes a serious result.
If it fails again, the better research target is likely not "permission manifold" but "preference vs permission algebra" with noisier, more distributed downstream evidence.

#v(2em)
#line(length: 100%, stroke: 0.5pt + rgb("#ccc"))
#v(0.3em)
#text(size: 8pt, fill: rgb("#999"))[
  Actionability Algebra Findings — 21 March 2026. Three synthetic policy phases, 96 prompts per phase, 48 layers, held-out scenario-group evaluation.
]
