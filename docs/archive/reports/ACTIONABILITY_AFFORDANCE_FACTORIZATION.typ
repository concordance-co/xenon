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

#align(left)[
  #text(size: 9pt, fill: rgb("#b33a2a"), tracking: 0.08em, weight: "medium")[XENON INTERPRETABILITY]
  #v(0.3em)
  #text(size: 22pt, weight: "bold")[Actionability Affordance Factorization]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    Cross-variant follow-up on `actionability_algebra_v3` and `v4`: the fused 4-way permission label is weaker than the primitive affordance bits.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[21 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[PHASES]\ #text(size: 9pt)[`v3` and `v4`]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[PROMPTS]\ #text(size: 9pt)[96 per phase]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[TEST GROUPS]\ #text(size: 9pt)[6 held-out groups]],
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
  #text(size: 12.5pt, weight: "medium")[`permission_mode` is probably the wrong probe target. The model appears to represent primitive affordance bits more cleanly than the fused 4-way mode.]
]

#v(1.2em)

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[V3 CAN BUY]\ #text(size: 16pt, weight: "bold")[0.875] #text(size: 8pt, fill: rgb("#888"))[\ `constraints_eos` L40]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[V3 OBSERVE VS ACT]\ #text(size: 16pt, weight: "bold")[0.833] #text(size: 8pt, fill: rgb("#888"))[\ `last_token` L40]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[V4 CAN SELL]\ #text(size: 16pt, weight: "bold")[0.750] #text(size: 8pt, fill: rgb("#888"))[\ `constraints_eos` L41]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[TOP-SYMBOL INVARIANCE]\ #text(size: 16pt, weight: "bold")[1.000] #text(size: 8pt, fill: rgb("#888"))[\ unchanged]],
)

= Why This Matters

The counting-manifolds paper explicitly warns that early probe attempts can fail when the target is a fused variable rather than the quantity the model actually represents cleanly.

That seems to be happening here.

- The fused 4-way `permission_mode` label is weaker under hardened prompt wording.
- But the primitive bits `can_buy`, `can_sell`, and `observe_vs_act` survive materially better on the same pooled activations in both `v3` and `v4`.

This is the first result in the actionability line that suggests a more precise mechanistic target rather than just a weaker or stronger version of the same story.


= Phase Progression

#align(center)[#image("assets/actionability_affordance/phase_progression.png", width: 95%)]
#text(size: 8pt, fill: rgb("#888"))[
  Fused downstream targets across `v1` through `v4`. `v2` looked strong, but prompt hardening in `v3` and `v4` degraded the fused permission-side reads sharply.
]


= `v4` Changes The Story

#align(center)[#image("assets/actionability_affordance/cross_variant_factorization.png", width: 94%)]
#text(size: 8pt, fill: rgb("#888"))[
  Across both hardened variants, decomposed affordance bits outperform the fused 4-way mode by a large margin.
]

#v(0.4em)

#table(
  columns: (1.3fr, 1.5fr, auto),
  align: (left, left, right),
  table.hline(stroke: 1pt),
  table.header([*Target*], [*Best State*], [*Balanced Accuracy*]),
  table.hline(stroke: 0.5pt),
  [`v3 permission_mode`], [`active_strategies_eos` L2], [0.625],
  [`v3 observe_vs_act`], [`last_token` L40], [0.833],
  [`v3 can_buy`], [`constraints_eos` L40], [0.875],
  [`v3 can_sell`], [`active_strategies_eos` L3], [0.792],
  [`v4 permission_mode`], [`portfolio_eos` L13], [0.458],
  [`v4 observe_vs_act`], [`active_strategies_eos` L2], [0.667],
  [`v4 can_buy`], [`portfolio_eos` L18], [0.792],
  [`v4 can_sell`], [`constraints_eos` L41], [0.750],
  table.hline(stroke: 1pt),
)


= Section Read

#align(center)[#image("assets/actionability_affordance/section_summary.png", width: 92%)]
#text(size: 8pt, fill: rgb("#888"))[
  `v4` still shows the primitive affordance bits landing in plausible different downstream sections. The exact best section moves between `v3` and `v4`, but the factorized affordance story itself remains.
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
  The actionability computation looks more factorized than the earlier fused-label framing suggested.
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
  A clean monolithic `permission_mode` representation is still not established.
]

#v(0.8em)

Best current interpretation:

- early market preference is still robust and easy
- downstream permission is not one crisp fused state
- instead, the model appears to carry primitive affordances separately
- that means the next probe family should target `can_buy`, `can_sell`, and `observe_vs_act`, not just `permission_mode`


= Next Step

The best next experiment is no longer another fused permission rerun. It is:

- carry the factorized labels into the next synthetic prompt-hardening pass
- test whether `can_buy` and `can_sell` survive additional paraphrase and row-format variation
- then bridge those binary affordance probes into real DX paired reruns

#v(2em)
#line(length: 100%, stroke: 0.5pt + rgb("#ccc"))
#v(0.3em)
#text(size: 8pt, fill: rgb("#999"))[
  Actionability Affordance Factorization — 21 March 2026. Synthetic `v3` and `v4` plus decomposed affordance probes on the same pooled residuals.
]
