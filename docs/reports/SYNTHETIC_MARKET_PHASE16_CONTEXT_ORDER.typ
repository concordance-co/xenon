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

// ── Title Block ─────────────────────────────────────────────────
#align(left)[
  #text(size: 9pt, fill: rgb("#b33a2a"), tracking: 0.08em, weight: "medium")[XENON INTERPRETABILITY]
  #v(0.3em)
  #text(size: 22pt, weight: "bold")[Synthetic Market Phase 16]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    Context-order comparison on the new DX-like synthetic prompt surface. This phase fixes the `market_eos` boundary,
    then asks three concrete questions: does later context leave the market encoding untouched, does context-before-market
    warp the market encoding itself, and do the two orderings converge again by the end of the prompt?
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[23 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[PROMPTS]\ #text(size: 9pt)[`920` synthetic prompts]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[MATCHED MARKETS]\ #text(size: 9pt)[`184` A/B/C markets]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[METHOD]\ #text(size: 9pt)[corrected boundary + A/B/C order test]],
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
    The corrected `market_eos` boundary makes Phase 2 clean. The sanity check passes exactly: `A vs B` is identical at both
    `market_mean` and corrected `market_eos`, because later context cannot change the market state once the market block is over.
    But `A vs C` diverges sharply once the same context is moved #emph[before] the market, especially at corrected `market_eos`
    around layers `40–42`. Downstream, `B vs C` mostly reconverges, which means the strongest order effect is on #emph[market perception],
    not on the final integrated state.
  ]
]

#v(1.2em)

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[RISK `market_eos`]\ #text(size: 16pt, weight: "bold")[`1.000 vs 0.939`] #text(size: 8pt, fill: rgb("#888"))[\ `A/B` vs `A/C` at `L42`]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[AFFORDANCE `market_eos`]\ #text(size: 16pt, weight: "bold")[`1.000 vs 0.930`] #text(size: 8pt, fill: rgb("#888"))[\ `A/B` vs `A/C` at `L40`]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[RISK `market_mean` GAP]\ #text(size: 16pt, weight: "bold")[`0.004`] #text(size: 8pt, fill: rgb("#888"))[\ best perception gap at `L39`]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[AFFORDANCE `market_mean` GAP]\ #text(size: 16pt, weight: "bold")[`0.006`] #text(size: 8pt, fill: rgb("#888"))[\ best perception gap at `L39`]],
)


= What Changed Before Phase 2

Two things changed before running this phase.

First, the synthetic prompt surface now matches the proposed DX-like format:

- six-asset markets
- real section order and slider language
- no explicit `synthetic` marker
- no `Archetype` field

Second, the `market_eos` boundary was fixed.

The old boundary ended at the next section header token, which meant the `market` section still included the separator line:

- blank lines
- the `------------------------------` divider

That made `market_eos` land on separator text instead of the last meaningful market token.

The new boundary trims trailing whitespace and divider lines first, then maps the trimmed end back to tokens. So in this report:

- `market_mean` = average state across the actual market block
- `market_eos` = state at the true end of the market block, before the divider and before later sections


= What A, B, And C Mean Here

Each matched market appears in five prompt variants. The synthetic market rows stay the same inside a matched set; only the context changes.

#table(
  columns: (0.8fr, 1.6fr, 3.6fr),
  align: (left, left, left),
  table.hline(stroke: 1pt),
  table.header([*Label*], [*Prompt variant*], [*Meaning*]),
  table.hline(stroke: 0.5pt),
  [`A`], [`market_only`], [Base DX-like prompt. The market comes first and the later sections are neutral / permissive. This is the baseline market read.],
  [`B`], [`risk_5_after_market` or `affordance_5_after_market`], [The market comes first. The edited context is placed after the market, so it should not change `market_mean` or corrected `market_eos`.],
  [`C`], [`risk_5_before_market` or `affordance_5_before_market`], [The same edited context is moved before the market. If context changes perception itself, the market states should move here.],
)

The three tests are:

- #text(weight: "medium")[Sanity check] `A vs B` at `market_mean` and corrected `market_eos`
- #text(weight: "medium")[Perception test] `A vs C` at `market_mean` and corrected `market_eos`
- #text(weight: "medium")[Integration test] `B vs C` at later section states and `last_token`


= What Data Was Used

The full Phase 16 cohort contains:

- `920` prompts total
- `184` matched base markets
- `5` variants per market
- `5,520` asset rows
- `27,600` pairwise asset rows

The prompt families are the same market sweeps from Phase 15, but wrapped in the new context-order scaffold:

- scalar sweeps over one market variable
- coupled sweeps over two market variables
- both with six-asset rosters and DX-like sections


= Market Perception: The Sanity Check And The Warp

#align(center)[#image("../../data/report_assets/synthetic_market_phase16_context_order/phase16_perception_curves.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
`A vs B` stays exactly flat because later context cannot affect the market section once the model has finished reading it. `A vs C` pulls away once the same context is moved before the market, and the strongest divergence appears at the corrected `market_eos` boundary.
]

This is the cleanest result in the phase.

For both risk and affordance:

- `A vs B` is exactly identical at `market_mean` and corrected `market_eos`
- `A vs C` is only mildly different at `market_mean`
- `A vs C` becomes sharply different at corrected `market_eos`

The best corrected-boundary layers are:

#table(
  columns: (1.2fr, 1.1fr, 0.9fr, 0.9fr, 0.9fr),
  align: (left, center, center, center, center),
  table.hline(stroke: 1pt),
  table.header([*Group*], [*State / layer*], [*A vs B*], [*A vs C*], [*Gap*]),
  table.hline(stroke: 0.5pt),
  [`Risk`], [`market_eos @ L42`], [`1.000`], [`0.939`], [`0.061`],
  [`Affordance`], [`market_eos @ L40`], [`1.000`], [`0.930`], [`0.070`],
  [`Risk`], [`market_mean @ L39`], [`1.000`], [`0.996`], [`0.004`],
  [`Affordance`], [`market_mean @ L39`], [`1.000`], [`0.994`], [`0.006`],
)

The interpretation is straightforward:

- the section-average market summary is fairly robust
- the precise state at the end of the market block is context-sensitive
- moving the same risk or affordance information before the market changes where the model #emph[finishes] reading the market


= Where The Warp Lives In The Discovered Basis

#align(center)[#image("../../data/report_assets/synthetic_market_phase16_context_order/phase16_basis_shift.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
At the best corrected `market_eos` layer for each group, the same context moved after the market barely changes the Phase 15 market PCs. The same context moved before the market shifts those same discovered PCs much more strongly.
]

The Phase 15 residualized discovery basis helps interpret the warp. At the strongest corrected `market_eos` layers:

- #text(weight: "medium")[Risk at `L42`]
  the big `A -> C` movement lands on PCs tied to `pct_1h_std`, `pct_5m_max`, and `net_flow_5m_max`
- #text(weight: "medium")[Affordance at `L40`]
  the big `A -> C` movement lands on PCs tied to `pct_5m_max`, `pct_1h_std`, and `pct_5m_std`

So this is not just a generic “the vectors moved” result. The shift is living inside the same market-linked directions discovered in Phase 15.


= Downstream Integration: Do B And C Converge Again?

#align(center)[#image("../../data/report_assets/synthetic_market_phase16_context_order/phase16_integration_curves.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Once both variants have read the full prompt, `B vs C` is still not identical, but it is much closer than the corrected `market_eos` perception split. The strongest order effect is upstream, at perception time.
]

The downstream picture is more forgiving than the market-boundary picture:

- `last_token`, `active_settings_eos`, `portfolio_eos`, and `constraints_eos` all remain high-cosine
- after the early embedding jump at `L0`, most downstream states stay near `0.98–1.00`
- the order effect is still present, but much smaller than the corrected `market_eos` split

That means:

- context order clearly changes the market read when context is seen first
- once the model reads the whole prompt, a large part of that difference gets absorbed downstream
- the sharpest order effect is on #emph[perception], not final integration


= Why The Boundary Fix Matters

This phase would have been much muddier with the old boundary.

If `market_eos` lands on the divider line or blank space after the market rows, then:

- the sanity check is less meaningful
- the perception test is less local to the market block
- any later-section leakage is harder to interpret

With the corrected boundary, the main claim is much sharper:

- later context does not retroactively change the market state
- earlier context changes how the market is encoded
- the strongest evidence for that sits exactly at the real end of the market block


= Raw Prompt Appendix

The appendix below uses one exact matched example from the Phase 16 capture set:

- example id: `context_coupled_pct_5m__net_flow_5m_r00_x00_y00`

No lines are paraphrased.

#pagebreak()

== System Prompt

#block(
  width: 100%,
  inset: (left: 10pt, right: 10pt, top: 10pt, bottom: 10pt),
  fill: rgb("#f7f7f7"),
  stroke: 0.5pt + rgb("#d0d0d0"),
)[#prompt-block("raw_prompts/phase16_context_order/phase16_system_prompt.txt")]

== A: `market_only` User Prompt

#block(
  width: 100%,
  inset: (left: 10pt, right: 10pt, top: 10pt, bottom: 10pt),
  fill: rgb("#f7f7f7"),
  stroke: 0.5pt + rgb("#d0d0d0"),
)[#prompt-block("raw_prompts/phase16_context_order/phase16_market_only_user_prompt.txt")]

== B: `risk_5_after_market` User Prompt

#block(
  width: 100%,
  inset: (left: 10pt, right: 10pt, top: 10pt, bottom: 10pt),
  fill: rgb("#f7f7f7"),
  stroke: 0.5pt + rgb("#d0d0d0"),
)[#prompt-block("raw_prompts/phase16_context_order/phase16_risk_5_after_market_user_prompt.txt")]

== C: `risk_5_before_market` User Prompt

#block(
  width: 100%,
  inset: (left: 10pt, right: 10pt, top: 10pt, bottom: 10pt),
  fill: rgb("#f7f7f7"),
  stroke: 0.5pt + rgb("#d0d0d0"),
)[#prompt-block("raw_prompts/phase16_context_order/phase16_risk_5_before_market_user_prompt.txt")]

== B: `affordance_5_after_market` User Prompt

#block(
  width: 100%,
  inset: (left: 10pt, right: 10pt, top: 10pt, bottom: 10pt),
  fill: rgb("#f7f7f7"),
  stroke: 0.5pt + rgb("#d0d0d0"),
)[#prompt-block("raw_prompts/phase16_context_order/phase16_affordance_5_after_market_user_prompt.txt")]

== C: `affordance_5_before_market` User Prompt

#block(
  width: 100%,
  inset: (left: 10pt, right: 10pt, top: 10pt, bottom: 10pt),
  fill: rgb("#f7f7f7"),
  stroke: 0.5pt + rgb("#d0d0d0"),
)[#prompt-block("raw_prompts/phase16_context_order/phase16_affordance_5_before_market_user_prompt.txt")]
