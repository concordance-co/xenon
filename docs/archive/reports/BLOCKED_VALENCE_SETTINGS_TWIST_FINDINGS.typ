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
  #text(size: 22pt, weight: "bold")[Blocked Valence + Settings Twist]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[Corrected `v2` kickoff findings for the top-ranked real-data rerun track. Research anchors: `RANKED_RESEARCH_ROADMAP.md`, `MARKET_COUNTING_MANIFOLDS_PLAN.md`, and `XENON_RESEARCH_ROADMAP_AND_KICKOFF.typ`.]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[20 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[BASE EXAMPLES]\ #text(size: 9pt)[154]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[PROMPT VARIANTS]\ #text(size: 9pt)[428]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[LAYERS]\ #text(size: 9pt)[48]],
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
  #text(size: 12.5pt, weight: "medium")[The first top-ranked rerun supports “early preference, late permission” more than a broad hidden blocked-valence pool.]
  #v(0.4em)
  #text(size: 9.5pt, fill: rgb("#555"))[In the corrected `v2` run, clearing strategies only reveals directional valence in 3 of 34 blocked-observe cases. But extreme settings do move downstream action state on a meaningful minority of examples without changing the preferred buy/sell asset identity in any of the 120 settings triplets.]
]

#v(1.2em)

// ── Summary Metrics ─────────────────────────────────────────────
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BLOCKED REVEALS]\ #text(size: 16pt, weight: "bold")[3 / 34] #text(size: 8pt, fill: rgb("#888"))[\ directional after strategy clear]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[SETTINGS FLIPS]\ #text(size: 16pt, weight: "bold")[17 / 120] #text(size: 8pt, fill: rgb("#888"))[\ all1 vs all5 valence flips]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[STRONG SHIFTS]\ #text(size: 16pt, weight: "bold")[13 / 120] #text(size: 8pt, fill: rgb("#888"))[\ |Δtrade prob| > 0.5]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[TOP-SYMBOL RERANKS]\ #text(size: 16pt, weight: "bold")[0 / 154] #text(size: 8pt, fill: rgb("#888"))[\ blocked pairs + settings triplets]],
)


= Scope

This report is the first completed pass on the top-ranked roadmap track: *blocked valence + settings twist* on real DX-style prompts.

- *Corrected rerun.* The earlier `v1` settings read was superseded after a prompt-rewrite bug was found in the slider editing path. This report uses only the corrected `v2` capture.
- *Transferred probes.* The rerun prompts are scored with probes transferred from the 918-tick decision-structure analysis: buy-target, sell-target, decision-type, and trade-side readouts.
- *Paired interventions.* The report compares paired prompt families rather than pooled averages:
  - `blocked_valence`: `original` vs `clear_strategies`
  - `settings_twist`: `original` vs `settings_all1` vs `settings_all5`
- *Dedicated research volume.* Captures and paired-analysis artifacts were written to the separate `xenon-research-data` volume, not the crowded `xenon-data` volume.


= Dataset

The kickoff set contains 154 base examples and 428 captured prompt variants:

- 34 blocked-observe examples rerun with strategies cleared
- 120 settings-sensitive examples rerun under all-sliders-low and all-sliders-high
- 48-layer pooled residuals for each captured variant

#align(center)[#image("../../data/report_assets/research_rerun_kickoff_v2/experiment_design.png", width: 90%)]
#text(size: 8pt, fill: rgb("#888"))[The kickoff is intentionally narrow. It focuses on the highest-value paired interventions from the research roadmap rather than trying to solve every policy and affordance question at once.]


= Main Quantitative Findings

#set table(stroke: none)
#table(
  columns: (1.5fr, 1.1fr, 1.7fr),
  align: (left, right, left),
  table.hline(stroke: 1pt),
  table.header(
    [*Question*], [*Metric*], [*Interpretation*],
  ),
  table.hline(stroke: 0.5pt),
  [`Blocked-valence reveal rate`], [3 / 34], [Clearing strategies rarely unmasks a hidden directional trade state.],
  [`Blocked-symbol reranks`], [0 / 34], [When a blocked case does reveal directional valence, it keeps the same top buy/sell asset identity.],
  [`Settings valence flips`], [17 / 120], [Extreme settings do change downstream action labels on a meaningful minority subset.],
  [`Strong settings shifts`], [13 / 120], [The settings effect is sparse rather than diffuse: most pairs stay near zero while a few flip hard.],
  [`Settings-symbol reranks`], [0 / 120], [All settings triplets preserve the same top buy and top sell symbols across `all1` and `all5`.],
  [`Row-state stability`], [CKA min 0.99992], [The market-row representation is almost unchanged under settings rewrites.],
  [`Downstream state movement`], [CKA min 0.748 / 0.741], [`last_token` and `active_settings_eos` diverge materially even when the row states stay fixed.],
  table.hline(stroke: 1pt),
)

#v(0.5em)

This is the central split in the results. The asset preference signals are extremely stable, but the execution state is not. That favors a decomposition in which settings and policy act *after* market preference formation instead of rewriting the underlying asset ranking.


= Blocked-Valence Read

#align(center)[#image("../../data/report_assets/research_rerun_kickoff_v2/blocked_valence.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[Blocked-observe cases remain mostly neutral even after strategy clearing. The pool is dominated by generic strategy pressure rather than clean “bullish but blocked” examples.]

- Only 3 of 34 blocked pairs reveal a directional state after clearing strategies.
- Those reveals are *not* arbitrary:
  - 2 bullish reveals match the original top-buy asset (`AIGF`)
  - 1 bearish reveal matches the original top-sell asset (`POOPCOIN`)
- No blocked pair changes its top buy symbol or top sell symbol.
- The current blocked pool is still dominated by `high_strategy_present`, which is a much noisier source of hidden valence than direct buy-block / sell-block conditions.

This weakens the strongest version of the blocked-valence hypothesis. Generic blocked-observe rows are *not* a rich latent-trade reservoir by default. The better interpretation is narrower: some blocked cases do carry hidden directional preference, but we need more targeted block modes to surface it reliably.


= Settings-Twist Read

#align(center)[#image("../../data/report_assets/research_rerun_kickoff_v2/settings_shift.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[The settings effect is real but sparse. Most triplets stay near zero, while a minority flip sharply between neutral and trade-like states.]

#align(center)[#image("../../data/report_assets/research_rerun_kickoff_v2/settings_cohort_effects.png", width: 88%)]
#text(size: 8pt, fill: rgb("#888"))[Strong settings sensitivity is concentrated in `policy_tension_observe` and `buy` cohorts. Sell-side examples move less often in this kickoff batch.]

- 17 of 120 settings triplets change predicted valence between `all1` and `all5`.
- 13 of 120 show *hard* trade-probability shifts with `|Δtrade probability| > 0.5`.
- The mean effect is modest (`all5 - all1 = +0.025` trade probability), but the median effect is zero. This is a sparse-intervention regime, not a smooth global drift.
- Strong settings changes concentrate in:
  - `policy_tension_observe`: 8 strong shifts, 8 valence flips
  - `buy`: 4 strong shifts, 8 valence flips
  - `sell`: 1 strong shift, 1 valence flip

Most importantly, *none* of the 120 settings triplets changes its top buy symbol or top sell symbol. The preferred asset identities stay fixed even when the trade / neutral / bearish / bullish classification changes.


= Layerwise Structure

#align(center)[#image("../../data/report_assets/research_rerun_kickoff_v2/settings_layerwise.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[`row_mean` stays almost perfectly stable under settings rewrites, but downstream representations diverge. The largest `last_token` divergence appears early, while the settings section remains separated through the end of the model.]

The layerwise metrics sharpen the qualitative picture:

- `row_mean` CKA stays essentially pinned at 1.0 across all layers.
- `last_token` CKA between `all1` and `all5` falls as low as `0.748` at layer 6.
- `active_settings_eos` CKA between `all1` and `all5` falls as low as `0.741` by layer 47.
- Parallel-fraction values remain low, which suggests settings changes are not just simple amplitude rescaling along one fixed direction.

This is the strongest support so far for *early preference, late permission*: settings are changing downstream action-state geometry more than they are changing the row-level market representation.


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
  Settings and policy mostly act downstream of a stable asset-preference signal. The model’s preferred buy/sell assets are far more stable than its final trade vs observe state.
]

#v(0.5em)

#block(
  width: 100%,
  inset: (left: 14pt, top: 10pt, bottom: 10pt, right: 10pt),
  stroke: (left: 3pt + rgb("#f57f17"), top: none, right: none, bottom: none),
  fill: rgb("#fff8e1"),
)[
  #text(size: 7.5pt, fill: rgb("#f57f17"), weight: "bold", tracking: 0.08em)[CHALLENGED NOW]
  #v(0.2em)
  The generic blocked-observe pool is not yet a broad source of hidden bullish/bearish labels. Clearing strategies alone reveals too few clean directional cases.
]

#v(0.5em)

#block(
  width: 100%,
  inset: (left: 14pt, top: 10pt, bottom: 10pt, right: 10pt),
  stroke: (left: 3pt + rgb("#16324F"), top: none, right: none, bottom: none),
  fill: rgb("#eef3f7"),
)[
  #text(size: 7.5pt, fill: rgb("#16324F"), weight: "bold", tracking: 0.08em)[REVISED EMPHASIS]
  #v(0.2em)
  The better near-term path is stronger settings and affordance interventions, plus narrower blocked cohorts that directly isolate buy-block, sell-block, funding, and hold-floor conflicts.
]


= What This Phase Accomplished

Relative to the research roadmap, this kickoff closes the first real-data pass on the top-ranked track:

- a corrected rerun workflow for paired real prompts
- a dedicated research capture volume and clean Neon tables
- paired scoring using transferred decision-structure probes
- first empirical separation between:
  - stable asset preference
  - downstream settings- and policy-dependent action state
- a concrete update to the blocked-valence hypothesis based on real data rather than intuition


= What Is Still Missing

This phase does *not* yet answer the entire decision-decomposition question.

- The blocked-valence half is underpowered because the current cohort is too generic.
- The settings half uses only `all1` and `all5`; it does not yet isolate which slider or policy axis is responsible for the observed flips.
- The current analysis is probe-based; it still needs causal patching or ablation on the settings-sensitive subset.
- The current batch covers 154 base examples, not the much larger real-log pool available in the database.


= Next Moves

The next steps should be narrower and more causal:

+ rebuild the blocked-valence cohort around direct block modes (`buy_blocked_only`, `sell_blocked_only`, `zero_eth`, `hold_floor`) instead of generic `high_strategy_present`
+ expand settings interventions from `all1` / `all5` to per-slider and per-policy rewrites
+ run causal interventions on the settings-sensitive subset, especially the early layers where `last_token` divergence is largest
+ use the extra full logs only after the cohort definitions are sharper, so scale buys us cleaner data instead of more of the same noise

If that next pass still preserves top asset identity while changing final action state, the “early preference, late permission” hypothesis becomes substantially stronger.

#v(2em)
#line(length: 100%, stroke: 0.5pt + rgb("#ccc"))
#v(0.3em)
#text(size: 8pt, fill: rgb("#999"))[Blocked Valence + Settings Twist Report — corrected `v2` kickoff, 20 March 2026. 154 base examples, 428 captured prompt variants, 48-layer paired analysis on the dedicated research volume.]
