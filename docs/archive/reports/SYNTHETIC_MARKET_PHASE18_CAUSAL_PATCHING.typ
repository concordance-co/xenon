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

#let summary = json("../../data/report_assets/synthetic_market_phase18_causal_patching/summary.json")
#let sample = summary.at("sample")
#let generation = summary.at("generation")
#let experiments = summary.at("experiments")

#let fmt3(x) = {
  let y = calc.round(x * 1000) / 1000
  str(y)
}

#let fmt1(x) = {
  let y = calc.round(x * 10) / 10
  str(y)
}

#let fmt6(x) = {
  let y = calc.round(x * 1000000) / 1000000
  str(y)
}

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

#let metric(exp_key, cond, field) = experiments.at(exp_key).at(cond).at("compare").at(field)
#let patch(exp_key, cond, field) = experiments.at(exp_key).at(cond).at("patch").at(field)

// ── Title Block ─────────────────────────────────────────────────
#align(left)[
  #text(size: 9pt, fill: rgb("#b33a2a"), tracking: 0.08em, weight: "medium")[XENON INTERPRETABILITY]
  #v(0.3em)
  #text(size: 22pt, weight: "bold")[Synthetic Market Phase 18]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    Deterministic causal patching on the discovered market-only representation. This version compares four interventions
    on the same stratified `24`-prompt cohort: the named one-dimensional `leader_axis` and `dispersion_axis`, plus the
    full top-`4` `market_mean` subspace at `L4` and `L35`, each against a matched orthogonal random control.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[#summary.at("date")]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[COHORT]\ #text(size: 9pt)[`24` prompts from #raw(str(sample.at("distinct_strata"))) strata]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[CONTEXT]\ #text(size: 9pt)[`market_only`]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[GENERATION]\ #text(size: 9pt)[`temp=0.0`, `top_p=0.95`, `max_tokens=15000`]],
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
    The `1D` named-axis story is weak, but the `4D` subspace story is stronger. Removing the named `leader_axis`
    changes asset choice in only `2/24` prompts, while its matched random control changes `7/24`. Removing the named
    `dispersion_axis` ties its control on asset changes (`3/24` each). But when the intervention expands to the full
    top-`4` subspace, the targeted effect grows: `L4` top-`4` changes asset choice in `6/24` prompts versus `4/24`
    for its control, and `L35` top-`4` changes asset choice in `8/24` prompts versus `3/24` for its control, with one
    `buy_token` → `record_observation` flip. The market representation looks more like a distributed subspace than a
    single causal knob.
  ]
]

#v(1.2em)

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [
    #summary-card(
      [L4 1D ASSET CHANGE],
      [#mono("2 / 24", size: 12.5pt)],
      [control = `7 / 24`],
    )
  ],
  [
    #summary-card(
      [L4 4D ASSET CHANGE],
      [#mono("6 / 24", size: 12.5pt)],
      [control = `4 / 24`],
    )
  ],
  [
    #summary-card(
      [L35 1D ASSET CHANGE],
      [#mono("3 / 24", size: 12.5pt)],
      [control = `3 / 24`],
    )
  ],
  [
    #summary-card(
      [L35 4D ASSET CHANGE],
      [#mono("8 / 24", size: 12.5pt)],
      [control = `3 / 24`],
    )
  ],
)


= Methodology

All runs in this report use the same prompt cohort and generation surface:

- Context variant: `market_only`
- Selection strategy: stratified round-robin over `(family, family_variant, roster_key)`
- Sample size: `24` prompts from #raw(str(sample.at("distinct_strata"))) strata
- Tool schema: `trading_v1`, tool choice: `required`

#table(
  columns: (1.2fr, 2fr),
  align: (left, left),
  table.hline(stroke: 1pt),
  table.header([*Parameter*], [*Value*]),
  table.hline(stroke: 0.5pt),
  [Temperature], [#raw(str(generation.at("temperature")))],
  [Top-p], [#raw(str(generation.at("top_p")))],
  [Top-k], [#raw(str(generation.at("top_k")))],
  [Max tokens], [#raw(str(generation.at("max_tokens")))],
)

The patch is applied to the market token span only:

+ locate the exact market span in the rendered chat prompt
+ compute the `market_mean` hidden state over that span
+ standardize with the Phase 15 residualized PCA mean and scale
+ remove either a named `1D` component or the top `4` components at the target layer
+ convert that standardized delta back to raw hidden space
+ add the same raw delta to every token in the market span

The matched random control removes an equal-norm random subspace orthogonal to the targeted one.

#table(
  columns: (1.5fr, 0.9fr, 1fr, 2.2fr),
  align: (left, center, center, left),
  table.hline(stroke: 1pt),
  table.header([*Intervention*], [*Layer*], [*Dim*], [*Target*]),
  table.hline(stroke: 0.5pt),
  [`Leader 1D`], [`4`], [`1D`], [`leader_axis`],
  [`Leader 4D`], [`4`], [`4D`], [top `4` `market_mean` PCs],
  [`Dispersion 1D`], [`35`], [`1D`], [`dispersion_axis`],
  [`Dispersion 4D`], [`35`], [`4D`], [top `4` `market_mean` PCs],
)


= Patch Sanity

The mechanical side is clean across all four experiments.

#table(
  columns: (1.5fr, 1.1fr, 1.1fr, 1.1fr),
  align: (left, center, center, center),
  table.hline(stroke: 1pt),
  table.header([*Intervention*], [*Project-out coeff after*], [*Random-control coeff after*], [*Rows with stats*]),
  table.hline(stroke: 0.5pt),
  [`Leader 1D @ L4`],
  [#fmt6(patch("leader_1d", "project_out", "coeff_after_mean"))],
  [#fmt3(patch("leader_1d", "random_control", "coeff_after_mean"))],
  [#raw(str(patch("leader_1d", "project_out", "rows_with_patch_stats"))) / 24],
  [`Leader 4D @ L4`],
  [#fmt6(patch("leader_4d", "project_out", "coeff_after_mean"))],
  [#fmt3(patch("leader_4d", "random_control", "coeff_after_mean"))],
  [#raw(str(patch("leader_4d", "project_out", "rows_with_patch_stats"))) / 24],
  [`Dispersion 1D @ L35`],
  [#fmt6(patch("dispersion_1d", "project_out", "coeff_after_mean"))],
  [#fmt3(patch("dispersion_1d", "random_control", "coeff_after_mean"))],
  [#raw(str(patch("dispersion_1d", "project_out", "rows_with_patch_stats"))) / 24],
  [`Dispersion 4D @ L35`],
  [#fmt6(patch("dispersion_4d", "project_out", "coeff_after_mean"))],
  [#fmt3(patch("dispersion_4d", "random_control", "coeff_after_mean"))],
  [#raw(str(patch("dispersion_4d", "project_out", "rows_with_patch_stats"))) / 24],
)

The patch fires cleanly in all cases. The remaining question is whether removing each target perturbs behavior more than the matched control.


= Behavioral Comparison

#align(center)[#image("../../data/report_assets/synthetic_market_phase18_causal_patching/phase18_change_rates.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Each experiment compares the targeted `project_out` against its matched orthogonal random control on the same deterministic `24`-prompt cohort.
]

#table(
  columns: (1.55fr, 1fr, 1fr, 1fr, 1fr),
  align: (left, center, center, center, center),
  table.hline(stroke: 1pt),
  table.header([*Intervention*], [*Tool-name change*], [*Asset change*], [*Mean spend delta*], [*Mean token-count delta*]),
  table.hline(stroke: 0.5pt),
  [`Leader 1D p/o`],
  [#fmt3(metric("leader_1d", "project_out", "tool_name_change_rate"))],
  [#fmt3(metric("leader_1d", "project_out", "tool_token_change_rate"))],
  [#fmt3(metric("leader_1d", "project_out", "mean_tool_spend_pct_delta"))],
  [#fmt1(metric("leader_1d", "project_out", "mean_generated_token_count_delta"))],
  [`Leader 1D ctrl`],
  [#fmt3(metric("leader_1d", "random_control", "tool_name_change_rate"))],
  [#fmt3(metric("leader_1d", "random_control", "tool_token_change_rate"))],
  [#fmt3(metric("leader_1d", "random_control", "mean_tool_spend_pct_delta"))],
  [#fmt1(metric("leader_1d", "random_control", "mean_generated_token_count_delta"))],
  table.hline(stroke: 0.3pt + rgb("#ddd")),
  [`Leader 4D p/o`],
  [#fmt3(metric("leader_4d", "project_out", "tool_name_change_rate"))],
  [#fmt3(metric("leader_4d", "project_out", "tool_token_change_rate"))],
  [#fmt3(metric("leader_4d", "project_out", "mean_tool_spend_pct_delta"))],
  [#fmt1(metric("leader_4d", "project_out", "mean_generated_token_count_delta"))],
  [`Leader 4D ctrl`],
  [#fmt3(metric("leader_4d", "random_control", "tool_name_change_rate"))],
  [#fmt3(metric("leader_4d", "random_control", "tool_token_change_rate"))],
  [#fmt3(metric("leader_4d", "random_control", "mean_tool_spend_pct_delta"))],
  [#fmt1(metric("leader_4d", "random_control", "mean_generated_token_count_delta"))],
  table.hline(stroke: 0.3pt + rgb("#ddd")),
  [`Dispersion 1D p/o`],
  [#fmt3(metric("dispersion_1d", "project_out", "tool_name_change_rate"))],
  [#fmt3(metric("dispersion_1d", "project_out", "tool_token_change_rate"))],
  [#fmt3(metric("dispersion_1d", "project_out", "mean_tool_spend_pct_delta"))],
  [#fmt1(metric("dispersion_1d", "project_out", "mean_generated_token_count_delta"))],
  [`Dispersion 1D ctrl`],
  [#fmt3(metric("dispersion_1d", "random_control", "tool_name_change_rate"))],
  [#fmt3(metric("dispersion_1d", "random_control", "tool_token_change_rate"))],
  [#fmt3(metric("dispersion_1d", "random_control", "mean_tool_spend_pct_delta"))],
  [#fmt1(metric("dispersion_1d", "random_control", "mean_generated_token_count_delta"))],
  table.hline(stroke: 0.3pt + rgb("#ddd")),
  [`Dispersion 4D p/o`],
  [#fmt3(metric("dispersion_4d", "project_out", "tool_name_change_rate"))],
  [#fmt3(metric("dispersion_4d", "project_out", "tool_token_change_rate"))],
  [#fmt3(metric("dispersion_4d", "project_out", "mean_tool_spend_pct_delta"))],
  [#fmt1(metric("dispersion_4d", "project_out", "mean_generated_token_count_delta"))],
  [`Dispersion 4D ctrl`],
  [#fmt3(metric("dispersion_4d", "random_control", "tool_name_change_rate"))],
  [#fmt3(metric("dispersion_4d", "random_control", "tool_token_change_rate"))],
  [#fmt3(metric("dispersion_4d", "random_control", "mean_tool_spend_pct_delta"))],
  [#fmt1(metric("dispersion_4d", "random_control", "mean_generated_token_count_delta"))],
)


= Asset Surface

#align(center)[#image("../../data/report_assets/synthetic_market_phase18_causal_patching/phase18_asset_distribution.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Targeted project-out conditions only. The full `4D` subspace patches move the chosen-asset distribution more than the named `1D` patches.
]

The concrete pattern is:

- `Leader 1D`: mostly preserves the baseline `MORI/LUMA` mix
- `Leader 4D`: pushes meaningfully toward `VEXA`
- `Dispersion 1D`: introduces one `record_observation` and modest extra `VEXA/TAVO`
- `Dispersion 4D`: produces the strongest asset redistribution, including `TAVO` and `KIRO`


= Interpretation

Three conclusions:

+ The named `1D` axes are real representation directions, but weak causal handles on final policy.
+ The broader `4D` subspaces are more causally relevant than the single named PCs.
+ The `L35` subspace is the strongest signal: `Dispersion 4D` clearly beats its matched control on asset changes (`8/24` vs `3/24`) and produces the largest targeted behavioral effect in the batch.

This is consistent with Phase 17's finding that the market encoding is broad and multi-dimensional. The `1D` patches underperform; the `4D` patches get stronger. That is what you would expect if the market summary is distributed across a small subspace rather than concentrated into one special direction.


= Next Steps

The clearest next experiment is a joint patch: `L4 top-4` + `L35 top-4` applied together on the same cohort.

That would test whether the early leader/prominence summary and the later dispersion/unevenness summary are jointly necessary for the downstream decision. If the current pattern holds, the joint patch should be stronger than either single-layer patch alone.

The same deterministic stratified cohort and matched orthogonal controls should be kept for comparability.
