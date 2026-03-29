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

#let summary = json("../../data/report_assets/synthetic_market_phase19_methodology_and_results/summary.json")
#let sample = summary.at("sample")
#let generation = summary.at("generation")
#let methodology = summary.at("methodology")
#let conditions = summary.at("conditions")
#let comparisons = summary.at("comparisons")
#let patch_data = summary.at("patch")

#let fmt3(x) = {
  let y = calc.round(x * 1000) / 1000
  str(y)
}

#let fmt1(x) = {
  let y = calc.round(x * 10) / 10
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

#let status-fill(status) = {
  if status == "meets" { rgb("#2c7c46") }
  else if status == "partial" { rgb("#a16b13") }
  else { rgb("#b33a2a") }
}

#let status-chip(status) = box(
  fill: status-fill(status),
  inset: (x: 4pt, y: 1pt),
  radius: 4pt,
)[#text(size: 7pt, fill: white, weight: "bold")[#status]]

#let proj = comparisons.at("project_out")
#let ctrl = comparisons.at("random_control")
#let n = sample.at("count")
#let proj_asset = calc.round(proj.at("tool_token_change_rate") * n)
#let ctrl_asset = calc.round(ctrl.at("tool_token_change_rate") * n)

// ── Title Block ─────────────────────────────────────────────────
#align(left)[
  #text(size: 9pt, fill: rgb("#b33a2a"), tracking: 0.08em, weight: "medium")[XENON INTERPRETABILITY]
  #v(0.3em)
  #text(size: 22pt, weight: "bold")[Synthetic Market Phase 19]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    Methodology review and causal-patching readout for the joint `L4 + L35` top-`4` market subspace
    intervention. This report evaluates the `48`-prompt deterministic batch against activation-patching
    best practices and characterizes what the result does and does not support.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[#summary.at("date")]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[COHORT]\ #text(size: 9pt)[#raw(str(n)) prompts from #raw(str(sample.at("distinct_strata"))) strata]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[GENERATION]\ #text(size: 9pt)[deterministic: `temp=0.0`, `top_p=0.95`]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[ENGINE]\ #text(size: 9pt)[unified worker, `batch=8`, chunked prefill]],
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
    The experiment follows good exploratory causal practice but not the strongest form of activation patching.
    Strengths: matched cohorts, deterministic decode, matched orthogonal random control, exact market-span
    targeting, patch coverage diagnostics, and a unified engine path. The main gap is conceptual: this is a
    #emph[necessity-style joint subspace ablation], not a clean source/base denoising mediation test with
    matched pairs, lambda sweeps, or path validation. The behavioral signal is modest --- `project_out` beats
    the control on asset changes (#raw(str(proj_asset)) vs #raw(str(ctrl_asset))) and spend delta --- but not
    large enough to claim definitive mechanistic identification.
  ]
]

#v(1.1em)

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [
    #summary-card(
      [ASSET-CHANGE RATE],
      [#mono(str(proj_asset) + " / " + str(n), size: 12.5pt)],
      [control = #raw(str(ctrl_asset)) / #raw(str(n))],
    )
  ],
  [
    #summary-card(
      [SPEND DELTA],
      [#mono(fmt3(proj.at("mean_tool_spend_pct_delta")), size: 12.5pt)],
      [control = #fmt3(ctrl.at("mean_tool_spend_pct_delta"))],
    )
  ],
  [
    #summary-card(
      [PATCH COVERAGE],
      [#mono("48 / 48", size: 12.5pt)],
      [both conditions; no skipped rows],
    )
  ],
  [
    #summary-card(
      [METHODOLOGY],
      [#mono("Exploratory+", size: 12.5pt)],
      [good hygiene, not yet full best-practice mediation],
    )
  ],
)


= Methodology Scorecard

#table(
  columns: (2.5fr, 0.7fr),
  align: (left, center),
  table.hline(stroke: 1pt),
  table.header([*Best-practice item*], [*Status*]),
  table.hline(stroke: 0.5pt),
  ..methodology.at("scorecard").map(row => (
    [#row.at("item")],
    [#status-chip(row.at("status"))],
  )).flatten(),
)

#v(0.4em)

The current experiment is:

- A deterministic, stratified, necessity-style `project_out` on the joint `L4 + L35` top-`4` `market_mean` subspace
- Compared against a matched orthogonal random control on the same prompts
- With explicit patch-applied, coverage, and norm diagnostics

It should #emph[not] be described as a full clean/corrupt activation-patching demonstration of a unique mechanistic variable.


= Experimental Design

== Cohort and Generation

#table(
  columns: (1.4fr, 2.2fr),
  align: (left, left),
  table.hline(stroke: 1pt),
  table.header([*Parameter*], [*Value*]),
  table.hline(stroke: 0.5pt),
  [Context variant], [`market_only`],
  [Selection strategy], [#raw(sample.at("selection_strategy"))],
  [Sample size], [#raw(str(n)) prompts from #raw(str(sample.at("distinct_strata"))) strata],
  [Tool schema / choice], [#raw(generation.at("tool_schema_mode")) / #raw(generation.at("tool_choice"))],
  [Temperature / top-p / top-k],
  [#raw(str(generation.at("temperature"))) / #raw(str(generation.at("top_p"))) / #raw(str(generation.at("top_k")))],
  [Max tokens], [#raw(str(generation.at("max_tokens")))],
  [Batch size], [#raw(str(generation.at("batch_size")))],
  [Compute], [#raw(str(generation.at("cpu"))) CPU, #raw(str(generation.at("memory_gb"))) GB],
  [Intervention],
  [joint `L4 + L35` top-`4` `market_mean` project-out vs matched orthogonal random control],
)

== Patch Procedure

The intervention targets the market token span only:

+ Locate the exact market span in the rendered chat prompt
+ Compute `market_mean` over that span at both `L4` and `L35`
+ Standardize with the Phase 15 residualized PCA mean and scale
+ Remove the top `4` standardized components at each layer
+ Convert the standardized delta back to raw hidden space
+ Add the same raw delta to every token in the market span

This is a distributed-subspace necessity probe --- broader than a single-position or single-coefficient source-swap, but well-targeted to the discovered market representation.


= Patch Sanity

#table(
  columns: (1.3fr, 0.8fr, 0.8fr, 0.9fr, 0.9fr, 0.9fr),
  align: (left, center, center, center, center, center),
  table.hline(stroke: 1pt),
  table.header([*Condition*], [*Layer*], [*Rows*], [*Coverage*], [*Δ norm*], [*Norm ratio*]),
  table.hline(stroke: 0.5pt),
  [`project_out`], [`L4`],
  [#raw(str(patch_data.at("project_out").at("l4").at("rows_with_stats")))],
  [#fmt3(patch_data.at("project_out").at("l4").at("mean_coverage_fraction"))],
  [#fmt3(patch_data.at("project_out").at("l4").at("mean_delta_norm_std"))],
  [#fmt3(patch_data.at("project_out").at("l4").at("mean_norm_ratio"))],
  [`project_out`], [`L35`],
  [#raw(str(patch_data.at("project_out").at("l35").at("rows_with_stats")))],
  [#fmt3(patch_data.at("project_out").at("l35").at("mean_coverage_fraction"))],
  [#fmt3(patch_data.at("project_out").at("l35").at("mean_delta_norm_std"))],
  [#fmt3(patch_data.at("project_out").at("l35").at("mean_norm_ratio"))],
  table.hline(stroke: 0.3pt + rgb("#ddd")),
  [`random_control`], [`L4`],
  [#raw(str(patch_data.at("random_control").at("l4").at("rows_with_stats")))],
  [#fmt3(patch_data.at("random_control").at("l4").at("mean_coverage_fraction"))],
  [#fmt3(patch_data.at("random_control").at("l4").at("mean_delta_norm_std"))],
  [#fmt3(patch_data.at("random_control").at("l4").at("mean_norm_ratio"))],
  [`random_control`], [`L35`],
  [#raw(str(patch_data.at("random_control").at("l35").at("rows_with_stats")))],
  [#fmt3(patch_data.at("random_control").at("l35").at("mean_coverage_fraction"))],
  [#fmt3(patch_data.at("random_control").at("l35").at("mean_delta_norm_std"))],
  [#fmt3(patch_data.at("random_control").at("l35").at("mean_norm_ratio"))],
)

The patch is mechanically clean:

- `patch_applied_rate = 1.0`, `patch_skipped_rate = 0.0` for both conditions
- Coverage is full at both layers
- Norm ratios stay close to `1.0`, providing evidence against gross distribution blow-up


= Behavioral Results

#align(center)[#image("../../data/report_assets/synthetic_market_phase19_methodology_and_results/phase19_change_rates.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Project-out versus matched random control on the same #raw(str(n))-prompt deterministic cohort.
]

#table(
  columns: (1.7fr, 1fr, 1fr),
  align: (left, center, center),
  table.hline(stroke: 1pt),
  table.header([*Metric*], [*Project-out*], [*Random control*]),
  table.hline(stroke: 0.5pt),
  [Tool-name change rate],
  [#fmt3(proj.at("tool_name_change_rate"))],
  [#fmt3(ctrl.at("tool_name_change_rate"))],
  [Asset change rate],
  [#fmt3(proj.at("tool_token_change_rate"))],
  [#fmt3(ctrl.at("tool_token_change_rate"))],
  [Mean spend delta],
  [#fmt3(proj.at("mean_tool_spend_pct_delta"))],
  [#fmt3(ctrl.at("mean_tool_spend_pct_delta"))],
  [Mean token-count delta],
  [#fmt1(proj.at("mean_generated_token_count_delta"))],
  [#fmt1(ctrl.at("mean_generated_token_count_delta"))],
)

#v(0.5em)

#align(center)[#image("../../data/report_assets/synthetic_market_phase19_methodology_and_results/phase19_token_distribution.png", width: 88%)]
#text(size: 8pt, fill: rgb("#888"))[
Chosen-asset distribution across baseline, project-out, and random control.
]

#v(0.5em)

#align(center)[#image("../../data/report_assets/synthetic_market_phase19_methodology_and_results/phase19_generated_token_boxplot.png", width: 82%)]
#text(size: 8pt, fill: rgb("#888"))[
Generated token counts are highly variable across conditions. Full-text change rate is `1.0` everywhere and not a useful discriminator.
]

The behavioral signal is modest but present:

- Project-out beats control on asset changes: #raw(str(proj_asset)) / #raw(str(n)) vs #raw(str(ctrl_asset)) / #raw(str(n))
- Project-out exceeds control on mean spend delta: #fmt3(proj.at("mean_tool_spend_pct_delta")) vs #fmt3(ctrl.at("mean_tool_spend_pct_delta"))
- `tool_presence_change_rate = 0.0` in both conditions
- `first_token_change_rate = 0.0` in both conditions


= Interpretation

This result supports a restrained conclusion:

+ The joint `L4 + L35` top-`4` market subspace is behaviorally relevant: the targeted project-out is modestly stronger than the matched random control on both asset choice and spend.
+ The experiment is materially cleaner than earlier batches: matched cohort, unified engine path, deterministic decode, and explicit patch diagnostics.
+ But the effect is not large enough, nor methodologically isolated enough, to claim that the PCA subspace is the unique causal variable the model uses.

The right label: #emph[good exploratory causal evidence for a distributed market subspace], not #emph[definitive mechanistic identification of the causal mediator].


= Next Steps

To move from exploratory ablation toward best-practice activation patching:

+ *Matched clean/corrupt pairs*: construct source/base prompt pairs where the market content differs in a controlled way, then test denoising and noising in both directions.
+ *Lambda sweep*: vary the intervention strength (e.g. `{0.25, 0.5, 1.0, 1.5}`) to test dose-response.
+ *Subspace-size controls*: sweep around the top-`4` choice to test whether `3` or `5` PCs change the result.
+ *Finer readout*: add a logit-difference or first-tool margin metric alongside the coarse behavioral surface.
+ *Path validation*: test whether alternate routes (e.g. later-layer subspaces alone) can explain the same behavioral shift.
