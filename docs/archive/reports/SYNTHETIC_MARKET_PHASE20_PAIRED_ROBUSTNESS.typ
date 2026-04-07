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

#let summary = json("../../data/report_assets/synthetic_market_phase20_paired_robustness/summary.json")
#let sample = summary.at("sample")
#let methodology = summary.at("methodology")
#let overall = summary.at("overall")
#let patch = summary.at("patch")
#let hypotheses = summary.at("hypotheses")

#let leader = hypotheses.at("leader")
#let dispersion = hypotheses.at("dispersion")

#let leader_denoise = leader.at("pair_modes").at("denoise")
#let leader_noise = leader.at("pair_modes").at("noise")
#let dispersion_denoise = dispersion.at("pair_modes").at("denoise")
#let dispersion_noise = dispersion.at("pair_modes").at("noise")

#let rows_per_hyp = sample.at("rows_per_cell") * sample.at("cell_count_per_hypothesis")
#let total_rows = rows_per_hyp * sample.at("hypothesis_count")

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

// ── Title Block ─────────────────────────────────────────────────
#align(left)[
  #text(size: 9pt, fill: rgb("#b33a2a"), tracking: 0.08em, weight: "medium")[XENON INTERPRETABILITY]
  #v(0.3em)
  #text(size: 22pt, weight: "bold")[Synthetic Market Phase 20]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    Paired denoise/noise robustness battery for the Leader (`L4`) and Dispersion (`L35`) market
    subspaces. This report summarizes the `20`-cell-per-hypothesis matrix, emphasizing matched
    random controls, lambda sweeps, and what the resulting evidence does and does not support.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[#summary.at("date")]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[SAMPLE]\ #text(size: 9pt)[#raw(str(total_rows)) requests across #raw(str(sample.at("hypothesis_count"))) hypotheses]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[GENERATION]\ #text(size: 9pt)[`batch=32`, `max_tokens=15000`, paired denoise/noise]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[ENGINE]\ #text(size: 9pt)[compiled for baseline/project-out, eager fallback for random control]],
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
    Phase 20 is materially stronger than the earlier exploratory batches: it uses matched paired
    cohorts, denoise and noise directions, a lambda sweep, and matched orthogonal random controls.
    The main empirical result is clean and consistent: `project_out` is #emph[less disruptive than
    matched random control in all `12 / 12`] tool-token and generated-token selectivity comparisons.
    Mechanically the patching is also clean (`1152 / 1152` applied, zero skips). The main remaining
    methodological gap is that source behaviors were not generated, so this is still #emph[paired
    robustness evidence], not a full source-restoration mediation demonstration.
  ]
]

#v(1.1em)

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [
    #summary-card(
      [SELECTIVITY WINS],
      [#mono(str(overall.at("tool_token_selectivity_gap_wins")) + " / " + str(overall.at("total_gap_comparisons")), size: 12.5pt)],
      [tool-token gap comparisons won by `project_out` versus random control],
    )
  ],
  [
    #summary-card(
      [TOKEN-DELTA WINS],
      [#mono(str(overall.at("generated_token_selectivity_gap_wins")) + " / " + str(overall.at("total_gap_comparisons")), size: 12.5pt)],
      [generated-token delta comparisons won by `project_out`],
    )
  ],
  [
    #summary-card(
      [PATCH COVERAGE],
      [#mono("1152 / 1152", size: 12.5pt)],
      [all patch entries applied; no skipped rows],
    )
  ],
  [
    #summary-card(
      [METHODOLOGY],
      [#mono("Paired+", size: 12.5pt)],
      [strong robustness battery, still short of full restoration mediation],
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

Phase 20 is best described as:

- A paired robustness battery over `Leader` (`L4`) and `Dispersion` (`L35`) subspaces
- With `denoise` and `noise` cohorts, lambda sweep `{0.5, 1.0, 1.5}`, and matched orthogonal random controls
- Backed by explicit per-row patch coverage and norm diagnostics

It is a substantially better causal test than the earlier single-condition ablations, but it still lacks source-behavior restoration metrics and path validation.


= Experimental Design

== Matrix Layout

#table(
  columns: (1.6fr, 2.2fr),
  align: (left, left),
  table.hline(stroke: 1pt),
  table.header([*Parameter*], [*Value*]),
  table.hline(stroke: 0.5pt),
  [Hypotheses], [Leader (`vol_1h_max`, `L4`) and Dispersion (`pct_1h_mad`, `L35`)],
  [Rows per cell], [#raw(str(sample.at("rows_per_cell")))],
  [Cells per hypothesis], [#raw(str(sample.at("cell_count_per_hypothesis"))): `2` baselines + `6` project-out + `12` random control],
  [Pair modes], [#raw(sample.at("pair_modes").join(", "))],
  [Lambda sweep], [#raw(sample.at("lambda_sweep").map(v => str(v)).join(", "))],
  [Random-control seeds], [#raw(sample.at("random_control_seeds").map(v => str(v)).join(", "))],
  [Batch size / max tokens], [#raw(str(sample.at("batch_size"))) / #raw(str(sample.at("max_tokens")))],
  [Runtime note], [baseline + `project_out` on compiled custom-op path; `random_control` on eager fallback because compiled support is currently `project_out`-only],
)

== What `denoise` And `noise` Mean Here

A `pair mode` label just tells you #emph[which member of a matched prompt pair we intervene on].

For each hypothesis, the code builds pairs of similar prompts from the same family / variant / roster group, then sorts the two prompts by the visible metric:

- `Leader`: `vol_1h_max`
- `Dispersion`: `pct_1h_mad`

Then:

- `denoise` means: #emph[start from the lower-metric prompt in the pair]
- `noise` means: #emph[start from the higher-metric prompt in the pair]

So in plain English:

- `denoise` asks, “what happens if we intervene on the weaker / lower-signal prompt?”
- `noise` asks, “what happens if we intervene on the stronger / higher-signal prompt?”

What these labels do #emph[not] mean in Phase 20:

- they do not mean we literally copy activations from one prompt into the other
- they do not mean we already have a full clean→corrupt or corrupt→clean restoration test

In this phase, the paired prompt only tells us which side of the matched pair we are testing. The actual intervention on that prompt is still one of:

- no patch baseline
- targeted `project_out`
- matched orthogonal `random_control`

== Patch Sanity

#table(
  columns: (1.2fr, 0.9fr, 0.9fr, 0.9fr, 1fr, 1fr),
  align: (left, center, center, center, center, center),
  table.hline(stroke: 1pt),
  table.header([*Hypothesis*], [*Entries*], [*Applied*], [*Coverage*], [*Std ratio*], [*Norm ratio*]),
  table.hline(stroke: 0.5pt),
  [Leader],
  [#raw(str(patch.at("leader").at("patch_entries")))],
  [#fmt3(patch.at("leader").at("patch_applied_rate"))],
  [#fmt3(patch.at("leader").at("mean_coverage_fraction"))],
  [#fmt3(patch.at("leader").at("mean_patch_mean_std_norm_ratio"))],
  [#fmt3(patch.at("leader").at("mean_patch_mean_norm_ratio"))],
  [Dispersion],
  [#raw(str(patch.at("dispersion").at("patch_entries")))],
  [#fmt3(patch.at("dispersion").at("patch_applied_rate"))],
  [#fmt3(patch.at("dispersion").at("mean_coverage_fraction"))],
  [#fmt3(patch.at("dispersion").at("mean_patch_mean_std_norm_ratio"))],
  [#fmt3(patch.at("dispersion").at("mean_patch_mean_norm_ratio"))],
)

The intervention is mechanically clean:

- `patch_applied_rate = 1.0` for both hypotheses
- `patch_skipped_rate = 0.0` for both hypotheses
- coverage is full
- average norm ratios stay close to `1.0`


= Behavioral Results

#align(center)[#image("../../data/report_assets/synthetic_market_phase20_paired_robustness/phase20_tool_token_change_curves.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Tool-token change rates across the lambda sweep. In every panel, `project_out` stays below the matched
random-control mean.
]

#v(0.4em)

#align(center)[#image("../../data/report_assets/synthetic_market_phase20_paired_robustness/phase20_generated_token_delta_curves.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Generated-token deltas show the same pattern: targeted project-out remains less disruptive than matched
random control in all `12` comparisons.
]

#v(0.5em)

#table(
  columns: (1fr, 0.9fr, 0.9fr, 1fr),
  align: (left, center, center, center),
  table.hline(stroke: 1pt),
  table.header([*Cell*], [*Tool-token gap*], [*Tool-name gap*], [*Token-delta gap*]),
  table.hline(stroke: 0.5pt),
  [Leader denoise, `lambda = 1.0`],
  [#fmt3(leader_denoise.at("headline_lambda_1").at("tool_token_selectivity_gap"))],
  [#fmt3(leader_denoise.at("headline_lambda_1").at("tool_name_selectivity_gap"))],
  [#fmt1(leader_denoise.at("headline_lambda_1").at("generated_token_delta_selectivity_gap"))],
  [Leader noise, `lambda = 1.0`],
  [#fmt3(leader_noise.at("headline_lambda_1").at("tool_token_selectivity_gap"))],
  [#fmt3(leader_noise.at("headline_lambda_1").at("tool_name_selectivity_gap"))],
  [#fmt1(leader_noise.at("headline_lambda_1").at("generated_token_delta_selectivity_gap"))],
  [Dispersion denoise, `lambda = 1.0`],
  [#fmt3(dispersion_denoise.at("headline_lambda_1").at("tool_token_selectivity_gap"))],
  [#fmt3(dispersion_denoise.at("headline_lambda_1").at("tool_name_selectivity_gap"))],
  [#fmt1(dispersion_denoise.at("headline_lambda_1").at("generated_token_delta_selectivity_gap"))],
  [Dispersion noise, `lambda = 1.0`],
  [#fmt3(dispersion_noise.at("headline_lambda_1").at("tool_token_selectivity_gap"))],
  [#fmt3(dispersion_noise.at("headline_lambda_1").at("tool_name_selectivity_gap"))],
  [#fmt1(dispersion_noise.at("headline_lambda_1").at("generated_token_delta_selectivity_gap"))],
)

#v(0.4em)

#align(center)[#image("../../data/report_assets/synthetic_market_phase20_paired_robustness/phase20_lambda1_selectivity_gaps.png", width: 88%)]
#text(size: 8pt, fill: rgb("#888"))[
Positive values mean random control is more disruptive than targeted project-out. The effect is strongest
for the dispersion hypothesis.
]

#v(0.4em)

#align(center)[#image("../../data/report_assets/synthetic_market_phase20_paired_robustness/phase20_patch_delta_norm_curves.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Patch delta norms scale with `lambda` as expected, while the mean hidden-state norm stays close to the
unpatched baseline on average.
]

The empirical story is consistent:

- `project_out` wins every tool-token and generated-token selectivity comparison (`12 / 12` each)
- the largest `lambda = 1.0` tool-token gaps appear in the dispersion cells (`0.344` in both denoise and noise)
- the generated-token selectivity gap is also strongest for dispersion, especially `noise` (`4715.8` tokens)
- the effect is selective rather than explosive: patch norms scale with `lambda`, but overall hidden-state norms remain near baseline


= How To Read The Random-Control Comparison

The key comparison in Phase 20 is not “does `project_out` cause more damage than random control?” It is the opposite question:

- does the targeted intervention behave differently from a matched arbitrary perturbation?

Here, “matched” means the control uses:

- the same prompt cohort
- the same layer and token span
- the same number of removed dimensions
- a comparable intervention scale

What we see is that `project_out` is #emph[consistently less disruptive] than the matched random control. For example, at `lambda = 1.0`:

- Leader denoise: tool-token change gap = `0.250`
- Leader noise: tool-token change gap = `0.125`
- Dispersion denoise: tool-token change gap = `0.344`
- Dispersion noise: tool-token change gap = `0.344`

This matters because it rules out a weak interpretation:

- “maybe any hidden-state perturbation of this size would produce the same behavioral effect”

If that weak interpretation were true, the random controls should look similar to the targeted patch. Instead, the random controls are #emph[more destructive] on every paired comparison. The most natural reading is:

- the discovered subspace is behaviorally special
- the targeted intervention is more selective and more on-manifold than random damage
- the observed effect is not just generic corruption

Just as important, this result should #emph[not] be overread. “Less disruptive than random control” does #emph[not] mean:

- the targeted subspace is the unique causal mediator
- the intervention is stronger than random control
- the full causal mechanism has been identified

The right interpretation is narrower:

- the subspace appears non-arbitrary and behaviorally meaningful
- removing it changes behavior in a structured way
- random perturbations produce larger off-manifold damage than the targeted project-out


= Interpretation

Phase 20 supports a stronger claim than Phase 19:

+ The targeted market subspaces are not interchangeable with matched random directions. Across both hypotheses,
  both pair modes, and all tested lambda values, the targeted intervention is systematically less disruptive
  than orthogonal controls.
+ That makes the subspaces look behaviorally special and more on-manifold than random perturbations.
+ The paired denoise/noise structure also reduces the risk that the earlier result was just a one-direction artifact.

But the report should still stay disciplined:

- source behaviors were not generated, so there is no direct restoration score
- neighboring-component and subspace-size sweeps were not run in this matrix
- there is still no path-level validation showing that the behavior flows through the hypothesized route
- because the targeted intervention is #emph[less] disruptive than random control, this is best read as evidence of selectivity and non-arbitrariness, not as evidence that we have fully ablated the model's only market representation

The right label is therefore: #emph[strong paired robustness evidence for behaviorally meaningful market subspaces], not yet #emph[a definitive mediation result for a unique causal variable].


= Next Steps

To close the remaining methodological gaps:

+ Generate explicit source behaviors for the paired cells so the analysis can report restoration and backfire rates directly.
+ Add neighboring-component and subspace-size sweeps around the selected top-`4` basis.
+ Extend the compiled custom-op path beyond `project_out` so all matrix cells can share the same runtime profile.
+ Add path-level validation or finer-grained site sweeps to test whether the effect localizes to the intended computational route.
