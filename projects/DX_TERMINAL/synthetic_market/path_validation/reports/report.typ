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

#let summary = json("../../../../../data/report_assets/synthetic_market_phase22_path_validation/summary.json")
#let sample = summary.at("sample")
#let experiment = summary.at("experiment")
#let methodology = summary.at("methodology")
#let overall = summary.at("overall")
#let metrics = summary.at("metrics")
#let counts = summary.at("counts")
#let tool_surface = summary.at("tool_surface")

#let fmt3(x) = {
  if x == none {
    return "n/a"
  }
  let y = calc.round(x * 1000) / 1000
  str(y)
}

#let pair_fmt(a, b) = fmt3(a) + " → " + fmt3(b)

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
  #text(size: 22pt, weight: "bold")[Synthetic Market Phase 22]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    Path validation for the Leader axis using an early lesion at L#raw(str(experiment.at("lesion_layer")))
    and a downstream rescue at L#raw(str(experiment.at("rescue_layer"))). The comparison is lesion only
    versus lesion plus rescue, both run on the compiled non-eager request-scoped patch path.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[#summary.at("date")]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[ROWS]\ #text(size: 9pt)[#raw(str(sample.at("count")))]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[PATCH]\ #text(size: 9pt)[L#raw(str(experiment.at("lesion_layer"))) `project_out` + L#raw(str(experiment.at("rescue_layer"))) `swap_components`]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[ENGINE]\ #text(size: 9pt)[#sample.at("engine_mode")]],
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
  #text(size: 12.5pt, weight: "medium")[#overall.at("main_read")]
]

#v(1.1em)

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [
    #summary-card(
      [TOOL-TOKEN RESTORE],
      [#mono(fmt3(metrics.at("source_tool_token_restoration_rate")), size: 12.5pt)],
      [#raw(str(counts.at("tool_token_restored_count"))) of #raw(str(counts.at("tool_token_restorable_count"))) restorable rows],
    )
  ],
  [
    #summary-card(
      [TOOL-TOKEN BACKFIRE],
      [#mono(fmt3(metrics.at("source_tool_token_backfire_rate")), size: 12.5pt)],
      [lower is better on rows already matching the source],
    )
  ],
  [
    #summary-card(
      [SPEND IMPROVEMENT],
      [#mono(fmt3(metrics.at("source_tool_spend_pct_improvement_rate")), size: 12.5pt)],
      [share of spend-mismatched rows that moved closer to the source],
    )
  ],
  [
    #summary-card(
      [LENGTH RESTORATION],
      [#mono(fmt3(metrics.at("mean_source_generated_token_count_normalized_restoration")), size: 12.5pt)],
      [negative means the rescue moved farther from the source length],
    )
  ],
)


= What Was Tested

Phase 22 is a path-validation test, not a plain ablation test.

For each matched pair, the experiment generated:

- the source prompt on its own
- the base prompt with the early lesion only
- the base prompt with the same lesion plus a downstream rescue

The practical question is:

- after damaging the early Leader signal, can a downstream rescue restore source-like behavior?

In this report:

- the #emph[baseline] is the lesion-only condition
- the #emph[intervention] is lesion plus rescue

That matters because a positive result would mean:

- the later rescue site still carries enough of the causal path to recover behavior after the earlier lesion


= Methodology In Plain English

`pair mode` here is just how matched examples are defined.

For `denoise`:

- the base row is the lower-valued member of the pair on `#sample.at("pair_metric")`
- the source row is the higher-valued member

`project_out` means:

- remove the selected Leader coefficients at the early layer

`swap_components` means:

- capture the source row
- average over the market span
- take the selected source-side coefficients
- write those coefficients into the base row at the rescue layer

So this is a specific directional test:

- break the early signal
- then try to restore the later signal


= Tool Surface Validity

#table(
  columns: (2.2fr, 1fr),
  align: (left, center),
  table.hline(stroke: 1pt),
  table.header([*Measure*], [*Count*]),
  table.hline(stroke: 0.5pt),
  [Lesion rows with parsed tool calls], [#raw(str(tool_surface.at("baseline_tool_rows")))],
  [Rescue rows with parsed tool calls], [#raw(str(tool_surface.at("intervention_tool_rows")))],
  [Source rows with parsed tool calls], [#raw(str(tool_surface.at("source_tool_rows")))],
  [Rows with parsed tool calls in all three conditions], [#raw(str(tool_surface.at("all_three_tool_rows")))],
  [Source rows that hit `max_tokens = 15000`], [#raw(str(tool_surface.at("source_length_cap_rows")))],
)

This is the key caveat for Phase 22:

- the patch path itself worked
- but the full-scale behavior surface did #emph[not] reliably reach parsed tool calls
- that means the action-choice metrics are only weakly interpretable at this scale


= Lesion Versus Rescue

#table(
  columns: (2.3fr, 1fr, 1.3fr),
  align: (left, center, left),
  table.hline(stroke: 1pt),
  table.header([*Measure*], [*Lesion → Rescue*], [*How to read it*]),
  table.hline(stroke: 0.5pt),
  [Source tool-name match], [#pair_fmt(metrics.at("source_tool_name_match_rate_baseline"), metrics.at("source_tool_name_match_rate_intervention"))], [Did the rescue move the chosen tool toward the source tool?],
  [Source tool-token match], [#pair_fmt(metrics.at("source_tool_token_match_rate_baseline"), metrics.at("source_tool_token_match_rate_intervention"))], [Did the rescue move the chosen token toward the source token?],
  [Tool-token restoration rate], [#fmt3(metrics.at("source_tool_token_restoration_rate"))], [Higher is better on rows that needed fixing],
  [Tool-token backfire rate], [#fmt3(metrics.at("source_tool_token_backfire_rate"))], [Lower is better on rows that were already correct],
  [Mean source spend gap], [#pair_fmt(metrics.at("mean_source_tool_spend_pct_delta_baseline"), metrics.at("mean_source_tool_spend_pct_delta_intervention"))], [Lower is better],
  [Spend improvement rate], [#fmt3(metrics.at("source_tool_spend_pct_improvement_rate"))], [Share of spend-mismatched rows that moved closer],
  [Mean source length gap], [#pair_fmt(metrics.at("mean_source_generated_token_count_delta_baseline"), metrics.at("mean_source_generated_token_count_delta_intervention"))], [Lower is better],
  [Mean normalized length restoration], [#fmt3(metrics.at("mean_source_generated_token_count_normalized_restoration"))], [Negative means the rescue moved farther away],
)


= Restorable Rows And Backfires

#table(
  columns: (1.7fr, 1.8fr),
  align: (left, left),
  table.hline(stroke: 1pt),
  table.header([*Metric*], [*Counts*]),
  table.hline(stroke: 0.5pt),
  [Tool name], [restorable #raw(str(counts.at("tool_name_restorable_count"))); restored #raw(str(counts.at("tool_name_restored_count"))); backfires #raw(str(counts.at("tool_name_backfire_count"))) / #raw(str(counts.at("tool_name_backfire_pool")))],
  [Tool token], [restorable #raw(str(counts.at("tool_token_restorable_count"))); restored #raw(str(counts.at("tool_token_restored_count"))); backfires #raw(str(counts.at("tool_token_backfire_count"))) / #raw(str(counts.at("tool_token_backfire_pool")))],
  [Spend pct], [restorable #raw(str(counts.at("spend_restorable_count"))); improved #raw(str(counts.at("spend_improved_count"))); backfires #raw(str(counts.at("spend_backfire_count"))) / #raw(str(counts.at("spend_backfire_pool")))],
  [Generated tokens], [restorable #raw(str(counts.at("generated_restorable_count"))); improved #raw(str(counts.at("generated_improved_count"))); backfires #raw(str(counts.at("generated_backfire_count"))) / #raw(str(counts.at("generated_backfire_pool")))],
)

This is the most useful operational readout #emph[when the tool surface is valid]:

- restoration asks whether rescue helps on rows that the lesion got wrong
- backfire asks whether rescue hurts rows that the lesion already had right


= Mechanical Validity

#table(
  columns: (1.8fr, 1fr),
  align: (left, center),
  table.hline(stroke: 1pt),
  table.header([*Measure*], [*Value*]),
  table.hline(stroke: 0.5pt),
  [Rows with patch stats], [#raw(str(metrics.at("rows_with_patch_stats")))],
  [Patch applied rate], [#fmt3(metrics.at("patch_applied_rate"))],
  [Patch skipped rate], [#fmt3(metrics.at("patch_skipped_rate"))],
)

Mechanically, this section answers whether the intervention path actually fired. A positive path-validation result only matters if the rescue was really applied.


= Methodology Scorecard

#table(
  columns: (2.8fr, 0.7fr),
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

The main methodological upgrade over Phase 21 is that this is not only a restoration test. It is a #emph[path validation] test: early lesion first, downstream rescue second.


= Interpretation

The most important question is not whether the rescue changes behavior at all. The question is whether it changes behavior in the #emph[source-consistent direction] after the lesion has already damaged the early signal.

So the strongest positive pattern would be:

- tool-token restoration meaningfully above zero
- tool-token backfire clearly lower than restoration
- spend moving closer to source
- patch application mechanically clean

But before reading those metrics causally, one precondition has to hold:

- lesion, rescue, and source all have to reach comparable parsed tool calls often enough to make the action surface observable

The weakest pattern would be:

- restoration near zero
- backfire comparable to or larger than restoration
- no improvement in spend or source-match metrics

If the tool surface is sparse or truncated, the safer interpretation is narrower:

- the current full-scale setup is not yet a valid path-validation benchmark for action choice
- the patching mechanics can still be correct while the readout is not strong enough
