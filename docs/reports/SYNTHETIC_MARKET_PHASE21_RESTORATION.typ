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

#let summary = json("../../data/report_assets/synthetic_market_phase21_restoration/summary.json")
#let sample = summary.at("sample")
#let methodology = summary.at("methodology")
#let overall = summary.at("overall")
#let axes = summary.at("axes")
#let leader = axes.at("leader")
#let dispersion = axes.at("dispersion")
#let leader_metrics = leader.at("metrics")
#let dispersion_metrics = dispersion.at("metrics")
#let leader_counts = leader.at("counts")
#let dispersion_counts = dispersion.at("counts")

#let fmt3(x) = {
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
  #text(size: 22pt, weight: "bold")[Synthetic Market Phase 21]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    Source-driven restoration across two axes on the compiled patch path: Leader (`L4`, top-4 components)
    and Dispersion (`L35`, top-4 components). Each axis uses `48` matched pairs, separately generated source
    behaviors, and a `swap_components` intervention that inserts source-side coefficients into the base prompt.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[#summary.at("date")]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[ROWS]\ #text(size: 9pt)[#raw(str(sample.at("count_per_axis"))) per axis]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[PATCH]\ #text(size: 9pt)[`swap_components`, source-driven donor means]],
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
      [LEADER TOOL-TOKEN RESTORE],
      [#mono(fmt3(leader_metrics.at("source_tool_token_restoration_rate")), size: 12.5pt)],
      [#raw(str(leader_counts.at("tool_token_restored_count"))) of #raw(str(leader_counts.at("tool_token_restorable_count"))) restorable rows],
    )
  ],
  [
    #summary-card(
      [DISPERSION TOOL-TOKEN RESTORE],
      [#mono(fmt3(dispersion_metrics.at("source_tool_token_restoration_rate")), size: 12.5pt)],
      [#raw(str(dispersion_counts.at("tool_token_restored_count"))) of #raw(str(dispersion_counts.at("tool_token_restorable_count"))) restorable rows],
    )
  ],
  [
    #summary-card(
      [LEADER LENGTH RESTORATION],
      [#mono(fmt3(leader_metrics.at("mean_source_generated_token_count_normalized_restoration")), size: 12.5pt)],
      [negative means the patched response moved away from the source length],
    )
  ],
  [
    #summary-card(
      [DISPERSION LENGTH RESTORATION],
      [#mono(fmt3(dispersion_metrics.at("mean_source_generated_token_count_normalized_restoration")), size: 12.5pt)],
      [much more negative than Leader],
    )
  ],
)


= What Was Tested

For each matched pair, the run produced three behaviors:

- the source prompt on its own
- the base prompt on its own
- the base prompt with a `swap_components` patch

The question is simple:

- did the patch make the base prompt behave more like its matched source prompt?

This report uses `denoise` pairs. In plain English, that means:

- the base row is the lower-valued member of the pair on the chosen metric
- the source row is the higher-valued member of the pair

Here the chosen metric depends on the axis:

- Leader uses `vol_1h_max`
- Dispersion uses `pct_1h_mad`

`swap_components` also has a plain-English meaning here:

- it does #emph[not] copy every hidden-state token from source to base
- it captures the source row, averages over the market span, and extracts the selected coefficients
- it then writes those source-side coefficients into the base row over the base market span

One operational note matters for interpretation:

- Leader was run earlier at `batch_size = 8`
- Dispersion was run later at `batch_size = 32`
- both still used the same compiled non-eager custom-op patch path


= Axis Comparison

#table(
  columns: (2.2fr, 1fr, 1fr, 1.25fr),
  align: (left, center, center, left),
  table.hline(stroke: 1pt),
  table.header([*Measure*], [*Leader*], [*Dispersion*], [*How to read it*]),
  table.hline(stroke: 0.5pt),
  [Source tool-name match], [#pair_fmt(leader_metrics.at("source_tool_name_match_rate_baseline"), leader_metrics.at("source_tool_name_match_rate_intervention"))], [#pair_fmt(dispersion_metrics.at("source_tool_name_match_rate_baseline"), dispersion_metrics.at("source_tool_name_match_rate_intervention"))], [Dispersion gets worse overall; Leader was already saturated],
  [Source tool-token match], [#pair_fmt(leader_metrics.at("source_tool_token_match_rate_baseline"), leader_metrics.at("source_tool_token_match_rate_intervention"))], [#pair_fmt(dispersion_metrics.at("source_tool_token_match_rate_baseline"), dispersion_metrics.at("source_tool_token_match_rate_intervention"))], [Leader improves; Dispersion gets worse],
  [Tool-token restoration rate], [#fmt3(leader_metrics.at("source_tool_token_restoration_rate"))], [#fmt3(dispersion_metrics.at("source_tool_token_restoration_rate"))], [Higher is better on rows that needed fixing],
  [Tool-token backfire rate], [#fmt3(leader_metrics.at("source_tool_token_backfire_rate"))], [#fmt3(dispersion_metrics.at("source_tool_token_backfire_rate"))], [Lower is better on rows that were already correct],
  [Spend improvement rate], [#fmt3(leader_metrics.at("source_tool_spend_pct_improvement_rate"))], [#fmt3(dispersion_metrics.at("source_tool_spend_pct_improvement_rate"))], [Dispersion looks perfect, but only on one restorable row],
  [Generated-token improvement rate], [#fmt3(leader_metrics.at("source_generated_token_count_improvement_rate"))], [#fmt3(dispersion_metrics.at("source_generated_token_count_improvement_rate"))], [Both are weak; Leader is less bad],
  [Mean normalized length restoration], [#fmt3(leader_metrics.at("mean_source_generated_token_count_normalized_restoration"))], [#fmt3(dispersion_metrics.at("mean_source_generated_token_count_normalized_restoration"))], [Negative means the patch moved away from the source],
)


= Restorable Rows And Backfires

#table(
  columns: (1.6fr, 1.6fr, 1.6fr),
  align: (left, left, left),
  table.hline(stroke: 1pt),
  table.header([*Metric*], [*Leader*], [*Dispersion*]),
  table.hline(stroke: 0.5pt),
  [Tool name], [restorable #raw(str(leader_counts.at("tool_name_restorable_count"))); restored #raw(str(leader_counts.at("tool_name_restored_count"))); backfires #raw(str(leader_counts.at("tool_name_backfire_count"))) / #raw(str(leader_counts.at("tool_name_backfire_pool")))], [restorable #raw(str(dispersion_counts.at("tool_name_restorable_count"))); restored #raw(str(dispersion_counts.at("tool_name_restored_count"))); backfires #raw(str(dispersion_counts.at("tool_name_backfire_count"))) / #raw(str(dispersion_counts.at("tool_name_backfire_pool")))],
  [Tool token], [restorable #raw(str(leader_counts.at("tool_token_restorable_count"))); restored #raw(str(leader_counts.at("tool_token_restored_count"))); backfires #raw(str(leader_counts.at("tool_token_backfire_count"))) / #raw(str(leader_counts.at("tool_token_backfire_pool")))], [restorable #raw(str(dispersion_counts.at("tool_token_restorable_count"))); restored #raw(str(dispersion_counts.at("tool_token_restored_count"))); backfires #raw(str(dispersion_counts.at("tool_token_backfire_count"))) / #raw(str(dispersion_counts.at("tool_token_backfire_pool")))],
  [Spend pct], [restorable #raw(str(leader_counts.at("spend_restorable_count"))); improved #raw(str(leader_counts.at("spend_improved_count"))); backfires #raw(str(leader_counts.at("spend_backfire_count"))) / #raw(str(leader_counts.at("spend_backfire_pool")))], [restorable #raw(str(dispersion_counts.at("spend_restorable_count"))); improved #raw(str(dispersion_counts.at("spend_improved_count"))); backfires #raw(str(dispersion_counts.at("spend_backfire_count"))) / #raw(str(dispersion_counts.at("spend_backfire_pool")))],
  [Generated tokens], [restorable #raw(str(leader_counts.at("generated_restorable_count"))); improved #raw(str(leader_counts.at("generated_improved_count"))); backfires #raw(str(leader_counts.at("generated_backfire_count"))) / #raw(str(leader_counts.at("generated_backfire_pool")))], [restorable #raw(str(dispersion_counts.at("generated_restorable_count"))); improved #raw(str(dispersion_counts.at("generated_improved_count"))); backfires #raw(str(dispersion_counts.at("generated_backfire_count"))) / #raw(str(dispersion_counts.at("generated_backfire_pool")))],
)

This table is the clearest practical summary:

- Leader helps on tool-token choice more often than it hurts
- Dispersion hurts tool-token choice almost as often as it helps
- Dispersion spend looks good only because there was just one spend mismatch to fix
- neither axis gives a clean length-restoration story


= Mechanical Validity

#table(
  columns: (1.8fr, 1fr, 1fr),
  align: (left, center, center),
  table.hline(stroke: 1pt),
  table.header([*Measure*], [*Leader*], [*Dispersion*]),
  table.hline(stroke: 0.5pt),
  [Rows with patch stats], [#raw(str(leader_metrics.at("rows_with_patch_stats")))], [#raw(str(dispersion_metrics.at("rows_with_patch_stats")))],
  [Patch applied rate], [#fmt3(leader_metrics.at("patch_applied_rate"))], [#fmt3(dispersion_metrics.at("patch_applied_rate"))],
  [Patch skipped rate], [#fmt3(leader_metrics.at("patch_skipped_rate"))], [#fmt3(dispersion_metrics.at("patch_skipped_rate"))],
)

Mechanically, both runs are clean. Every row has patch diagnostics and nothing was skipped.


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

The main methodological upgrade over Phase 20 is that Phase 21 uses #emph[generated source behaviors] and a #emph[source-driven swap]. That makes this a real restoration test, not only a disruption test.


= Interpretation

Leader and Dispersion do not behave the same way.

Leader is the better causal candidate:

- it improves source match on the action surface we care about most
- tool-token restoration is materially above backfire
- spend moves toward the source more often than not

Dispersion is weaker:

- source match on tool name and tool token gets worse overall
- tool-token restoration is small and backfire is high
- the one clean positive is spend, but that result rests on a one-row restorable subset

The combined takeaway is not “restoration failed.” It is more specific:

- the market representation is not uniform
- Leader looks more like a usable causal handle on action choice
- Dispersion may matter for a narrower calibration-like quantity, but not for clean action restoration
- neither axis yet restores whole-response length cleanly


= Next Step

The next experiment should push the restoration story where it is currently strongest:

- keep `batch_size = 32` as the default going forward unless a run errors
- extend the Leader restoration surface beyond `denoise` to other pairing directions
- keep Dispersion in the suite, but treat it as a contrast case rather than the primary success case
- if possible, move from span-mean coefficient swapping toward a more exact source-activation transplant

If Leader keeps restoring tool-token choice under those stronger tests, the causal claim becomes materially harder to dismiss.
