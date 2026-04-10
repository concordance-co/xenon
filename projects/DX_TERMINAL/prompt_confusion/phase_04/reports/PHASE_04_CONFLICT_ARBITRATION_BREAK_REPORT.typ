#set page(
  paper: "us-letter",
  margin: (top: 2.4cm, bottom: 2.4cm, left: 2.6cm, right: 2.6cm),
  numbering: "1",
  number-align: right,
)
#set text(font: "Georgia", size: 10.5pt)
#set par(justify: true, leading: 0.7em)
#set heading(numbering: none)

#let ink = rgb("#182028")
#let muted = rgb("#5A6772")
#let accent = rgb("#B33A2A")
#let soft = rgb("#F7F2EF")
#let rule = rgb("#D7DEE3")
#let lift = rgb("#EAF4EE")

#show heading.where(level: 1): it => {
  set text(size: 13pt, weight: "bold", fill: ink)
  v(1.1em)
  it
  v(0.35em)
}

#show heading.where(level: 2): it => {
  set text(size: 11pt, weight: "bold", fill: ink)
  v(0.8em)
  it
  v(0.25em)
}

#let stat(label, value, note) = block(
  fill: white,
  stroke: (paint: rule, thickness: 0.7pt),
  radius: 10pt,
  inset: 12pt,
  width: 100%,
)[
  #text(size: 8pt, fill: muted, weight: "bold")[#label]
  #v(4pt)
  #text(size: 18pt, fill: ink, weight: "bold")[#value]
  #v(4pt)
  #text(size: 8.8pt, fill: muted)[#note]
]

#align(left)[
  #text(size: 9pt, fill: accent, tracking: 0.08em, weight: "medium")[XENON INTERPRETABILITY]
  #v(0.3em)
  #text(size: 22pt, weight: "bold", fill: ink)[Prompt Confusion Phase 04 Arbitration Break Report]
  #v(0.4em)
  #text(size: 11pt, fill: muted)[
    Checkpoint memo for the first full-sequence section-attribution and causal-patching pass on April 10, 2026.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.4pt + ink)
]

#v(1em)

#block(
  width: 100%,
  inset: (left: 14pt, top: 12pt, bottom: 12pt, right: 12pt),
  stroke: (left: 3pt + accent, top: none, right: none, bottom: none),
  fill: soft,
)[
  #text(size: 7.5pt, fill: accent, weight: "bold", tracking: 0.08em)[MAIN READ]
  #v(0.3em)
  #text(size: 12.5pt, fill: ink, weight: "medium")[
    The full-sequence arbitration pass found a real conflict-side signal inside the prompt, with the cleanest policy-source readout in the `SETTINGS` section at layer 24. The first causal generation pass did not validate yet because the harness returned reasoning-style `<think>` completions instead of parseable JSON actions.
  ]
]

= Run Summary

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 8pt,
  stat([Capture run], [`7e71cf742002`], [123 conflict-resolution rows with full-sequence residual capture.]),
  stat([Section probe], [`123 rows`], [Grouped by `arbitration_group_id`.]),
  stat([Best policy readout], [`SETTINGS mean @ L24`], [`0.7647` balanced accuracy.]),
  stat([Causal sweep], [`3 conditions`], [Baseline plus signed section-direction patches.]),
)

This checkpoint uses the conflict-only publication `workflow_dataset_conflict_probe_v3_conflict_readout_side_v1` and the downloaded analysis outputs in `projects/DX_TERMINAL/prompt_confusion/phase_04/outputs/conflict_arbitration_stage1`.

= Section Attribution

The strongest policy-source readouts from the residual stream were:

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, center, center, center, center),
  inset: 6pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#E6EEF2") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

  [*Section*], [*Pooling*], [*Layer*], [*Balanced Accuracy*], [*Interpretation*],

  [`SETTINGS`], [`mean`], [`24`], [`0.7647`], [Best policy-source readout selected for direction construction.],
  [`SETTINGS`], [`eos`], [`36`], [`0.7639`], [Nearly tied prompt-end section anchor.],
  [`STRATEGY`], [`mean`], [`36`], [`0.7426`], [Clear strategy-side readout, but weaker than settings.],
  [`TASK`], [`eos`], [`36`], [`0.7157`], [Some arbitration spillover into the instruction boundary.],
)

#v(0.6em)

#block(
  fill: lift,
  stroke: (paint: rule, thickness: 0.6pt),
  radius: 10pt,
  inset: 12pt,
)[
  #text(size: 9pt, fill: muted, weight: "bold")[Interpretation]
  #v(4pt)
  #text(size: 10.2pt, fill: ink)[
    The strongest overall residual section readouts were actually `PORTFOLIO mean @ L28 = 0.7924` and `MARKET mean @ L24 = 0.7844`, which is a useful confound warning. For the arbitration-specific direction, the right choice is still the best policy-source section rather than the absolute best section, because the goal is to separate `SETTINGS` versus `STRATEGY`, not merely to decode contextual pressure.
  ]
]

#v(0.8em)

#figure(
  image("../outputs/conflict_arbitration_stage1/section_attribution/residual_section_heatmap.png", width: 100%),
  caption: [Residual grouped balanced accuracy by section, pooling target, and captured layer.]
)

= Router Note

Router section attribution did not produce a usable sweep in this pass. Only `25` rows had router tensors whose token coverage still aligned cleanly with the rendered full prompt spans. That is enough to say the router path is currently partial coverage for this full-sequence setup, but not enough to support a stable grouped section analysis.

= Causal Check

The causal patch run targeted the selected arbitration feature:

- section: `SETTINGS`
- pooling anchor: `mean`
- layer: `24`
- patch strength: `1.0`

The run completed operationally, but the behavioral outputs were not usable. All three conditions returned `neither_rate = 1.0` because the model emitted reasoning-style `<think>` completions instead of the expected JSON action object, so the row-level classifier could not map them onto `strategy` or `setting`.

#block(
  fill: soft,
  stroke: (paint: rule, thickness: 0.6pt),
  radius: 10pt,
  inset: 12pt,
)[
  #text(size: 9pt, fill: accent, weight: "bold")[Why this is not a null result]
  #v(4pt)
  #text(size: 10.2pt, fill: ink)[
    The failed causal readout does not mean the patch direction was ineffective. The row-level artifacts show truncated reasoning traces starting with `<think>` rather than valid JSON, which means the causal harness did not yet reproduce the same non-thinking generation surface used in the earlier behavior audit. This is a generation-harness mismatch, not evidence against the arbitration feature.
  ]
]

= Local Artifacts

The current checkpoint is downloaded locally:

- `projects/DX_TERMINAL/prompt_confusion/phase_04/outputs/conflict_arbitration_stage1/section_attribution/summary.json`
- `projects/DX_TERMINAL/prompt_confusion/phase_04/outputs/conflict_arbitration_stage1/section_attribution/section_probe_results.parquet`
- `projects/DX_TERMINAL/prompt_confusion/phase_04/outputs/conflict_arbitration_stage1/section_attribution/residual_section_heatmap.png`
- `projects/DX_TERMINAL/prompt_confusion/phase_04/outputs/conflict_arbitration_stage1/causal_check/summary.json`
- `projects/DX_TERMINAL/prompt_confusion/phase_04/outputs/conflict_arbitration_stage1/causal_check/row_level.parquet`

= Next Step

The immediate follow-up is narrow:

1. rerun the causal harness with thinking explicitly disabled so baseline generation returns parseable JSON again
2. keep the same selected arbitration feature unless the clean rerun says otherwise: `SETTINGS mean @ layer 24`
3. only after that clean rerun compare baseline versus signed patch shifts by `readout_side` and by family

This checkpoint is enough to justify the next causal iteration, but not enough to claim a successful behavioral intervention yet.

#v(2em)
#line(length: 100%, stroke: 0.5pt + rgb("#CCC"))
#v(0.3em)
#text(size: 8pt, fill: rgb("#999"))[
  Prompt Confusion Phase 04 Arbitration Break Report - generated from the full-sequence conflict capture `7e71cf742002` and the downloaded checkpoint outputs in `projects/DX_TERMINAL/prompt_confusion/phase_04/outputs/conflict_arbitration_stage1`.
]
