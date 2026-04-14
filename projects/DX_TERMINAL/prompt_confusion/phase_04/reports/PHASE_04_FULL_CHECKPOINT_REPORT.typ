#set page(
  paper: "us-letter",
  margin: (top: 2.35cm, bottom: 2.35cm, left: 2.55cm, right: 2.55cm),
  numbering: "1",
  number-align: right,
)
#set text(font: "Georgia", size: 10.4pt)
#set par(justify: true, leading: 0.68em)
#set heading(numbering: none)

#let ink = rgb("#182028")
#let muted = rgb("#5A6772")
#let accent = rgb("#B33A2A")
#let soft = rgb("#F7F2EF")
#let rule = rgb("#D7DEE3")
#let lift = rgb("#EAF4EE")
#let mist = rgb("#EEF4F8")

#show heading.where(level: 1): it => {
  set text(size: 13pt, weight: "bold", fill: ink)
  v(1.2em)
  it
  v(0.4em)
}

#show heading.where(level: 2): it => {
  set text(size: 11pt, weight: "bold", fill: ink)
  v(0.9em)
  it
  v(0.3em)
}

#show figure.caption: set text(size: 8.4pt, fill: muted)

#let stat(label, value, note, tone: white) = block(
  fill: tone,
  stroke: (paint: rule, thickness: 0.7pt),
  radius: 10pt,
  inset: 12pt,
  width: 100%,
)[
  #text(size: 8pt, fill: muted, weight: "bold")[#label]
  #v(4pt)
  #text(size: 18pt, fill: ink, weight: "bold")[#value]
  #v(4pt)
  #text(size: 8.7pt, fill: muted)[#note]
]

#let callout(tag, body, fill-color: soft, tag-color: accent) = block(
  width: 100%,
  inset: (left: 14pt, top: 12pt, bottom: 12pt, right: 12pt),
  stroke: (left: 3pt + tag-color, top: none, right: none, bottom: none),
  fill: fill-color,
)[
  #text(size: 7.5pt, fill: tag-color, weight: "bold", tracking: 0.08em)[#tag]
  #v(0.3em)
  #text(size: 12.3pt, fill: ink, weight: "medium")[#body]
]

// ── Title Block ──────────────────────────────────────────────

#align(left)[
  #text(size: 9pt, fill: accent, tracking: 0.08em, weight: "medium")[XENON INTERPRETABILITY]
  #v(0.3em)
  #text(size: 22pt, weight: "bold", fill: ink)[Prompt Confusion Phase 04 Full Checkpoint]
  #v(0.5em)
  #text(size: 11pt, fill: muted)[
    April 10, 2026 --- behavior, conflict probing, arbitration readouts, exploratory PCA, section attribution, and the first causal follow-up.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.4pt + ink)
]

#v(1.2em)

// ── Top-Line Summary ─────────────────────────────────────────

#callout(
  [BOTTOM LINE],
  [
    Phase 04 is mechanistically usable. We can reliably detect _whether_ a prompt contains a conflict (peak accuracy 92%), and we have a weaker but real signal for _which side is winning_ (peak accuracy 71%). A plausible causal intervention target exists at `SETTINGS mean @ layer 24`, but the first causal test was invalid --- the model emitted reasoning traces instead of JSON, so we have no causal result yet.
  ],
)

#v(0.6em)

#block(
  fill: mist,
  stroke: (paint: rule, thickness: 0.6pt),
  radius: 10pt,
  inset: 14pt,
)[
  #text(size: 9pt, fill: ink, weight: "bold")[Plain-language summary]
  #v(6pt)
  #text(size: 10.2pt, fill: ink)[
    We gave Qwen a batch of 288 synthetic trading prompts --- half with internally consistent instructions (aligned), half where the `SETTINGS` and `STRATEGY` sections deliberately contradict each other (conflict). Three things came back clean:

    1. *Behavior is usable.* Every prompt produced valid structured output. Conflict prompts behave differently from aligned ones, and they don't all collapse to the same answer --- exactly what we need for probing.

    2. *The model knows when it's conflicted.* A linear probe on the residual stream can tell conflict from aligned prompts at 92% accuracy (layer 36). This is a strong Level-2 representational finding.

    3. *The model partially knows which side is winning.* On conflict-only rows, a probe can decode whether the model is favoring `SETTINGS` or `STRATEGY` at 71% accuracy (layer 20). Weaker, but real.

    The one thing that _didn't_ work: we tried pushing the `SETTINGS` direction into the residual stream to see if it steers the output. All outputs came back invalid because the model started emitting `<think>` traces. That's a harness bug, not a negative result. Next step is rerunning with thinking disabled.
  ]
]

#v(1.4em)

// ── Run Summary ──────────────────────────────────────────────

= Run Summary

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 8pt,
  stat([Behavior], [`16474bce`], [288 / 288 valid structured outputs.]),
  stat([Conflict Probe], [`62fc89fd`], [Aligned vs. conflict residual probe.], tone: mist),
  stat([Arbitration Probe], [`6924c2b3`], [Conflict-only: strategy vs. setting.], tone: lift),
  stat([Attribution], [`7e71cf74`], [123 conflict rows, section-local analysis.]),
)

#v(0.3em)

All artifacts live under `phase_04/outputs`:

#text(size: 9pt, fill: muted)[
  `behavior_audit_16474bceae4e.json` · `analysis_full_prompt_eos_probe/62fc89fd861b` · `analysis_full_prompt_eos_pca/f4c70fcbac6d` · `analysis_conflict_readout_residual/6924c2b36f43` · `analysis_conflict_readout_pca/331de685cbc1` · `analysis_conflict_readout_router/4aeb882aea9d` · `conflict_arbitration_stage1`
]

#v(0.8em)

// ── Behavioral Sanity ────────────────────────────────────────

= 1. Behavioral Sanity

The behavior audit checks whether the synthetic prompts produce usable, non-collapsed differences --- not benchmark accuracy. We need conflict rows that behave differently from aligned rows, and conflict rows that don't all resolve the same way.

#v(0.4em)

#table(
  columns: (1.15fr, auto, auto, auto, auto, auto),
  align: (left, center, center, center, center, center),
  inset: 7pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#E6EEF2") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

  [*Split*], [*Rows*], [*Valid*], [*Exact Match*], [*→ Strategy*], [*→ Setting*],

  [`overall`], [`288`], [`100%`], [`51.4%`], [`66.7%`], [`24.0%`],
  [`aligned`], [`144`], [`100%`], [`70.8%`], [`83.3%`], [`12.5%`],
  [`conflict`], [`144`], [`100%`], [`31.9%`], [`50.0%`], [`35.4%`],
)

#v(0.4em)

#block(
  fill: lift,
  stroke: (paint: rule, thickness: 0.6pt),
  radius: 10pt,
  inset: 12pt,
)[
  #text(size: 9pt, fill: muted, weight: "bold")[Key takeaway]
  #v(4pt)
  #text(size: 10.2pt, fill: ink)[
    Every row parsed. Aligned and conflict rows separate behaviorally. Conflict rows show a `72 strategy / 51 setting / 21 neither` readout mix --- enough diversity for arbitration analysis.
  ]
]

#v(0.6em)

Conflict rows broken out by family:

#v(0.3em)

#table(
  columns: (1.7fr, auto, auto, auto, auto),
  align: (left, center, center, center, center),
  inset: 7pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#E6EEF2") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

  [*Family*], [*Rows*], [*Strategy*], [*Setting*], [*Neither*],

  [`activity_force_observe`], [`36`], [`31`], [`5`], [`0`],
  [`activity_force_trade`], [`36`], [`15`], [`21`], [`0`],
  [`trade_size_force_large`], [`36`], [`3`], [`18`], [`15`],
  [`trade_size_force_small`], [`36`], [`23`], [`7`], [`6`],
)

#v(0.4em)

The softest family (`activity_force_trade`) still moves with contextual pressure rather than collapsing --- its `balanced` bucket goes `3 strategy / 9 setting`, while `setting_favored` goes `12 / 0` and `strategy_favored` goes `0 / 12`. Arbitration responds to pressure, not noise.

#v(0.8em)

// ── Conflict Probe ───────────────────────────────────────────

= 2. Conflict Detection (Aligned vs. Conflict)

Can we read _whether a conflict exists_ from the residual stream? Yes --- strongly.

#v(0.4em)

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (center, center, center, center, center),
  inset: 7pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#E6EEF2") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

  [*Layer*], [*Balanced Accuracy*], [*Mean*], [*Std Dev*], [*Selectivity*],

  [`0`], [`0.784`], [`0.784`], [`0.038`], [`0.310`],
  [`20`], [`0.788`], [`0.788`], [`0.024`], [`0.281`],
  [`24`], [`0.864`], [`0.864`], [`0.037`], [`0.399`],
  [`28`], [`0.861`], [`0.861`], [`0.021`], [`0.368`],
  [*`36`*], [*`0.920`*], [*`0.920`*], [*`0.042`*], [*`0.417`*],
  [`44`], [`0.837`], [`0.837`], [`0.072`], [`0.354`],
)

#v(0.4em)

#block(
  fill: soft,
  stroke: (paint: rule, thickness: 0.6pt),
  radius: 10pt,
  inset: 12pt,
)[
  #text(size: 9pt, fill: accent, weight: "bold")[Level-2 result]
  #v(4pt)
  #text(size: 10.2pt, fill: ink)[
    Strong linear readout of conflict state from mid-to-late residual stream, peaking at layer 36 with 92% grouped balanced accuracy. The model builds an increasingly clear internal representation of "these instructions contradict each other" as it processes the prompt.
  ]
]

#v(0.5em)

#figure(
  image("../outputs/analysis_full_prompt_eos_pca/f4c70fcbac6d/pca_layer28_workflow_label.png", width: 100%),
  caption: [PCA of conflict vs. aligned residual activations at layer 28. Descriptive only, but consistent with the strong linear probe result.]
)

#v(0.8em)

// ── Arbitration ──────────────────────────────────────────────

= 3. Arbitration (Which Side Is Winning?)

The harder question: on conflict-only rows, can we tell whether the model is favoring `SETTINGS` or `STRATEGY`?

#v(0.4em)

#table(
  columns: (1.2fr, auto, auto, auto, auto),
  align: (left, center, center, center, center),
  inset: 7pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#E6EEF2") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

  [*Stream*], [*Best Layer*], [*Balanced Accuracy*], [*Std Dev*], [*Note*],

  [*residual*], [*`20`*], [*`0.712`*], [*`0.062`*], [Peak arbitration readout.],
  [`residual`], [`24`], [`0.708`], [`0.083`], [Nearly tied.],
  [`residual`], [`44`], [`0.700`], [`0.079`], [Late residual still carries signal.],
  [`router`], [`16`], [`0.648`], [`0.069`], [Best router layer.],
  [`router`], [`28`], [`0.647`], [`0.087`], [Comparable.],
  [`router`], [`44`], [`0.622`], [`0.117`], [Above baseline, but noisy.],
)

#v(0.4em)

#block(
  fill: lift,
  stroke: (paint: rule, thickness: 0.6pt),
  radius: 10pt,
  inset: 12pt,
)[
  #text(size: 9pt, fill: muted, weight: "bold")[Key takeaway]
  #v(4pt)
  #text(size: 10.2pt, fill: ink)[
    Arbitration is materially weaker than conflict detection (71% vs. 92%) --- expected, since it's a harder question. But it's real: the residual stream carries genuine information about which side of the conflict is winning. Router signal is weaker (~65%) but enough to keep in the follow-up stack.
  ]
]

#v(0.5em)

#figure(
  image("../outputs/analysis_conflict_readout_pca/331de685cbc1/pca_layer28_workflow_label.png", width: 100%),
  caption: [PCA of strategy vs. setting residual activations at layer 28 (conflict rows only).]
)

#v(0.8em)

// ── Section Attribution ──────────────────────────────────────

= 4. Section Attribution and Causal Status

Where in the prompt is the arbitration signal coming from? The section-attribution pass probed each prompt section independently.

#v(0.4em)

#table(
  columns: (1.15fr, auto, auto, auto, auto),
  align: (left, center, center, center, center),
  inset: 7pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#E6EEF2") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

  [*Section*], [*Pooling*], [*Layer*], [*Balanced Acc.*], [*Note*],

  [`PORTFOLIO`], [`mean`], [`28`], [`0.792`], [Highest absolute, but confound risk.],
  [`MARKET`], [`mean`], [`24`], [`0.784`], [Strong contextual-pressure confound.],
  [*`SETTINGS`*], [*`mean`*], [*`24`*], [*`0.765`*], [*Best policy-source readout → selected target.*],
  [`SETTINGS`], [`eos`], [`36`], [`0.764`], [Nearly tied.],
  [`STRATEGY`], [`mean`], [`36`], [`0.743`], [Clear but slightly weaker.],
)

#v(0.4em)

Selected intervention target:

#block(
  fill: mist,
  stroke: (paint: rule, thickness: 0.6pt),
  radius: 10pt,
  inset: 12pt,
)[
  #text(size: 10.2pt, fill: ink)[
    *Section:* `SETTINGS` · *Pooling:* `mean` · *Layer:* `24` · *Direction norm:* `1.4996`
  ]
]

#v(0.5em)

#figure(
  image("../outputs/conflict_arbitration_stage1/section_attribution/residual_section_heatmap.png", width: 100%),
  caption: [Grouped balanced accuracy by section, pooling, and layer for the conflict-only attribution pass.]
)

#v(0.5em)

== Causal Test: Invalid

#block(
  fill: soft,
  stroke: (paint: rule, thickness: 0.6pt),
  radius: 10pt,
  inset: 12pt,
)[
  #text(size: 9pt, fill: accent, weight: "bold")[Not a null result --- a harness failure]
  #v(4pt)
  #text(size: 10.2pt, fill: ink)[
    All three causal conditions (baseline, `setting_push`, `strategy_push`) returned `neither_rate = 1.0`. But the raw outputs show the model emitting `<think> Okay, let's break this down ...` instead of the expected JSON. The intervention may or may not have worked --- we can't tell, because the output surface broke. Fix: rerun with thinking disabled.
  ]
]

#v(0.5em)

Two additional caveats:

- Only 25 rows had usable full-span router coverage, so the router section sweep is not yet interpretable.
- `PORTFOLIO` and `MARKET` carry strong arbitration-adjacent signal. Confound control matters in follow-up work.

#v(1.2em)

// ── Status Grid ──────────────────────────────────────────────

= What We Know vs. What We Don't

#v(0.3em)

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  block(
    fill: lift,
    stroke: (paint: rule, thickness: 0.6pt),
    radius: 10pt,
    inset: 14pt,
  )[
    #text(size: 8.5pt, fill: rgb("#2D7A4F"), weight: "bold", tracking: 0.06em)[ESTABLISHED]
    #v(6pt)
    #text(size: 10.1pt, fill: ink)[
      - Synthetic prompts are behaviorally usable
      - Conflict is strongly represented in the residual stream (92%)
      - Arbitration is linearly decodable above baseline (71%)
      - Plausible causal target: `SETTINGS mean @ L24`
    ]
  ],
  block(
    fill: soft,
    stroke: (paint: rule, thickness: 0.6pt),
    radius: 10pt,
    inset: 14pt,
  )[
    #text(size: 8.5pt, fill: accent, weight: "bold", tracking: 0.06em)[OPEN]
    #v(6pt)
    #text(size: 10.1pt, fill: ink)[
      - No valid causal result yet (harness bug)
      - Router section attribution too sparse (N=25)
      - `PORTFOLIO` / `MARKET` confound not ruled out
      - No attention or gradient-based follow-up yet
    ]
  ],
)

#v(1.2em)

// ── Next Steps ───────────────────────────────────────────────

= Next Steps

+ Rerun the causal harness with thinking disabled so outputs match the JSON-only behavioral surface.
+ Keep `SETTINGS mean @ layer 24` as the intervention target unless the rerun disproves it.
+ After a valid causal result, move into attention and attribution follow-up conditioned on conflict-side labels.
+ Maintain explicit confound checks: section length, family identity, and the strong `PORTFOLIO` / `MARKET` readouts.

#v(1.2em)

// ── Artifact Index ───────────────────────────────────────────

= Local Artifact Index

#text(size: 9pt, fill: muted)[
  All paths relative to `projects/DX_TERMINAL/prompt_confusion/phase_04/outputs/`.
]

#v(0.3em)

#text(size: 9pt)[
  - `behavior_audit_16474bceae4e.json`
  - `analysis_full_prompt_eos_probe/62fc89fd861b/results.json`
  - `analysis_full_prompt_eos_pca/f4c70fcbac6d/pca_layer28_workflow_label.png`
  - `analysis_conflict_readout_residual/6924c2b36f43/results.json`
  - `analysis_conflict_readout_pca/331de685cbc1/pca_layer28_workflow_label.png`
  - `analysis_conflict_readout_router/4aeb882aea9d/results.json`
  - `conflict_arbitration_stage1/section_attribution/summary.json`
  - `conflict_arbitration_stage1/section_attribution/section_probe_results.parquet`
  - `conflict_arbitration_stage1/causal_check/summary.json`
  - `conflict_arbitration_stage1/causal_check/row_level.parquet`
]

#v(2em)
#line(length: 100%, stroke: 0.5pt + rgb("#CCC"))
#v(0.3em)
#text(size: 8pt, fill: rgb("#999"))[
  Prompt Confusion Phase 04 Full Checkpoint --- generated from local outputs under `phase_04/outputs`, including behavior run `16474bce`, conflict probe `62fc89fd`, arbitration probes `6924c2b3` and `4aeb882e`, PCA runs `f4c70fca` and `331de685`, and section-attribution checkpoint `conflict_arbitration_stage1`.
]
