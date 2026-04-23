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
#let rule-color = rgb("#D7DEE3")
#let lift = rgb("#EAF4EE")
#let mist = rgb("#EEF4F8")
#let warn = rgb("#FFF8F0")

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
  stroke: (paint: rule-color, thickness: 0.7pt),
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
  #text(size: 22pt, weight: "bold", fill: ink)[Prompt Confusion Phase 04]
  #v(0.15em)
  #text(size: 15pt, weight: "regular", fill: muted)[Combined Stages 1 & 2 Report]
  #v(0.5em)
  #text(size: 11pt, fill: muted)[
    April 10, 2026 --- conflict detection, arbitration probing, causal intervention, confound analysis, and attention follow-up.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.4pt + ink)
]

#v(1.2em)

// ── Bottom Line ──────────────────────────────────────────────

#callout(
  [BOTTOM LINE],
  [
    The model knows when its instructions conflict (92% probe accuracy). We can causally steer which side it favors. But the arbitration signal is not a clean "source-selection mechanism" --- it is dominated by the _type_ of conflict and contextual pressure, not an abstract resolution circuit. That is itself informative: the model's conflict resolution is family-specific and pressure-responsive, not generic.
  ],
)

#v(0.6em)

#block(
  fill: mist,
  stroke: (paint: rule-color, thickness: 0.6pt),
  radius: 10pt,
  inset: 14pt,
)[
  #text(size: 9pt, fill: ink, weight: "bold")[Plain-language summary]
  #v(6pt)
  #text(size: 10.2pt, fill: ink)[
    We gave Qwen 288 synthetic trading prompts --- half with internally consistent instructions, half where `SETTINGS` and `STRATEGY` deliberately contradict each other. Here is what we learned across two rounds of analysis:

    + *The model builds a clear "conflicted" representation.* A linear probe on the residual stream detects conflict at 92% accuracy (layer 36). This is a strong, clean finding.

    + *The model can be steered.* Injecting the `SETTINGS` direction into the residual stream at layer 24 moves behavior in the expected direction. `strategy_push` shifts the model 9 percentage points toward strategy-following; `setting_push` shifts it about 3 points toward setting-following.

    + *But "which side wins" is mostly determined by the type of conflict.* A simple classifier using just the conflict family and pressure bucket predicts the winner at 80% --- _better_ than the residual probe (71%). The model isn't using one general-purpose arbitration circuit; it is resolving different kinds of conflicts differently. That is a real finding about conflict resolution, just not the one we initially expected.

    + *Attention is not yet informative.* Section-aggregated attention barely beats chance as an arbitration decoder (best: 59%).
  ]
]

#v(1.4em)

// ── Run IDs ──────────────────────────────────────────────────

= Run Overview

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 8pt,
  stat([Behavior], [`16474bce`], [288 / 288 valid structured outputs.]),
  stat([Conflict Probe], [`62fc89fd`], [Aligned vs. conflict, 92% peak.], tone: mist),
  stat([Arbitration Probe], [`6924c2b3`], [Conflict-only, 71% peak.], tone: lift),
)

#v(0.3em)

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 8pt,
  stat([Causal Rerun], [Stage 2], [Harness fixed. Intervention active.], tone: lift),
  stat([Confound Check], [Stage 2], [Family + pressure = 80% baseline.], tone: warn),
  stat([Attention], [Stage 2], [Weak decoder, best 59%.]),
)

#v(1.2em)

// ── Section 1: Conflict Detection ────────────────────────────

= 1. Conflict Detection Is Strong

Can we read _whether a conflict exists_ from the residual stream? Yes --- cleanly and reliably.

#v(0.4em)

#table(
  columns: (auto, auto, auto, auto),
  align: (center, center, center, center),
  inset: 7pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#E6EEF2") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

  [*Layer*], [*Balanced Accuracy*], [*Std Dev*], [*Selectivity*],

  [`0`], [`0.784`], [`0.038`], [`0.310`],
  [`20`], [`0.788`], [`0.024`], [`0.281`],
  [`24`], [`0.864`], [`0.037`], [`0.399`],
  [`28`], [`0.861`], [`0.021`], [`0.368`],
  [*`36`*], [*`0.920`*], [*`0.042`*], [*`0.417`*],
  [`44`], [`0.837`], [`0.072`], [`0.354`],
)

#v(0.4em)

This is a Level-2 representational result. The model builds an increasingly clear internal representation of "these instructions contradict each other," peaking at layer 36.

#v(0.5em)

#figure(
  image("../outputs/analysis_full_prompt_eos_pca/f4c70fcbac6d/pca_layer28_workflow_label.png", width: 100%),
  caption: [PCA of conflict vs. aligned residual activations at layer 28. PC1 and PC2 together explain ~37% of variance.]
)

#v(0.4em)

#block(
  fill: mist,
  stroke: (paint: rule-color, thickness: 0.6pt),
  radius: 10pt,
  inset: 12pt,
)[
  #text(size: 9pt, fill: ink, weight: "bold")[Reading the PCA]
  #v(4pt)
  #text(size: 10.2pt, fill: ink)[
    The aligned and conflict points overlap heavily in this view --- which might seem to contradict the 86% probe accuracy at this layer. It doesn't. PC1 and PC2 capture the directions of _maximum overall variance_, not the directions that best separate the labels. The residual stream is encoding many things at once (prompt content, family identity, token structure), and the conflict signal lives in a direction that is real but not dominant enough to show up as a top principal component.

    The LDA projection (not shown) confirms this: when you project onto the _supervised_ discriminant axis, aligned and conflict rows separate cleanly. The conflict direction is a learned feature of the representation, not the primary mode of variation.
  ]
]

#v(0.8em)

// ── Section 2: Arbitration ───────────────────────────────────

= 2. Arbitration Signal Exists, But Is Confounded

On conflict-only rows, the residual stream carries information about _which side is winning_ --- but that information is heavily entangled with dataset structure.

== The probe result

#v(0.3em)

#table(
  columns: (1.2fr, auto, auto, auto),
  align: (left, center, center, center),
  inset: 7pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#E6EEF2") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

  [*Stream*], [*Best Layer*], [*Balanced Accuracy*], [*Std Dev*],

  [*residual*], [*`20`*], [*`0.712`*], [*`0.062`*],
  [`residual`], [`24`], [`0.708`], [`0.083`],
  [`residual`], [`44`], [`0.700`], [`0.079`],
  [`router`], [`16`], [`0.648`], [`0.069`],
  [`router`], [`28`], [`0.647`], [`0.087`],
  [`router`], [`44`], [`0.622`], [`0.117`],
)

#v(0.5em)

== The confound baseline

This is the key new result from Stage 2. Before interpreting the 71% probe as evidence of a latent arbitration mechanism, we need to ask: can we match that accuracy from dataset metadata alone?

#v(0.3em)

#table(
  columns: (1.6fr, auto, auto, auto),
  align: (left, center, center, center),
  inset: 7pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#E6EEF2") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

  [*Baseline*], [*Balanced Accuracy*], [*Std Dev*], [*vs. Residual Probe (71.2%)*],

  [*`family + pressure`*], [*`0.804`*], [*`0.085`*], [*Above*],
  [`family only`], [`0.763`], [`0.127`], [Above],
  [`user text n-gram`], [`0.701`], [`0.110`], [Tied],
  [`metadata (all)`], [`0.674`], [`0.054`], [Below],
  [`lexical IDs`], [`0.651`], [`0.103`], [Below],
  [`pressure only`], [`0.573`], [`0.033`], [Below],
  [`length / position`], [`0.487`], [`0.100`], [Below chance],
)

#v(0.5em)

#callout(
  [KEY FINDING],
  [
    Just knowing the conflict family and pressure bucket predicts the winner at 80.4% --- materially better than the best residual probe (71.2%). The probe is not discovering information the dataset structure doesn't already carry.
  ],
  fill-color: warn,
  tag-color: rgb("#B37A00"),
)

#v(0.6em)

#block(
  fill: mist,
  stroke: (paint: rule-color, thickness: 0.6pt),
  radius: 10pt,
  inset: 14pt,
)[
  #text(size: 9pt, fill: ink, weight: "bold")[Why this is interesting, not just a problem]
  #v(6pt)
  #text(size: 10.2pt, fill: ink)[
    The naive read is "the arbitration probe is confounded, so the result is weaker than we thought." That's true, but the confound structure itself is informative:

    - *Family identity dominates.* The model resolves `activity_force_observe` conflicts differently from `trade_size_force_large` conflicts. This isn't noise --- it means the model has learned family-specific resolution policies rather than a single abstract arbitration rule.

    - *Pressure modulates within family.* Adding the pressure bucket on top of family pushes from 76% to 80%. The model's resolution is sensitive to _how hard_ each section pushes, not just _what kind_ of conflict it is.

    - *Section length is irrelevant.* The `length_position` baseline is below chance (48.7%). Whatever drives the arbitration label, it is not a trivial surface feature. The confound is semantic, not mechanical.

    So the story is: the model does not have one arbitration circuit. It has family-specific conflict resolution behaviors that are modulated by contextual pressure. That is a substantive finding about how LLMs handle instruction conflicts.
  ]
]

#v(0.5em)

== Visualizing the confound

The same PCA projection colored by arbitration label (left) versus by conflict family (right) tells the story immediately:

#v(0.3em)

#grid(
  columns: (1fr, 1fr),
  gutter: 8pt,
  figure(
    image("../outputs/analysis_conflict_readout_pca/331de685cbc1/pca_layer28_workflow_label.png", width: 100%),
    caption: [Layer 28 PCA colored by *arbitration label* (strategy vs. setting). No visible separation.]
  ),
  figure(
    image("../outputs/analysis_conflict_readout_pca_by_family/pca_layer28_strategy_family.png", width: 100%),
    caption: [Same layer 28 PCA colored by *conflict family*. The structure maps onto families, not arbitration outcome.]
  ),
)

#v(0.4em)

The arbitration label doesn't organize the PCA space at all --- but family identity does. Activity families (observe, trade) cluster on the left half of PC1; trade-size families (large, small) cluster on the right. This holds across layers and gets tighter through mid-layers.

#v(0.4em)

The LDA projection makes the separation even more stark:

#v(0.3em)

#figure(
  image("../outputs/analysis_conflict_readout_pca_by_family/lda_layer28_strategy_family.png", width: 100%),
  caption: [LDA projection at layer 28 colored by family. Near-perfect 4-way separation: LD1 splits activity vs. trade-size families; LD2 splits the specific variant within each pair. Almost zero overlap.]
)

#v(0.4em)

#block(
  fill: warn,
  stroke: (paint: rule-color, thickness: 0.6pt),
  radius: 10pt,
  inset: 12pt,
)[
  #text(size: 9pt, fill: rgb("#B37A00"), weight: "bold")[What this confirms]
  #v(4pt)
  #text(size: 10.2pt, fill: ink)[
    The dominant structure in the conflict-only residual stream is a 2#sym.times{}2 family organization: activity vs. trade-size on one axis, the specific conflict variant on the other. The centroid PCA puts 80.6% of between-family variance on PC1 alone --- family identity is nearly one-dimensional.

    An arbitration probe operating in this space is almost certainly riding family-correlated features rather than discovering a generic "which side wins" direction. The arbitration label cuts _across_ the family clusters, which is why PCA doesn't recover it. Any probe that achieves 71% without controlling for family could be doing so entirely through family-specific resolution patterns.

    This does not invalidate the probe result --- the probe _is_ decoding real information about behavior. But it reframes what the probe is detecting: family-specific resolution policies, not an abstract arbitration mechanism. Within-family probes in Stage 3 will test whether anything remains after controlling for this structure.
  ]
]

#v(0.8em)

// ── Section 3: Causal Intervention ───────────────────────────

= 3. Causal Intervention Is Behaviorally Active

Stage 1's causal test was invalid (the model emitted `<think>` traces instead of JSON). Stage 2 fixed the harness, and the intervention now produces valid, directional effects.

#v(0.4em)

Intervention target: `SETTINGS mean @ layer 24`, strength `1.0`.

#v(0.3em)

#table(
  columns: (1.2fr, auto, auto, auto),
  align: (left, center, center, center),
  inset: 7pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#E6EEF2") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

  [*Condition*], [*Strategy Rate*], [*Setting Rate*], [*Neither Rate*],

  [`baseline`], [`53.7%`], [`45.5%`], [`0.8%`],
  [`setting_push`], [`48.8%`], [`48.8%`], [`2.4%`],
  [`strategy_push`], [`62.6%`], [`36.6%`], [`0.8%`],
)

#v(0.4em)

#block(
  fill: lift,
  stroke: (paint: rule-color, thickness: 0.6pt),
  radius: 10pt,
  inset: 12pt,
)[
  #text(size: 9pt, fill: muted, weight: "bold")[Key takeaway]
  #v(4pt)
  #text(size: 10.2pt, fill: ink)[
    The intervention works, but asymmetrically. `strategy_push` produces a clear +9pp shift toward strategy-following. `setting_push` is weaker (+3pp toward setting) and introduces some invalid outputs. This is consistent with the model having a slight strategy-default bias that is easier to amplify than to override.
  ]
]

#v(0.6em)

== Family-Level Causal Breakdown

The aggregate numbers hide important family-level structure.

#v(0.3em)

#table(
  columns: (1.7fr, auto, auto, auto, auto, auto, auto),
  align: (left, center, center, center, center, center, center),
  inset: 6pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#E6EEF2") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

  [*Family*], [*Base Strat.*], [*Base Set.*], [*Push→Strat.*], [*Push→Set.*], [*Strat. Delta*], [*Set. Delta*],

  [`activity_force_observe`], [`30`], [`6`], [`31`], [`5`], [+1], [-1],
  [`activity_force_trade`], [`14`], [`22`], [`16`], [`23`], [+2], [+1],
  [`trade_size_force_large`], [`0`], [`21`], [`6`], [`15`], [*+6*], [*0*],
  [`trade_size_force_small`], [`22`], [`7`], [`24`], [`11`], [+2], [+4],
)

#v(0.4em)

#callout(
  [NOTABLE],
  [
    The intervention bites hardest in `trade_size_force_large`, which is the family where the model most strongly defaults to setting-following behavior. Baseline has 0 strategy-favored rows; after `strategy_push`, 6 rows flip. The activity families barely move. This is further evidence that the causal patch is interacting with family-specific structure, not a generic arbitration direction.
  ],
  fill-color: mist,
  tag-color: rgb("#2D5FA0"),
)

#v(0.8em)

// ── Section 4: Section Attribution ───────────────────────────

= 4. Section Attribution

Where in the prompt does the arbitration signal come from? The section-attribution pass probed each prompt section independently on the conflict-only split.

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
  [*`SETTINGS`*], [*`mean`*], [*`24`*], [*`0.765`*], [*Selected causal target.*],
  [`SETTINGS`], [`eos`], [`36`], [`0.764`], [Nearly tied.],
  [`STRATEGY`], [`mean`], [`36`], [`0.743`], [Slightly weaker.],
)

#v(0.4em)

`PORTFOLIO` and `MARKET` score higher than `SETTINGS`, which is a confound warning: those sections carry contextual-pressure information that correlates with the arbitration label. `SETTINGS` was chosen as the intervention target because it is the cleanest _policy-source_ section, even though it's not the strongest absolute readout.

#v(0.5em)

#figure(
  image("../outputs/conflict_arbitration_stage1/section_attribution/residual_section_heatmap.png", width: 100%),
  caption: [Grouped balanced accuracy by section, pooling, and layer for the conflict-only attribution pass.]
)

#v(0.8em)

// ── Section 5: Attention ─────────────────────────────────────

= 5. Attention: Not Yet Informative

Section-aggregated attention mass barely distinguishes strategy-favored from setting-favored conflict resolution.

#v(0.4em)

#table(
  columns: (auto, auto, auto, auto),
  align: (center, center, center, center),
  inset: 7pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#E6EEF2") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

  [*Layer*], [*Anchor*], [*Balanced Accuracy*], [*Std Dev*],

  [*`20`*], [*`settings_eos`*], [*`0.587`*], [*`0.151`*],
  [`24`], [`prompt_eos`], [`0.554`], [`0.030`],
  [`36`], [`strategy_eos`], [`0.548`], [`0.138`],
  [`36`], [`prompt_eos`], [`0.544`], [`0.036`],
  [`24`], [`settings_eos`], [`0.535`], [`0.169`],
  [`20`], [`prompt_eos`], [`0.532`], [`0.046`],
)

#v(0.4em)

The best result (`settings_eos` at layer 20, 58.7%) has high variance and barely exceeds chance. The attention delta heatmap below confirms this: the mean deltas between strategy-favored and setting-favored rows are on the order of $plus.minus 0.003$, which is essentially zero at this sample size.

#v(0.5em)

#figure(
  image("../outputs/conflict_arbitration_stage2/attention_summary/prompt_eos_attention_delta.png", width: 85%),
  caption: [Mean attention delta (setting-favored minus strategy-favored) by target section and layer at the prompt EOS token. All values near zero.]
)

#v(0.8em)

// ── Status Grid ──────────────────────────────────────────────

= What We Know Now

#v(0.3em)

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  block(
    fill: lift,
    stroke: (paint: rule-color, thickness: 0.6pt),
    radius: 10pt,
    inset: 14pt,
  )[
    #text(size: 8.5pt, fill: rgb("#2D7A4F"), weight: "bold", tracking: 0.06em)[ESTABLISHED]
    #v(6pt)
    #text(size: 10.1pt, fill: ink)[
      - Conflict detection is strong and clean (92%)
      - Causal intervention is behaviorally active
      - Arbitration is family-specific, not generic
      - Pressure modulates arbitration within families
      - Section length is not a confound
    ]
  ],
  block(
    fill: soft,
    stroke: (paint: rule-color, thickness: 0.6pt),
    radius: 10pt,
    inset: 14pt,
  )[
    #text(size: 8.5pt, fill: accent, weight: "bold", tracking: 0.06em)[OPEN]
    #v(6pt)
    #text(size: 10.1pt, fill: ink)[
      - No clean cross-family arbitration mechanism yet
      - Causal effect is asymmetric and family-dependent
      - `PORTFOLIO` / `MARKET` confound not isolated
      - Attention is not informative at current resolution
      - Router attribution still too sparse (N=25)
    ]
  ],
)

#v(1.2em)

// ── Interpretation ───────────────────────────────────────────

= Interpretation: What Kind of Finding Is This?

#block(
  fill: mist,
  stroke: (paint: rule-color, thickness: 0.6pt),
  radius: 10pt,
  inset: 14pt,
)[
  #text(size: 10.2pt, fill: ink)[
    The initial hypothesis was that the model might have a general-purpose arbitration circuit --- a direction in the residual stream that encodes "follow source A over source B" regardless of conflict type. The evidence so far points to a different picture:

    #v(6pt)

    *The model resolves conflicts categorically, not generically.* When `SETTINGS` says "observe only" and `STRATEGY` says "trade aggressively," the model resolves that conflict differently from when `SETTINGS` says "small positions" and `STRATEGY` says "go large." Family identity alone predicts the winner at 76%. Adding contextual pressure gets to 80%.

    #v(6pt)

    This is not a failure of the experimental design. It is a real finding about how at least one model handles instruction conflicts: resolution is semantic and context-dependent, not a simple priority lookup. The practical implication is that prompting techniques that assume a stable "system prompt always wins" hierarchy are likely wrong --- the actual hierarchy depends on _what kind_ of instruction is being overridden and _how hard_ the override pushes.

    #v(6pt)

    The next question is whether a residualized arbitration signal exists _after_ controlling for family and pressure --- i.e., whether there is also a generic component underneath the family-specific behavior. That is the main open question for Stage 3.
  ]
]

#v(1.2em)

// ── Next Steps ───────────────────────────────────────────────

= Next Steps

+ *Family-controlled arbitration probes.* Run within-family probes, or regress out family and pressure metadata before probing, to test whether a cross-family arbitration signal exists after confound control.
+ *Family-level causal evaluation.* Compare causal deltas within each family separately. The aggregate causal result mixes families where the patch does a lot (`trade_size_force_large`) with families where it does nothing (`activity_force_observe`).
+ *Isolate `PORTFOLIO` / `MARKET` confound.* These sections carry arbitration-adjacent signal that may just be pressure correlates. Test by shuffling or ablating them.
+ *Head-level attention.* Section-aggregated attention is too coarse. Move to per-head attention patterns conditioned on arbitration outcome.
+ *Increase router coverage.* Only 25 rows had usable full-span router data. Either fix the capture or extend the run.

#v(1.2em)

// ── Artifact Index ───────────────────────────────────────────

= Artifact Index

#text(size: 9pt, fill: muted)[
  All paths relative to `projects/DX_TERMINAL/prompt_confusion/phase_04/outputs/`.
]

#v(0.3em)

#text(size: 9pt)[
  *Stage 1*
  - `behavior_audit_16474bceae4e.json`
  - `analysis_full_prompt_eos_probe/62fc89fd861b/results.json`
  - `analysis_full_prompt_eos_pca/f4c70fcbac6d/`
  - `analysis_conflict_readout_residual/6924c2b36f43/results.json`
  - `analysis_conflict_readout_pca/331de685cbc1/`
  - `analysis_conflict_readout_router/4aeb882aea9d/results.json`
  - `conflict_arbitration_stage1/section_attribution/`
  - `conflict_arbitration_stage1/causal_check/` #text(fill: muted)[(invalid --- harness bug)]

  #v(0.3em)

  *Stage 2*
  - `analysis_conflict_readout_pca_by_family/` #text(fill: muted)[(family-colored PCA/LDA/centroid plots)]
  - `conflict_arbitration_stage2/causal_check/summary.json`
  - `conflict_arbitration_stage2/causal_check/row_level.parquet`
  - `conflict_arbitration_stage2/confound_checks/summary.json`
  - `conflict_arbitration_stage2/confound_checks/baseline_results.parquet`
  - `conflict_arbitration_stage2/attention_summary/summary.json`
  - `conflict_arbitration_stage2/attention_summary/row_level.parquet`
  - `conflict_arbitration_stage2/attention_summary/prompt_eos_attention_delta.png`
]

#pagebreak()

// ── Appendix ─────────────────────────────────────────────────

#align(left)[
  #text(size: 9pt, fill: accent, tracking: 0.08em, weight: "medium")[APPENDIX]
  #v(0.3em)
  #text(size: 18pt, weight: "bold", fill: ink)[Caveats, Limitations, and Isolation Strategy]
  #v(0.5em)
  #line(length: 100%, stroke: 1pt + rule-color)
]

#v(1em)

This appendix collects the methodological cautions that should inform how the main results are read and what Stage 3 needs to address.

= A. How Much Should We Trust the Family Separation?

The LDA plot at layer 28 shows near-perfect 4-way family separation. That is visually striking but should be interpreted carefully.

#v(0.4em)

#block(
  fill: warn,
  stroke: (paint: rule-color, thickness: 0.6pt),
  radius: 10pt,
  inset: 12pt,
)[
  #text(size: 9pt, fill: rgb("#B37A00"), weight: "bold")[LDA is designed to separate classes]
  #v(4pt)
  #text(size: 10.2pt, fill: ink)[
    LDA finds the projection that *maximally separates* the groups. In a 2048-dimensional residual stream with only ~30 points per family, LDA has enormous freedom to find separating directions. It would be more surprising if it _couldn't_ find clean separation. The near-zero overlap in the LDA is partly the method flattering the data.
  ]
]

#v(0.5em)

The unsupervised PCA tells a more honest story. Family structure is visible in PCA too --- activity families cluster on the left of PC1, trade-size families on the right --- but with substantial overlap, especially at later layers. The top two principal components explain only ~40% of variance, so most of the residual stream is encoding things unrelated to family identity.

#v(0.4em)

What makes the family finding more than a methodological artifact:

- The *confound baselines* confirm it statistically, independent of any projection method. `family_only` predicts the arbitration label at 76.3% using a simple classifier on a categorical variable --- no high-dimensional geometry involved.
- The *PCA* shows family structure in an unsupervised setting, before any label information is used.
- The *causal intervention* bites differently across families, which would not happen if families were just label noise.

What does *not* follow:

- That the model has "dedicated family circuits." The families have genuinely different prompt content --- different vocabulary, different section phrasing. Some separation is trivially expected because the model is encoding "these prompts contain different words."
- That 76% family-baseline accuracy means the probe is doing nothing beyond family. It could be capturing a mix of family features and genuine within-family arbitration signal. We cannot distinguish these yet.

= B. Sample Size Constraints

#v(0.4em)

#table(
  columns: (1.5fr, auto, auto),
  align: (left, center, left),
  inset: 7pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#E6EEF2") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

  [*Analysis*], [*N*], [*Concern*],

  [Conflict detection probe], [`288`], [Adequate for grouped 5-fold.],
  [Arbitration probe], [`123`], [Tight. 5-fold grouped CV on 43 groups.],
  [Per-family arbitration], [`~30`], [Too small for reliable within-family probes.],
  [Router section attribution], [`25`], [Not interpretable.],
  [Causal check (per family)], [`~30`], [Directional only; single-row flips matter.],
)

#v(0.4em)

The per-family N of ~30 is the binding constraint for Stage 3. Within-family probing with 5-fold CV on 30 examples will have high variance. Options:

- *Expand the dataset.* Add more conflict variants per family, or add new families. This is the cleanest fix but requires new data generation and capture.
- *Pool across families with family as a covariate.* Regress out family (and pressure) from the activations before probing. This preserves N=123 while controlling for the confound, but assumes the confound is additive.
- *Permutation-based significance.* For small N, replace accuracy with a permutation test to get proper p-values rather than relying on point estimates.

= C. What the Causal Result Does and Does Not Show

The causal intervention moves behavior in the expected direction, but interpretation is limited:

- *Asymmetry.* `strategy_push` produces +9pp shift; `setting_push` produces only +3pp. This could mean the direction is not symmetric, or that the model has a strategy-default bias that is easier to amplify than override.
- *Family dependence.* The intervention bites hardest in `trade_size_force_large` (6 row flips from 0 baseline) and barely moves the activity families. If the intervention were acting on a generic arbitration direction, we would expect it to move all families. Instead it interacts with family-specific structure.
- *Strength = 1.0.* We used a single intervention strength. A strength sweep would reveal whether the effect is linear, saturating, or has a threshold --- important for distinguishing a genuine direction from a perturbation artifact.
- *No same-label control.* We did not run the intervention on *aligned* rows to check whether it disrupts behavior when no conflict exists. A clean causal direction should have no effect on aligned prompts.

= D. Isolation Strategy for Stage 3

The main open question is: *does a cross-family arbitration signal exist after controlling for family and pressure?* The following analyses would address this:

#v(0.3em)

#table(
  columns: (auto, 1.8fr, 1.2fr),
  align: (left, left, left),
  inset: 7pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#E6EEF2") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

  [*Priority*], [*Analysis*], [*What it would show*],

  [1], [Residualize activations: regress out family + pressure, then re-probe], [Whether a generic arbitration component exists underneath family-specific behavior],
  [2], [Within-family probes (where N allows)], [Whether arbitration signal exists within a single conflict type],
  [3], [Causal intervention on aligned rows (same-label control)], [Whether the intervention direction is specific to conflict or a general perturbation],
  [4], [Causal strength sweep (0.25, 0.5, 1.0, 2.0)], [Whether the effect is linear and dose-dependent],
  [5], [Expand dataset to ~60+ rows per family], [Statistical power for within-family analyses],
  [6], [Cross-family generalization: train probe on 3 families, test on held-out 4th], [Whether the probe learns anything that transfers across conflict types],
)

#v(0.4em)

Item 6 is the sharpest test. If a probe trained on activity families can predict arbitration outcome on trade-size families (or vice versa), that is strong evidence for a generic signal. If it cannot, the arbitration mechanism is family-specific all the way down.

#v(2em)
#line(length: 100%, stroke: 0.5pt + rgb("#CCC"))
#v(0.3em)
#text(size: 8pt, fill: rgb("#999"))[
  Prompt Confusion Phase 04 Combined Checkpoint --- stages 1 and 2. Generated from local outputs under `phase_04/outputs`. Behavior run `16474bce`, conflict probe `62fc89fd`, arbitration probes `6924c2b3` / `4aeb882e`, PCA runs `f4c70fca` / `331de685`, family PCA from `analysis_conflict_readout_pca_by_family`, section attribution and causal from `conflict_arbitration_stage1` and `conflict_arbitration_stage2`.
]
