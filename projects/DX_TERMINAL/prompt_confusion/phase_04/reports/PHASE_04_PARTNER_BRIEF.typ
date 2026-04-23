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
  #text(size: 15pt, weight: "regular", fill: muted)[Partner Brief]
  #v(0.5em)
  #text(size: 11pt, fill: muted)[
    April 10, 2026 --- key findings from two rounds of mechanistic analysis on how an LLM resolves contradictory instructions.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.4pt + ink)
]

#v(1em)

// ── Setup ────────────────────────────────────────────────────

= The Experiment

We gave Qwen3-30B-A3B a batch of 288 synthetic trading prompts. Half had internally consistent instructions (aligned). The other half had a deliberate contradiction: the `SETTINGS` section said one thing, the `STRATEGY` section said the opposite. We then used linear probes and causal interventions on the model's internal activations to study how it detects and resolves these conflicts.

#v(0.8em)

// ── Finding 1 ────────────────────────────────────────────────

= Finding 1: The Model Knows When It's Conflicted

A linear probe on the residual stream can distinguish conflict from aligned prompts at *92% accuracy* (layer 36, grouped balanced accuracy). The signal builds progressively through the model's layers.

#v(0.4em)

#table(
  columns: (auto, auto, auto, auto),
  align: (center, center, center, center),
  inset: 7pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#E6EEF2") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

  [*Layer*], [*Balanced Accuracy*], [*Std Dev*], [*Selectivity*],

  [`0`], [`0.784`], [`0.038`], [`0.310`],
  [`24`], [`0.864`], [`0.037`], [`0.399`],
  [*`36`*], [*`0.920`*], [*`0.042`*], [*`0.417`*],
  [`44`], [`0.837`], [`0.072`], [`0.354`],
)

#v(0.4em)

This is a clean Level-2 representational result. The model builds an increasingly clear internal representation of "these instructions contradict each other" as it processes the prompt.

#v(0.5em)

#figure(
  image("../outputs/analysis_full_prompt_eos_pca/f4c70fcbac6d/pca_layer28_workflow_label.png", width: 85%),
  caption: [PCA of residual activations at layer 28, colored by conflict status. Aligned and conflict prompts occupy overlapping but statistically separable regions.]
)

#v(1em)

// ── Finding 2 ────────────────────────────────────────────────

= Finding 2: Conflict Resolution Is Category-Specific, Not Generic

We initially expected to find a general-purpose "who wins" signal --- a direction in activation space that encodes whether `SETTINGS` or `STRATEGY` is being followed, regardless of the conflict type. Instead, we found that the model's internal representations are organized by conflict *type*, not by resolution outcome.

#v(0.4em)

The evidence converges from two directions. Externally, a simple classifier using only the conflict family (what kind of contradiction) and pressure bucket (how strong the contextual push) predicts the winner at *80.4%*. Internally, PCA on the model's activations reveals that the same family structure dominates the representation space, while the arbitration label (who won) shows no structure at all. Both views point to the same conclusion: the model is encoding what kind of conflict it is facing, not maintaining an independent resolution signal.

#v(0.3em)

#table(
  columns: (1.6fr, auto, auto),
  align: (left, center, center),
  inset: 7pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#E6EEF2") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

  [*Predictor*], [*Balanced Accuracy*], [*vs. Best Probe (71.2%)*],

  [*Family + pressure metadata*], [*80.4%*], [*Above*],
  [Family metadata only], [76.3%], [Above],
  [Best residual-stream probe (layer 20)], [71.2%], [---],
  [Section length / position], [48.7%], [Below chance],
)

#v(0.5em)

The PCA visualization makes this concrete. The same activations plotted twice with different coloring:

#v(0.3em)

#grid(
  columns: (1fr, 1fr),
  gutter: 8pt,
  figure(
    image("../outputs/analysis_conflict_readout_pca/331de685cbc1/pca_layer28_workflow_label.png", width: 100%),
    caption: [Colored by *who won* (strategy vs. setting). No visible structure.]
  ),
  figure(
    image("../outputs/analysis_conflict_readout_pca_by_family/pca_layer28_strategy_family.png", width: 100%),
    caption: [Colored by *conflict type* (family). Clear clustering.]
  ),
)

#v(0.4em)

The supervised LDA projection sharpens this further: four near-separable family clusters, organized along two interpretable axes.

#v(0.3em)

#figure(
  image("../outputs/analysis_conflict_readout_pca_by_family/lda_layer28_strategy_family.png", width: 75%),
  caption: [LDA at layer 28. LD1 separates activity-type conflicts from trade-size-type conflicts. LD2 separates the specific variant within each pair.]
)

#v(0.5em)

#callout(
  [KEY TAKEAWAY],
  [
    The model does not have one universal tiebreaker for conflicting instructions. It categorizes the conflict first, then resolves it differently depending on the category. "Should I trade or observe?" is resolved through a different pathway than "should I go big or small?" The practical implication: instruction priority in LLMs is not a stable hierarchy --- it depends on what kind of instruction is being overridden.
  ],
  fill-color: mist,
  tag-color: rgb("#2D5FA0"),
)

#v(1em)

// ── Finding 3 ────────────────────────────────────────────────

= Finding 3: We Can Causally Steer the Resolution

Injecting the `SETTINGS` direction into the residual stream at layer 24 moves behavior in the expected direction. The model's conflict resolution responds to targeted internal perturbation.

#v(0.4em)

#table(
  columns: (1.2fr, auto, auto, auto),
  align: (left, center, center, center),
  inset: 7pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#E6EEF2") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

  [*Condition*], [*Strategy Rate*], [*Setting Rate*], [*Neither*],

  [`baseline`], [`53.7%`], [`45.5%`], [`0.8%`],
  [`setting_push`], [`48.8%`], [`48.8%`], [`2.4%`],
  [`strategy_push`], [`62.6%`], [`36.6%`], [`0.8%`],
)

#v(0.4em)

The effect is asymmetric (strategy_push is stronger than setting_push) and family-dependent --- it bites hardest in trade-size families and barely moves activity families. This is consistent with the category-specific resolution finding: the causal patch interacts with family-specific structure rather than a single generic direction.

#v(1em)

// ── What's Next ──────────────────────────────────────────────

= What's Next

The main open question is whether a *cross-family* arbitration signal exists underneath the family-specific behavior. Stage 3 will focus on isolation:

+ *Residualized probing* --- regress out family and pressure from activations, then re-probe. Tests whether any generic arbitration component remains.
+ *Cross-family generalization* --- train a probe on three families, test on the held-out fourth. The sharpest test: if it transfers, the signal is generic; if not, it's category-specific all the way down.
+ *Same-label causal control* --- run the intervention on aligned (non-conflict) prompts. A clean direction should have no effect when there's no conflict to resolve.
+ *Dataset expansion* --- current N per family (~30) limits within-family statistical power. More variants per family would unlock finer-grained analysis.

#v(1.2em)

// ── Status Grid ──────────────────────────────────────────────

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
      - Conflict detection: strong, clean (92%)
      - Conflict resolution is category-specific
      - Causal intervention is behaviorally active
      - Pressure modulates resolution within categories
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
      - Whether a generic cross-family signal exists
      - Causal specificity (no same-label control yet)
      - Within-family statistical power (N~30)
      - Attention-level follow-up not yet informative
    ]
  ],
)

#v(2em)
#line(length: 100%, stroke: 0.5pt + rgb("#CCC"))
#v(0.3em)
#text(size: 8pt, fill: rgb("#999"))[
  Prompt Confusion Phase 04 Partner Brief --- condensed from the full combined checkpoint report. Model: Qwen3-30B-A3B. Dataset: 288 synthetic trading prompts (144 aligned, 144 conflict, 4 conflict families). Full methodology and caveats available in the internal report.
]
