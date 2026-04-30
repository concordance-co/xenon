#set page(
  paper: "us-letter",
  margin: (top: 1.35cm, bottom: 1.25cm, left: 1.55cm, right: 1.55cm),
  numbering: "1",
  number-align: right,
)
#set text(font: "Georgia", size: 9.1pt)
#set par(justify: true, leading: 0.54em)
#set heading(numbering: none)

#let ink = rgb("#182028")
#let muted = rgb("#5a6670")
#let accent = rgb("#b33a2a")
#let cool = rgb("#285e9e")
#let rule = rgb("#ccd6de")
#let head = rgb("#e8eef3")
#let band = rgb("#f8fafb")
#let note = rgb("#eef4f8")
#let todo-fill = rgb("#fff3d9")
#let good = rgb("#eaf4ee")

#show heading.where(level: 1): it => {
  set text(size: 15.5pt, weight: "bold", fill: ink)
  v(0.65em)
  it
  v(0.15em)
  line(length: 100%, stroke: 0.8pt + rule)
  v(0.22em)
}

#show heading.where(level: 2): it => {
  set text(size: 11.2pt, weight: "bold", fill: ink)
  v(0.45em)
  it
  v(0.10em)
}

#let callout(tag, body, fill-color: note, tag-color: cool) = block(
  width: 100%,
  inset: (left: 9pt, top: 6.5pt, bottom: 6.5pt, right: 8pt),
  stroke: (left: 2.4pt + tag-color, top: none, right: none, bottom: none),
  fill: fill-color,
)[
  #set par(justify: false)
  #text(size: 6.7pt, fill: tag-color, weight: "bold", tracking: 0.08em)[#tag]
  #v(0.12em)
  #text(size: 8.9pt, fill: ink)[#body]
]

#let todo(body) = callout([TODO], body, fill-color: todo-fill, tag-color: accent)

#let source(body) = callout([SOURCE], body, fill-color: note, tag-color: muted)

#let data-table(cols, ..args) = table(
  columns: cols,
  inset: 4.0pt,
  stroke: 0.32pt + rule,
  fill: (x, y) => if y == 0 { head } else if calc.odd(y) { band } else { white },
  ..args,
)

#let tight-table(cols, ..args) = {
  set text(size: 7.8pt)
  table(
    columns: cols,
    inset: 3.0pt,
    stroke: 0.28pt + rule,
    fill: (x, y) => if y == 0 { head } else if calc.odd(y) { band } else { white },
    ..args,
  )
}

#let fig(path, caption) = figure(
  image(path, width: 100%),
  caption: text(size: 7.4pt, fill: muted)[#caption],
)

#let twofig(left-path, left-caption, right-path, right-caption) = grid(
  columns: (1fr, 1fr),
  gutter: 0.36cm,
  fig(left-path, left-caption),
  fig(right-path, right-caption),
)

#align(left)[
  #text(size: 7.4pt, fill: accent, tracking: 0.08em, weight: "medium")[DX TERMINAL / PROMPT CONFUSION]
  #v(0.14em)
  #text(size: 21pt, weight: "bold", fill: ink)[Policy Conflict Internals in Real Agentic Contexts]
  #v(0.10em)
  #text(size: 10pt, fill: muted)[Working Typst draft. Prose target: under 2000 words, excluding tables, captions, and TODO blocks.]
  #v(0.34em)
  #line(length: 100%, stroke: 1pt + ink)
]

#todo[
  Write the final opener. Keep the claim narrow: this is not a production
  detector or causal mechanism claim. The strongest story is that synthetic
  policy-conflict directions can recover a shape-specific real production
  signal after the ontology is narrowed.
]

= Problem

This work follows up on our recent DX Terminal collaboration, where we used
mechanistic interpretability in real-world financial contexts. In part 1, we
looked for early signs of interpretable market-perception geometry in LLM
activations over real market data.

Part 2 looks at a different problem DXRG saw in their agents: strange behavior
when policy sources collide. A trading agent sees system rules, user strategies,
active vault settings, portfolio state, market data, and prior decisions. Those
sources are not always mutually consistent.

#todo[
  Add one concrete DX Terminal example in plain language: strategy says buy
  aggressively while settings cap size; strategy says sell all while settings
  imply partial exits; complaint refers to a strategy no longer active.
]

Our question was whether these policy-source conflicts leave a readable
internal signal. The useful version of that question is not "can we classify
all complaints?" It is narrower: can we train clean synthetic directions for
specific conflict shapes, then see whether those directions fire on analogous
real prompts?

= Why This Matters

One of Concordance's goals is to make interpretability useful to agent builders
working on real systems. The DX Terminal policy-conflict problem was valuable
because it was concrete, recurring, and operationally meaningful: if an agent's
internal state reflects competing constraints, that signal could eventually
support monitoring, audits, prompt redesign, settings semantics, and strategy
lifecycle debugging.

#todo[
  Add your voice here. The current paragraph is deliberately functional. You
  may want one paragraph on why "real data -> synthetic abstraction -> real
  data" is the loop this post is trying to demonstrate.
]

= Synthetic Abstraction

Real prompts are entangled, so we started with controlled prompt families. The
synthetic prompts isolate conflicts between user-configured strategies and
vault-configured active settings while keeping the decision surface simple.

#source[
  Dataset and workflow sources: `phase_09/scripts/build_phase_09_dataset.py`
  for `trade_size`, `phase_10/scripts/build_phase_10_dataset.py` for
  `risk_preference`, `phase_12/scripts/build_phase_12_dataset.py` for
  `diversification_preference`, and `phase_12/specs/workflow.py` plus
  `phase_12/specs/marshalls_strict_workflow.py` for lexical holdout probes.
]

#block(fill: band, inset: 7pt, stroke: 0.3pt + rule)[
  #text(size: 8.1pt)[
    #raw(
      "[system]\nRole: trading agent.\nCore rule: each prompt contains STRATEGY and ACTIVE SETTINGS.\nPriority rule: ACTIVE SETTINGS are binding execution constraints.\nDecision order: activity, asset/risk/diversification posture, then size.\n\n[user]\nTASK: choose exactly one action for this tick.\nSTRATEGY: compact user preference.\nACTIVE SETTINGS: slider-like constraints.\nPORTFOLIO: controlled portfolio state.\nMARKET: controlled synthetic assets and evidence.\nOutput: strict JSON.",
      lang: "text",
    )
  ]
]

We tested three families:

- `trade_size`: buy small vs large, with the output size/action as the axis.
- `risk_preference`: asset selection by allowed risk posture.
- `diversification_preference`: concentration vs broadening, conditioned on the portfolio.

Each family used paraphrased strategy templates and settings-label templates.
Those templates were split into train/test groups, then probes were evaluated
on held-out strategy wording, held-out settings wording, and stricter both-axis
holdouts. The point was to test the conflict relation rather than a phrase
family.

#todo[
  Optional: replace the prompt sketch with a more narrative paragraph. The
  current version is data-dense and edit-friendly, but not yet blog-polished.
]

= Synthetic Probe Results

After iterating on prompt structure and confound controls, all three families
were linearly readable under lexical holdouts. Values below are balanced
accuracy / AUROC where paired.

#source[
  Canonical numbers: `phase_12/reports/PROMPT_CONFLICT_FAMILY_CHECKPOINT_2026_04_16.typ`.
  Blog-friendly figures and captions: `phase_12/reports/DX_TERMINAL_CONFLICT_GEOMETRY_BRIEF_2026_04_17.typ`.
]

#data-table(
  (1.05fr, 2.75fr, 1.05fr),
  [*Family*], [*Standard probe results*], [*Strict holdout*],
  [`trade_size`], [XOR `0.9948 / 1.0000`; strategy holdout `1.0000 / 1.0000`; settings holdout `0.9948 / 1.0000`], [`0.990 / 1.000` at L40],
  [`risk_preference`], [XOR `0.9635 / 0.9766`; strategy holdout `0.9844 / 0.9937`; settings holdout `0.9740 / 0.9839`], [`0.8854 / 0.9119`],
  [`diversification_preference`], [behavior: aligned `1.0000`, conflict `0.8542`; XOR `0.9896 / 0.9995`; strategy holdout `1.0000 / 1.0000`; settings holdout `0.9792 / 0.9957`], [`0.8333 / 0.8819`],
)

#twofig(
  "../phase_12/reports/dx_terminal_brief_assets/family_within_auroc_by_layer.png",
  [Representative within-family AUROC curves across layers.],
  "../phase_12/reports/dx_terminal_brief_assets/strict_family_auroc_by_layer.png",
  [Strict lexical-holdout AUROC curves.],
)

The families are related but not identical. Same-capture mean-difference
directions at L36 showed moderate cosine similarity, strongest between
`risk_preference` and `trade_size`.

#data-table(
  (2.4fr, 0.9fr),
  [*Pair*], [*L36 cosine*],
  [`risk_preference` vs `trade_size`], [`0.6449`],
  [`diversification_preference` vs `risk_preference`], [`0.4684`],
  [`diversification_preference` vs `trade_size`], [`0.4883`],
)

#twofig(
  "../phase_12/reports/three_family_visuals/shared_axis_distributions.png",
  [Shared-axis distributions: separation exists, but family baselines are offset.],
  "../phase_12/reports/three_family_visuals/directed_subspace_scatter_by_family_conflict_v2.png",
  [Directed subspace view: related conflict geometry, not one collinear axis.],
)

#todo[
  Write the transition from "synthetic readout works" to "transfer is hard."
  Important claim boundary: high AUROC supports a representational readout, not
  a causal mechanism.
]

= First Real Transfer Attempt

The first direct transfer pass projected synthetic conflict directions onto
full production prompts at coarse global sites. It did not cleanly separate
complaint rows from baseline controls.

#source[
  Bridge source: `phase_12/reports/REAL_TRANSFER_BRIDGE_PLAN_2026_04_22.md`.
  The plan records the initial null/unclear full-prompt transfer and frames the
  next experiments as tests of concept failure, format failure, and site
  failure.
]

That failure mattered. It showed that synthetic success does not automatically
become production success. Real prompts differ in template, label semantics,
time horizon, complaint noise, and where the relevant evidence appears inside
the prompt.

#todo[
  Add one paragraph with your interpretation of the failure. A good frame:
  this was not "the probe failed," it was the point where the ontology had to
  get sharper.
]

= Bridge Program

We then used bridge datasets to separate template mismatch from content and
ontology mismatch.

#data-table(
  (2.05fr, 0.65fr, 0.75fr, 0.75fr),
  [*Dataset*], [*Rows*], [*Aligned*], [*Conflict*],
  [Stage 1a template control], [`768`], [`384`], [`384`],
  [Stage 1b loose adapter], [`258`], [`168`], [`90`],
  [Stage 1b strict adapter], [`118`], [`81`], [`37`],
  [Stage 1b strict buy-only], [`33`], [`27`], [`6`],
)

#source[
  Neon tables: `dx_terminal_trade_size_stage1a_template_control_v1`,
  `dx_terminal_trade_size_stage1b_adapter_loose_v1`,
  `dx_terminal_trade_size_stage1b_adapter_strict_v1`,
  `dx_terminal_trade_size_stage1b_adapter_strict_buy_only_v1`.
  Summary JSONs live in `phase_12/outputs/transfer_bridge/`.
]

The bridge evidence was real but weak. Stage 1b was noisy; buy-only filtering
helped, but probe-to-synthetic cosine remained near zero. Sell/liquidation
contamination was part of the problem, but the deeper issue was an unresolved
ontology and representation mismatch.

#todo[
  Decide how much bridge detail belongs in the final post. If the post is for
  a broad audience, this section can be a short failure-analysis interlude
  before Phase 13.
]

= Phase 13 Real Signal Discovery

Phase 13 asked a simpler question: if we do not train a classifier and do not
set thresholds, do fixed synthetic directions produce scalar structure anywhere
on real DX Terminal prompts?

Medium validation run: `wr_14f78308dbac_dbc78513`. Corpus:
`dx_terminal_signal_discovery_phase13_v1`. Scope: aggressive tier, end-of-section
sites, 500 complaints, 300 structure-matched controls, 118 Stage 1b strict
anchors, and 33 buy-only anchors. Primary cell: L32 `settings_end`.

#source[
  Phase 13 source: `phase_13/reports/PHASE13_REAL_TRANSFER_SIGNAL_BRIEF_2026_04_24.typ`.
]

#data-table(
  (1.05fr, 0.72fr, 0.82fr, 0.98fr, 0.98fr, 1.05fr),
  [*Direction*], [*Anchor*], [*Complaint*], [*Structure control*], [*Anchor-control*], [*Complaint-control*],
  [`trade_size`], [`4.425`], [`3.803`], [`3.278`], [`+1.147`], [`+0.526`],
  [`shared_mean`], [`3.462`], [`3.137`], [`2.760`], [`+0.703`], [`+0.377`],
)

At this cell, `trade_size` and `shared_mean` separated complaint prompts from
structure-matched controls. `risk_preference` was weaker and
`diversification_preference` was not clean, so this is not a generic "all
synthetic directions light up on all complaints" result.

#todo[
  Add the intuitive explanation of L32 `settings_end`: by this point the model
  has seen ACTIVE SETTINGS, but has not yet consumed the whole prompt.
]

= Row Reading And Ontology Correction

The most important interpretability move was reading the top and bottom rows.
The preregistered proxy expected `USER_CONFIG_CONFLICT` to rank high and
`RULE_FABRICATION` or non-config rows to rank low. That proxy was wrong for the
`trade_size` target. Root-cause labels diagnose why a complaint happened; the
probe target is visible current-prefix conflict shape.

#data-table(
  (1.1fr, 0.85fr, 1fr, 0.95fr, 1.15fr),
  [*Direction*], [*Top action/size*], [*Top strategy ignored*], [*Bottom action/size*], [*Bottom strategy ignored*],
  [`trade_size`], [`20/25`], [`5/25`], [`15/25`], [`10/25`],
  [`shared_mean`], [`20/25`], [`5/25`], [`9/25`], [`16/25`],
)

#data-table(
  (1.5fr, 0.85fr),
  [*Top `trade_size` complaint type*], [*Count*],
  [`UNWANTED_BUY`], [`10/25`],
  [`UNWANTED_SELL`], [`6/25`],
  [`WRONG_SIZE`], [`4/25`],
  [Concrete action/size combined], [`20/25`],
)

High `trade_size` rows were mostly concrete current action or size complaints:
"why did you buy HOTDOGZ?", "why did you buy so much POOPCOIN?", or "Buy
available balance 30%, not 10 ETH." Low rows could still be valid complaints,
but often depended on history, lifecycle state, prior strategy fulfillment, or
rule interpretation rather than a clean current-prefix size conflict.

#todo[
  Expand this into the core narrative section. The lesson is strong: looking at
  internals forced the label ontology to change from root cause to conflict
  shape.
]

= Claim Boundary

#callout(
  [CLAIM],
  [
    Fixed synthetic directions recover a real production signal at L32
    `settings_end`. `trade_size` is selective for current-prefix concrete
    sized-action conflict. `shared_mean` appears to track broader policy
    tension, but the shared-family interpretation still needs more audit.
  ],
  fill-color: good,
  tag-color: cool,
)

This is not a final detector, not a deployment monitor, and not a causal
mechanism claim. It is a representational transfer result over projection
extremes with structure-matched controls, not a gold complaint dataset.

= Larger Thesis

The useful loop is: real data exposes a messy failure mode; synthetic prompts
isolate a clean abstraction; probes find a candidate internal signal; bridge
tests expose transfer mismatch; real-data projection finds a narrower
shape-specific signal; row reading improves the ontology.

#todo[
  Write the closing in your voice. Keep it practical: this is a template for
  using mech interp as an engineering feedback loop, not just as a benchmark
  result.
]

= Next Steps

- Hand-label top and bottom rows with `current_action_size_conflict`,
  `retrospective_history_conflict`, `strategy_fulfillment_conflict`,
  `interpretation_or_rule_conflict`, and `unclear_or_label_mismatch`.
- Inspect neighboring settings cells after hand-labeling: L28, L32, and L36
  `settings_end`.
- Find true non-complaint production controls.
- Defer causal tests and interventions until the label ontology is cleaner.
