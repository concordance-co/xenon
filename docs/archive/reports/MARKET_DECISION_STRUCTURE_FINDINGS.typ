#set page(
  paper: "us-letter",
  margin: (x: 0.62in, y: 0.68in),
  footer: context [
    #align(center)[
      #text(size: 8pt, fill: rgb("#6A7A89"))[
        Market Decision Structure Report - March 20, 2026
      ]
    ]
  ],
)

#set par(justify: false, leading: 0.58em)
#set text(font: "Libertinus Serif", size: 10pt, fill: rgb("#16202A"))

#let ink = rgb("#16202A")
#let muted = rgb("#5E6F82")
#let navy = rgb("#16324F")
#let teal = rgb("#2E6A69")
#let gold = rgb("#CA9440")
#let rose = rgb("#B56662")
#let cream = rgb("#F6EFE3")
#let mist = rgb("#EAF2F2")
#let line = rgb("#D6DEE3")
#let softline = rgb("#E7ECEF")
#let pillnavy = rgb("#35506A")

#show heading.where(level: 1): it => block(
  above: 1.15em,
  below: 0.35em,
  text(16pt, weight: "bold", fill: navy)[#it.body],
)

#show heading.where(level: 2): it => block(
  above: 0.85em,
  below: 0.28em,
  text(12pt, weight: "bold", fill: teal)[#it.body],
)

#show figure.caption: set text(size: 8.5pt, fill: muted)

#let pill(content, fill-color: rgb("#FFFFFF"), text-color: ink) = box(
  fill: fill-color,
  stroke: (paint: softline, thickness: 0.6pt),
  radius: 999pt,
  inset: (x: 8pt, y: 4pt),
)[
  #text(size: 8pt, fill: text-color)[#content]
]

#let stat(label, value, note, tone: white) = block(
  fill: tone,
  stroke: (paint: softline, thickness: 0.6pt),
  radius: 12pt,
  inset: 12pt,
  width: 100%,
)[
  #text(size: 8pt, fill: muted, weight: "bold")[#label]
  #v(5pt)
  #text(size: 20pt, fill: navy, weight: "bold")[#value]
  #v(4pt)
  #text(size: 8.8pt, fill: muted)[#note]
]

#let signal(title, body, tone: white) = block(
  fill: tone,
  stroke: (paint: softline, thickness: 0.6pt),
  radius: 12pt,
  inset: 12pt,
  width: 100%,
)[
  #text(size: 8pt, fill: muted, weight: "bold")[#title]
  #v(6pt)
  #text(size: 11pt, fill: ink)[#body]
]

#let step(letter, title, body) = block(
  fill: white,
  stroke: (paint: softline, thickness: 0.6pt),
  radius: 12pt,
  inset: 12pt,
  width: 100%,
)[
  #box(
    fill: navy,
    radius: 999pt,
    inset: (x: 7pt, y: 4pt),
  )[
    #text(size: 8pt, fill: white, weight: "bold")[#letter]
  ]
  #v(7pt)
  #text(size: 11pt, fill: navy, weight: "bold")[#title]
  #v(5pt)
  #text(size: 9pt, fill: muted)[#body]
]

#let quote(body) = block(
  fill: rgb("#FBF7EF"),
  stroke: (paint: softline, thickness: 0.6pt),
  radius: 12pt,
  inset: 14pt,
  width: 100%,
)[
  #text(size: 10.5pt, fill: ink)[#body]
]

#block(
  fill: navy,
  inset: 18pt,
  radius: 16pt,
  width: 100%,
)[
  #text(size: 21pt, weight: "bold", fill: white)[Market Decision Structure]
  #v(5pt)
  #text(size: 11pt, fill: luma(245))[
    Fresh report for the 918-tick run. This supersedes the earlier 101-tick pilot memo.
  ]
  #v(10pt)
  #text(size: 9pt, fill: luma(235))[
    Research anchors: `MARKET_MANIFOLD_RESEARCH_PLAN.md` and `MARKET_MANIFOLD_IMPLEMENTATION_PLAN.md`
  ]
  #v(10pt)
  #grid(
    columns: (auto, auto, auto, auto),
    gutter: 6pt,
    pill([March 20, 2026], fill-color: pillnavy, text-color: white),
    pill([48 layers], fill-color: pillnavy, text-color: white),
    pill([Held-out grouped evaluation], fill-color: pillnavy, text-color: white),
    pill([Decision-structure slice], fill-color: pillnavy, text-color: white),
  )
]

#v(12pt)

This report addresses one narrower question from the market-manifold program: *when does the model bind an action-relevant preference to a specific asset?* The new 918-tick decision-structure run points to a mixed answer. Buy-side preference remains strongly early, but target-asset and sell-target signals both receive modest downstream sharpening from later policy and affordance context.

= Executive Read

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 8pt,
  stat([Ticks], [918], [559 trades and 359 observations.]),
  stat([Asset rows], [5,719], [Row-level pooled states across the held-out evaluation.]),
  stat([Best pre AUROC], [0.878], [`is_buy_target` from `row_eos` at layer 25.], tone: mist),
  stat([Largest post lift], [+0.019], [Target-asset and sell-target both improve modestly post-context.], tone: cream),
)

#v(8pt)

#quote[
  *Main read:* the architecture does not look like "all preference early" or "all policy late." It looks staged. Market rows already carry a strong asset preference signal, but downstream sections such as `constraints_eos` and `active_settings_eos` can sharpen that signal when legality or actionability matters.
]

= Dataset

This is the first decision-structure run that is large enough to move beyond the pilot.

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 8pt,
  signal([Decision mix], [559 trades vs 359 observations.]),
  signal([Trade direction], [280 buys vs 279 sells.]),
  signal([Positive rows], [558 target-asset rows, 279 buy-target rows, 279 sell-target rows.]),
)

#v(8pt)

Why this matters:

- the run is no longer bottlenecked by sell scarcity
- the bearish-side readout is now meaningful
- the result is a real decision-structure test rather than a buy-heavy pilot

#figure(
  image("data/report_assets/decision_structure/dataset_composition.png", width: 100%),
  caption: [Dataset composition for the 918-tick run. The class balance is much healthier than in the pilot, though target-asset rows still concentrate in a handful of symbols.]
)

= Method

The report focuses on the decision-structure slice of the broader research plan. The analysis compares what is decodable from market rows alone against what becomes decodable after downstream sections are added.

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 8pt,
  step([01], [Pool the sequence], [Reduce full-sequence captures into `row_mean_i`, `row_eos_i`, and downstream section states such as `active_settings_eos`, `portfolio_eos`, `constraints_eos`, `prev_decisions_eos`, and `last_token`.]),
  step([02], [Train held-out probes], [Fit linear probes on grouped splits so multiple rows from the same tick do not leak across train and test.]),
  step([03], [Compare pre vs post], [Treat row states as pre-context and row-plus-section combinations as post-context representations.]),
  step([04], [Read staging], [If post states beat pre states, downstream context adds usable structure; if pre states dominate, the preference is already formed while the market is read.]),
)

= Main Quantitative Findings

#table(
  columns: (1.2fr, 2.1fr, auto, 2.2fr, auto, auto),
  align: (left, left, center, left, center, center),
  inset: 6pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#DDEBF0") } else if calc.odd(y) { rgb("#F8FBFC") } else { white },

  [*Target*], [*Best pre state*], [*AUROC*], [*Best post state*], [*AUROC*], [*Post - Pre*],

  [`is_target_asset`], [`row_mean` @ layer 35], [0.815], [`row_mean + constraints_eos` @ layer 26], [0.834], [#text(fill: teal, weight: "bold")[+0.019]],
  [`is_buy_target`], [`row_eos` @ layer 25], [0.878], [`row_mean + constraints_eos` @ layer 29], [0.870], [#text(fill: rose, weight: "bold")[-0.008]],
  [`is_sell_target`], [`row_mean` @ layer 2], [0.839], [`row_mean + active_settings_eos` @ layer 2], [0.857], [#text(fill: teal, weight: "bold")[+0.018]],
)

Interpretation of the table:

- `is_buy_target` remains strongest in a pre-settings row state
- `is_target_asset` improves slightly once constraints are added
- `is_sell_target` improves slightly once active settings are added

This is the key reason the new report differs from the pilot. The earlier "everything is early" story no longer holds as a blanket conclusion.

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,

  figure(
    image("data/report_assets/decision_structure/best_pre_post.png", width: 100%),
    caption: [Best pre-row vs best post-context AUROC by target. Post-context wins are small rather than dramatic, but they are now real for target-asset and sell-target.]
  ),
  figure(
    image("data/report_assets/decision_structure/representation_heatmap.png", width: 100%),
    caption: [Maximum AUROC by representation and target. `row_mean` and `row_eos` remain central, but `+ constraints` and `+ settings` now matter for some tasks.]
  ),
)

#v(8pt)

#figure(
  image("data/report_assets/decision_structure/layerwise_auroc.png", width: 100%),
  caption: [Layerwise AUROC curves. The buy-target signal remains distinctly early, while target-asset and sell-target gain from downstream context without requiring a deep late reconstruction.]
)

= Interpretation

#grid(
  columns: (1fr, 1fr),
  gutter: 8pt,
  signal([Supported now], [Early asset preference is real. The model does not need to wait for the end of the prompt to identify an attractive buy candidate.]),
  signal([Also supported now], [Downstream context is not cosmetic. Constraints and settings can sharpen the decision signal for some targets.], tone: mist),
)

#v(8pt)

Best current interpretation:

- the model forms a strong early market preference while reading asset rows
- later sections reweight that preference when legality, sizing, or actionability matters
- the result is consistent with the research plan's split between the asset-in-context manifold, the asset-valence manifold, and the final decision manifold

#quote[
  *Best current interpretation:* Xenon appears to use an early market preference layer followed by a later policy-and-affordance refinement layer. That is closer to "early read, later reweighting" than to either extreme of "all preference at the start" or "all policy at the end."
]

= What We Accomplished

Relative to the root plans, this run closes the first serious decision-structure milestone.

- balanced cohort selection and large-scale full-sequence capture on Modal
- reliable pooled row and downstream section states
- held-out pre/post probe analysis for `is_target_asset`, `is_buy_target`, and `is_sell_target`
- a result strong enough to constrain the next research step instead of merely validating pipeline health

= What Is Still Missing

This report does *not* finish the broader market-manifold program.

- blocked-bullish and blocked-bearish labels for observe cases
- base market-geometry work: intrinsic dimension, RSA, pairwise probes, and router specialization
- stronger settings interventions for a clean pre/post reinterpretation test
- causal necessity tests such as row masking and rank-vs-magnitude corruption

= Recommended Next Steps

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 8pt,
  step([A], [Build blocked-valence labels], [Use constrained observe examples to separate latent bullishness or bearishness from mere non-action. This is the missing bridge from executed action to the actual asset-valence manifold.]),
  step([B], [Run market geometry in parallel], [Start intrinsic dimension, RSA against raw and rank spaces, and router specialization on current row activations.]),
  step([C], [Design stronger settings interventions], [Replace weak prompt edits with real policy changes so pre/post reinterpretation is tested against meaningful shifts rather than near-identity prompts.]),
  step([D], [Then run causal tests], [Once the label program is stronger, use row masking, pairwise swaps, and rank-preserving vs magnitude-preserving corruptions to measure what the model actually needs.]),
)

#v(10pt)

#quote[
  The main value of this run is that it narrows the search space. We no longer need to argue about whether the decision-structure pipeline works. It does. The next question is how to move from executed action labels to true asset-valence labels, while building the base market-geometry measurements in parallel.
]
