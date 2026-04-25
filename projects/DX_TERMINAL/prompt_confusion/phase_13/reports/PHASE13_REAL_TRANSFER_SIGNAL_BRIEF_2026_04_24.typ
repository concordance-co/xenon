#set page(
  paper: "us-letter",
  margin: (top: 1.25cm, bottom: 1.20cm, left: 1.45cm, right: 1.45cm),
  numbering: "1",
  number-align: right,
)
#set text(font: "Georgia", size: 8.35pt)
#set par(justify: true, leading: 0.43em)
#set heading(numbering: none)

#let ink = rgb("#182028")
#let muted = rgb("#5A6772")
#let accent = rgb("#B33A2A")
#let cool = rgb("#285E9E")
#let soft = rgb("#F7F2EF")
#let mist = rgb("#EEF4F8")
#let lift = rgb("#EAF4EE")
#let warn = rgb("#FFF3D9")

#show heading.where(level: 1): it => {
  set text(size: 10.8pt, weight: "bold", fill: ink)
  v(0.45em)
  it
  v(0.12em)
}

#show heading.where(level: 2): it => {
  set text(size: 9.2pt, weight: "bold", fill: ink)
  v(0.32em)
  it
  v(0.08em)
}

#let callout(tag, body, fill-color: mist, tag-color: cool) = block(
  width: 100%,
  inset: (left: 9pt, top: 6.5pt, bottom: 6.5pt, right: 8pt),
  stroke: (left: 2.6pt + tag-color, top: none, right: none, bottom: none),
  fill: fill-color,
)[
  #text(size: 6.4pt, fill: tag-color, weight: "bold", tracking: 0.08em)[#tag]
  #v(0.14em)
  #text(size: 9.15pt, fill: ink)[#body]
]

#let compact-table(cols, ..args) = table(
  columns: cols,
  align: (left, center, center, center, center, center),
  inset: 4.1pt,
  stroke: 0.32pt + rgb("#CFD8DF"),
  fill: (x, y) => if y == 0 { rgb("#E7EEF3") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },
  ..args,
)

#align(left)[
  #text(size: 7.3pt, fill: accent, tracking: 0.08em, weight: "medium")[DX TERMINAL / PHASE 13]
  #v(0.12em)
  #text(size: 18pt, weight: "bold", fill: ink)[Real Transfer Signal Brief]
  #v(0.04em)
  #text(size: 10.3pt, fill: muted)[Synthetic conflict probes on production prompts]
  #v(0.20em)
  #text(size: 7.6pt, fill: muted)[April 24, 2026. Medium validation run `wr_14f78308dbac_dbc78513`.]
  #v(0.35em)
  #line(length: 100%, stroke: 1pt + ink)
]

#v(0.35em)

#callout(
  [HEADLINE],
  [
    Synthetic probes trained on clean policy-source conflict transfer to real
    DX Terminal prompts at `L32 settings_end`. `trade_size` fires most strongly
    on current-prefix concrete sized-action conflict: wrong buy/sell side,
    wrong allocation, or explicit size mismatch. `shared_mean` captures a
    broader policy-tension component, while both directions fire less strongly
    on rows whose evidence is temporal, bookkeeping-based, or interpretive.
  ],
)

= Setup

Medium corpus: `dx_terminal_signal_discovery_phase13_v1` in Neon. Scope:
aggressive tier only, end-of-section sites, 500 complaints, 300
structure-matched controls, 118 Stage 1b strict anchors, and 33 buy-only
anchors. Primary preregistered cell: layer `32`, position `settings_end`.
This is late enough for settings to be visible and close to the synthetic
probe discovery band.

= Definitions

#table(
  columns: (1fr, 2.55fr),
  align: (left, left),
  inset: 3.7pt,
  stroke: 0.30pt + rgb("#CFD8DF"),
  fill: (x, y) => if y == 0 { rgb("#E7EEF3") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

  [*Term*], [*Meaning in this brief*],
  [`probe` / `direction`], [fixed vector learned from synthetic conflict prompts; real prompts are scored by projection onto it],
  [`trade_size`], [probe family for concrete buy/sell/position-size disagreement],
  [`shared_mean`], [mean of the validated family directions; intended to capture common policy-source tension],
  [`L32 settings_end`], [model layer 32 activation at the end of the ACTIVE SETTINGS section],
  [`anchor positive`], [bridge rows designed to contain known conflict signal],
  [`structure-matched control`], [non-complaint production prompt with similar DX Terminal prompt shape],
  [`top` / `bottom`], [highest complaint projections / complaints closest to the control mean],
)

The run did not train classifiers or thresholds. It only asked whether fixed
synthetic directions produce meaningful scalar structure on real prompts.

= Cohort Result

#compact-table(
  (1.2fr, 0.75fr, 0.85fr, 0.85fr, 0.85fr, 0.85fr),
  [*Direction*], [*Anchor*], [*Complaint*], [*Control*], [*Anchor-control*], [*Complaint-control*],
  [`trade_size`], [`4.425`], [`3.803`], [`3.278`], [`+1.147`], [`+0.526`],
  [`shared_mean`], [`3.462`], [`3.137`], [`2.760`], [`+0.703`], [`+0.377`],
)

Interpretation: the transfer signal replicated at `L32 settings_end` for
`trade_size` and `shared_mean`. Family specificity still matters:
`risk_preference` was weaker at this cell and `diversification_preference` did
not show clean ordering, so this is not every synthetic vector lighting up on
every complaint prompt.

= Why The First Proxy Was Too Coarse

The preregistered category proxy was `USER_CONFIG_CONFLICT` high and
`RULE_FABRICATION` low. That is not the right target for `trade_size`. Those
labels diagnose root cause; the probe target is visible conflict shape.

#callout(
  [CORRECTION],
  [
    Use action-shape labels for this audit: concrete current-prefix
    buy/sell/size conflict versus temporal, bookkeeping, fulfillment, or
    interpretation-layer conflict. Root-cause labels are metadata, not the
    primary semantic target.
  ],
  fill-color: warn,
  tag-color: accent,
)

#pagebreak()

#align(left)[
  #text(size: 7.3pt, fill: accent, tracking: 0.08em, weight: "medium")[DX TERMINAL / PHASE 13]
  #v(0.12em)
  #text(size: 15.5pt, weight: "bold", fill: ink)[Top/Bottom Readout]
  #v(0.25em)
  #line(length: 100%, stroke: 0.9pt + ink)
]

#v(0.25em)

= What The Probe Ranked High

Top/bottom slices were pulled from complaint rows only.

#table(
  columns: (1fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr),
  align: (left, center, center, center, center),
  inset: 4.1pt,
  stroke: 0.32pt + rgb("#CFD8DF"),
  fill: (x, y) => if y == 0 { rgb("#E7EEF3") } else if y == 1 or y == 2 { lift } else { soft },

  [*Direction*], [*Top action/size*], [*Top strategy*], [*Bottom action/size*], [*Bottom strategy*],
  [`trade_size`], [`20/25`], [`5/25`], [`15/25`], [`10/25`],
  [`shared_mean`], [`20/25`], [`5/25`], [`9/25`], [`16/25`],
)

Top `trade_size` complaint types: `UNWANTED_BUY` 10, `UNWANTED_SELL` 6,
`WRONG_SIZE` 4. The high `shared_mean` slice is the same row set as high
`trade_size`, but with lower projection magnitude. The clearest contrast is
low `shared_mean`, where diffuse `STRATEGY_IGNORED` rows dominate.

= Shape Of Low-Projection Conflicts

The bottom rows are not "agent did nothing." Reviewed strategy-ignored bottom
rows still contain buy/sell actions. Their conflict is different:

- no active strategy is visible, but the complaint references an old or
  expected strategy
- the current decision is already taking the requested action, so the complaint
  is retrospective
- the current action is an intermediate step in a multi-step strategy
- the disagreement is about entry price, timing, strategy fulfillment state, or
  rule interpretation rather than current visible trade size

#table(
  columns: (1.1fr, 1.45fr, 1.55fr),
  align: (left, left, left),
  inset: 4.1pt,
  stroke: 0.32pt + rgb("#CFD8DF"),
  fill: (x, y) => if y == 0 { rgb("#E7EEF3") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

  [*Slice*], [*Complaint shape*], [*Visible prompt/action shape*],
  [high `trade_size`], [wrong buy/sell or explicit size complaint], [current-prefix action/size tension],
  [high `shared_mean`], [same high row set as `trade_size`], [broader policy tension on concrete action conflicts],
  [low `trade_size`], [missed entry, old strategy, "why not execute?"], [no active strategy, aligned action, or history-dependent conflict],
  [low `shared_mean`], [strategy ignored / rule narrative], [temporal or interpretation-layer evidence dominates],
)

= Claim And Next Move

#callout(
  [CLAIM],
  [
    Fixed synthetic directions recover a real production signal at
    `L32 settings_end`. `trade_size` is selective for current-prefix concrete
    sized-action conflict. `shared_mean` tracks broader policy tension, while
    firing less on temporal, bookkeeping, and interpretation-layer complaints.
  ],
)

Residual risk: top/bottom labels are still an audit over projection extremes,
not a new gold dataset. Recommended next move: hand-label the same rows with
`current_action_size_conflict`, `retrospective_history_conflict`,
`strategy_fulfillment_conflict`, `interpretation_or_rule_conflict`, and
`unclear_or_label_mismatch`; then report enrichment against those labels before
expanding cells or moving toward causal tests.
