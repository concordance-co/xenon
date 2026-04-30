#set page(
  paper: "us-letter",
  margin: (top: 1.15cm, bottom: 1.10cm, left: 1.35cm, right: 1.35cm),
  numbering: "1",
  number-align: right,
)
#set text(font: "Georgia", size: 8.2pt)
#set par(leading: 0.42em)
#set heading(numbering: none)

#let ink = rgb("#192129")
#let muted = rgb("#5d6871")
#let rule = rgb("#cfd8df")
#let head = rgb("#e8eef3")
#let band = rgb("#f8fafb")

#show heading.where(level: 1): it => {
  set text(size: 13pt, weight: "bold", fill: ink)
  v(0.58em)
  it
  v(0.18em)
  line(length: 100%, stroke: 0.7pt + rule)
  v(0.26em)
}

#show heading.where(level: 2): it => {
  set text(size: 9.2pt, weight: "bold", fill: ink)
  v(0.36em)
  it
  v(0.12em)
}

#let data-table(cols, ..args) = table(
  columns: cols,
  inset: 4.0pt,
  stroke: 0.32pt + rule,
  fill: (x, y) => if y == 0 { head } else if calc.odd(y) { band } else { white },
  ..args,
)

#let small-table(cols, ..args) = {
  set text(size: 6.9pt)
  table(
    columns: cols,
    inset: 2.7pt,
    stroke: 0.26pt + rule,
    fill: (x, y) => if y == 0 { head } else if calc.odd(y) { band } else { white },
    ..args,
  )
}

#let fig(path, title) = block(width: 100%)[
  #image(path, width: 100%)
  #v(0.08em)
  #text(size: 6.9pt, fill: muted)[#title]
]

#let codecell(body) = text(size: 6.4pt)[#body]

#align(left)[
  #text(size: 7.2pt, fill: muted, tracking: 0.08em)[DX TERMINAL / PROMPT CONFUSION]
  #v(0.10em)
  #text(size: 18pt, weight: "bold", fill: ink)[Policy Conflict Blog Data Packet]
  #v(0.06em)
  #text(size: 8.2pt, fill: muted)[Data-only packet from `POLICY_CONFLICT_BLOG_OUTLINE.md`.]
]

= 3. Synthetic Abstraction

#data-table(
  (1.05fr, 3.2fr),
  [*Family*], [*Results*],
  [`trade_size`], [XOR `0.9948 / 1.0000`; strategy holdout `1.0000 / 1.0000`; settings holdout `0.9948 / 1.0000`; strict combined `0.990 / 1.000` at L40],
  [`risk_preference`], [XOR `0.9635 / 0.9766`; strategy holdout `0.9844 / 0.9937`; settings holdout `0.9740 / 0.9839`; strict both-axes `0.8854 / 0.9119`],
  [`diversification_preference`], [aligned behavior `1.0000`; conflict behavior `0.8542`; XOR `0.9896 / 0.9995`; strategy holdout `1.0000 / 1.0000`; settings holdout `0.9792 / 0.9957`; strict both-axes `0.8333 / 0.8819`],
)

#grid(
  columns: (1fr, 1fr),
  gutter: 0.35cm,
  fig("../../phase_12/reports/dx_terminal_brief_assets/family_within_auroc_by_layer.png", [Family-within AUROC by layer]),
  fig("../../phase_12/reports/dx_terminal_brief_assets/strict_family_auroc_by_layer.png", [Strict family AUROC by layer]),
)

#fig("../../phase_12/reports/dx_terminal_brief_assets/strict_asset_family_auroc_by_layer.png", [Strict asset-family AUROC by layer])

= 4. Clean-Setting Result

#data-table(
  (2.4fr, 1.0fr),
  [*Pair*], [*L36 same-capture cosine*],
  [`risk_preference` vs `trade_size`], [`0.6449`],
  [`diversification_preference` vs `risk_preference`], [`0.4684`],
  [`diversification_preference` vs `trade_size`], [`0.4883`],
)

#data-table(
  (0.55fr, 1.0fr, 1.0fr, 1.0fr),
  [*Layer*], [*div_vs_risk*], [*div_vs_size*], [*risk_vs_size*],
  [`28`], [`0.4666`], [`0.4204`], [`0.5802`],
  [`36`], [`0.4684`], [`0.4883`], [`0.6449`],
  [`40`], [`0.3886`], [`0.4638`], [`0.5239`],
)

#grid(
  columns: (1fr, 1fr),
  gutter: 0.35cm,
  fig("../../phase_12/reports/three_family_visuals/shared_axis_distributions.png", [Shared-axis distributions]),
  fig("../../phase_12/reports/three_family_visuals/directed_subspace_scatter_by_family_conflict_v2.png", [Directed subspace scatter by family/conflict]),
)

= 5. Optional: Joint Prompt Result

#data-table(
  (1.5fr, 1fr),
  [*Readout*], [*Balanced accuracy / AUROC*],
  [`size_conflict_present`], [`0.9414 / 0.9862`],
  [`risk_conflict_present`], [`0.9388 / 0.9871`],
  [`any_conflict_present`], [`0.9306 / 0.9503`],
  [`double_conflict_present`], [`0.8898 / 0.9687`],
)

#fig("../../phase_12/reports/dx_terminal_brief_assets/phase11_joint_prompt_auroc_by_layer.png", [Phase 11 joint prompt AUROC by layer])

= 7. Bridge Program

#data-table(
  (1.35fr, 2.75fr, 0.48fr, 0.55fr, 0.55fr),
  [*Dataset*], [*Table*], [*Rows*], [*Aligned*], [*Conflict*],
  [Stage 1a template control], [#codecell[dx_terminal_trade_size_stage1a_template_control_v1]], [`768`], [`384`], [`384`],
  [Stage 1b loose adapter], [#codecell[dx_terminal_trade_size_stage1b_adapter_loose_v1]], [`258`], [`168`], [`90`],
  [Stage 1b strict adapter], [#codecell[dx_terminal_trade_size_stage1b_adapter_strict_v1]], [`118`], [`81`], [`37`],
  [Stage 1b strict buy-only], [#codecell[dx_terminal_trade_size_stage1b_adapter_strict_buy_only_v1]], [`33`], [`27`], [`6`],
)

== Stage 1b Loose Adapter Counts

#grid(
  columns: (1fr, 1fr),
  gutter: 0.35cm,
  small-table(
    (1.3fr, 0.55fr, 0.55fr),
    [*Complaint type*], [*Aligned*], [*Conflict*],
    [`GENERAL_PERFORMANCE`], [`59`], [`29`],
    [`NOT_TRADING`], [`30`], [`18`],
    [`STRATEGY_IGNORED`], [`21`], [`11`],
    [`UNWANTED_BUY`], [`15`], [`7`],
    [`CONFUSION`], [`13`], [`11`],
    [`UNWANTED_SELL`], [`12`], [`10`],
    [`FEATURE_EXPECTATION`], [`8`], [`0`],
    [`OVERTRADING`], [`4`], [`0`],
    [`WRONG_SIZE`], [`4`], [`3`],
    [`HOLDING_VIOLATION`], [`2`], [`1`],
  ),
  small-table(
    (1.5fr, 0.55fr, 0.55fr),
    [*Root cause*], [*Aligned*], [*Conflict*],
    [`RULE_FABRICATION`], [`44`], [`25`],
    [`USER_CONFIG_CONFLICT`], [`43`], [`20`],
    [`USER_EXPECTATION_MISMATCH`], [`32`], [`22`],
    [`CORRECT_BEHAVIOR`], [`21`], [`9`],
    [`PROMPT_FAILURE`], [`11`], [`7`],
    [`STRATEGY_SLIDER_LOCKOUT`], [`8`], [`2`],
    [`OVERTRADING`], [`5`], [`0`],
    [`MARKET_LEGITIMATE`], [`4`], [`5`],
  ),
)

== Stage 1b Strict Adapter Counts

#grid(
  columns: (1fr, 1fr),
  gutter: 0.35cm,
  small-table(
    (1.3fr, 0.55fr, 0.55fr),
    [*Complaint type*], [*Aligned*], [*Conflict*],
    [`GENERAL_PERFORMANCE`], [`21`], [`9`],
    [`NOT_TRADING`], [`16`], [`7`],
    [`STRATEGY_IGNORED`], [`11`], [`3`],
    [`UNWANTED_BUY`], [`9`], [`4`],
    [`UNWANTED_SELL`], [`6`], [`5`],
    [`FEATURE_EXPECTATION`], [`6`], [`0`],
    [`CONFUSION`], [`5`], [`5`],
    [`OVERTRADING`], [`3`], [`0`],
    [`WRONG_SIZE`], [`3`], [`3`],
    [`HOLDING_VIOLATION`], [`1`], [`1`],
  ),
  small-table(
    (1.5fr, 0.55fr, 0.55fr),
    [*Root cause*], [*Aligned*], [*Conflict*],
    [`RULE_FABRICATION`], [`22`], [`7`],
    [`USER_CONFIG_CONFLICT`], [`21`], [`11`],
    [`USER_EXPECTATION_MISMATCH`], [`17`], [`11`],
    [`CORRECT_BEHAVIOR`], [`8`], [`5`],
    [`PROMPT_FAILURE`], [`7`], [`1`],
    [`OVERTRADING`], [`2`], [`0`],
    [`MARKET_LEGITIMATE`], [`2`], [`1`],
    [`STRATEGY_SLIDER_LOCKOUT`], [`2`], [`1`],
  ),
)

= 8. Phase 13 Real Signal Discovery

#data-table(
  (1.05fr, 0.72fr, 0.82fr, 0.98fr, 0.98fr, 1.05fr),
  [*Direction*], [*Anchor*], [*Complaint*], [*Structure control*], [*Anchor-control*], [*Complaint-control*],
  [`trade_size`], [`4.425`], [`3.803`], [`3.278`], [`+1.147`], [`+0.526`],
  [`shared_mean`], [`3.462`], [`3.137`], [`2.760`], [`+0.703`], [`+0.377`],
)

= 9. Row Reading / Ontology Correction

#data-table(
  (1.2fr, 0.9fr, 0.9fr),
  [*Direction*], [*Top-25*], [*Bottom-25*],
  [`trade_size`: root cause `USER_CONFIG_CONFLICT`], [`17/25`], [`20/25`],
)

#data-table(
  (1.1fr, 0.85fr, 1fr, 0.95fr, 1.15fr),
  [*Direction*], [*Top action/size*], [*Top strategy ignored*], [*Bottom action/size*], [*Bottom strategy ignored*],
  [`trade_size`], [`20/25`], [`5/25`], [`15/25`], [`10/25`],
  [`shared_mean`], [`20/25`], [`5/25`], [`9/25`], [`16/25`],
)

#data-table(
  (1.55fr, 0.85fr),
  [*Top `trade_size` complaint type*], [*Count*],
  [`UNWANTED_BUY`], [`10/25`],
  [`UNWANTED_SELL`], [`6/25`],
  [`WRONG_SIZE`], [`4/25`],
  [Concrete action/size combined], [`20/25`],
)

= 10. What The Probe Seems To Read

#data-table(
  (1fr, 2.4fr),
  [*Bucket*], [*Examples / counts*],
  [High projection examples], ["why did you buy HOTDOGZ?"; "why did you buy so much POOPCOIN?"; "Buy available balance 30%, not 10 ETH"; "Quit buying tokens. Liquidate..."; "You are under allocated to POOPCOIN"],
  [No active strategy visible], [complaint: "why didn't you lock in some gains when i asked you to?"; prompt: `No active strategies.`; decision: sells HOTDOGZ 100%],
  [Agent already taking requested action], [complaint: "why you not buying AIGF as I said?"; active strategy: allocate all available ETH to AIGF; decision: buys AIGF 100%],
  [Multi-step execution], [complaint wants HOLE; strategy: sell POOPCOIN first, then deploy into HOLE; decision: sells POOPCOIN],
  [Strategy/rule interpretation], ["sell every time you have a profit"; "why is strategy 30 minutes, not 30 seconds"; "you bought higher than my entry"],
  [Partial execution], [complaint: "SELL IT FULL"; decision: sells 50%],
)

= Figure / Artifact Index

#small-table(
  (1.2fr, 2.8fr),
  [*Group*], [*Local artifact*],
  [Synthetic performance], [`phase_12/reports/dx_terminal_brief_assets/family_within_auroc_by_layer.png`],
  [Synthetic performance], [`phase_12/reports/dx_terminal_brief_assets/strict_family_auroc_by_layer.png`],
  [Synthetic performance], [`phase_12/reports/dx_terminal_brief_assets/strict_asset_family_auroc_by_layer.png`],
  [Three-family geometry], [`phase_12/reports/three_family_visuals/shared_axis_distributions.png`],
  [Three-family geometry], [`phase_12/reports/three_family_visuals/directed_subspace_scatter_by_family_conflict_v2.png`],
  [Joint prompt], [`phase_12/reports/dx_terminal_brief_assets/phase11_joint_prompt_auroc_by_layer.png`],
  [Bridge summaries], [`phase_12/outputs/transfer_bridge/*.json`],
  [Real transfer], [`phase_13/reports/PHASE13_REAL_TRANSFER_SIGNAL_BRIEF_2026_04_24.typ`],
)
