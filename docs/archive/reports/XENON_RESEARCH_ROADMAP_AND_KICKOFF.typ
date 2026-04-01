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

#show heading.where(level: 3): it => {
  set text(size: 10pt, weight: "bold")
  v(0.5em)
  it
  v(0.2em)
}

// ── Title Block ─────────────────────────────────────────────────
#align(left)[
  #text(size: 9pt, fill: rgb("#b33a2a"), tracking: 0.08em, weight: "medium")[XENON INTERPRETABILITY]
  #v(0.3em)
  #text(size: 22pt, weight: "bold")[Ranked Research Roadmap & Kickoff]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[A re-ranked research program after the synthetic geometry phases. This report argues for a shift from manifold-first work toward blocked-valence, settings-twist, and causal-decomposition work, then shows the live DB audit and kickoff manifests that make that shift executable immediately.]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[20 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[LIVE DECISIONS]\ #text(size: 9pt)[121,352]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[TOP TRACK]\ #text(size: 9pt)[Blocked valence + settings twist]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[KICKOFF BATCH]\ #text(size: 9pt)[154 rows]],
  )
  #v(0.3em)
  #line(length: 100%, stroke: 0.5pt + rgb("#ccc"))
]

#v(1em)

// ── Verdict ─────────────────────────────────────────────────────
#block(
  width: 100%,
  inset: (left: 14pt, top: 12pt, bottom: 12pt, right: 12pt),
  stroke: (left: 3pt + rgb("#b33a2a"), top: none, right: none, bottom: none),
  fill: rgb("#faf5f3"),
)[
  #text(size: 7.5pt, fill: rgb("#b33a2a"), weight: "bold", tracking: 0.08em)[MAIN READ]
  #v(0.3em)
  #text(size: 12.5pt, weight: "medium")[The best research path is no longer manifold-first.]
  #v(0.4em)
  #text(size: 9.5pt, fill: rgb("#555"))[The highest-value program now is: determine what the model wants to do, determine what stops it, and determine whether settings reweight that preference or merely gate execution. Geometry still matters, but as a simplifying lens for a behavioral and causal story rather than as the main scientific target.]
]

#v(1.2em)

// ── Summary Metrics ─────────────────────────────────────────────
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BLOCKED POOL]\ #text(size: 16pt, weight: "bold")[59,468] #text(size: 8pt, fill: rgb("#888"))[\ live candidate rows]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[POLICY-TENSION POOL]\ #text(size: 16pt, weight: "bold")[30,115] #text(size: 8pt, fill: rgb("#888"))[\ live candidate rows]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BLOCKED KICKOFF]\ #text(size: 16pt, weight: "bold")[34] #text(size: 8pt, fill: rgb("#888"))[\ unique-vault starter batch]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[SETTINGS KICKOFF]\ #text(size: 16pt, weight: "bold")[120] #text(size: 8pt, fill: rgb("#888"))[\ 72 observe + 48 refs]],
)


= Why The Program Should Shift

The current evidence stack no longer supports treating manifold discovery as the center of gravity.

- Decision-structure work showed *where* action-relevant signal lives, but not what intermediate quantities the model actually computes.
- Synthetic scalar and coupled geometry work showed some low-dimensional order, but not a single clean market manifold.
- The strongest current unresolved questions are behavioral and causal:
  - is `observe` neutral or blocked?
  - are settings late gates or true reinterpretation operators?
  - what variables are actually necessary for the model to choose an action?

This report therefore re-ranks the research tracks around those questions.


= Ranked Research Tracks

#align(center)[#image("../../data/report_assets/research_kickoff/roadmap_scores.png", width: 94%)]
#text(size: 8pt, fill: rgb("#888"))[Priority scores for the next research tracks. The top-ranked work is blocked valence plus settings twist, followed by causal necessity of the strongest market variables.]

#set table(stroke: none)
#table(
  columns: (0.7fr, 1.6fr, 0.8fr, 2.4fr),
  align: (left, left, right, left),
  table.hline(stroke: 1pt),
  table.header(
    [*Rank*], [*Track*], [*Score*], [*Why it belongs here*],
  ),
  table.hline(stroke: 0.5pt),
  [`1`], [`Blocked valence + settings twist`], [9.7], [Shortest path from localization to mechanism; creates the missing labels for asset valence and directly tests whether settings reweight preference or only gate execution.],
  [`2`], [`Causal necessity of market variables`], [9.2], [The synthetic work has already surfaced the strongest candidate variables; the next gain comes from patching and ablation rather than more passive decoding.],
  [`3`], [`Real-data decision decomposition`], [8.8], [Reconnects synthetic findings to real DX-terminal prompts and separates true neutral observe cases from blocked-latent-sentiment cases.],
  [`4`], [`Policy vs perception routing`], [7.9], [Potentially cleaner mechanistic handle if experts split along market parsing versus policy handling.],
  [`5`], [`Geometry support track`], [6.6], [Still useful, but should simplify causal stories rather than define the main agenda.],
  table.hline(stroke: 1pt),
)


= Top Hypotheses

The most interesting hypotheses to explore next are:

- *Early preference, late permission.* Market preference forms before legality and affordance gating.
- *Blocked observe hides valence.* Many observe cases are actually bullish or bearish states that cannot be executed.
- *Settings reweight, not rewrite.* Settings mostly rotate or sharpen existing preference structure rather than creating preference from scratch.
- *Flow matters jointly.* Momentum × flow is a stronger candidate mechanism than flow alone.
- *Participation modulates confidence.* Participation shapes whether momentum is trusted rather than acting as a primary preference axis.
- *Concentration is policy-facing.* Concentration matters more through risk or policy pathways than as a clean early perceptual variable.


= Live Candidate Audit

The live Neon pool is already large enough to support the top-ranked program.

#align(center)[#image("../../data/report_assets/research_kickoff/candidate_pools.png", width: 92%)]
#text(size: 8pt, fill: rgb("#888"))[The raw pool is not the bottleneck anymore. There are already tens of thousands of blocked-observe and policy-tension examples available for the next research phase.]

#align(center)[#image("../../data/report_assets/research_kickoff/blocked_reasons.png", width: 92%)]
#text(size: 8pt, fill: rgb("#888"))[The blocked-observe pool is huge but structurally narrow: most cases are strategy-driven, which means blocked-valence reruns should focus on deconstraint and strategy-removal variants rather than assuming a broad natural mix.]

#align(center)[#image("../../data/report_assets/research_kickoff/policy_regimes.png", width: 92%)]
#text(size: 8pt, fill: rgb("#888"))[Policy-tension cases cluster heavily in a few extreme settings regimes, especially `R1:A1` and `R5:A5`. That is useful rather than bad: it means the first settings-twist pass can focus on high-contrast cells instead of diffuse mid-range settings.]

Most important audit reads:

- The blocked pool is large (`59,468`) but dominated by `high_strategy_present` cases (`53,559`), with smaller secondary blocks from explicit buy/sell restrictions.
- The policy-tension pool is also large (`30,115`) and highly actionable: `28,500` cases still have both buy and sell available.
- Natural settings diversity is concentrated, not uniform. Extreme settings cells dominate the pool, which is exactly what the first twist experiments should exploit.


= Kickoff Manifests

The kickoff is already underway, and the live audit changed the operational design: one combined manifest was the wrong abstraction because observe-heavy rows were competing with trade-reference rows for the same vault budget. The correct shape is two manifests.

#align(center)[#image("../../data/report_assets/research_kickoff/kickoff_manifests.png", width: 88%)]
#text(size: 8pt, fill: rgb("#888"))[The kickoff now splits into two manifests: a blocked-valence starter batch and a larger settings-twist batch with explicit trade references.]

#table(
  columns: (1.4fr, auto, auto, auto),
  align: (left, right, right, left),
  table.hline(stroke: 1pt),
  table.header(
    [*Manifest*], [*Rows*], [*Unique vaults*], [*Use*],
  ),
  table.hline(stroke: 0.5pt),
  [`Blocked valence kickoff`], [34], [34], [Deconstraint / strategy-removal reruns on blocked observe cases.],
  [`Settings twist kickoff`], [120], [102], [72 policy-tension observe cases plus 24 buy and 24 sell reference trades.],
  table.hline(stroke: 1pt),
)

#v(0.4em)

Interpretation:

- The blocked-valence starter is intentionally smaller because the highest-priority blocked pool is structurally concentrated.
- The settings-twist starter is larger because the candidate pool is both large and highly actionable.
- This is enough to begin the top-ranked research track without waiting for more raw data or more geometry work.


= Immediate Next Experiments

The next concrete experiments should be:

+ rerun the blocked-valence kickoff batch under deconstraint or strategy removal
+ rerun the settings-twist kickoff batch under high-contrast settings rewrites
+ compare `row_mean_i`, `active_settings_eos`, `constraints_eos`, and `last_token`
+ use the rerun outcomes to build the first blocked-bullish and blocked-bearish labels

Only after those experiments are in place should the program widen again into broader geometry or routing work.


= What To Deprioritize

- broad manifold search without a behavioral decomposition
- more generic buy/sell probes without blocked-valence labels
- treating `observe` as uniformly neutral
- expanding synthetic geometry breadth before testing the top behavioral hypotheses


= Conclusion

The live data and the current result stack now point in the same direction:

- the research program should be organized around *decision decomposition*, not just geometry
- the best immediate target is *blocked valence plus settings twist*
- the kickoff is already materially underway with live candidate audits and runnable starter manifests

This is a better place to spend the next iteration than another round of broad manifold hunting.

#v(2em)
#line(length: 100%, stroke: 0.5pt + rgb("#ccc"))
#v(0.3em)
#text(size: 8pt, fill: rgb("#999"))[Ranked Research Roadmap & Kickoff — 20 March 2026. Based on live Neon audit results plus prior decision-structure and synthetic geometry findings.]
