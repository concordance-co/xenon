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
  #text(size: 22pt, weight: "bold")[Post-Market Context Geometry Evidence]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    A synthesis report focused on one claim: even when a clean global real market frame does not transfer across examples,
    the model's #emph[post-market section states] can still carry clear, context-shaped geometry changes. The strongest real case is affordance.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[23 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[SCOPE]\ #text(size: 9pt)[synthetic ladders + real DX bridges]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[FOCUS]\ #text(size: 9pt)[where real context effects appear]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[MAIN CASE]\ #text(size: 9pt)[real affordance in post-market states]],
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
  #text(size: 7.5pt, fill: rgb("#b33a2a"), weight: "bold", tracking: 0.08em)[BOTTOM LINE]
  #v(0.3em)
  #text(size: 12.5pt, weight: "medium")[
    The clean real effect is #emph[not] “one market row changes its geometry when the setting changes.” The clean real effect is:
    after the model has read the market and moves through the settings, portfolio, and constraints blocks, the recovered geometry
    bends toward the context-shaped layout. That is strongest for #emph[affordance], especially in `constraints_eos`.
  ]
]


= What This Report Is Trying To Show

This report is narrower than the phase-by-phase writeups.

It is built around one question:

- when we say "post-market section states still carry real context-conditioned geometry changes," what exactly is the evidence?

The answer is spread across several experiments, so this report puts them into one chain:

1. controlled synthetic ladders show that context #emph[can] bend market geometry
2. an early real row-local bridge shows that reading only isolated market rows is too weak
3. the later real post-market bridge shows where the real signal actually appears


= A Plain-English Metric Definition

The main number in this report is the #emph[score-over-base margin].

You can read it like this:

- positive margin = the recovered geometry is closer to the #emph[context-shaped layout] than to the unchanged base layout
- zero margin = no clear preference
- negative margin = the recovered geometry still looks more like the unchanged base market

So bigger positive numbers mean:

- the context edit is showing up more clearly in the model's internal geometry


= The Evidence Chain

#align(center)[#image("../../data/report_assets/postmarket_context_geometry_evidence/evidence_chain.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Across the program, the strongest real signal is not the earlier row-local bridge. It is the later post-market affordance signal in downstream section states.
]

#v(0.4em)

This chart compresses the key story into six numbers:

- the synthetic ladders are positive, which shows that controlled context changes can bend geometry
- the real #emph[row-local] risk bridge is slightly negative (`-0.006`)
- the real #emph[post-market] risk bridge becomes positive (`0.076`)
- the real #emph[post-market affordance] bridge is the clearest result by far (`0.401`)

The important step is the middle one:

- moving from row-local reads to post-market section states changes the real story from "basically nothing" to "clear affordance-sensitive geometry"


= Why The Row-Local Real Bridge Was Not Enough

#align(center)[#image("../../data/report_assets/postmarket_context_geometry_evidence/rowlocal_vs_postmarket.png", width: 86%)]
#text(size: 8pt, fill: rgb("#888"))[
The early real bridge looked weak because it asked the wrong question. The stronger real effect appears after the market block, not inside single market-row states.
]

#v(0.4em)

The earlier real risk bridge used:

- `30` real examples
- row-local pooled states like `row_mean` and `row_eos`
- one ladder family: risk

Its best margin was still slightly negative:

- `-0.006` at `row_eos`, `risk_3`, `L4`

That does #emph[not] mean real context does nothing.

It means:

- real context is probably not applied in a strong, easy-to-read way inside isolated market-row states

That negative result was useful because it pushed the program to a better question:

- what happens #emph[after] the model has read the market and started combining it with settings, portfolio, and constraints?


= What Changed In The Post-Market Bridge

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 0.8em,
  [#block(inset: 9pt, fill: rgb("#faf7f2"), stroke: 1pt + rgb("#d6dee3"))[*Keep the market fixed* \ Same exact real 6-asset roster inside one laddered example.]],
  [#block(inset: 9pt, fill: rgb("#faf7f2"), stroke: 1pt + rgb("#d6dee3"))[*Edit only context* \ Change the risk text or route-availability text, not the market rows.]],
  [#block(inset: 9pt, fill: rgb("#faf7f2"), stroke: 1pt + rgb("#d6dee3"))[*Read later states* \ Measure `market_eos`, `active_settings_eos`, `portfolio_eos`, `constraints_eos`, and `last_token`.]],
  [#block(inset: 9pt, fill: rgb("#faf7f2"), stroke: 1pt + rgb("#d6dee3"))[*Compare two layouts* \ Ask whether the recovered geometry stays base-like or moves toward the context-shaped layout.]],
)

#v(0.5em)

That bridge used:

- `24` real risk base examples
- `24` real affordance base examples
- exact `6`-asset rosters, held fixed inside each ladder

This matters because the later section states have seen more of the prompt:

- first the market
- then the active settings
- then the portfolio
- then the constraints

If context is going to reshape the model's internal market picture, those states are where we should expect to see it.


= Concrete Affordance-Ladder Prompt Examples

The earlier inline prompt section used condensed excerpts to keep the main report readable. That was a presentation choice, not a source claim.

The #emph[raw verbatim prompts] are now included in an appendix at the end of this report:

- synthetic `market_only`
- synthetic `affordance_4`
- real `market_only`
- real `affordance_4`

Those appendix blocks are copied directly from the stored prompt data with no summarization.


= Where The Real Signal Lives

#align(center)[#image("../../data/report_assets/postmarket_context_geometry_evidence/real_state_summary.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
For the real post-market bridge, affordance is strong across many downstream states and strongest in `constraints_eos`. Risk is weaker and peaks later.
]

#v(0.4em)

The state-by-state picture is clean:

- #text(weight: "medium")[risk]
  becomes positive, but only modestly
- #text(weight: "medium")[affordance]
  becomes strongly positive in several downstream states

The best post-market real numbers are:

#table(
  columns: (1.25fr, 1.35fr, 1fr, 1fr, 1fr),
  align: (left, left, center, center, center),
  table.hline(stroke: 1pt),
  table.header([*Family*], [*Best state*], [*Context*], [*Layer*], [*Margin*]),
  table.hline(stroke: 0.5pt),
  [`real risk`], [`last_token`], [`risk_5`], [`37`], [`0.076`],
  [`real affordance`], [`constraints_eos`], [`affordance_4`], [`0`], [`0.401`],
)

#v(0.4em)

And the best state inside the real affordance bridge is not just one lucky point:

- `market_mean`: `0.262`
- `market_eos`: `0.249`
- `active_settings_eos`: `0.290`
- `portfolio_eos`: `0.246`
- `constraints_eos`: `0.401`
- `last_token`: `0.264`

So the real affordance result is broad across downstream section states, not a one-state fluke.


= The Full State-by-Context Picture

#align(center)[#image("../../data/report_assets/postmarket_context_geometry_evidence/real_context_heatmaps.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Each cell shows the best positive margin for one context and one prompt state. Affordance gets stronger as the ladder hardens and is concentrated in downstream section endpoints. Risk is weaker and much less uniform.
]

#v(0.4em)

These heatmaps make two things easier to see.

First, affordance grows in the way we would hope:

- `affordance_1` is already positive
- the harder affordance steps stay positive
- `constraints_eos` is especially strong at `affordance_4` and `affordance_5`

Second, risk is real but much weaker:

- small positives exist
- the cleanest values are concentrated at `risk_5`
- there is no equally broad, ladder-wide real risk pattern

So the real bridge does #emph[not] say "all context families behave the same."

It says:

- affordance reshapes the downstream geometry clearly
- risk does so only weakly and mostly at the extreme end


= How The Synthetic Phases Support This Read

The synthetic ladders matter because they tell us the geometry method itself is capable of seeing context-shaped movement.

Best synthetic margins were:

#table(
  columns: (1.6fr, 1.15fr, 1.2fr, 1fr, 1fr),
  align: (left, left, left, center, center),
  table.hline(stroke: 1pt),
  table.header([*Synthetic family*], [*Best state*], [*Context*], [*Layer*], [*Margin*]),
  table.hline(stroke: 0.5pt),
  [`risk ladder`], [`row_eos`], [`risk_3`], [`13`], [`0.030`],
  [`portfolio ladder`], [`row_eos`], [`portfolio_4`], [`14`], [`0.034`],
  [`affordance ladder`], [`row_eos`], [`affordance_4`], [`36`], [`0.072`],
)

#v(0.4em)

This ranking already hinted at what the real bridge later showed:

- affordance is the strongest context family
- portfolio is intermediate
- risk is weaker

So the real affordance result is not appearing from nowhere. It is consistent with the stronger synthetic family.


= What This Does And Does Not Claim

This synthesis report #emph[does] support the following claim:

- real post-market section states still carry context-shaped geometry changes
- that effect is strongest for affordance
- the signal is concentrated after the market block, especially in `constraints_eos` and nearby states

This report #emph[does not] claim:

- that one global real market coordinate system transfers cleanly across examples
- that risk and affordance behave the same
- that the effect is equally strong in isolated market-row states

That distinction matters.

The successful claim is the narrower one:

- the real model keeps carrying a market picture after the market section
- and later context can still bend that picture in a structured way


= The Cleanest One-Paragraph Reading

Across the earlier experiments, the row-local real bridge looked weak enough that it could have been tempting to conclude that real settings were not shaping geometry at all. This synthesis shows that would have been the wrong takeaway. The better reading is: the real signal mostly appears #emph[after] the market block, once the model has had time to combine the market with the settings, portfolio, and constraints. That later signal is strongest for affordance, especially in `constraints_eos`, where the best margin reaches `0.401`. So the strongest current claim is not "the model has one reusable real market frame." It is "post-market section states still carry real, context-shaped geometry changes, and affordance is the clearest case."


#pagebreak()

= Appendix: Raw Prompt Pair For Synthetic Affordance Ladder

#text(size: 9pt, fill: rgb("#666"))[
These blocks are verbatim raw prompt files. No wording has been shortened or rewritten.
]

== Synthetic `market_only` System

#raw(read("../../data/report_assets/postmarket_context_geometry_evidence/raw_prompts/synthetic_market_only_system.txt"), block: true, lang: "text")

== Synthetic `market_only` User

#raw(read("../../data/report_assets/postmarket_context_geometry_evidence/raw_prompts/synthetic_market_only_user.txt"), block: true, lang: "text")

== Synthetic `affordance_4` System

#raw(read("../../data/report_assets/postmarket_context_geometry_evidence/raw_prompts/synthetic_affordance_4_system.txt"), block: true, lang: "text")

== Synthetic `affordance_4` User

#raw(read("../../data/report_assets/postmarket_context_geometry_evidence/raw_prompts/synthetic_affordance_4_user.txt"), block: true, lang: "text")


#pagebreak()

= Appendix: Raw Prompt Pair For Real Affordance Ladder

#text(size: 9pt, fill: rgb("#666"))[
These blocks are verbatim raw prompt files exported from `research_rerun_prompts` for the matched real affordance example used in the report.
]

== Real `market_only` System

#raw(read("../../data/report_assets/postmarket_context_geometry_evidence/raw_prompts/real_market_only_system.txt"), block: true, lang: "text")

== Real `market_only` User

#raw(read("../../data/report_assets/postmarket_context_geometry_evidence/raw_prompts/real_market_only_user.txt"), block: true, lang: "text")

== Real `affordance_4` System

#raw(read("../../data/report_assets/postmarket_context_geometry_evidence/raw_prompts/real_affordance_4_system.txt"), block: true, lang: "text")

== Real `affordance_4` User

#raw(read("../../data/report_assets/postmarket_context_geometry_evidence/raw_prompts/real_affordance_4_user.txt"), block: true, lang: "text")
