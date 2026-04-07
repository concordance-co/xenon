#set page(
  paper: "us-letter",
  margin: (top: 1.8cm, bottom: 1.8cm, left: 2.0cm, right: 2.0cm),
  numbering: "1",
  number-align: right,
)

#set text(font: "Georgia", size: 10.5pt)
#set par(justify: true, leading: 0.72em)
#set heading(numbering: none)

#show heading.where(level: 1): it => {
  set text(size: 17pt, weight: "bold")
  v(1.0em)
  it
  v(0.3em)
}

#show heading.where(level: 2): it => {
  set text(size: 12.2pt, weight: "bold")
  v(0.8em)
  it
  v(0.25em)
}

#let summary = json("../../data/report_assets/synthetic_market_public_story/summary.json")
#let headline = summary.at("headline_numbers")
#let claims = summary.at("validated_claims")
#let figures = summary.at("figures")
#let client = summary.at("client_summary")

#let mono(text_value, size: 11pt, weight: "bold", fill: black) = {
  text(font: "Menlo", size: size, weight: weight, fill: fill)[#text_value]
}

#let pct(x) = {
  let y = calc.round(x * 1000) / 10
  str(y) + "%"
}

#let pp(x) = {
  let y = calc.round(x * 10000) / 100
  if y > 0 {
    "+" + str(y) + " pp"
  } else {
    str(y) + " pp"
  }
}

#let callout(title, body, fill: rgb("#f7f4f1"), accent: rgb("#9d3c2a")) = block(
  width: 100%,
  inset: (left: 12pt, right: 12pt, top: 10pt, bottom: 10pt),
  fill: fill,
  stroke: (left: 3pt + accent, top: none, right: none, bottom: none),
)[
  #text(size: 7.5pt, tracking: 0.08em, fill: accent, weight: "bold")[#title]
  #v(0.25em)
  #body
]

#align(left)[
  #text(size: 9pt, fill: rgb("#9d3c2a"), tracking: 0.08em, weight: "medium")[CLIENT BRIEF]
  #v(0.25em)
  #text(size: 24pt, weight: "bold")[The Model Reads the Market. Here Is What We Can Prove.]
  #v(0.35em)
  #text(size: 11pt, fill: rgb("#47433f"))[
    A concise summary of what held up --- and what did not --- across nine phases of research into how a trading model forms and acts on its internal picture of a market.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.1pt + black)
]

#v(0.9em)

#callout(
  [BOTTOM LINE],
  [
    #text(size: 12.4pt, weight: "medium")[
      The model builds a genuine internal picture of the market, and that picture shifts when earlier context changes the frame. But no single internal signal we isolated was enough to explain the model's final trading choice. The honest conclusion: real structure, no simple switch.
    ]
  ],
)

= What Held Up

#table(
  columns: (2.0fr, 0.9fr, 3.0fr),
  align: (left, center, left),
  table.hline(stroke: 1pt),
  table.header([*Claim*], [*Support*], [*Why it matters*]),
  table.hline(stroke: 0.5pt),
  ..claims.map(row => (
    [#row.at("claim")],
    [#row.at("support")],
    [#row.at("evidence")],
  )).flatten(),
)

= The Three Most Important Results

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 0.9em,
  [
    #callout(
      [CONTEXT SHAPES THE READ],
      [
        The same market, read through different framing, produced a different internal picture. Context placed before the market changed the reading. Context placed after did not.
        #v(0.35em)
        Risk framing gap: #mono(pp(headline.at("risk_gap")), size: 10pt)
        #linebreak()
        Opportunity framing gap: #mono(pp(headline.at("affordance_gap")), size: 10pt)
      ],
      fill: rgb("#f8f5f1"),
      accent: rgb("#8a5030"),
    )
  ],
  [
    #callout(
      [THE SIGNALS ARE REAL],
      [
        Targeted edits to the model's internal market signals behaved differently from equally-sized random edits --- in every single main comparison.
        #v(0.35em)
        Selectivity: #mono(str(headline.at("phase20_selectivity_wins")) + " / " + str(headline.at("phase20_total_comparisons")), size: 10pt) comparisons won
      ],
      fill: rgb("#f4f6f3"),
      accent: rgb("#2f6b4f"),
    )
  ],
  [
    #callout(
      [NO SINGLE DECISIVE LEVER],
      [
        When we tried to restore the source behavior by putting matching signal back in, the effect was modest at best. The model's decision draws on a richer internal summary.
        #v(0.35em)
        Stronger signal: #mono(pp(headline.at("phase21_leader_match_delta")), size: 10pt)
      ],
      fill: rgb("#f6f1f0"),
      accent: rgb("#b6523a"),
    )
  ],
)

#v(0.6em)

#align(center)[#image(figures.at("phase16_perception"), width: 100%)]
#text(size: 8pt, fill: rgb("#7a746e"))[
The context-ordering effect in detail. When context came before the market (solid lines), the model's internal market read shifted. When the same context came after (dashed lines), the read stayed put. This is the clearest evidence that the model's first impression of the market depends on prior framing.
]

#v(0.4em)

#align(center)[#image("../../data/report_assets/synthetic_market_phase20_paired_robustness/phase20_lambda1_selectivity_gaps.png", width: 90%)]
#text(size: 8pt, fill: rgb("#7a746e"))[
Selectivity gaps from the robustness battery. Positive values mean the random edit was more disruptive than the targeted edit. In every main comparison, the targeted edit was gentler --- meaning the internal market signals are structured, not arbitrary.
]

= What The Restoration Test Showed

The strongest test was simple in concept and demanding in practice: take the internal signal from a source example and transplant it into the base prompt. If that signal is the reason the model chose a particular asset, restoring it should push the model back toward the source's choice.

#align(center)[#image(figures.at("phase21_comparison"), width: 92%)]
#text(size: 8pt, fill: rgb("#7a746e"))[
The restoration test. The stronger candidate signal produced a modest positive shift. The weaker signal did not. Neither looked like the decisive cause of the model's trading choice.
]

#table(
  columns: (1.7fr, 1fr, 1fr, 1fr, 1.5fr),
  align: (left, center, center, center, left),
  table.hline(stroke: 1pt),
  table.header([*Signal tested*], [*Change in source agreement*], [*Help rate*], [*Harm rate*], [*Public reading*]),
  table.hline(stroke: 0.5pt),
  [Stronger signal],
  [#pp(headline.at("phase21_leader_match_delta"))],
  [#pct(headline.at("phase21_leader_fix_rate"))],
  [#pct(headline.at("phase21_leader_harm_rate"))],
  [Interesting, but still too modest for a strong causal claim],
  [Weaker signal],
  [#pp(headline.at("phase21_dispersion_match_delta"))],
  [#pct(headline.at("phase21_dispersion_fix_rate"))],
  [#pct(headline.at("phase21_dispersion_harm_rate"))],
  [Not a convincing driver of the final choice],
)

= Recommended Public Framing

#callout(
  [SAFE EXTERNAL FRAMING],
  [#text(size: 12pt, weight: "medium")[#client.at("recommended_public_framing")]],
  fill: rgb("#f3f0eb"),
  accent: rgb("#2f6b4f"),
)

#v(0.3em)

A final path-validation test (lesion at one layer, rescue at a later one) was mechanically clean but produced too sparse a behavioral readout to draw conclusions. This does not weaken the results above; it marks where the next round of work begins.

= Talking Points By Audience

*For a technical audience:*

- The model builds a measurable internal market picture that persists across prompt variants and survives format-decontamination.
- Earlier framing demonstrably shifts that picture; later framing does not.
- The two strongest candidate signals are real and selective --- but they did not survive the strongest causal test (source-matching restoration).
- The decision surface appears richer than any single internal direction can capture.

*For a business audience:*

- This work produced a clearer map of how the model reads and reacts to markets.
- It also rules out the simplest story: there is no one hidden switch behind the trading choice.
- The conclusions are narrower than a headline claim because each finding was tested against tighter controls until it either held or broke.

= Why This Outcome Is Strong

A narrower conclusion backed by data is more useful than a dramatic claim that cannot survive scrutiny.

The data supports three things clearly: the model's internal market picture is real, prior framing changes it, and the internal signals we found are structured rather than arbitrary. The data also shows one thing clearly negative: neither candidate signal, on its own, explains the final trading choice.

That gives future work a precise starting point and a clear boundary.
