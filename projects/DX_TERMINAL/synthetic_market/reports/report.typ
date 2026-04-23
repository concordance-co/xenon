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
  set text(size: 18pt, weight: "bold")
  v(1.2em)
  it
  v(0.35em)
}

#show heading.where(level: 2): it => {
  set text(size: 12.5pt, weight: "bold")
  v(0.9em)
  it
  v(0.25em)
}

#let summary = json("../../../../data/report_assets/synthetic_market_public_story/summary.json")
#let phases = summary.at("phases")
#let headline = summary.at("headline_numbers")
#let claims = summary.at("validated_claims")
#let figures = summary.at("figures")

#let fmt3(x) = {
  let y = calc.round(x * 1000) / 1000
  str(y)
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

#let mono(text_value, size: 11pt, weight: "bold", fill: black) = {
  text(font: "Menlo", size: size, weight: weight, fill: fill)[#text_value]
}

#let note-card(title, body, fill: rgb("#f6f2ef"), stroke_fill: rgb("#b6523a")) = block(
  width: 100%,
  inset: (left: 12pt, right: 12pt, top: 10pt, bottom: 10pt),
  fill: fill,
  stroke: (left: 3pt + stroke_fill, top: none, right: none, bottom: none),
)[
  #text(size: 7.5pt, tracking: 0.08em, fill: stroke_fill, weight: "bold")[#title]
  #v(0.25em)
  #body
]

#let summary-card(title, value, detail) = block(
  width: 100%,
  inset: (left: 8pt, right: 8pt, top: 8pt, bottom: 8pt),
  fill: rgb("#f9f7f4"),
  stroke: 0.4pt + rgb("#ddd7d2"),
)[
  #text(size: 7pt, fill: rgb("#7a746e"), weight: "bold", tracking: 0.08em)[#title]
  #v(0.2em)
  #value
  #v(0.25em)
  #text(size: 8pt, fill: rgb("#625d58"))[#detail]
]

#align(left)[
  #text(size: 9pt, fill: rgb("#9d3c2a"), tracking: 0.08em, weight: "medium")[PUBLIC RESEARCH STORY]
  #v(0.25em)
  #text(size: 25pt, weight: "bold")[The Model Sees the Market. We Tried to Find Out Why It Trades the Way It Does.]
  #v(0.35em)
  #text(size: 11pt, fill: rgb("#47433f"))[#summary.at("deck")]
  #v(0.8em)
  #line(length: 100%, stroke: 1.2pt + black)
  #v(0.45em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.9em,
    [#text(size: 7pt, fill: rgb("#7a746e"), weight: "bold")[DATE]\ #text(size: 9pt)[#summary.at("date")]],
    [#text(size: 7pt, fill: rgb("#7a746e"), weight: "bold")[SPAN]\ #text(size: 9pt)[Nine phases]],
    [#text(size: 7pt, fill: rgb("#7a746e"), weight: "bold")[RESULT]\ #text(size: 9pt)[Real internal market picture; no single decisive cause]],
    [#text(size: 7pt, fill: rgb("#7a746e"), weight: "bold")[STATUS]\ #text(size: 9pt)[Validated through path test]],
  )
]

#v(0.9em)

#note-card(
  [TOP LINE],
  [#text(size: 12.5pt, weight: "medium")[#summary.at("top_line")]],
)

#v(0.9em)

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 0.8em,
  [
    #summary-card(
      [EARLY CONTEXT EFFECT],
      [#mono(pp(headline.at("risk_gap")), size: 12pt)],
      [Risk framing changed the market read when it appeared before the market block.],
    )
  ],
  [
    #summary-card(
      [OPPORTUNITY EFFECT],
      [#mono(pp(headline.at("affordance_gap")), size: 12pt)],
      [Opportunity framing produced a similarly sized shift.],
    )
  ],
  [
    #summary-card(
      [STRONGEST ROBUSTNESS RESULT],
      [#mono(str(headline.at("phase20_selectivity_wins")) + " / " + str(headline.at("phase20_total_comparisons")), size: 12pt)],
      [Targeted edits beat matched random edits in every main comparison.],
    )
  ],
  [
    #summary-card(
      [FINAL SOURCE-MATCH TEST],
      [#mono(pp(headline.at("phase21_leader_match_delta")), size: 12pt)],
      [The stronger signal helped a little, but not enough to count as the cause of the final choice.],
    )
  ],
)

= Why This Study Matters

Most interpretability work stops at discovery. You find a pattern inside a model, you name it, and the pattern becomes the explanation.

That is not enough. A signal that looks meaningful might just be wallpaper --- something the model computes along the way but never actually leans on when it reaches for a decision. Finding a readable internal pattern and proving that it drives behavior are very different things. Confusing them is the easiest mistake in this field, and the most consequential.

This study applied a sequence of increasingly hard tests to the same set of internal signals:

- find a pattern
- confirm it survives after removing prompt-format confounds
- test whether earlier context bends it
- test whether editing it changes behavior
- test whether restoring it moves behavior back toward a known source
- test whether a surgical lesion-and-rescue reveals a causal path

That last test is where many appealing stories fall apart. In our case, that is largely what happened --- and that is why the result is useful.

#align(center)[#image("../../../../data/report_assets/synthetic_market_phase16_17_combined/combined_methodology_overview.png", width: 100%)]
#text(size: 8pt, fill: rgb("#7a746e"))[
The methodology at a glance. Each phase tightened controls and raised the standard of evidence required before the next test.
]

= Phase Summary

#table(
  columns: (0.8fr, 1.3fr, 2.0fr, 2.9fr, 1.3fr),
  align: (left, left, left, left, left),
  table.hline(stroke: 1pt),
  table.header([*Phase*], [*What we asked*], [*What we tested*], [*What held up*], [*Confidence*]),
  table.hline(stroke: 0.5pt),
  ..phases.map(phase => (
    [#phase.at("phase")],
    [#phase.at("title")],
    [#phase.at("question")],
    [#phase.at("what_held_up")],
    [#phase.at("confidence")],
  )).flatten(),
)

#v(0.45em)

Notice the confidence column: early optimism, growing precision, and a final reckoning with what the evidence actually supports. The rest of this report fills in the data behind each row.

= Discovery: The Model Builds An Internal Market Picture

The synthetic prompts mirror the real task: the same system prompt style, the same section order, the same market framing, six assets per market. The question was whether the model's hidden activations contain a recognizable picture of the market it just read.

They do. Two strong candidate patterns appeared consistently across the prompt set.

#align(center)[#image(figures.at("phase15_discovery"), width: 92%)]
#text(size: 8pt, fill: rgb("#7a746e"))[
The first discovery pass found clear market-linked internal patterns. Two recurring signals stood out, both linked to visible market features.
]

Some of that initial signal was tangled with prompt-format effects. After removing the part of the internal signal predictable from prompt shape, the same two patterns survived --- cleaner and stronger. The average-market read was especially robust. The end-of-market read was noisier.

#align(center)[#image(figures.at("phase15_rerun"), width: 92%)]
#text(size: 8pt, fill: rgb("#7a746e"))[
After removing format confounds, the model's average-market summary held up cleanly. The same two candidate patterns survived the decontamination.
]

That gave us two confirmed internal signals worth testing further.

= Context Sensitivity: Prior Framing Changes The Market Read

Is the model's market picture fixed, or does it depend on what the model has already been told?

We held the market constant and moved the same block of additional context either before or after the market section.

When the context came after the market, the market read stayed the same. When the same context came before the market, the model's reading shifted.

#align(center)[#image(figures.at("phase16_perception"), width: 100%)]
#text(size: 8pt, fill: rgb("#7a746e"))[
The same context matters when it appears before the market, not when it appears after. This is direct evidence that the model's first impression of the market depends on prior framing.
]

The size of that shift was not enormous, but it was real:

- risk framing produced a gap of #mono(pp(headline.at("risk_gap")), size: 10pt)
- opportunity framing produced a gap of #mono(pp(headline.at("affordance_gap")), size: 10pt)

The key result is the asymmetry:

- after-market context did not retroactively rewrite the market picture
- before-market context changed how the model finished reading the market block

The model's market read is not a passive snapshot. It is shaped by the lens through which the model approaches the data. Change the lens, and the same numbers land differently.

#align(center)[#image("../../../../data/report_assets/synthetic_market_phase16_17_combined/combined_basis_shift.png", width: 92%)]
#text(size: 8pt, fill: rgb("#7a746e"))[
The internal basis shifted measurably when context appeared before the market. When the same context appeared after, the basis stayed put. This is the fingerprint of a framing effect, not a retroactive rewrite.
]

= Axis Decomposition: What The Two Signals Mean

What do the two strongest recurring internal patterns correspond to in the visible market?

To answer this, we matched the model's internal patterns only against features a human reader could also see in the prompt --- no hidden labels, no synthetic ground truth.

The result was intuitive, but not reductive:

- one recurring pattern looked like a standout asset with strong activity around it
- the other looked like how uneven the market was overall

#align(center)[#image(figures.at("phase17_breakdown"), width: 100%)]
#text(size: 8pt, fill: rgb("#7a746e"))[
The two strongest internal patterns line up with ordinary market descriptions, but neither collapses into one perfect formula. That turned out to matter later.
]

Two concrete markers stood out:

- the clearest visible clue for the first pattern was #mono(str(headline.at("leader_feature").at("feature")), size: 10pt)
- the clearest visible clue for the second pattern was #mono(str(headline.at("dispersion_feature").at("feature")), size: 10pt)

These visible features are not the full explanation --- but they show that the model's hidden market summary can be described in ordinary market language.

The natural next question: do these signals actually drive the model's decisions, or are they just a readable byproduct?

= Intervention: Does Editing The Signal Change The Choice?

We weakened the candidate signals directly inside the model's hidden state and measured whether the trading choice changed.

Single-direction edits were weak. Broader edits across a cluster of related directions produced stronger effects.

#align(center)[#image(figures.at("phase18_change_rates"), width: 100%)]
#text(size: 8pt, fill: rgb("#7a746e"))[
Single narrow edits were weak. Broader groups of related edits mattered more. That pushed us away from the idea of one neat hidden switch.
]

In practical terms:

- the broader early-market edit changed the chosen asset at a rate of #mono(pct(headline.at("phase18_leader_4d_targeted")), size: 10pt), versus #mono(pct(headline.at("phase18_leader_4d_control")), size: 10pt) for its matched random edit
- the broader later-market edit changed the chosen asset at a rate of #mono(pct(headline.at("phase18_dispersion_4d_targeted")), size: 10pt), versus #mono(pct(headline.at("phase18_dispersion_4d_control")), size: 10pt) for its matched random edit

That pointed toward a broader internal market summary rather than a single hidden dial.

With tighter controls and deterministic decoding, the result held but remained modest:

- targeted edits changed the chosen asset at #mono(pct(headline.at("phase19_targeted_choice_change")), size: 10pt)
- matched random edits changed it at #mono(pct(headline.at("phase19_control_choice_change")), size: 10pt)

Real signal, but not enough for a strong causal claim. The obvious question: are the targeted edits actually special, or would any perturbation of the same size do the same thing?

= Robustness: Targeted Edits Versus Matched Random Edits

We tested this with matched prompt pairs, both weaker-side and stronger-side variants, several edit strengths, and matched random controls.

The targeted edits behaved differently from equally sized arbitrary edits.

#align(center)[#image(figures.at("phase20_selectivity"), width: 90%)]
#text(size: 8pt, fill: rgb("#7a746e"))[
Positive values mean the random edit was more disruptive than the targeted edit. This was the cleanest evidence that the internal market signals were real and selective rather than arbitrary noise.
]

In all #mono(str(headline.at("phase20_selectivity_wins")) + " / " + str(headline.at("phase20_total_comparisons")), size: 10pt) main choice comparisons, the targeted edit was less disruptive than the matched random edit.

This rules out the cheapest counter-explanation. The targeted edits behaved in a structurally different way from arbitrary matched edits of the same magnitude.

#align(center)[#image("../../../../data/report_assets/synthetic_market_phase20_paired_robustness/phase20_tool_token_change_curves.png", width: 92%)]
#text(size: 8pt, fill: rgb("#7a746e"))[
Tool-token change rates across edit strengths. The targeted and random curves separate cleanly: the model's response to a targeted edit follows a different pattern than its response to a random one of the same size.
]

But selectivity is not causation. The signals are real and structured --- but are they the reason the model makes the choices it makes?

= Restoration: Can We Move Behavior Back Toward A Known Source?

The previous tests all weakened signals and watched for damage. The restoration test reversed the logic: take the internal market signal from a matched source example and transplant it into the base prompt. If the signal is truly the reason the model chose a particular asset, restoring it should push the model back toward the source example's behavior.

This is the gold standard for causal claims, and it is where most appealing stories fail.

#align(center)[#image(figures.at("phase21_comparison"), width: 92%)]
#text(size: 8pt, fill: rgb("#7a746e"))[
The stronger candidate helped a little. The weaker candidate did not. Neither behaved like the single decisive cause of the final trading choice.
]

The stronger candidate signal gave a modest positive result:

- agreement with the source choice rose by #mono(pp(headline.at("phase21_leader_match_delta")), size: 10pt)
- on rows that needed help, it moved the model toward the source choice #mono(pct(headline.at("phase21_leader_fix_rate")), size: 10pt) of the time
- on rows that were already right, it made things worse #mono(pct(headline.at("phase21_leader_harm_rate")), size: 10pt) of the time

The weaker candidate signal did not hold up:

- agreement with the source choice moved by #mono(pp(headline.at("phase21_dispersion_match_delta")), size: 10pt)
- its help rate was #mono(pct(headline.at("phase21_dispersion_fix_rate")), size: 10pt)
- its harm rate was #mono(pct(headline.at("phase21_dispersion_harm_rate")), size: 10pt)

#table(
  columns: (1.8fr, 1fr, 1fr, 1fr, 1.4fr),
  align: (left, center, center, center, left),
  table.hline(stroke: 1pt),
  table.header([*Signal tested*], [*Change in source agreement*], [*Help rate*], [*Harm rate*], [*Verdict*]),
  table.hline(stroke: 0.5pt),
  [Stronger market signal],
  [#pp(headline.at("phase21_leader_match_delta"))],
  [#pct(headline.at("phase21_leader_fix_rate"))],
  [#pct(headline.at("phase21_leader_harm_rate"))],
  [Interesting, but still modest],
  [Weaker market signal],
  [#pp(headline.at("phase21_dispersion_match_delta"))],
  [#pct(headline.at("phase21_dispersion_fix_rate"))],
  [#pct(headline.at("phase21_dispersion_harm_rate"))],
  [Not persuasive],
)

The restoration test is where the overall conclusion narrowed. The internal market structure is real. Context bends it. Targeted edits are selective. But none of the candidate signals we isolated could reliably move behavior back toward a known source.

= Path Validation: Lesion And Rescue

One final test: instead of restoring a signal directly, damage it at an early layer and attempt a downstream rescue at a later layer using source-side coefficients. If the two sites are part of the same causal pathway, the rescue should partially undo the lesion.

The interventions applied cleanly, but the behavioral surface was too sparse to measure action-choice changes with confidence --- the lesioned model did not reliably produce parsed tool calls. The path-validation question remains open for a future test at a scale where the readout is dense enough to interpret.

= What We Can Now Say With Confidence

#table(
  columns: (2.0fr, 0.9fr, 3.0fr),
  align: (left, center, left),
  table.hline(stroke: 1pt),
  table.header([*Claim*], [*Support*], [*Why this held up*]),
  table.hline(stroke: 0.5pt),
  ..claims.map(row => (
    [#row.at("claim")],
    [#row.at("support")],
    [#row.at("evidence")],
  )).flatten(),
)

The most important positive result is also the simplest: the model builds a real internal picture of the market, and that picture changes when earlier context shifts the frame. Everything else --- the intervention tests, the robustness batteries, the restoration attempts --- hangs from that central finding.

= What The Evidence Does Not Support

We should not claim:

- that we found the one hidden cause of the final trading choice
- that either of the two named signals by itself explains the decision
- that every readable internal pattern is also a decisive behavioral lever

The data rules these out. The restoration test was designed specifically to catch overreach, and it did.

= The Final Conclusion

#note-card(
  [FINAL CONCLUSION],
  [
    #text(size: 12.4pt, weight: "medium")[
      The model builds a meaningful internal summary of the market, and that summary is influenced by context.
      We can also find recurring internal signals that line up with visible market features. But the specific
      signals we isolated are not enough, on their own, to explain the model's final trading choice.
    ]
  ],
  fill: rgb("#f4f1ec"),
  stroke_fill: rgb("#2f6b4f"),
)

#v(0.7em)

Narrower is not weaker. The data supports four things:

- the model's internal market picture is real and measurable
- prior context shapes that picture
- the model's decision is not governed by one simple hidden switch
- the two strongest candidate signals are real and selective, but not sufficient on their own to explain the final choice

These are the claims the evidence actually supports. The next round of work starts from a clear map: which questions have been answered and which remain open.
