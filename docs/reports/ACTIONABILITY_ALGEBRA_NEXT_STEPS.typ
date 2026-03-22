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

#align(left)[
  #text(size: 9pt, fill: rgb("#b33a2a"), tracking: 0.08em, weight: "medium")[XENON INTERPRETABILITY]
  #v(0.3em)
  #text(size: 22pt, weight: "bold")[Actionability Next Steps Memo]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    Short planning memo following the `actionability_algebra_v1/v2/v3` sequence.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
]

#v(1em)

#block(
  width: 100%,
  inset: (left: 14pt, top: 12pt, bottom: 12pt, right: 12pt),
  stroke: (left: 3pt + rgb("#b33a2a"), top: none, right: none, bottom: none),
  fill: rgb("#faf5f3"),
)[
  #text(size: 7.5pt, fill: rgb("#b33a2a"), weight: "bold", tracking: 0.08em)[DECISION]
  #v(0.3em)
  #text(size: 12.5pt, weight: "medium")[Push one harder synthetic step, then bridge only the surviving claim back to real DX data.]
]

= Situation

After `v3`, the synthetic program has a split result:

- the early market-preference result is robust enough to keep
- the clean downstream permission result is not yet robust enough to headline

That means the synthetic line is still worth continuing, but only in a way that increases semantic pressure and reduces surface-form shortcutting.


= Recommended `v4`

The best immediate synthetic experiment is `actionability_algebra_v4`.

Core changes:

- Keep the `v3` paraphrases.
- Keep shuffled bullet order within sections.
- Add distractor numeric lines to both `## PORTFOLIO CONTEXT` and `## EXECUTION CONSTRAINTS`.
- Increase scenario-group diversity so held-out splits are less tied to a single phrasing template.
- Rewrite local permission cues in multiple equivalent ways.

Target question:

Can any downstream permission or actionability read survive once the permission-relevant numbers and phrases are no longer uniquely salient?


= Success Criteria

`v4` would count as genuinely interesting if all three hold:

- `market_best_asset` stays near ceiling in row states
- at least one downstream permission-side classifier stays materially above chance under held-out scenario groups
- the best downstream section remains stable across prompt variants instead of migrating arbitrarily with phrasing changes

If those fail, the cleaner research story becomes:

- market preference is early and robust
- permission is weaker, distributed, and presentation-sensitive


= Real-Data Bridge

Only bridge the strongest surviving claim:

1. Validate early preference on real DX prompts.
2. Test permission as a noisy downstream gating signal, not as a crisp semantic classifier.
3. Prefer paired reruns and controlled perturbations over generic observe-class probing.


= What To Deprioritize

- Do not over-invest in a pure "permission manifold" story yet.
- Do not treat `v2` alone as sufficient evidence.
- Do not expand into large real-data capture for permission until the synthetic side survives `v4`.


= Why This Is Still Good Progress

`v3` is not a failure. It did exactly what a robustness phase should do: it showed which part of the `v2` story survives and which part does not.

That is a better research position than having only the convenient `v2` result.

#v(2em)
#line(length: 100%, stroke: 0.5pt + rgb("#ccc"))
#v(0.3em)
#text(size: 8pt, fill: rgb("#999"))[
  Actionability Next Steps Memo — 21 March 2026.
]
