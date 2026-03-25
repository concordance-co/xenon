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
  #text(size: 22pt, weight: "bold")[Real DX Post-Market Geometry Bridge]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    This bridge keeps the real 6-asset market snapshot fixed and asks a narrower question than the earlier row-local report:
    after the model reads the market, where do real risk and affordance edits show up in the pooled section states?
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[22 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[REAL DATA]\ #text(size: 9pt)[42 base examples / 264 prompts]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[OBJECT]\ #text(size: 9pt)[6-asset post-market geometry]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[STATES]\ #text(size: 9pt)[market + section EOS pooled states]],
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
  #text(size: 12.5pt, weight: "medium")[
    The clean real effect does not look like a reusable cross-example market coordinate frame. It looks like #emph[post-market realignment] in downstream section states, with
    #emph[affordance] much clearer than #emph[risk].
  ]
  #v(0.4em)
  #text(size: 9.5pt, fill: rgb("#555"))[
    Across the full six-asset cohort, the best cross-example coordinate transfer is still negative (`-0.190 R²` for affordance, `-0.580 R²` for risk),
    so there is no convincing global coordinate system that transfers cleanly from one real DX prompt to another. But once we stop demanding that global frame and instead ask whether
    the recovered geometry moves #emph[within each example] toward the context-adjusted score geometry, a strong signal appears: affordance reaches a best margin of `0.401` at
    `constraints_eos @ L0`, while risk is weaker and mostly concentrated at the extreme end of the ladder (`0.076` at `risk_5`, `last_token @ L37`).
  ]
]

#v(1.2em)

// ── Summary Metrics ─────────────────────────────────────────────
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BASE DATA]\ #text(size: 16pt, weight: "bold")[`42 / 264`] #text(size: 8pt, fill: rgb("#888"))[\ base examples / prompts]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[FRAME TRANSFER]\ #text(size: 16pt, weight: "bold")[`-0.190 R²`] #text(size: 8pt, fill: rgb("#888"))[\ best affordance transfer is still negative]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST RISK REALIGNMENT]\ #text(size: 16pt, weight: "bold")[`0.076`] #text(size: 8pt, fill: rgb("#888"))[\ `risk_5`, `last_token @ L37`]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST AFFORDANCE REALIGNMENT]\ #text(size: 16pt, weight: "bold")[`0.401`] #text(size: 8pt, fill: rgb("#888"))[\ `affordance_4`, `constraints_eos @ L0`]],
)


= Why This Bridge Exists

The earlier real bridge asked whether changing the real risk setting deformed #emph[row-local] market geometry. That answer was mostly “no.” This phase moves one step downstream:

- keep the same real market snapshot fixed
- keep the same ladder edits
- stop reading only individual market rows
- instead read pooled section states after the model has seen the market and begun integrating settings, portfolio, and constraints

This is a better match to the current hypothesis:

- real market rows may stay fairly stable
- real context effects may only become visible once the model starts combining the market with the rest of the prompt


= Experimental Design

#align(center)[#image("../../data/report_assets/real_postmarket_geometry_bridge_v2/experiment_design.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
The real post-market bridge dataset recipe. Each base example is a real HQ observation prompt with a fixed 6-asset roster, rerendered through a risk ladder or an affordance ladder.
]

#v(0.4em)

The dataset contains:

- `42` total base examples
- `264` total prompts
- `24` risk base examples and `24` affordance base examples
- `4` repeated 6-asset roster families across `13` vaults

The two context families are:

- #text(weight: "medium")[risk ladder]: `risk_1 .. risk_5`
- #text(weight: "medium")[affordance ladder]: `market_only`, then `affordance_1 .. affordance_5`

Important change from the older bridge:

- this is now a #emph[full 6-asset object], not a 4-asset slice inside larger rosters


= Plain-Language Definitions

- #text(weight: "medium")[post-market geometry]
  the recovered asset layout after reading the whole market block and then measuring pooled section states later in the same prompt
- #text(weight: "medium")[cross-example coordinate transfer]
  whether one shared coordinate probe trained on some real examples generalizes to held-out real examples
- #text(weight: "medium")[within-example realignment]
  whether the recovered geometry for a given prompt moves closer to the context-adjusted score geometry than to the unchanged base geometry
- #text(weight: "medium")[market_mean]
  the mean residual over the full market section
- #text(weight: "medium")[market_eos]
  the final token of the market block
- #text(weight: "medium")[active_settings_eos]
  the final token of the `## ACTIVE SETTINGS` block
- #text(weight: "medium")[portfolio_eos]
  the final token of the `## PORTFOLIO CONTEXT` block
- #text(weight: "medium")[constraints_eos]
  the final token of the `## CONSTRAINTS` block
- #text(weight: "medium")[last_token]
  the final input token before generation


= Cross-Example Coordinate Transfer

#align(center)[#image("../../data/report_assets/real_postmarket_geometry_bridge_v2/transfer_summary.png", width: 97%)]
#text(size: 8pt, fill: rgb("#888"))[
For both ladders, the best cross-example coordinate-transfer `R²` stays negative. That means the report should not be read as evidence for one clean global real market frame.
]

#v(0.4em)

This is the first important limit:

- the best risk transfer is `-0.580 R²`
- the best affordance transfer is `-0.190 R²`
- neither ladder supports a reusable, cross-example real coordinate frame the way the synthetic phases did

So if we stopped here, the bridge would look weak. But the real question is narrower:

- even without a strong global frame
- do the #emph[within-example] recovered geometries still move in the expected direction after the market block?


= Realignment By Context

#align(center)[#image("../../data/report_assets/real_postmarket_geometry_bridge_v2/realignment_contexts.png", width: 97%)]
#text(size: 8pt, fill: rgb("#888"))[
Risk only becomes mildly positive at the most extreme context. Affordance becomes strongly positive and stays that way for the harder ladder steps.
]

#v(0.4em)

This is the core result.

Risk:

- mostly hovers near zero
- only becomes clearly positive at `risk_5`
- best margin is `0.076` at `last_token @ L37`

Affordance:

- turns positive immediately
- gets substantially stronger in the harder contexts
- peaks at `0.401` for `affordance_4` in `constraints_eos @ L0`

So the real post-market bridge says:

- #emph[risk] exists, but weakly and mostly at the extreme end of the ladder
- #emph[affordance] exists much more clearly, and the signal is already visible in downstream section endpoints


= Where The Real Signal Lives

#align(center)[#image("../../data/report_assets/real_postmarket_geometry_bridge_v2/state_heatmaps.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Heatmaps of mean score-over-base margin by state and context. The useful real signal concentrates in post-market section states, especially `constraints_eos`.
]

#v(0.4em)

The heatmaps make the localization clearer.

For risk:

- no state has a strongly positive mean margin across the ladder
- the best averages are still close to zero
- the real effect looks sparse and context-specific, not like a clean ladder-wide deformation

For affordance:

- `constraints_eos` is the strongest state overall
- `active_settings_eos` and `portfolio_eos` are also clearly positive
- `market_mean` itself is positive too, but the downstream section states are cleaner

That is exactly what we would expect if:

- the market representation is encoded first
- then route availability and execution constraints reshape the geometry during post-market integration


= Layerwise View

#align(center)[#image("../../data/report_assets/real_postmarket_geometry_bridge_v2/selected_layerwise.png", width: 97%)]
#text(size: 8pt, fill: rgb("#888"))[
Selected layerwise curves for the most informative contexts. `risk_5` only weakly rises late, while `affordance_5` is positive earlier and in more clearly interpretable section states.
]

#v(0.4em)

The layerwise view helps separate the two context families:

- `risk_5` only develops a modest positive effect late, especially at `last_token`
- `affordance_5` is already positive in interpretable section states like `constraints_eos` and `active_settings_eos`

That makes the taxonomy sharper:

- #text(weight: "medium")[risk] is real but weak, late, and concentrated at the extreme end
- #text(weight: "medium")[affordance] is real, earlier, and much easier to localize to post-market sections


= Interpretation

The strongest reading of the full real bridge is:

1. The real DX prompts do #emph[not] yield a clean shared coordinate system that transfers from one example to another.
2. That does #emph[not] mean there is no real geometry signal.
3. The real signal appears after the market block, in pooled section states that integrate context.
4. Affordance is the cleaner family; risk is weaker and more diffuse.

So the right update is:

- move away from “one reusable real market frame”
- move toward “post-market integration geometry over a relatively stable market encoding”

That is a stronger and more specific claim than the earlier row-local bridge allowed.


= How To Read The Charts

The key metrics in this report are:

- #text(weight: "medium")[coord `R²`]
  how well a probe trained on some real examples predicts a target coordinate on held-out examples
- #text(weight: "medium")[score-over-base margin]
  how much closer the recovered geometry is to the context-adjusted score geometry than to the unchanged base geometry

Interpretation guide:

- #text(weight: "medium")[negative transfer `R²`]
  there is no clean shared cross-example coordinate frame
- #text(weight: "medium")[positive margin]
  within a prompt, the recovered geometry moves in the expected contextual direction
- #text(weight: "medium")[near-zero margin]
  the state is mostly unchanged by the ladder edit

That means a chart can still be scientifically useful even if transfer is weak:

- weak transfer + positive margin
  says “the global frame is messy, but the contextual movement is still real”


= What To Do Next

The next best move is not another broad synthetic phase. It is a cleaner real follow-up:

- keep the same 6-asset post-market object
- keep the section-level state lens
- test whether the same effect appears under matched #emph[portfolio] edits
- and then test simple causal manipulations on the strongest real affordance state (`constraints_eos`)

That follows directly from the current result:

- affordance is the cleanest real post-market geometry family we have seen so far
- it should be the first real family we try to intervene on
