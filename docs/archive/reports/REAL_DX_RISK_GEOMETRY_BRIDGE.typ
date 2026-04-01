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
  #text(size: 22pt, weight: "bold")[Real DX Risk Geometry Bridge]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    Phase 15 moves the 4-asset geometry program from synthetic prompts to real DX observation prompts. The question is not whether
    risk changes the final action, but whether changing #emph[Asset Risk Preference] from `1` to `5` deforms the #emph[row-local]
    market geometry carried by the selected asset rows.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[22 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[REAL DATA]\ #text(size: 9pt)[30 matched HQ observations]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[LADDER]\ #text(size: 9pt)[`risk_1 .. risk_5`, anchored on `risk_3`]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[OBJECT]\ #text(size: 9pt)[row-local 4-asset geometry]],
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
    The synthetic risk-deformation story does #emph[not] transfer to row-local geometry on matched real DX prompts. In the real bridge,
    the selected 4-asset row geometry stays effectively fixed across `risk_1 .. risk_5`.
  ]
  #v(0.4em)
  #text(size: 9.5pt, fill: rgb("#555"))[
    Cross-example coordinate probes are only moderate (`0.576 R²` for the best x-axis read; `0.466 R²` for the best y-axis read), so the
    real hand-built coordinate frame is not decoded nearly as cleanly as in synthetic phases. But the more important result is stronger:
    the decoded row-local geometry does not move with risk. Realignment margins are slightly negative at every context, and the decoded
    geometry-step norm is exactly `0.0` for every adjacent real risk pair. That implies the row-local market frame is stable and the
    risk setting is likely being applied somewhere beyond the individual market-row states.
  ]
]

#v(1.2em)

// ── Summary Metrics ─────────────────────────────────────────────
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BASE DATA]\ #text(size: 16pt, weight: "bold")[`30 / 150`] #text(size: 8pt, fill: rgb("#888"))[\ base examples / prompts]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST X TRANSFER]\ #text(size: 16pt, weight: "bold")[`0.576 R²`] #text(size: 8pt, fill: rgb("#888"))[\ `row_mean @ L21`]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST REALIGNMENT]\ #text(size: 16pt, weight: "bold")[`-0.0059`] #text(size: 8pt, fill: rgb("#888"))[\ still negative, `row_eos @ L4`]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[RISK-STEP DELTA]\ #text(size: 16pt, weight: "bold")[`0.0`] #text(size: 8pt, fill: rgb("#888"))[\ decoded geometry-step norm]],
)


= Why This Bridge Exists

Phases 9-14 built a synthetic story around a shared 4-asset market frame:

- the model preserved a reusable coordinate system over multiple assets
- synthetic risk behaved like a structured ladder on top of that frame
- synthetic portfolio and affordance produced different families of deformations

Phase 15 asks the obvious external-validation question:

- if we take #emph[real DX observation prompts]
- keep the market snapshot fixed
- and change only the risk preference setting

do the same row-local geometry effects appear?

This is a bridge test, not a restatement of the synthetic results. If the bridge fails, that is scientifically useful because it tells us
where the synthetic abstraction stopped matching the real system.


= Experimental Design

#align(center)[#image("../../data/report_assets/real_risk_geometry_bridge/experiment_design.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
The real bridge dataset recipe. Each base example is a real HQ observation prompt. A stable 4-asset slice is selected inside that larger roster,
then the prompt is rerendered at `risk_1 .. risk_5` with the market snapshot held fixed.
]

#v(0.4em)

The dataset contains:

- `30` matched real base examples
- `150` total prompts
- `5` real roster families, each contributing `6` examples
- `4` selected assets per base example, chosen from a larger real roster

Important constraint:

- there is no natural supply of HQ prompts with exactly `4` assets
- so this bridge works with a #emph[4-asset slice] inside larger real rosters
- the slice is kept fixed across the whole risk ladder for that example

In plain English:

- same market rows
- same 4 selected assets
- same row order and symbols inside the selected slice
- only the risk preference setting changes


= Plain-Language Definitions

- #text(weight: "medium")[row-local geometry]
  the multi-asset shape recovered only from the selected asset rows, not from later policy sections or the final action token
- #text(weight: "medium")[risk ladder]
  the ordered family `risk_1, risk_2, risk_3, risk_4, risk_5` for the same underlying prompt
- #text(weight: "medium")[coordinate transfer]
  how well a probe trained to recover a real coordinate axis from one risk context generalizes to other risk contexts
- #text(weight: "medium")[realignment margin]
  how much closer the recovered geometry is to the risk-adjusted score geometry than to the unchanged base geometry
- #text(weight: "medium")[risk-adjusted score geometry]
  the asset layout implied if the same market is rescored under a more conservative or more aggressive risk preference
- #text(weight: "medium")[bridge]
  a synthetic-to-real validation step, not a new synthetic phase


= Coordinate Transfer

#align(center)[#image("../../data/report_assets/real_risk_geometry_bridge/coordinate_transfer.png", width: 97%)]
#text(size: 8pt, fill: rgb("#888"))[
Mean coordinate-transfer curves by layer, averaged over the real risk ladder. The x-axis is model layer; the y-axis is transfer `R²`.
]

#v(0.4em)

The coordinate-transfer chart says two things at once:

- the real row states do carry some stable multi-asset structure
- but the real hand-built coordinate frame is much noisier than the synthetic one

The best reads are:

- `base_x`: `0.576 R²` from `row_mean @ L21`
- `base_y`: `0.466 R²` from `row_eos @ L6`

That is enough to say the real 4-asset object is not random. But it is not enough to claim a crisp low-dimensional real market frame in the
same sense as the synthetic phases. So the stronger evidence in this report does #emph[not] come from cross-example coordinate quality. It comes
from what happens #emph[across the risk ladder within the same example].


= Realignment Margin

#align(center)[#image("../../data/report_assets/real_risk_geometry_bridge/realignment_margin.png", width: 97%)]
#text(size: 8pt, fill: rgb("#888"))[
If row-local geometry were moving toward risk-adjusted score geometry, the margin would turn positive. Instead it stays slightly negative.
]

#v(0.4em)

This is the first strong bridge result.

Across all five real risk contexts:

- the best realignment margin is still negative: `-0.0059`
- there is no layer where the row-local geometry clearly shifts toward risk-adjusted score geometry
- `row_eos` gets closest, but it still does not cross into positive territory

That means the real risk preference edit is #emph[not visibly deforming the selected row-local geometry in the expected direction].


= Identity Across Real Risk Steps

#align(center)[#image("../../data/report_assets/real_risk_geometry_bridge/invariance_identity.png", width: 97%)]
#text(size: 8pt, fill: rgb("#888"))[
At the selected row-local state, decoded geometry is unchanged across adjacent real risk steps, while the score geometry implied by the prompt does change.
]

#v(0.4em)

This is the strongest result in the bridge.

At the selected row-local state (`row_eos @ L2`):

- each adjacent risk step is fit perfectly by the identity map (`coord_r2_mean = 1.0`)
- the decoded geometry-step norm is `0.0` for every pair
- but the risk-adjusted score geometry would prefer nontrivial movement (`0.33 .. 0.43` mean distance-Spearman)

So the bridge does #emph[not] say “risk has no effect in the real system.”

It says something narrower and more useful:

- risk does not show up as a deformation of the selected row-local 4-asset geometry

That strongly suggests the risk setting is applied:

- later than the market-row states we are reading
- or in a different representation than the row-local market geometry


= Interpretation

The cleanest reading of Phase 15 is:

1. Synthetic phases were real and useful, but they were still synthetic.
2. Real DX row-local geometry seems to preserve the market slice across the risk ladder.
3. The risk instruction is therefore likely being integrated somewhere beyond those row-local states.

This is a meaningful narrowing of the problem:

- if we want the real risk effect, more row-local market probing will probably not be enough
- the next place to look is #emph[context integration over the market frame], not the market rows in isolation

In other words, the bridge did not fail because it found nothing. It failed in a specific way:

- the selected row-local market geometry is too stable
- so risk is probably applied downstream of it


= What To Do Next

The next move should be narrower than another large synthetic expansion.

Best next step:

- run the same bridge lens on #emph[context-integration states], not only row-local row states

Concretely:

- keep the same matched real risk ladder
- keep the same selected 4-asset slices
- move the representation object to section-level or later aggregate states

Examples:

- portfolio or settings section EOS states
- later token positions after the model has seen both the market and the user settings
- pairwise or set-level comparisons formed after the full market block, not inside individual rows

That would directly test the current hypothesis:

- the real risk effect exists
- but it is applied after the shared market slice has already been encoded


= How To Read The Layerwise Charts

The layerwise charts in this report all follow the same pattern:

- x-axis = layer
- y-axis = average metric value
- a peak means “this read is strongest here”

The two main y-axis metrics are:

- #text(weight: "medium")[transfer `R²`]
  how well a coordinate probe recovers a real axis from held-out rows
- #text(weight: "medium")[realignment margin]
  whether recovered geometry is closer to risk-adjusted score geometry or to unchanged base geometry

Interpretation guide:

- high transfer + positive margin
  would support real row-local deformation
- high transfer + negative margin
  means the representation is stable, but not moving with the edited context
- flat zero deformation
  means the row-local geometry is effectively unchanged across the ladder

That third case is what dominates this report.
