#set page(
  paper: "us-letter",
  margin: (top: 1.9cm, bottom: 1.9cm, left: 2.0cm, right: 2.0cm),
  numbering: "1",
  number-align: right,
)
#set text(font: "Georgia", size: 10.1pt)
#set par(justify: true, leading: 0.64em)
#set heading(numbering: none)

#let ink = rgb("#182028")
#let muted = rgb("#5A6772")
#let accent = rgb("#AF3E2F")
#let cool = rgb("#295E9E")
#let soft = rgb("#F6F1ED")
#let mist = rgb("#EDF4F8")
#let lift = rgb("#EAF4EE")

#show heading.where(level: 1): it => {
  set text(size: 13pt, weight: "bold", fill: ink)
  v(0.95em)
  it
  v(0.28em)
}

#show heading.where(level: 2): it => {
  set text(size: 11pt, weight: "bold", fill: ink)
  v(0.72em)
  it
  v(0.18em)
}

#show figure.caption: set text(size: 8.3pt, fill: muted)

#let callout(tag, body, fill-color: soft, tag-color: accent) = block(
  width: 100%,
  inset: (left: 14pt, top: 12pt, bottom: 12pt, right: 12pt),
  stroke: (left: 3pt + tag-color, top: none, right: none, bottom: none),
  fill: fill-color,
)[
  #text(size: 7.6pt, fill: tag-color, weight: "bold", tracking: 0.08em)[#tag]
  #v(0.3em)
  #text(size: 11.8pt, fill: ink)[#body]
]

#align(left)[
  #text(size: 9pt, fill: accent, tracking: 0.08em, weight: "medium")[DX TERMINAL / XENON]
  #v(0.25em)
  #text(size: 22pt, weight: "bold", fill: ink)[Prompt Conflict Geometry]
  #v(0.08em)
  #text(size: 15pt, fill: muted)[DX Terminal Highlights Brief]
  #v(0.4em)
  #text(size: 10.3pt, fill: muted)[April 17, 2026]
  #v(0.75em)
  #line(length: 100%, stroke: 1.3pt + ink)
]

#v(0.9em)

#callout(
  [MAIN TAKEAWAY],
  [
    We can now reliably read internal strategy/settings disagreement from
    the model across three execution dimensions: `trade_size`,
    `risk_preference`, and `diversification_preference`. The strongest pair
    is `trade_size` and `risk_preference`, but all three families share
    meaningful geometry. In practical terms, DX Terminal now has a credible
    path toward detecting when the model sees tension between strategy
    guidance and user settings, identifying which axis is in tension, and
    eventually steering or debugging those cases instead of treating them as
    black-box output quirks.
  ],
  fill-color: mist,
  tag-color: cool,
)

= Main Findings

- Three prompt-local conflict families now validate cleanly enough to use as
  internal diagnostics: `trade_size`, `risk_preference`, and
  `diversification_preference`.
- Text-only baselines stay at chance across the strong families, so the model
  is not solving these tasks by raw lexical shortcut.
- Single-family probes are very strong, with best balanced accuracy ranging
  from `0.9635` to `1.0000` on standard holdouts.
- The mixed joint prompt shows both `size` and `risk` conflicts can be
  read out from the same forward pass:
  - `size_conflict_present`: `0.9414 / 0.9862`
  - `risk_conflict_present`: `0.9388 / 0.9871`
- Same-capture geometry at `L36` shows a real three-family structure rather
  than isolated benchmark-specific features:
  - `risk` vs `trade_size`: `0.6449`
  - `diversification` vs `risk`: `0.4684`
  - `diversification` vs `trade_size`: `0.4883`
- The main technical caveat is calibration, not absence of signal:
  `diversification` carries a family-specific baseline offset that breaks raw
  threshold transfer even when ranking transfer remains strong.

= Probe Highlights

These numbers are best read in three buckets:

- *Behavior*: whether the model actually followed the intended setting on the
  benchmark rows
- *Standard probe*: how well a linear readout decodes conflict under the
  normal lexical holdouts
- *Strict*: the harder novelty test, where the readout has to survive a more
  aggressive split

The important pattern is that all three families are strongly readable, but
they are not equally clean behaviorally. `trade_size` is the anchor,
`risk_preference` is representationally strong but behaviorally asymmetric,
and `diversification_preference` is the cleanest asset-selection family so
far.

#block(breakable: false)[
  #table(
    columns: (1.3fr, 0.8fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center),
    inset: 7pt,
    stroke: 0.4pt,
    fill: (x, y) => if y == 0 { rgb("#E6EEF2") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

    [*Family / Readout*], [*Behavior*], [*Standard Probe*], [*Strict*], [*Note*],

    [`trade_size`], [`~97%` conflict], [`0.9948 / 1.0000`], [`0.9896 / 1.0000`], [cleanest benchmark],
    [`risk_preference`], [`100%` aligned], [`0.9844 / 0.9937`], [`0.8854 / 0.9119`], [behaviorally asymmetric],
    [`diversification_preference`], [`100%` aligned / `85.4%` conflict], [`1.0000 / 1.0000` strategy holdout], [`0.8333 / 0.8819`], [portfolio-conditioned third family],
  )
]

#v(0.35em)

Format above is `balanced accuracy / AUROC`.

#figure(
  image("dx_terminal_brief_assets/family_within_auroc_by_layer.png", width: 100%),
  caption: [
    Representative within-family AUROC curves across layers. All three
    families rise sharply out of chance and become highly readable in the late
    middle layers rather than only at the final layer.
  ],
)

#figure(
  image("dx_terminal_brief_assets/strict_family_auroc_by_layer.png", width: 100%),
  caption: [
    The harder strict split drops performance, but the validated family set
    still stays well above chance and peaks in the same late-middle region.
  ],
)

= Geometry Highlights

The geometry result is the part that makes this more useful than "we trained a
few good probes." It says the model is not building three unrelated benchmark
features. It is building a family of related conflict representations with a
shared internal structure.

At `L36`, the current best same-capture cosine picture is:

#table(
  columns: (1.45fr, 0.75fr),
  align: (left, center),
  inset: 7pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#E6EEF2") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

  [*Pair*], [*Cosine*],
  [`risk_preference` vs `trade_size`], [`0.6449`],
  [`diversification_preference` vs `risk_preference`], [`0.4684`],
  [`diversification_preference` vs `trade_size`], [`0.4883`],
)

This is the current best high-level read:

- `trade_size` and `risk_preference` are the tightest pair
- `diversification_preference` is not an outlier; it is a real third member
- the three families share a conflict-relevant subspace, but not identical
  calibration

#figure(
  image("three_family_visuals/shared_axis_distributions.png", width: 100%),
  caption: [
    Shared-axis view of the three-family capture. Each family has a clear
    aligned-to-conflict separation, but the family baselines are shifted
    relative to each other. This is why cross-family AUROC can stay high while
    raw threshold transfer breaks.
  ],
)

#figure(
  image("three_family_visuals/directed_subspace_scatter_by_family_conflict_v2.png", width: 100%),
  caption: [
    Directed 2D subspace view at the best geometry layer. `risk_preference`
    and `trade_size` remain the tightest pair, while
    `diversification_preference` occupies a nearby but offset region rather
    than an isolated one.
  ],
)

= Joint Prompt Result

== Simultaneous Conflict Readout

The joint-prompt run was the key compositional result. When both `size` and `risk`
conflicts appear in the same prompt, both remain strongly readable from the
same activation stream:

- `size_conflict_present`: `0.9414 / 0.9862`
- `risk_conflict_present`: `0.9388 / 0.9871`
- `any_conflict_present`: `0.9306 / 0.9503`
- `double_conflict_present`: `0.8898 / 0.9687`

That means the model is not just building one narrow benchmark feature per
prompt type. It can maintain multiple conflict signals simultaneously.

#figure(
  image("dx_terminal_brief_assets/phase11_joint_prompt_auroc_by_layer.png", width: 100%),
  caption: [
    Joint-prompt AUROC curves. Multiple conflict labels stay strong across the
    same forward pass, with the cleanest readout window in the same late-middle
    layers as the single-family benchmarks.
  ],
)

= Diversification Geometry Update

The original three-family transfer workflow made diversification look farther
away than it really was. The follow-up same-capture projection diagnostic
showed that the bigger issue was prompt-structure offset, likely from the
extra `PORTFOLIO` block at the last prompt token.

That changed the interpretation from:

- "diversification may be too far from the family to matter"

to:

- "diversification shares real structure, but family-specific baseline shift
  breaks raw threshold transfer"

That is a materially better story for DX Terminal because it means the signal
is more reusable than the first transfer numbers implied.

= Why This Matters

- We can now distinguish two different failure modes:
  - the model never represented the strategy/settings disagreement
  - the model represented it, but resolved it oddly downstream
- We have axis-level readouts, not just a vague "something is off" detector.
  The current families already cover:
  - how much to buy
  - which asset fits the allowed risk posture
  - which asset fits the desired concentration vs broadening posture
- The geometry work gives us something stronger than a one-off classifier:
  a reusable conflict subspace we can score, compare, and eventually
  intervene on.
- This opens a path to product-facing diagnostics on real prompts:
  - detect hidden strategy/settings tension
  - identify which execution axis is in conflict
  - debug template changes mechanistically instead of by anecdote

= Recommended Next Steps

1. Validate transfer on real DX Terminal prompts, starting with the cleanest
   axis: `trade_size`.
2. Use the current conflict directions as monitors on real prompts before
   trying any intervention.
3. Return to activation patching once the infra is ready, using the shared
   conflict directions as the causal target.

#v(0.8em)
#line(length: 100%, stroke: 0.9pt + rgb("#C9D3DA"))
#v(0.4em)
#text(size: 8.5pt, fill: muted)[
  Metric format is `balanced accuracy / AUROC` unless otherwise noted. All
  results shown here come from `Qwen3-30B-A3B` prompt-EOS residual analyses
  across Phases 09--12, with emphasis on the strongest validated findings and
  the corrected same-capture geometry result.
]
