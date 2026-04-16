#set page(
  paper: "us-letter",
  margin: (top: 2.1cm, bottom: 2.1cm, left: 2.35cm, right: 2.35cm),
  numbering: "1",
  number-align: right,
)
#set text(font: "Georgia", size: 10.2pt)
#set par(justify: true, leading: 0.67em)
#set heading(numbering: none)

#let ink = rgb("#182028")
#let muted = rgb("#5A6772")
#let accent = rgb("#B33A2A")
#let soft = rgb("#F7F2EF")
#let rule = rgb("#D7DEE3")
#let lift = rgb("#EAF4EE")
#let mist = rgb("#EEF4F8")

#show heading.where(level: 1): it => {
  set text(size: 13pt, weight: "bold", fill: ink)
  v(1.1em)
  it
  v(0.35em)
}

#show heading.where(level: 2): it => {
  set text(size: 11pt, weight: "bold", fill: ink)
  v(0.8em)
  it
  v(0.25em)
}

#show figure.caption: set text(size: 8.4pt, fill: muted)

#let callout(tag, body, fill-color: soft, tag-color: accent) = block(
  width: 100%,
  inset: (left: 14pt, top: 12pt, bottom: 12pt, right: 12pt),
  stroke: (left: 3pt + tag-color, top: none, right: none, bottom: none),
  fill: fill-color,
)[
  #text(size: 7.5pt, fill: tag-color, weight: "bold", tracking: 0.08em)[#tag]
  #v(0.3em)
  #text(size: 12pt, fill: ink)[#body]
]

#align(left)[
  #text(size: 9pt, fill: accent, tracking: 0.08em, weight: "medium")[XENON INTERPRETABILITY]
  #v(0.3em)
  #text(size: 22pt, weight: "bold", fill: ink)[Prompt Confusion]
  #v(0.1em)
  #text(size: 15pt, fill: muted)[Three-Family Conflict Geometry Checkpoint]
  #v(0.45em)
  #text(size: 10.4pt, fill: muted)[
    April 16, 2026 --- summary of the current prompt-local conflict benchmarks,
    shared geometry results, and the projection diagnostic that clarified the
    diversification story.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.4pt + ink)
]

#v(1em)

#callout(
  [BOTTOM LINE],
  [
    We now have three validated prompt-local conflict families:
    `trade_size`, `risk_preference`, and `diversification_preference`.
    All three are text-blind under lexical controls and linearly readable in
    the residual stream. The updated same-capture geometry story is that
    `trade_size` and `risk_preference` form the tightest pair, while
    `diversification_preference` is a real but somewhat more distant third
    member. The main caveat is no longer "is diversification real?" but
    "how much prompt-structure offset contaminates thresholded transfer?"
    This is valuable because it gives Concordance a real benchmark suite for
    multi-source policy reasoning inside an LLM, and it gives DX Terminal a
    concrete path toward detecting, explaining, and eventually steering cases
    where strategy guidance and user settings pull the model in different
    directions.
  ],
  fill-color: mist,
  tag-color: rgb("#2D5FA0"),
)

= Background

The project started as an attempt to study how a trading-oriented LLM
represents internal prompt conflicts between `STRATEGY` and `ACTIVE SETTINGS`.
Several early designs were too confounded, semantically unstable, or dependent
on temporal context the prompt did not actually contain. The main lesson from
that process was methodological:

- strong conflict benchmarks need to be prompt-local
- conflict labels need clean aligned rows
- lexical controls need to be at chance
- and geometry claims should not rely on structurally broken families

The current checkpoint focuses on the three families that survived that
filtering. It is primarily a datasets / analysis / results document, not a
full retrospective on every discarded design.

= Datasets

The current benchmark family set is:

#table(
  columns: (1.25fr, 0.7fr, 0.7fr, 0.9fr, 1.2fr),
  align: (left, center, center, center, left),
  inset: 7pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#E6EEF2") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

  [*Family*], [*Rows*], [*Aligned*], [*Conflict*], [*Decision axis*],

  [`trade_size` (Phase 09)], [384], [clean], [very clean], [output `size` field],
  [`risk_preference` (Phase 10)], [384], [100% aligned exact], [asymmetric], [asset selection by risk posture],
  [`diversification_preference` (Phase 12)], [384], [100% aligned exact], [85.4% exact], [asset selection by portfolio fit],
)

#v(0.5em)

Each family is a matched-pair benchmark with an aligned / conflict label.
The families differ in what the disagreement is *about*:

- `trade_size` asks whether the model should buy `small` or `large`
- `risk_preference` asks which viable asset matches the allowed risk posture
- `diversification_preference` asks which viable asset best matches the
  allowed concentration vs broadening posture given the existing book

Two design decisions are worth highlighting.

First, `risk_preference` and `diversification_preference` both use asset
selection rather than activity gating. That keeps the task prompt-local and
lets us ask whether conflict representations generalize beyond a single output
field.

Second, `diversification_preference` is explicitly portfolio-conditioned.
`ALPHA` deepens an already-owned sleeve; `BETA` broadens into a more distinct
one. This makes `PORTFOLIO` semantically operative rather than decorative.

= Analysis

The analysis stack across these phases is now fairly consistent:

1. Behavioral smoke on aligned and conflict rows
2. Cheap and formal lexical gates on raw prompt text
3. Standard probe workflows with lexical holdouts
4. Marshall-style strict both-axes holdouts where available
5. Cross-family geometry and transfer
6. Same-capture projection / threshold diagnostics when transfer results look
   calibration-sensitive

All probe analyses here use:

- model: `Qwen3-30B-A3B`
- site: `resid_post`
- readout token: last prompt token
- layers sampled from `L0` to `L44`

The current geometry checkpoint relies on two complementary cross-family
estimators:

- *TransferProbeSpec geometry*:
  logistic-regression probe directions transferred between families
- *same-capture mean-diff geometry*:
  family-specific conflict centroid differences computed inside one combined
  capture

The second estimator turned out to matter a lot for interpreting
`diversification_preference`.

= Results

== 1. Each family is independently real

`trade_size` remains the anchor benchmark.
Its single-family probe results are near ceiling:

- XOR split: `0.9948 / 1.0000`
- strategy holdout: `1.0000 / 1.0000`
- settings holdout: `0.9948 / 1.0000`

Behavioral compliance is also the strongest: `~97%` follow-setting on
conflict rows. This family is the baseline reference point for the project.

`risk_preference` is the second strong family.
Its standard holdouts were:

- XOR split: `0.9635 / 0.9766`
- strategy holdout: `0.9844 / 0.9937`
- settings holdout: `0.9740 / 0.9839`
- strict both-axes: `0.8854 / 0.9119`

The main caveat is behavioral asymmetry on conflict rows: conservative
constraints bind much more strongly than aggressive permissions.

`diversification_preference` is the third validated family.
Its first-pass behavior is cleaner than risk:

- aligned rows: `1.0000` exact
- conflict rows: `0.8542` exact

And its probe results are strong:

- XOR split: `0.9896 / 0.9995`
- strategy holdout: `1.0000 / 1.0000`
- settings holdout: `0.9792 / 0.9957`
- strict both-axes: `0.8333 / 0.8819`

The strict result is softer than the easy splits, but still comfortably above
chance with text baselines pinned at `0.500 / 0.500`.

== 2. The original three-family geometry story was too pessimistic about diversification

The first three-family transfer workflow suggested:

- `risk_preference` and `trade_size` were the tightest pair
- `diversification_preference` was much farther away
- outgoing diversification transfer had high AUROC but often collapsed to
  `0.500` balanced accuracy

At `L36`, the original transfer-probe cosine picture was:

#table(
  columns: (1.45fr, 0.75fr),
  align: (left, center),
  inset: 7pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#E6EEF2") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

  [*Pair*], [*Cosine*],
  [`risk vs trade_size`], [`0.5341`],
  [`diversification vs risk`], [`0.2996`],
  [`diversification vs trade_size`], [`0.2846`],
)

Those numbers were real as probe-direction comparisons, but they mixed shared
conflict structure with family-specific classification features. That mattered
most for diversification.

== 3. The projection diagnostic clarified the geometry

The follow-up projection diagnostic recomputed family-specific conflict
directions inside the *same combined capture* and then evaluated all families
against those directions.

At `L36`, the updated mean-difference cosine matrix is:

#table(
  columns: (1.45fr, 0.75fr),
  align: (left, center),
  inset: 7pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#E6EEF2") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

  [*Pair*], [*Cosine*],
  [`risk vs trade_size`], [`0.6449`],
  [`diversification vs risk`], [`0.4684`],
  [`diversification vs trade_size`], [`0.4883`],
)

This changes the diversification story materially.

The right picture now is not:

- risk and size are related
- diversification is barely connected

It is:

- risk and size are the tightest pair
- diversification is a real third member
- but it sits on a stronger family-specific baseline offset

#figure(
  image("three_family_visuals/directed_subspace_scatter_by_family_conflict.png", width: 100%),
  caption: [
    Directed 2D projection of all 1152 rows onto the conflict-relevant subspace
    derived from the three same-capture mean-diff directions. `L36` gives the
    clearest overall view: risk and trade size are the tightest pair, while
    diversification is visibly related but shifted.
  ],
)

== 4. The big diversification problem is threshold transfer, not absence of signal

The projection diagnostic also answered why diversification looked so odd under
cross-family transfer.

On the diversification direction at `L36`, the within-family score bands are:

- aligned mean: `-2.1717`
- conflict mean: `-1.6130`
- threshold: `-1.8923`

Applying that same diversification direction to other families gives decent
ranking but broken threshold transfer:

- diversification -> `trade_size`:
  - AUROC `0.8471`
  - balanced accuracy `0.5000`
  - FNR `1.0000`
- diversification -> `risk_preference`:
  - AUROC `0.8165`
  - balanced accuracy `0.5599`
  - FNR `0.8802`

The reverse direction shows the complementary effect:

- `risk_preference` -> diversification:
  - AUROC `0.8185`
  - balanced accuracy `0.5000`
  - FPR `1.0000`
- `trade_size` -> diversification:
  - AUROC `0.8302`
  - balanced accuracy `0.5859`
  - FPR `0.8281`

That is the signature of an additive baseline shift:

- scores preserve ordering well enough for AUROC
- but the source-family threshold is badly miscalibrated on the target family

#figure(
  image("three_family_visuals/shared_axis_distributions.png", width: 100%),
  caption: [
    The most informative current figure. All three families separate aligned
    from conflict rows on the shared axis, but their overall baselines are
    shifted relative to one another. This explains why cross-family AUROC can
    stay high while thresholded balanced accuracy collapses.
  ],
)

The likely source is prompt structure.
`diversification_preference` uniquely includes a semantically active
`PORTFOLIO` block, and because we are reading the last prompt token, that extra
section appears to shift the prompt-EOS residual in a way that projects onto
the conflict direction.

= Implications

The project-level claim is now stronger and cleaner than it was a few days ago.

#callout(
  [CURRENT CLAIM],
  [
    Qwen3-30B-A3B builds readable internal representations of strategy/settings
    disagreement across multiple prompt-local execution dimensions.
    `trade_size`, `risk_preference`, and `diversification_preference` are all
    independently decodable, text-blind under lexical controls, and linked by
    shared conflict geometry (cross-family transfer AUROC up to `0.98`).
    `risk_preference` and `trade_size` form the tightest pair;
    `diversification_preference` is a real third member with a larger
    family-specific baseline offset. The main remaining external-validity
    question is transfer to real DX Terminal prompts and other models.
  ],
  fill-color: lift,
  tag-color: rgb("#2F6B4F"),
)

== Why the geometry matters

A probe trained entirely on size conflict can detect risk conflict at `0.98`
AUROC --- near-perfect ranking transfer. The model is not building three
unrelated detectors. It is building related features that share substantial
structure.

That is what the geometry work adds on top of the individual probe results.
Probe accuracy says conflict is readable on each axis. The cross-family
transfer says the underlying signal is structured enough to generalize across
execution dimensions.

That matters because it changes how we can apply the result:

- for Concordance, it gives us a reusable benchmark and subspace for testing
  monitoring, transfer, and causal tooling
- for DX Terminal, it gives us a path to detect when strategy and settings are
  in tension, identify *which axis* is in tension, and distinguish missing
  conflict detection from strange downstream resolution

Three concrete implications follow, and they matter at two levels: the
Concordance research stack and the DX Terminal product.

First, we should stop treating thresholded cross-family balanced accuracy as
the main geometry metric. For these cross-family comparisons, the more honest
signals are:

- same-capture mean-diff cosine
- cross-family AUROC
- explicit score-offset / threshold-shift diagnostics

Second, no additional synthetic family is needed right now.
The geometry claim already has enough structure:

- one tight pair (`risk_preference`, `trade_size`)
- one moderately related third member (`diversification_preference`)
- one well-understood calibration caveat (`PORTFOLIO` offset)

Third, the project is now better positioned for *qualitative upgrades* rather
than simple family expansion. The natural next directions are:

- real-data transfer, probably beginning with `trade_size`
- activation patching once the infrastructure is ready
- a shareable combined report that anchors the geometry story with the new
  projection figures

== Why this is valuable to Concordance

For Concordance, this upgrades prompt-conflict analysis from an anecdotal
debugging exercise into a reusable interpretability benchmark.

What we have now is not just a single interesting probe, but a benchmark
family with:

- prompt-local labels
- text-blind lexical controls
- matched-pair behavioral grounding
- shared cross-family geometry
- and a clear diagnostic for separating shared signal from prompt-structure
  offset

That matters to Concordance in three ways.

First, it is a credible internal benchmark for evaluating whether model
representations preserve distinctions that matter in instruction-composed
systems.

Second, it is a testbed for new mechanistic tooling. The next causal or
transfer method we build can be checked against a benchmark where the geometry
is already partially mapped, instead of being tried first on messy live data.

Third, it is a concrete product-facing demo of what Concordance is for:
turning invisible prompt interactions into measurable internal structure,
rather than only scoring outputs after the fact.

== Why this is valuable to DX Terminal

For DX Terminal, the value is operational.

DX Terminal already asks the model to combine multiple policy sources:

- strategy preferences
- active settings
- market information
- and sometimes portfolio state

The current result says the model does not merely emit a final answer; it
builds readable internal representations of *where those policy sources
disagree*. That matters because many product failures in a system like DX
Terminal are not raw capability failures. They are policy-composition failures:

- the strategy says one thing
- the settings imply another
- the user sees a surprising execution choice
- and it is hard to tell whether the model missed the conflict, noticed it but
  resolved it strangely, or got nudged by prompt structure

This work gives us the technical foundation to build toward a DX Terminal
workflow where we can:

- flag prompts whose internal conflict score is unusually high
- explain which execution axis is in tension (`size`, `risk`, or
  `diversification`)
- distinguish "the model did not see the disagreement" from "the model saw it
  but resolved it in a product-surprising way"
- and eventually intervene or rewrite prompts before a bad execution reaches
  the user

One product-facing lesson is already visible in the benchmark family: settings
phrased as hard constraints bind more reliably than settings phrased as
permissions. The exact behavioral asymmetry differs by family, but the broader
pattern is stable enough to matter for DX Terminal prompt design. If we want
settings to actually control execution, restrictive wording is much more
trustworthy than permissive wording.

More broadly, the product payoff is better observability into policy alignment
inside the prompt, not just better hindsight after an odd trade recommendation
has already appeared.

= Recommendations

Near-term:

- use the new same-capture visuals as the anchor figures for team discussion
- treat `trade_size`, `risk_preference`, and `diversification_preference` as
  the settled benchmark family set
- do not spend more cycles inventing additional synthetic settings families

Next experiments:

- apply the cleanest family directions to real DX Terminal prompts, starting
  with `trade_size`
- return to activation patching when the new infra is stable

This checkpoint gives us a coherent multi-family benchmark story with one clean
anchor (`trade_size`), one strong but asymmetric family (`risk_preference`),
and one portfolio-conditioned family (`diversification_preference`) whose
geometry now looks interpretable rather than anomalous.
