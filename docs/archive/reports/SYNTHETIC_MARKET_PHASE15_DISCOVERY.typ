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

#let prompt-line(line) = {
  if line == "" {
    linebreak()
  } else if line.starts-with("   - ") {
    h(1.6em)
    [- #line.slice(5)]
    linebreak()
  } else if line.starts-with("  - ") {
    h(1.2em)
    [- #line.slice(4)]
    linebreak()
  } else {
    [#line]
    linebreak()
  }
}

#let prompt-block(path) = {
  set text(font: "Menlo", size: 7.1pt)
  set par(justify: false, leading: 0.5em)
  for line in read(path).split("\n") {
    prompt-line(line)
  }
}

// ── Title Block ─────────────────────────────────────────────────
#align(left)[
  #text(size: 9pt, fill: rgb("#b33a2a"), tracking: 0.08em, weight: "medium")[XENON INTERPRETABILITY]
  #v(0.3em)
  #text(size: 22pt, weight: "bold")[Synthetic Market Phase 15]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    Phase 1 discovery on the new DX-like synthetic prompt surface. This pass does not assume a hand-built latent basis.
    It asks a simpler question: when the model finishes reading the market block, which dominant directions in activation
    space track real market variables, and which are still just prompt-shape artifacts?
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[23 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATA]\ #text(size: 9pt)[184 market-only prompts]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[STATES]\ #text(size: 9pt)[`market_mean`, `market_eos`]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[METHOD]\ #text(size: 9pt)[PCA + nuisance / market correlations]],
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
  #text(size: 7.5pt, fill: rgb("#b33a2a"), weight: "bold", tracking: 0.08em)[MAIN READ]
  #v(0.3em)
  #text(size: 12.5pt, weight: "medium")[
    The new DX-like synthetic prompts are good enough to surface #emph[clean market-linked directions] in `market_mean`, but
    not good enough yet to eliminate prompt-length confounds entirely. The strongest clean directions track leader and dispersion
    variables like `pct_1h_max`, `pct_1h_std`, and concentration extremes. `market_eos` is more compressed and still informative,
    but weaker than `market_mean`.
  ]
]

#v(1.2em)

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[PROMPTS]\ #text(size: 16pt, weight: "bold")[`184`] #text(size: 8pt, fill: rgb("#888"))[\ market-only synthetic prompts]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[ROSTER WIDTH]\ #text(size: 16pt, weight: "bold")[`6`] #text(size: 8pt, fill: rgb("#888"))[\ assets per market]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST CLEAN `market_mean`]\ #text(size: 16pt, weight: "bold")[`0.710 vs 0.071`] #text(size: 8pt, fill: rgb("#888"))[\ `pct_1h_std` at `L35`]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST CLEAN `market_eos`]\ #text(size: 16pt, weight: "bold")[`0.510 vs 0.143`] #text(size: 8pt, fill: rgb("#888"))[\ `top20_holder_pct_max` at `L1`]],
)


= Why Phase 15 Exists

The earlier synthetic market phases were good for geometry method development, but the prompt surface was too artificial:

- explicit synthetic framing
- latent labels like `Archetype`
- section order and wording that did not match the real DX prompts closely enough

Phase 15 fixes that first. It keeps the synthetic control, but changes the surface form:

- same trading-agent system prompt style
- same section order as the real task
- same slider language
- same constraint and price-impact framing
- six-asset markets instead of four

Only after making that surface more realistic do we ask a discovery-style question:

- when the model reads the market block on this new surface, what directions in activation space actually line up with market variables?


= What Data Was Used

The full Phase 1 discovery cohort contains:

- `184` prompts total
- `1,104` asset rows
- `5,520` pairwise asset comparisons

In this table:

- #text(weight: "medium")[family] means the experiment template
- #text(weight: "medium")[variant] means the specific market variable or variable pair that gets swept inside that template

So:

- `market_basis_scalar` means one market variable is changed while the rest of the roster stays in a fixed background market
- `market_basis_coupled` means two market variables are changed together on a small grid while the rest of the roster stays in a fixed background market

The actual Phase 1 prompt families are:

#table(
  columns: (1.45fr, 1.55fr, 3.3fr, 0.8fr),
  align: (left, left, right),
  table.hline(stroke: 1pt),
  table.header([*Family*], [*Variant*], [*What changes across prompts*], [*Prompts*]),
  table.hline(stroke: 0.5pt),
  [`market_basis_scalar`], [`pct_5m`], [Only the first asset's 5-minute price change is swept from weak to strong while the other five assets come from one of three fixed background rosters.], [`21`],
  [`market_basis_scalar`], [`net_flow_5m`], [Only the first asset's 5-minute net flow is swept while the other five assets stay as background distractors.], [`21`],
  [`market_basis_scalar`], [`unique_traders_5m`], [Only the first asset's participation level is swept while the rest of the roster stays fixed.], [`21`],
  [`market_basis_scalar`], [`top20_holder_pct`], [Only the first asset's holder concentration is swept while the rest of the roster stays fixed.], [`21`],
  [`market_basis_coupled`], [`pct_5m × net_flow_5m`], [The first asset moves over a 2D grid where short-horizon price change and net flow vary together; the other five assets provide background market context.], [`50`],
  [`market_basis_coupled`], [`unique_traders_5m × top20_holder_pct`], [The first asset moves over a 2D grid where participation and concentration vary together; the other five assets provide background market context.], [`50`],
)

#v(0.4em)

Another way to say this:

- every prompt has the same #emph[shape] at the surface level: one six-asset market in DX-like format
- what changes is the first asset's market profile
- the other five assets keep the prompt from collapsing into a one-row toy example

Every prompt in this phase is `market_only`. There is no later context ladder yet. The point of this phase is to discover the market basis first.

#align(center)[#image("../../data/report_assets/synthetic_market_phase15_discovery/phase15_discovery_summary.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
The new cohort combines four scalar sweeps and two coupled sweeps. The summary charts compare the strongest market-linked PC at each layer against its prompt-length correlation.
]


= What The Analysis Actually Does

For each prompt, the pipeline pools two section-level states:

- `market_mean`: the mean residual state across the full market section
- `market_eos`: the residual state at the end of the market section

Then, for each layer and each state:

1. run PCA across prompts
2. keep the top five principal components
3. correlate each component with two feature groups

The two feature groups are:

- #text(weight: "medium")[nuisance features]
  `seq_len`, `user_chars`, and `n_rows`
- #text(weight: "medium")[market features]
  prompt-level summaries built from the real rendered market rows, including:
  `pct_5m_mean`, `pct_1h_max`, `net_flow_5m_std`, `unique_traders_5m_max`, `top20_holder_pct_min`, `attractiveness_max`, and related gap / mean / std summaries

The main readout in this report is the #emph[market-over-nuisance margin]:

- positive margin = the PC looks more like a market direction than a prompt-shape direction
- small or negative margin = the PC is not clean enough to trust as a market basis candidate


= What Looks Real Already

The best `market_mean` directions are encouraging because several of them are strongly market-linked while staying nearly independent of prompt length.

Representative clean `market_mean` PCs:

#table(
  columns: (1fr, 0.8fr, 2fr, 1fr, 1fr),
  align: (left, center, left, center, center),
  table.hline(stroke: 1pt),
  table.header([*State*], [*Layer / PC*], [*Top market feature(s)*], [*Market*], [*Nuisance*]),
  table.hline(stroke: 0.5pt),
  [`market_mean`], [`L4 / PC1`], [`pct_1h_max`, `top20_holder_pct_min`, `attractiveness_max`], [`0.626`], [`0.037`],
  [`market_mean`], [`L35 / PC1`], [`pct_1h_std`, `pct_5m_std`, `pct_1h_mean`], [`0.710`], [`0.071`],
  [`market_mean`], [`L30 / PC2`], [`pct_1h_std`, `pct_5m_std`, `pct_5m_max`], [`0.690`], [`0.094`],
  [`market_mean`], [`L26 / PC3`], [`pct_1h_gap`, `unique_traders_5m_max`], [`0.550`], [`0.019`],
)

The pattern is not random. These clean directions cluster around two kinds of market questions:

- #text(weight: "medium")[leader strength]
  how strong the best asset looks relative to the roster
- #text(weight: "medium")[dispersion]
  how spread out or uneven the market is across the six assets

That is already a better place to be than the old hand-built `strength / quality` scaffold. The model appears to care about summary variables we can name from the rendered rows themselves.


= What Is Still Confounded

Not every high-variance direction is a useful market basis direction.

The clearest failure mode is:

- some dominant PCs track `vol_5m_mean` or `vol_1h_mean`
- but they also track `seq_len` and `user_chars` almost one-for-one

Representative confounded PCs:

#table(
  columns: (1fr, 0.8fr, 2fr, 1fr, 1fr),
  align: (left, center, left, center, center),
  table.hline(stroke: 1pt),
  table.header([*State*], [*Layer / PC*], [*Top market feature(s)*], [*Market*], [*Nuisance*]),
  table.hline(stroke: 0.5pt),
  [`market_mean`], [`L6 / PC3`], [`vol_5m_mean`, `vol_1h_mean`, `net_flow_5m_std`], [`0.868`], [`0.867`],
  [`market_mean`], [`L5 / PC3`], [`vol_5m_mean`, `vol_1h_mean`, `net_flow_5m_std`], [`0.856`], [`0.879`],
  [`market_eos`], [`L3 / PC1`], [`vol_5m_mean`, `vol_1h_mean`, `net_flow_5m_std`], [`0.822`], [`0.828`],
  [`market_eos`], [`L7 / PC2`], [`vol_5m_mean`, `vol_1h_mean`, `net_flow_5m_std`], [`0.811`], [`0.802`],
)

So the honest read is:

- some discovered directions are real market candidates
- some are still partly prompt-shape or formatting directions

That is not a failure. It is exactly why this discovery phase exists.


= `market_mean` Vs `market_eos`

The two pooled states are not equally good.

`market_mean` is the stronger discovery object:

- stronger positive margins
- cleaner leader and dispersion reads
- several mid/late-layer PCs with market correlation well above nuisance correlation

`market_eos` is still useful, but more compressed:

- its best clean margins are smaller
- its strongest directions often emphasize a single leader or concentration extreme
- it looks more like a condensed section summary than a broad market basis

#align(center)[#image("../../data/report_assets/synthetic_market_phase15_discovery/phase15_explained_variance.png", width: 92%)]
#text(size: 8pt, fill: rgb("#888"))[
At selected layers, `market_eos` concentrates more variance into the first one or two PCs than `market_mean`, which is consistent with a more compressed section summary.
]


= What This Means For The Research Program

Phase 15 is a good first discovery pass, but not yet the final market-basis story.

The strongest conclusions are:

- the new DX-like synthetic surface is materially better than the old synthetic prompt surface
- `market_mean` already contains clean market-linked directions without needing the old hand-built latent coordinates
- the most plausible empirical basis candidates are about:
  - leader strength
  - market dispersion
  - concentration extremes
- prompt-length confounds are still real, especially in volume-like directions

So the next step should be:

1. reduce or orthogonalize prompt-length variation across background rosters
2. rerun this same discovery analysis
3. use the discovered clean directions, not the old hand-built axes, as the basis for the context-order comparison phase (`A`, `B`, `C`)


= Raw Prompt Appendix

The raw prompts below are copied directly from the actual Phase 1 cohort. They are not paraphrased.

== Shared System Prompt

#text(size: 8pt, fill: rgb("#888"))[The exact system prompt used for both example prompts below.]
#v(0.3em)
#prompt-block("raw_prompts/phase15_market_basis_discovery_system.txt")

== Example 1: Coupled Prompt (`pct_5m × net_flow_5m`)

#text(size: 8pt, fill: rgb("#888"))[A verbatim user prompt from `log_id = 2147260000` in the Phase 15 discovery cohort.]
#v(0.3em)
#prompt-block("raw_prompts/phase15_market_basis_discovery_coupled_user.txt")

== Example 2: Scalar Prompt (`net_flow_5m`)

#text(size: 8pt, fill: rgb("#888"))[A verbatim user prompt from `log_id = 2147260100` in the Phase 15 discovery cohort.]
#v(0.3em)
#prompt-block("raw_prompts/phase15_market_basis_discovery_scalar_user.txt")
