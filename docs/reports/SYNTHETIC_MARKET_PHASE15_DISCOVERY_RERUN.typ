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

#align(left)[
  #text(size: 9pt, fill: rgb("#b33a2a"), tracking: 0.08em, weight: "medium")[XENON INTERPRETABILITY]
  #v(0.3em)
  #text(size: 22pt, weight: "bold")[Synthetic Market Phase 15 Re-Run]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    Same 184 captured prompts, same two pooled market states, but a cleaner analysis pass: regress nuisance variables out of the activations
    before PCA, then rerun the discovery readout on the residual activations.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[23 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[CAPTURES]\ #text(size: 9pt)[No new capture]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[STATES]\ #text(size: 9pt)[`market_mean`, `market_eos`]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[METHOD]\ #text(size: 9pt)[Residualize nuisances, then PCA]],
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
    The re-run validates the strongest Phase 15 market basis claim, but only for `market_mean`. After nuisance residualization,
    the old leader and dispersion directions survive and get substantially cleaner. `market_eos` improves too, but it fails the
    strict sanity test: too many top PCs still correlate with prompt-shape variables above `0.15`.
  ]
]

#v(1.2em)

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[SAME DATA]\ #text(size: 16pt, weight: "bold")[`184`] #text(size: 8pt, fill: rgb("#888"))[\ prompts reused from Phase 15 baseline]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[`market_mean` SANITY]\ #text(size: 16pt, weight: "bold")[`176 → 1`] #text(size: 8pt, fill: rgb("#888"))[\ PCs above `0.15` nuisance]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST `market_mean`]\ #text(size: 16pt, weight: "bold")[`0.723 vs 0.019`] #text(size: 8pt, fill: rgb("#888"))[\ `pct_1h_std` at `L36 / PC1`]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[`market_eos` SANITY]\ #text(size: 16pt, weight: "bold")[`177 → 21`] #text(size: 8pt, fill: rgb("#888"))[\ still too many noisy PCs]],
)


= What Changed

This is an analysis-only re-run. Nothing about the prompts or the captures changed.

The only new step is:

1. build a nuisance matrix from `seq_len`, `user_chars`, and `n_rows`
2. fit a linear regression from those nuisance variables into the full activation vector
3. subtract the nuisance prediction from the activation vector
4. run PCA on the residual activations, not the raw activations

In code, the heart of the change is:

#block(
  fill: rgb("#f8f8f8"),
  inset: 10pt,
  radius: 4pt,
  stroke: 0.5pt + rgb("#ddd"),
)[
  #text(font: "Menlo", size: 8pt)[
    reg = LinearRegression()
    reg.fit(nuisance, activations)
    residuals = activations - reg.predict(nuisance)
  ]
]

The goal is simple: if a high-variance direction only exists because prompt length changes with a market variable, remove that prompt-shape part first and then ask what the top directions still track.


= What The Re-Run Confirms

The strongest clean directions from the first run survive:

- the early leader read at `market_mean`, `L4 / PC1`
- the late dispersion read at `market_mean`, `L35 / PC1`

After residualization, both get cleaner:

#table(
  columns: (1.2fr, 1.2fr, 1fr, 1fr, 1fr, 1fr),
  align: (left, center, center, center, center, center),
  table.hline(stroke: 1pt),
  table.header([*State*], [*Layer / PC*], [*Top market feature*], [*Baseline*], [*Re-run*], [*Re-run nuisance*]),
  table.hline(stroke: 0.5pt),
  [`market_mean`], [`L4 / PC1`], [`pct_1h_max`], [`0.626`], [`0.622`], [`0.006`],
  [`market_mean`], [`L35 / PC1`], [`pct_1h_std`], [`0.710`], [`0.699`], [`0.022`],
)

So the clean Phase 15 story was not a fluke. The residualized pass says:

- there really is an early leader-strength direction
- there really is a later dispersion direction
- those directions do not depend on the prompt-length confound

#align(center)[#image("../../data/report_assets/synthetic_market_phase15_discovery_rerun/phase15_rerun_compare.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
The baseline and residualized layer profiles tell two different stories. `market_mean` becomes cleaner almost everywhere. `market_eos` improves, but still retains several nuisance-heavy layers.
]


= What Changes After Residualization

The re-run does not just clean the old picture. It also sharpens it.

For `market_mean`, the best layer-by-layer feature becomes much more consistent:

- baseline: a mix of `pct_1h_max`, `pct_1h_gap`, `pct_1h_std`, and a few one-off features
- re-run: `pct_1h_std` dominates `36` of `48` layers, with `pct_1h_max` taking the other `12`

That is a better discovery result than the first report. It suggests the model's broad market summary is organized mainly around:

- leader strength
- market dispersion

For `market_eos`, the new signal is different:

- baseline: a noisy mix of dispersion, gaps, volume-like features, and concentration
- re-run: `pct_5m_max` dominates `37` of `48` layers

That looks like a real short-horizon leader signal becoming visible only after the volume confound is stripped away.

#align(center)[#image("../../data/report_assets/synthetic_market_phase15_discovery_rerun/phase15_rerun_feature_shift.png", width: 100%)]
#text(size: 8pt, fill: rgb("#888"))[
Residualization collapses the noisy best-feature distribution into a much simpler picture: `market_mean` mostly tracks dispersion and leader strength, while `market_eos` mostly tracks a compressed short-horizon leader read.
]


= The Sanity Check

The user-specified sanity test was:

- after residualization, top PC nuisance correlations should be near zero
- if a PC still correlates with `seq_len` above about `0.15`, something is still wrong

This passes for `market_mean` and fails for `market_eos`.

Specifically:

- `market_mean`
  - baseline: `176 / 240` PCs above `0.15`
  - re-run: `1 / 240`
  - only one outlier remains: `L24 / PC5` at `0.176`
- `market_eos`
  - baseline: `177 / 240` PCs above `0.15`
  - re-run: `21 / 240`
  - the worst layers are still clustered early and mid-section, peaking at `L8 / PC2 = 0.712`

So the correct read is:

- `market_mean` is now clean enough to serve as the discovered market basis for Phase 2
- `market_eos` is improved, but still too entangled with prompt-shape variation to trust as the primary basis state


= What This Sets Up

This re-run makes the next phase much clearer.

For the context-order comparison (`A`, `B`, `C`):

- use `market_mean` as the primary discovery state
- keep `market_eos` as a secondary compressed summary state
- track the clean discovered directions first:
  - leader strength
  - market dispersion
  - short-horizon leader read

That is a better position than the original Phase 15 baseline because the main basis candidates are now defended against the most obvious nuisance confound.


= Raw Prompt Appendix

The raw prompts below are copied directly from the actual Phase 15 cohort. They are unchanged between the baseline and the re-run.

== Shared System Prompt

#text(size: 8pt, fill: rgb("#888"))[The exact system prompt used for both example prompts below.]
#v(0.3em)
#prompt-block("raw_prompts/phase15_market_basis_discovery_system.txt")

== Example 1: Coupled Prompt (`pct_5m × net_flow_5m`)

#text(size: 8pt, fill: rgb("#888"))[A verbatim user prompt from the discovery cohort.]
#v(0.3em)
#prompt-block("raw_prompts/phase15_market_basis_discovery_coupled_user.txt")

== Example 2: Scalar Prompt (`net_flow_5m`)

#text(size: 8pt, fill: rgb("#888"))[A verbatim user prompt from the discovery cohort.]
#v(0.3em)
#prompt-block("raw_prompts/phase15_market_basis_discovery_scalar_user.txt")
