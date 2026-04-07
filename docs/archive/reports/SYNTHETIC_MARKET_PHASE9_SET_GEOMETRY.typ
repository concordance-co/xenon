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
  #text(size: 22pt, weight: "bold")[Synthetic Market Phase 9]
  #v(0.4em)
  #text(size: 11pt, fill: rgb("#4a4a4a"))[
    Four-asset set geometry. This phase keeps rank order fixed while changing the latent shape of the market across four assets,
    then asks whether the model preserves coordinates, within-snapshot geometry, or whole-shape identity.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.8em,
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DATE]\ #text(size: 9pt)[22 March 2026]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[PROMPTS]\ #text(size: 9pt)[96 market-only]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[DESIGN]\ #text(size: 9pt)[4 shapes × 4 layouts × 2 styles × 3 scales]],
    [#text(size: 7.5pt, fill: rgb("#888"), weight: "bold")[OBJECT]\ #text(size: 9pt)[4-asset latent market geometry]],
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
  #text(size: 12.5pt, weight: "medium")[Phase 9 finds a better object than row retrieval: the model preserves latent asset coordinates very strongly and within-snapshot shape moderately, but not a nuisance-invariant whole-shape family code.]
  #v(0.4em)
  #text(size: 9.5pt, fill: rgb("#555"))[
    This is a useful split result. The model does not appear to store a robust "same-rank market template" for each 4-asset shape. Instead,
    it looks more like it preserves asset positions in a latent coordinate system, from which some geometry can be reconstructed.
  ]
]

#v(1.2em)

// ── Summary Metrics ─────────────────────────────────────────────
#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 1em,
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST LATENT X]\ #text(size: 16pt, weight: "bold")[0.99967] #text(size: 8pt, fill: rgb("#888"))[\ row_mean L2]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST LATENT Y]\ #text(size: 16pt, weight: "bold")[0.99977] #text(size: 8pt, fill: rgb("#888"))[\ row_mean L1]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST ALIGNMENT]\ #text(size: 16pt, weight: "bold")[0.35238] #text(size: 8pt, fill: rgb("#888"))[\ middle gap, row_eos L25]],
  [#text(size: 7pt, fill: rgb("#888"), weight: "bold")[BEST IDENTITY]\ #text(size: 16pt, weight: "bold")[0.00097] #text(size: 8pt, fill: rgb("#888"))[\ dominant outlier, style-only]],
)


= Why Phase 9 Exists

Phase 8 showed that a fixed anchor-pair relation is too brittle once the surrounding roster changes. That suggested a better object:

- not a single row
- not one pair
- but the relative positioning of multiple assets inside the same market

Phase 9 uses a 4-asset synthetic market and holds the rank order fixed while changing the latent shape. That creates a cleaner test of whether the model is preserving:

1. per-asset market coordinates
2. the geometry of the whole set
3. a reusable "market family" identity


= The Four Latent Market Shapes

#align(center)[#image("../../data/report_assets/synthetic_market_phase9_set_geometry/latent_shapes.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[
The Phase 9 scenarios deliberately share the same winner-to-loser order while changing pairwise spacing and clustering structure.
]

#v(0.4em)

The key design choice is that rank order alone is no longer sufficient. If the model only preserves ordinal structure, the scenarios should collapse together. If it preserves richer market geometry, those scenarios should separate.


= Latent Coordinates Are Explicit

#align(center)[#image("../../data/report_assets/synthetic_market_phase9_set_geometry/coordinate_regression.png", width: 82%)]
#text(size: 8pt, fill: rgb("#888"))[
Each asset row can be mapped back to its latent 2D market coordinates with almost perfect held-out accuracy.
]

#v(0.4em)

#table(
  columns: (1.4fr, 1.2fr, auto, auto),
  align: (left, left, right, right),
  table.hline(stroke: 1pt),
  table.header(
    [*Target*], [*Best State*], [*Metric*], [*Layer*],
  ),
  table.hline(stroke: 0.5pt),
  [`latent_x`], [`row_mean`], [R² 0.99967], [L2],
  [`latent_y`], [`row_mean`], [R² 0.99977], [L1],
  table.hline(stroke: 1pt),
)

#v(0.5em)

This is the strongest result in the phase. It says the model does not merely keep coarse rankings or local heuristics: the individual row states make the latent asset positions almost perfectly recoverable.


= Within-Snapshot Geometry Partly Survives

#align(center)[#image("../../data/report_assets/synthetic_market_phase9_set_geometry/alignment.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[
Distance structure is partially preserved, but exact pair identification remains weak even at the best layer for each scenario.
]

#v(0.4em)

#table(
  columns: (1.5fr, 1.1fr, auto, auto, auto),
  align: (left, left, right, right, right),
  table.hline(stroke: 1pt),
  table.header(
    [*Scenario*], [*Best State*], [*Distance Spearman*], [*Closest Pair*], [*Farthest Pair*],
  ),
  table.hline(stroke: 0.5pt),
  [`even ladder`], [`row_eos` L25], [0.31887], [0.125], [0.500],
  [`top pair cluster`], [`row_eos` L23], [0.34524], [0.1667], [0.500],
  [`dominant outlier`], [`row_eos` L12], [0.15476], [0.500], [0.375],
  [`middle gap`], [`row_eos` L25], [0.35238], [0.0000], [0.500],
  table.hline(stroke: 1pt),
)

#v(0.5em)

This is a real but limited positive:

- the internal row geometry tracks the latent distances better than chance
- later `row_eos` states are consistently best for that job
- but the model does not cleanly recover the exact closest/farthest pair structure

So the geometry is there, but not as a crisp canonical shape code.


= Whole-Shape Identity Does Not Survive

#align(center)[#image("../../data/report_assets/synthetic_market_phase9_set_geometry/identity_modes.png", width: 96%)]
#text(size: 8pt, fill: rgb("#888"))[
Same-rank geometry-family retrieval is nearly flat. Style and scale leave tiny positive margins; layout changes erase them.
]

#v(0.4em)

#table(
  columns: (1.6fr, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  table.hline(stroke: 1pt),
  table.header(
    [*Scenario*], [*Full*], [*Style-only*], [*Layout-only*], [*Magnitude-only*],
  ),
  table.hline(stroke: 0.5pt),
  [`even ladder`], [0.00011], [0.00016], [-0.00024], [0.00014],
  [`top pair cluster`], [0.00018], [0.00023], [-0.00027], [0.00020],
  [`dominant outlier`], [0.00062], [0.00097], [-0.00015], [0.00072],
  [`middle gap`], [0.00011], [0.00018], [-0.00012], [0.00015],
  table.hline(stroke: 1pt),
)

#v(0.5em)

This is the clearest negative result:

- the same-rank market families do *not* look like reusable, nuisance-invariant templates
- `layout_only` is negative in every scenario
- even the best positive margins are tiny

That sharply limits what Phase 9 can support. It does not justify claiming a clean family-level market manifold.


= Interpretation

#block(
  width: 100%,
  inset: (left: 14pt, top: 10pt, bottom: 10pt, right: 10pt),
  stroke: (left: 3pt + rgb("#2e7d32"), top: none, right: none, bottom: none),
  fill: rgb("#e8f5e9"),
)[
  #text(size: 7.5pt, fill: rgb("#2e7d32"), weight: "bold", tracking: 0.08em)[SUPPORTED NOW]
  #v(0.2em)
  The model preserves coordinate-like information about multiple assets, and some of the set-level geometry can be reconstructed from those row states.
]

#v(0.5em)

#block(
  width: 100%,
  inset: (left: 14pt, top: 10pt, bottom: 10pt, right: 10pt),
  stroke: (left: 3pt + rgb("#f57f17"), top: none, right: none, bottom: none),
  fill: rgb("#fff8e1"),
)[
  #text(size: 7.5pt, fill: rgb("#f57f17"), weight: "bold", tracking: 0.08em)[NOT SUPPORTED]
  #v(0.2em)
  The model does not appear to encode each same-rank 4-asset market as a stable whole-shape identity that survives layout changes.
]

#v(0.8em)

The best synthesis is:

- asset-level positions are very real
- whole-set spacing is partly real
- family retrieval is mostly the wrong target

That is a better place to land than another easy pairwise win, because it narrows the right research object.


= What To Do Next

Phase 9 suggests the next step should keep the 4-asset object but change the target:

1. recover or compare the pair-distance matrix directly
2. study set-level geometric alignment instead of family retrieval
3. add settings back in and ask whether settings *deform* an existing market geometry
4. test whether later context preserves, rotates, or compresses the latent coordinate system

The most promising follow-up is not "which family is this?" but "how does the market geometry move?"


= Conclusion

Phase 9 is a useful refinement of the market-representation program.

It rejects a stronger, cleaner story:

- there is no good evidence here for a reusable same-rank shape code

But it supports a better one:

- the model appears to maintain asset positions in a latent coordinate system, and some multi-asset market geometry is recoverable from those positions

That is a stronger foundation for the next stage than row retrieval or fixed-pair identity.
