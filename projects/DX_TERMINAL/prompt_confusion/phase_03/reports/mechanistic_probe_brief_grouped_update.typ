// ─────────────────────────────────────────────────────────────────
// Prompt Confusion Phase 03 — Mechanistic Probe Brief (Grouped Update)
// Self-contained: no external JSON dependencies.
// ─────────────────────────────────────────────────────────────────

#set page(
  paper: "us-letter",
  margin: (top: 1.75cm, bottom: 1.75cm, left: 2.0cm, right: 2.0cm),
  numbering: "1 / 1",
  number-align: right,
)

#set text(font: "Georgia", size: 9.6pt)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: none)

#show heading.where(level: 1): it => {
  set text(size: 12.5pt, weight: "bold")
  v(0.78em)
  it
  v(0.18em)
}

#show heading.where(level: 2): it => {
  set text(size: 10pt, weight: "bold")
  v(0.5em)
  it
  v(0.12em)
}

#let accent = rgb("#9d3c2a")
#let accent_green = rgb("#2f6b4f")
#let accent_blue = rgb("#33567b")
#let accent_gold = rgb("#8a6b1f")
#let muted = rgb("#7a746e")
#let cream = rgb("#f7f4f1")
#let stone = rgb("#f9f7f4")
#let border = rgb("#e0d9d3")
#let ink = rgb("#1b1815")

#let mono(body, size: 10pt, weight: "bold", fill: ink) = {
  text(font: "Menlo", size: size, weight: weight, fill: fill)[#body]
}

#let callout(title, body, fill: cream, accent_color: accent) = block(
  width: 100%,
  inset: (left: 12pt, right: 12pt, top: 6pt, bottom: 6pt),
  fill: fill,
  stroke: (left: 3pt + accent_color, top: none, right: none, bottom: none),
)[
  #text(size: 7.5pt, tracking: 0.08em, fill: accent_color, weight: "bold")[#title]
  #v(0.15em)
  #body
]

#let stat-card(label, value, sub) = block(
  width: 100%,
  inset: (left: 9pt, right: 9pt, top: 8pt, bottom: 8pt),
  fill: stone,
  stroke: 0.4pt + border,
)[
  #text(size: 7pt, fill: muted, weight: "bold", tracking: 0.08em)[#label]
  #v(0.25em)
  #text(size: 17pt, weight: "bold", fill: ink)[#value]
  #v(0.2em)
  #text(size: 8pt, fill: muted)[#sub]
]

#let cap(body) = text(size: 7.8pt, fill: muted, style: "italic")[#body]

#align(left)[
  #text(size: 8.5pt, fill: accent, tracking: 0.10em, weight: "medium")[CONCORDANCE · PHASE 03 BRIEF]
  #v(0.2em)
  #text(size: 20pt, weight: "bold")[Prompt Confusion Mechanistic Slice — Grouped Update]
  #v(0.3em)
  #text(size: 10.5pt, fill: rgb("#47433f"))[
    Update to the earlier brief after rerunning the probe analysis with grouped cross-validation by #mono([matched_pair_id]) where available.
  ]
  #v(0.55em)
  #line(length: 100%, stroke: 0.9pt + black)
  #v(0.3em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.6em,
    [#text(size: 6.8pt, fill: muted, weight: "bold")[DATE]\ #text(size: 8.5pt)[April 2026]],
    [#text(size: 6.8pt, fill: muted, weight: "bold")[MODEL]\ #text(size: 8.5pt)[Qwen3-30B-A3B]],
    [#text(size: 6.8pt, fill: muted, weight: "bold")[CAPTURE RUN]\ #text(size: 8.5pt)[#mono([09036218e3ae])]],
    [#text(size: 6.8pt, fill: muted, weight: "bold")[STATUS]\ #text(size: 8.5pt)[Grouped rerun complete]],
  )
]

#v(0.7em)

#callout(
  [BOTTOM LINE],
  [
    #text(size: 11pt, weight: "medium")[
      The grouped rerun did #emph[not] knock down the conflict-vs-non-conflict result. That means the earlier concern about pair leakage was not the main issue for that task. The conflict probe still looks almost perfectly separable, which shifts suspicion toward broader prompt-family or task-construction differences. The source probe remains strong, but the grouped rerun did not really test a harder split there because each row was already its own group.
    ]
  ],
)

#v(0.7em)

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 0.6em,
  stat-card([CONFLICT GROUPS], mono([972], size: 13pt), [1,944 rows; paired split applied]),
  stat-card([SOURCE GROUPS], mono([666], size: 13pt), [666 rows; one row per group]),
  stat-card([BEST CONFLICT LAYER], mono([L12], size: 13pt), [Grouped balanced acc 0.996]),
  stat-card([BEST SOURCE LAYER], mono([L20], size: 13pt), [Grouped balanced acc 0.921]),
)

#v(0.7em)

= What Changed

#table(
  columns: (1.2fr, 0.75fr, 0.75fr, 1.2fr),
  align: (left, center, center, left),
  stroke: 0.4pt + border,
  inset: (x: 6pt, y: 4pt),
  table.header(
    text(size: 8pt, weight: "bold")[Task],
    text(size: 8pt, weight: "bold")[Old Best Bal],
    text(size: 8pt, weight: "bold")[Grouped Best Bal],
    text(size: 8pt, weight: "bold")[Interpretation],
  ),
  [Conflict vs non-conflict], [0.9979], [0.9959], [Essentially unchanged; pair leakage was not the main driver.],
  [Strategy vs setting], [0.9058], [0.9213], [Nominally improved, but this grouped split is not materially stronger because groups = rows.],
)

#v(0.45em)

#callout(
  [KEY UPDATE],
  [
    The first rerun answered one question clearly: keeping matched pairs together does #emph[not] explain away the conflict readout. The second rerun answered a more awkward question: using #mono([matched_pair_id]) as the group key does almost nothing for the source task, because the strong-conflict-only slice contains one row per pair.
  ],
  fill: rgb("#f4f1ec"),
  accent_color: accent_blue,
)

= Probe Readout After Grouping

#grid(
  columns: (1fr, 1fr),
  gutter: 0.9em,
  [
    == Conflict Vs Non-Conflict

    Grouped evaluation still gives near-perfect separation. The best layer stays at #mono([L12]), with balanced accuracy #mono([0.9959]). The weakest layer is still very high at #mono([0.9717]). That is the real update: the coarse conflict task remains trivially linearly separable even when aligned and strong-conflict rows from the same pair are forced into the same fold.

    #table(
      columns: (0.7fr, 1fr, 1fr, 1fr),
      align: (left, center, center, center),
      stroke: 0.4pt + border,
      inset: (x: 5pt, y: 3pt),
      table.header(
        text(size: 7.8pt, weight: "bold")[Layer],
        text(size: 7.8pt, weight: "bold")[Grouped Bal],
        text(size: 7.8pt, weight: "bold")[Old Bal],
        text(size: 7.8pt, weight: "bold")[Delta],
      ),
      [12], [0.9959], [0.9979], [-0.0020],
      [40], [0.9943], [0.9841], [+0.0102],
      [8],  [0.9933], [0.9856], [+0.0077],
      [0],  [0.9717], [0.9666], [+0.0051],
    )

    #v(0.25em)
    #cap[Grouped conflict probe: still too easy. The leakage story has moved up a level, from matched-pair leakage to likely family/template/task leakage.]
  ],
  [
    == Strategy Vs Setting

    The source probe remains strong, with grouped balanced accuracy peaking at #mono([0.9213]) on #mono([L20]). But the grouping key is not doing real work here: the export reports #mono([666 groups]) for #mono([666 rows]). So this rerun is still informative about signal strength, not yet a tougher generalization test.

    #table(
      columns: (0.7fr, 1fr, 1fr, 1fr),
      align: (left, center, center, center),
      stroke: 0.4pt + border,
      inset: (x: 5pt, y: 3pt),
      table.header(
        text(size: 7.8pt, weight: "bold")[Layer],
        text(size: 7.8pt, weight: "bold")[Grouped Bal],
        text(size: 7.8pt, weight: "bold")[Old Bal],
        text(size: 7.8pt, weight: "bold")[Delta],
      ),
      [20], [0.9213], [0.8875], [+0.0338],
      [4],  [0.9173], [0.9015], [+0.0158],
      [12], [0.9150], [0.8983], [+0.0167],
      [0],  [0.8751], [0.8866], [-0.0115],
    )

    #v(0.25em)
    #cap[Grouped source probe: good signal, but not yet a stricter control because #mono([matched_pair_id]) is one-to-one with rows in this slice.]
  ],
)

= What This Means Now

#callout(
  [REVISED TAKE],
  [
    The pair-grouping fix was necessary, but it did not change the main picture. The conflict task is still suspiciously easy. That means the next control should not be another generic rerun of the same setup; it should be a more meaningful split, such as leaving out a whole family, template, or pressure bucket. For the source task, the current result is encouraging, but it still needs a grouping key that actually bundles related rows together.
  ],
  fill: rgb("#f8f1ee"),
  accent_color: accent_gold,
)

#v(0.45em)

#table(
  columns: (1.25fr, 1.75fr),
  align: (left, left),
  stroke: 0.4pt + border,
  inset: (x: 6pt, y: 4pt),
  table.header(
    text(size: 8pt, weight: "bold")[Question],
    text(size: 8pt, weight: "bold")[Answer After Grouped Rerun],
  ),
  [Did matched-pair leakage explain the conflict result?], [No. The grouped conflict probe remained near-perfect.],
  [Did the grouped rerun make the source test meaningfully stricter?], [Not yet. Each source row was already its own group under #mono([matched_pair_id]).],
  [What is the next real control?], [Leave out broader shared structure: family, template, or pressure bucket; or define a better group key for the source slice.],
)

= Recommended Next Step

The next pass should focus on #emph[harder generalization], not more of the same:

1. For conflict vs non-conflict, try leave-one-family-out or leave-one-template-out probes.
2. For strategy vs setting, define a better grouping key than #mono([matched_pair_id]) for the strong-conflict-only slice.
3. Only after one of those survives should we treat the readout as strong enough to motivate finer localization or causal work.

#v(0.45em)
#cap[This update supersedes the earlier interpretation of the grouped rerun, but does not replace the earlier brief file.]
