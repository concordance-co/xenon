// ─────────────────────────────────────────────────────────────────
// Prompt Confusion Phase 03 — Mechanistic Probe Brief
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
  #text(size: 20pt, weight: "bold")[Prompt Confusion Mechanistic Slice]
  #v(0.3em)
  #text(size: 10.5pt, fill: rgb("#47433f"))[
    First-pass readout on two promising families: #mono([trade_size_force_large]) and #mono([activity_force_observe]). The slice uses aligned vs strong-conflict rows plus a strong-conflict-only strategy-vs-setting split.
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
    [#text(size: 6.8pt, fill: muted, weight: "bold")[STATUS]\ #text(size: 8.5pt)[Probe-only sanity pass]],
  )
]

#v(0.7em)

#callout(
  [BOTTOM LINE],
  [
    #text(size: 11pt, weight: "medium")[
      The slice is clearly linearly decodable, but the current probe setup is probably too easy. The conflict-vs-aligned result is so strong that it is more likely reading prompt-family or formatting differences than a clean conflict representation. The strategy-vs-setting split is more believable and worth following up, but still needs grouped controls before it supports a mechanistic claim.
    ]
  ],
)

#v(0.7em)

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 0.6em,
  stat-card([CONFLICT SLICE], mono([1,944], size: 13pt), [Aligned vs strong-conflict rows]),
  stat-card([SOURCE SLICE], mono([666], size: 13pt), [Strong-conflict only; strategy vs setting readout]),
  stat-card([BEST CONFLICT LAYER], mono([L12], size: 13pt), [Balanced acc 0.998]),
  stat-card([BEST SOURCE LAYER], mono([L8], size: 13pt), [Balanced acc 0.906]),
)

#v(0.7em)

#callout(
  [SLICE DEFINITION],
  [
    The probe slice comes from two Neon views over the same capture. The first view labels rows as #mono([aligned]) or #mono([conflict]) and contains #mono([1,944]) examples. The second keeps only strong-conflict rows with usable readout-side labels and relabels them as #mono([strategy]) or #mono([setting]); it contains #mono([666]) examples. Both runs use residual-stream last-token activations on the captured layers #mono([0, 4, 8, ..., 44]).
  ],
  fill: rgb("#f4f1ec"),
  accent_color: accent_blue,
)

= Probe Readout

#grid(
  columns: (1fr, 1fr),
  gutter: 0.9em,
  [
    == Conflict Vs Non-Conflict

    The global readout is almost perfectly separable at every captured layer. Accuracy ranges from #mono([0.967]) at layer 0 to #mono([0.998]) at layer 12; balanced accuracy is effectively the same because the slice is balanced. That is too good to trust at face value.

    *Best layers.*
    #table(
      columns: (0.6fr, 1fr, 1fr, 1fr),
      align: (left, center, center, center),
      stroke: 0.4pt + border,
      inset: (x: 5pt, y: 3pt),
      table.header(
        text(size: 7.8pt, weight: "bold")[Layer],
        text(size: 7.8pt, weight: "bold")[Accuracy],
        text(size: 7.8pt, weight: "bold")[Balanced],
        text(size: 7.8pt, weight: "bold")[Selectivity],
      ),
      [12], [0.9979], [0.9979], [0.4830],
      [24], [0.9964], [0.9964], [0.4856],
      [36], [0.9938], [0.9938], [0.5324],
      [4],  [0.9877], [0.9877], [0.5329],
    )

    #v(0.25em)
    #cap[Conflict run: strong signal, but likely contaminated by prompt-level differences under random CV.]
  ],
  [
    == Strategy Vs Setting

    The strong-conflict-only readout is weaker but still strong across the whole layer sweep. Balanced accuracy ranges from roughly #mono([0.885]) to #mono([0.906]) on #mono([666]) examples, peaking in the early-mid stack. This looks more plausible than the conflict split, but it still shares the same leakage risk.

    *Best layers.*
    #table(
      columns: (0.6fr, 1fr, 1fr, 1fr),
      align: (left, center, center, center),
      stroke: 0.4pt + border,
      inset: (x: 5pt, y: 3pt),
      table.header(
        text(size: 7.8pt, weight: "bold")[Layer],
        text(size: 7.8pt, weight: "bold")[Accuracy],
        text(size: 7.8pt, weight: "bold")[Balanced],
        text(size: 7.8pt, weight: "bold")[Selectivity],
      ),
      [8],  [0.9054], [0.9058], [0.4521],
      [36], [0.9024], [0.9038], [0.4415],
      [4],  [0.9009], [0.9015], [0.4475],
      [32], [0.8964], [0.8974], [0.4475],
    )

    #v(0.25em)
    #cap[Source run: useful signal, but still not a clean mechanistic result without grouped evaluation.]
  ],
)

= Why The First Result Is Suspicious

#callout(
  [MAIN CAVEAT],
  [
    The current analysis uses ordinary cross-validation over rows. That means aligned and strong-conflict examples from the same prompt family can land in both train and test. A linear probe can therefore succeed by reading stable formatting or family-level artifacts rather than a clean "conflict state." The near-perfect conflict-vs-non-conflict numbers are the signature to worry about, not celebrate.
  ],
  fill: rgb("#f8f1ee"),
  accent_color: accent,
)

#v(0.45em)

#table(
  columns: (1.3fr, 1.7fr),
  align: (left, left),
  stroke: 0.4pt + border,
  inset: (x: 6pt, y: 4pt),
  table.header(
    text(size: 8pt, weight: "bold")[What Looks Real],
    text(size: 8pt, weight: "bold")[What Still Needs Control],
  ),
  [There is definitely linearly decodable structure in the slice.], [Whether the readout survives grouped CV by #mono([matched_pair_id]) or template family.],
  [The strategy-vs-setting task is harder than the coarse conflict task.], [Whether the signal remains once family and pressure-bucket shortcuts are blocked.],
  [Early-mid layers are already sufficient for source readout.], [Whether the best read layer is also the right write/localization layer.],
)

= Recommended Next Step

The next pass should be narrower and stricter rather than broader:

1. Re-run the two probes with grouped CV by #mono([matched_pair_id]).
2. Break out #mono([trade_size_force_large]) and #mono([activity_force_observe]) separately.
3. Add a matched-pair delta analysis: compare #mono([strong_conflict - aligned]) inside the same pair.
4. Only after that, decide whether PCA or causal patching is worth the time.

#v(0.45em)
#cap[Current status: good enough to continue, not good enough to claim a mechanism.]
