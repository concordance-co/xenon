// ─────────────────────────────────────────────────────────────────
// Synthetic Market Research — Client Brief
// Self-contained: no external JSON dependencies. All numerics inlined
// from blog_post_draft.md and client_summary_draft.md.
// ─────────────────────────────────────────────────────────────────

#set page(
  paper: "us-letter",
  margin: (top: 1.5cm, bottom: 1.5cm, left: 1.8cm, right: 1.8cm),
  numbering: "1 / 1",
  number-align: right,
)

#set text(font: "Georgia", size: 9.6pt)
#set par(justify: true, leading: 0.6em)
#set heading(numbering: none)

#show heading.where(level: 1): it => {
  set text(size: 12.5pt, weight: "bold")
  v(0.55em)
  it
  v(0.12em)
}

#show heading.where(level: 2): it => {
  set text(size: 10pt, weight: "bold")
  v(0.4em)
  it
  v(0.08em)
}

#let accent = rgb("#9d3c2a")
#let accent_green = rgb("#2f6b4f")
#let accent_blue = rgb("#33567b")
#let muted = rgb("#7a746e")
#let cream = rgb("#f7f4f1")
#let stone = rgb("#f9f7f4")
#let border = rgb("#e0d9d3")
#let ink = rgb("#1b1815")

// ── helpers ─────────────────────────────────────────────────────
#let mono(body, size: 10pt, weight: "bold", fill: ink) = {
  text(font: "Menlo", size: size, weight: weight, fill: fill)[#body]
}

#let pp(x) = {
  let y = calc.round(x * 100) / 100
  if y > 0 { "+" + str(y) + " pp" } else { str(y) + " pp" }
}

#let callout(title, body, fill: cream, accent_color: accent) = block(
  width: 100%,
  inset: (left: 12pt, right: 12pt, top: 9pt, bottom: 9pt),
  fill: fill,
  stroke: (left: 3pt + accent_color, top: none, right: none, bottom: none),
)[
  #text(size: 7.5pt, tracking: 0.08em, fill: accent_color, weight: "bold")[#title]
  #v(0.2em)
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

// ── Page 1: Cover + verdict ────────────────────────────────────

#align(left)[
  #text(size: 8.5pt, fill: accent, tracking: 0.10em, weight: "medium")[CONCORDANCE · RESEARCH BRIEF]
  #v(0.2em)
  #text(size: 20pt, weight: "bold")[Reading the Mind of an Autonomous Trading Agent]
  #v(0.3em)
  #text(size: 10.5pt, fill: rgb("#47433f"))[
    We tested whether the model's internal picture of the market is real, where it comes from, and whether it actually drives the trading choice. Six findings, one honest verdict.
  ]
  #v(0.55em)
  #line(length: 100%, stroke: 0.9pt + black)
  #v(0.3em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.6em,
    [#text(size: 6.8pt, fill: muted, weight: "bold")[DATE]\ #text(size: 8.5pt)[April 2026]],
    [#text(size: 6.8pt, fill: muted, weight: "bold")[MODEL]\ #text(size: 8.5pt)[Qwen3-30B-A3B (MoE)]],
    [#text(size: 6.8pt, fill: muted, weight: "bold")[SCOPE]\ #text(size: 8.5pt)[6 findings · 9 phases]],
    [#text(size: 6.8pt, fill: muted, weight: "bold")[STATUS]\ #text(size: 8.5pt)[Validated through restoration test]],
  )
]

#v(0.7em)

#callout(
  [BOTTOM LINE],
  [
    #text(size: 11pt, weight: "medium")[
      The model builds a precise, structured internal picture of the market, and prior context measurably shapes that picture. The two specific internal signals we isolated are real and selective — but on their own, they only partially explain the model's final trading choice. Narrower than the headline story, and stronger because of it.
    ]
  ],
)

#v(0.7em)

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 0.6em,
  stat-card([MARKET DATA FIDELITY], mono([> 0.99 R²], size: 13pt), [Every tested factor recoverable from internal state]),
  stat-card([RELATIONAL ROBUSTNESS], mono([≈ 20×], size: 13pt), [Pairwise vs single-asset stability under reformatting]),
  stat-card([SELECTIVITY], mono([12 / 12], size: 13pt), [Targeted edits beat matched random edits]),
  stat-card([RESTORATION], mono([+ 4.2 pp], size: 13pt), [Choice agreement shift on the strongest signal]),
)

#v(0.7em)

#callout(
  [WHY THIS MATTERS],
  [
    Most interpretability work stops at discovery: find a pattern, name it, call it an explanation. That is not enough — a signal that looks meaningful might just be wallpaper the model computes but never actually leans on. This study applied progressively harder tests to the same set of internal signals: discovery, format-decontamination, context-sensitivity, intervention, robustness, and source-driven restoration. Where those tests held, the claims got stronger. Where they broke, we say so.
  ],
  fill: rgb("#f4f1ec"),
  accent_color: accent_green,
)

#v(0.55em)

#align(center)[#image("assets/public_story/charts/08_research_arc.png", width: 92%)]

#pagebreak()

// ── Page 2: Findings 1-3 ───────────────────────────────────────

= Finding 1 — The Model Preserves Market Data With High Fidelity

#grid(
  columns: (1.5fr, 1fr),
  gutter: 0.9em,
  [
    Linear probes on early-layer activations recover every market metric we tested with near-perfect accuracy. Price changes at two timescales, trading volume at two timescales, net capital flow, unique trader counts, holder concentration, and the composite scores all decode at #mono([R² > 0.994]). The model does not compress, summarize, or hallucinate the input — downstream decisions are grounded in the actual numbers it was given.

    *High points.* Holder concentration #mono([0.999]); 1-hour price, 1-hour volume, unique traders, and both composite scores #mono([0.998]); net capital flow #mono([0.994]). Whatever the model is doing later, it is doing it on top of a faithful copy of what it read.
  ],
  align(center)[
    #image("assets/public_story/charts/01_market_decodability.png", width: 100%)
    #v(0.15em)
    #cap[Recovery R² for nine market factors. All > 0.994.]
  ],
)

= Finding 2 — The Representation Is Relational, Not Row-By-Row

#grid(
  columns: (1.5fr, 1fr),
  gutter: 0.9em,
  [
    When the model evaluates a token, it does so relative to the full field. We reformatted prompts (rearranging row order, swapping ticker symbols, altering text style) and watched what survived. Single-asset identity broke easily under layout changes. Pairwise relationships between assets were approximately #mono([20×]) more robust on the same prompt set.

    *Scale.* 384 controlled prompt variations across four scenario families, multiple formatting styles, and three magnitude scales. The model's internal market is encoded as a structure of comparisons, not a list of independent rows.
  ],
  align(center)[
    #image("assets/public_story/charts/02_relational_stability.png", width: 100%)
    #v(0.15em)
    #cap[Single-asset vs pairwise stability under reformatting.]
  ],
)

= Finding 3 — Pre-Market Context Shapes the Market Read

#grid(
  columns: (1.5fr, 1fr),
  gutter: 0.9em,
  [
    Identical markets land differently when the framing comes first. We held the market section constant and moved the same context block either before or after it — *same words, only the placement changes.* When context appeared after the market, the internal market read was unchanged (#mono([0.000]) shift). When it appeared before the market, the read shifted: risk framing #mono([0.061]) (peak at L42), opportunity framing #mono([0.070]) (peak at L40).

    The asymmetry is the result. Not "more text moves the representation" — only context that arrives in time to act as a lens does. Confirmed on the full five-level risk ladder: the base coordinate system stays nearly perfect (#mono([R² > 0.995]) cross-context) while the *interpretation* shifts in a structured, ladder-appropriate way.
  ],
  align(center)[
    #image("assets/public_story/charts/03_context_effect.png", width: 100%)
    #v(0.15em)
    #cap[Phase 16: same context, only the placement changes.]
  ],
)

#pagebreak()

// ── Page 3: Findings 4-6 ───────────────────────────────────────

= Finding 4 — Two Named Internal Signals

#grid(
  columns: (1fr, 1fr),
  gutter: 0.7em,
  align(center)[#image("assets/public_story/charts/04a_leader_signal.png", width: 100%)],
  align(center)[#image("assets/public_story/charts/04b_dispersion_signal.png", width: 100%)],
)

After controlling for prompt-formatting artifacts (sequence length, character count, asset count), statistical decomposition of the model's market-section activations identifies two recurring internal patterns. *Leader* (early layers) tracks the standout asset: best single feature #mono([R² 0.46]), best pair #mono([R² 0.67]). *Dispersion* (later layers) tracks how uneven the market is: best single feature #mono([R² 0.52]), best pair #mono([R² 0.84]). Both collapsed below #mono([R² 0.06]) on shuffled-data controls — confirming they reflect market content, not statistical noise — and both decode from features a human trader could also read off the screen.

= Finding 5 — The Signals Are Real (Selectivity Test)

#grid(
  columns: (1.15fr, 1fr),
  gutter: 0.9em,
  [
    We edited the model's internal state at each signal site and measured behavioral disruption against matched random edits of the same magnitude. In every tested condition, the targeted edit produced *less* collateral disruption than the matched random one — the signature of a meaningful internal feature, not a generic perturbation effect.

    #v(0.2em)

    #table(
      columns: (1.7fr, 0.8fr, 0.8fr, 0.7fr),
      align: (left, center, center, center),
      stroke: 0.4pt + border,
      inset: (x: 5pt, y: 3pt),
      table.header(
        text(size: 7.8pt, weight: "bold")[Condition],
        text(size: 7.8pt, weight: "bold")[Targeted],
        text(size: 7.8pt, weight: "bold")[Random],
        text(size: 7.8pt, weight: "bold")[Gap],
      ),
      [Leader, constructive],   [43.8%], [68.8%], [25.0 pp],
      [Leader, destructive],    [56.3%], [68.8%], [12.5 pp],
      [Dispersion, constructive], [40.6%], [75.0%], [34.4 pp],
      [Dispersion, destructive],  [31.3%], [65.6%], [34.4 pp],
    )

    Across three strengths × four conditions, *all 12 comparisons* showed the same pattern. Both signals carry specific, non-redundant information.
  ],
  align(center)[
    #image("assets/public_story/charts/05_selectivity.png", width: 100%)
    #v(0.15em)
    #cap[Targeted vs matched random edits at strength 1.0.]
  ],
)

= Finding 6 — Restoration: The Honest Causal Test

#grid(
  columns: (1.15fr, 1fr),
  gutter: 0.9em,
  [
    Earlier tests damaged signals and watched for harm. Restoration reverses the logic: take each signal from a donor scenario, transplant it into a base scenario, and measure whether the model shifts toward the donor's choice. *Gold standard for causal claims, and where most appealing stories fall apart.*

    #v(0.2em)

    #table(
      columns: (1.4fr, 0.95fr, 0.7fr, 0.75fr),
      align: (left, center, center, center),
      stroke: 0.4pt + border,
      inset: (x: 5pt, y: 3pt),
      table.header(
        text(size: 7.8pt, weight: "bold")[Signal],
        text(size: 7.8pt, weight: "bold")[Δ choice agreement],
        text(size: 7.8pt, weight: "bold")[Fix rate],
        text(size: 7.8pt, weight: "bold")[Backfire],
      ),
      [*Leader*],     [#text(fill: accent_green, weight: "bold")[+4.2 pp]], [25.0%], [6.3%],
      [*Dispersion*], [#text(fill: accent, weight: "bold")[−2.1 pp]],       [13.6%], [#text(fill: accent, weight: "bold")[15.4%]],
    )

    *Leader* is a real but partial causal handle: modest positive shift, fix rate 4× the backfire rate, spend improvement on 66.7% of cases. *Dispersion* failed: backfire exceeded fix rate. 48 paired scenarios per signal.
  ],
  align(center)[
    #image("assets/public_story/charts/06_restoration.png", width: 100%)
    #v(0.15em)
    #cap[Restoration outcomes for the two signals.]
  ],
)

#pagebreak()

// ── Page 4: Verdict + framing ───────────────────────────────────

= What Held Up

#table(
  columns: (1.7fr, 0.65fr, 2.7fr),
  align: (left, center, left),
  stroke: 0.4pt + border,
  inset: (x: 5pt, y: 4pt),
  table.header(
    text(size: 8pt, weight: "bold")[Claim],
    text(size: 8pt, weight: "bold")[Support],
    text(size: 8pt, weight: "bold")[Why this rating],
  ),
  [The model preserves raw market data internally with high fidelity.],
  text(size: 8pt, weight: "bold", fill: accent_green)[Strong],
  [All 9 tested factors decode at R² > 0.994. Consistent across prompt variants.],

  [The model's market understanding is comparative, not row-by-row.],
  text(size: 8pt, weight: "bold", fill: accent_green)[Strong],
  [Pairwise relationships ~20× more stable than single-asset identity. 384 prompt variations.],

  [Pre-market context shapes how the model reads the market.],
  text(size: 8pt, weight: "bold", fill: accent_green)[Strong],
  [Asymmetry: 0.000 shift when context comes after, 0.061–0.070 when before. Confirmed across full risk ladder.],

  [The two named signals (Leader, Dispersion) reflect genuine market content.],
  text(size: 8pt, weight: "bold", fill: accent_green)[Strong],
  [Targeted edits beat matched random edits in 12/12 comparisons. Both collapse below R² 0.06 on shuffled controls.],

  [The Leader signal partially drives the model's trading choice.],
  text(size: 8pt, weight: "bold", fill: accent_blue)[Partial],
  [Restoration produces +4.2 pp shift toward donor scenario, fix rate 4× the backfire rate. Real but modest.],

  [The Dispersion signal does *not* meaningfully drive the trading choice.],
  text(size: 8pt, weight: "bold", fill: accent)[Negative],
  [Restoration backfire (15.4%) exceeds fix rate (13.6%). The signal is real (Finding 5), but not a causal handle.],
)

#v(0.4em)

#grid(
  columns: (1fr, 1fr),
  gutter: 0.7em,
  [
    #callout(
      [WHAT WE ARE NOT CLAIMING],
      [
        - We did *not* find the single hidden cause of the trading choice.
        - We did *not* show that either named signal alone determines a decision.
        - We did *not* prove every readable internal pattern is also a behavioral lever.

        The restoration test was designed to catch overreach, and it did. The complete decision pathway integrates the model's market read with user-specific factors — portfolio, constraints, strategy framing — that this study did not isolate. That is the next phase.
      ],
      fill: rgb("#f6f1f0"),
      accent_color: accent,
    )
  ],
  [
    #callout(
      [VERDICT],
      [
        #text(size: 10pt, weight: "medium")[
          The model builds a meaningful, structured internal summary of the market, and that summary is influenced by prior context. We can identify recurring internal signals that line up with visible market features. One of those signals (Leader) partially steers the trading choice. The other (Dispersion) does not. Narrower is not weaker — these are the claims the evidence actually supports.
        ]
      ],
      fill: rgb("#f3f0eb"),
      accent_color: accent_green,
    )
  ],
)

#v(0.4em)

#callout(
  [RECOMMENDED PUBLIC FRAMING],
  [
    #text(size: 9.8pt, weight: "medium", style: "italic")[
      "Concordance built a precise, verifiable map of how an autonomous trading agent reads the market. The model preserves raw data faithfully, builds a comparative internal picture, and that picture is shaped by the framing the model has already been given. We identified two specific internal signals — a 'standout asset' detector that has a real but partial influence on the final trading choice, and a 'market unevenness' signal that does not. The work narrows the next research question to a clear target: where the model's market read meets its decision."
    ]
  ],
  fill: rgb("#f4f1ec"),
  accent_color: accent_green,
)

#pagebreak()

// ── Page 5: Appendix ───────────────────────────────────────────

= Appendix

#grid(
  columns: (1fr, 1fr),
  gutter: 1.0em,
  [
    == A.1 · Model and Scale

    #table(
      columns: (1.4fr, 1.4fr),
      align: (left, left),
      stroke: 0.4pt + border,
      inset: (x: 5pt, y: 3pt),
      [*Model*], [Qwen3-30B-A3B (MoE)],
      [*Total parameters*], [~30 B],
      [*Active per token*], [~3 B],
      [*Layers examined*], [48],
      [*Expert routing*], [top-8 of 60 per MoE],
      [*Synthetic scenarios / experiment*], [184–920],
      [*Real inference logs (reference)*], [203,292],
      [*Real activation captures*], [11,579],
      [*Intervention conditions*], [12 (2 × 2 × 3)],
      [*Paired scenarios / condition*], [48],
      [*Capture infra*], [Modal · A100-80GB],
      [*Inference engine*], [vLLM + capture hooks],
    )
  ],
  [
    == A.2 · Methodology

    - *Linear probing.* Ridge regression, L1-regularized logistic regression, and SGD with balanced class weights, on grouped held-out splits (no row leakage across train/test).
    - *Activation capture.* Full-sequence pooled into row, market-mean, market-eos, and downstream-section states. Format-confound residualization removes the slice predictable from sequence length, character count, and asset count.
    - *Decomposition.* PCA on residualized market-section activations; the two dominant directions are regressed against human-readable market features to assign Leader and Dispersion labels.
    - *Intervention.* Source-driven `swap_components` patching at the residual stream — source row's market-span activations are averaged and the selected coefficients are inserted into the base prompt's market span.
    - *Restoration.* Matched-pair denoise design: the lower-valued member of each pair is the base, the higher-valued is the source. The patch tries to move the base toward the source's behavior.
  ],
)

#v(0.2em)

== A.3 · Validation and Controls

- *Format controls.* Layout permutations, ticker swaps, row reordering, and style changes confirm findings reflect market content rather than surface text patterns.
- *Matched random controls.* Every targeted intervention is paired with a random edit of the same magnitude at the same site. The 12/12 selectivity result is defined against this control.
- *Shuffle baselines.* Both named signals collapse below R² 0.06 on shuffled inputs, ruling out spurious statistical relationships.
- *Bootstrap CIs.* All behavioral metrics include bootstrap confidence intervals at 2,000 resamples.
- *Decode-validity check.* For Phase 21 restoration, all 48/48 base and source rows on both axes produced parsed tool calls with `finish_reason=stop` and zero `max_tokens` cap hits — the action surface is observable, not truncated.

== A.4 · Path-Validation Status

We ran a Phase 22 path-validation test on the Leader axis: an early lesion (`project_out` at L4) followed by a downstream rescue (`swap_components` at L35) using source-side coefficients. *The patch path itself worked* — every row produced patch diagnostics, no rows were skipped, both lesion and rescue applied cleanly. *But* the lesioned model did not reliably produce parsed tool calls at scale, so the action-choice readout was too sparse to interpret causally. This is not a refutation of any finding above; it is where the next round of work begins. Two natural follow-ups: (a) a denser behavioral readout regime that survives the lesion, and (b) moving from span-mean coefficient swapping toward exact source-activation transplantation.

#v(0.4em)

#line(length: 100%, stroke: 0.4pt + border)
#v(0.15em)
#text(size: 7.2pt, fill: muted)[
  Concordance · Synthetic Market Research Brief · April 2026 · Self-contained; numerics traceable to `client_summary_draft.md` and `blog_post_draft.md`.
]
