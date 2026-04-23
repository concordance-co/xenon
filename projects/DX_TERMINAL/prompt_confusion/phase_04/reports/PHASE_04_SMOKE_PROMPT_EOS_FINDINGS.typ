#set page(
  paper: "us-letter",
  margin: (top: 2.4cm, bottom: 2.4cm, left: 2.6cm, right: 2.6cm),
  numbering: "1",
  number-align: right,
)
#set text(font: "Georgia", size: 10.5pt)
#set par(justify: true, leading: 0.7em)
#set heading(numbering: none)

#let ink = rgb("#182028")
#let muted = rgb("#5A6772")
#let accent = rgb("#B33A2A")
#let soft = rgb("#F7F2EF")
#let rule = rgb("#D7DEE3")
#let lift = rgb("#EAF4EE")

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

#let stat(label, value, note) = block(
  fill: white,
  stroke: (paint: rule, thickness: 0.7pt),
  radius: 10pt,
  inset: 12pt,
  width: 100%,
)[
  #text(size: 8pt, fill: muted, weight: "bold")[#label]
  #v(4pt)
  #text(size: 18pt, fill: ink, weight: "bold")[#value]
  #v(4pt)
  #text(size: 8.8pt, fill: muted)[#note]
]

#align(left)[
  #text(size: 9pt, fill: accent, tracking: 0.08em, weight: "medium")[XENON INTERPRETABILITY]
  #v(0.3em)
  #text(size: 22pt, weight: "bold", fill: ink)[Prompt Confusion Phase 04 Smoke Findings]
  #v(0.4em)
  #text(size: 11pt, fill: muted)[
    Prompt-EOS smoke memo for the grouped conflict probe run on April 10, 2026.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.4pt + ink)
]

#v(1em)

#block(
  width: 100%,
  inset: (left: 14pt, top: 12pt, bottom: 12pt, right: 12pt),
  stroke: (left: 3pt + accent, top: none, right: none, bottom: none),
  fill: soft,
)[
  #text(size: 7.5pt, fill: accent, weight: "bold", tracking: 0.08em)[MAIN READ]
  #v(0.3em)
  #text(size: 12.5pt, fill: ink, weight: "medium")[
    The `prompt_eos` smoke path is operational end to end, and the grouped Phase 04 conflict signal is still clearly decodable after fixing the prompt-end pooling target.
  ]
]

= Run Summary

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 8pt,
  stat([Capture run], [`86287b7c78a3`], [48 of 48 smoke examples captured.]),
  stat([Analysis run], [`f844453d2c91`], [Grouped `workflow_label` probe completed.]),
  stat([Evaluation], [24 groups], [`matched_pair_id` grouped 5-fold evaluation.]),
  stat([Pooling], [`prompt_eos`], [Prompt-side EOS token instead of raw last token.]),
)

This smoke run uses the updated Phase 04 workflow snapshot with:

- `capture.pooling = "prompt_eos"`
- `analysis.pooling = "prompt_eos"`
- `analysis.group_column = "matched_pair_id"`
- `capture.add_generation_prompt = false`

= Accuracy Table

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, center, center, center, center),
  inset: 6pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#E6EEF2") } else if calc.odd(y) { rgb("#F9FBFC") } else { white },

  [*Layer*], [*Accuracy Mean*], [*Balanced Accuracy*], [*Std Dev*], [*Selectivity*],

  [`0`], [`0.770`], [`0.770`], [`0.0748`], [`+0.375`],
  [`12`], [`0.685`], [`0.685`], [`0.0943`], [`+0.105`],
  [`24`], [`0.785`], [`0.785`], [`0.0889`], [`+0.265`],
  [`36`], [`0.820`], [`0.820`], [`0.1166`], [`+0.325`],
  [`44`], [`0.725`], [`0.725`], [`0.0922`], [`+0.250`],
)

#v(0.5em)

#block(
  fill: lift,
  stroke: (paint: rule, thickness: 0.6pt),
  radius: 10pt,
  inset: 12pt,
)[
  #text(size: 9pt, fill: muted, weight: "bold")[Smoke interpretation]
  #v(4pt)
  #text(size: 10.2pt, fill: ink)[
    The best grouped readout in this smoke run is layer 36 at `0.820` balanced accuracy. Layer 24 and layer 0 are also strong, while layer 12 weakens materially. On a 48-example smoke slice, this is enough to say the Phase 04 signal survives the prompt-end pooling correction, but not enough to headline a stable geometry claim yet.
  ]
]

= What Changed

Two capture-path fixes matter for this run:

- pooling now anchors on the last prompt-side EOS token instead of the final rendered newline
- vLLM router capture now accumulates logits across prefill chunks rather than overwriting the buffer with the final chunk only

Those changes remove the earlier router indexing failures and make the `prompt_eos` feature meaningfully aligned with the intended prompt boundary.

= Stability Note

The major operational bug from the previous smoke run is fixed: the capture job now completes all 48 rows instead of dying mid-run with router indexing errors.

There is still one stability concern worth treating as active:

- the capture worker has been finishing successfully, but Modal/vLLM still emits `EngineCore_DP0 died unexpectedly` during teardown

The most obvious risk factor in the old configuration was that capture still forced `enable_chunked_prefill = false` even though Qwen3 MoE warns against that mode. That has now been patched so capture workers use chunked prefill by default in the workflow orchestrator and related DX terminal copies. A fresh smoke rerun on capture run `acf299a8c4ae` confirms that the old unsupported-mode warning is gone.

I also added explicit worker teardown that calls `llm_engine.shutdown()` and clears CUDA caches. A second rerun on capture run `c8911ffa6b50` still reports `EngineCore_DP0 died unexpectedly` during shutdown. So the remaining issue is not prompt capture correctness and not the old unsupported prefill mode; it is a post-success worker teardown problem.

= Next Step

The immediate next step is:

1. isolate why vLLM workers still die noisily on shutdown after a successful flush, most likely by reducing worker fan-out, shrinking worker reuse, or changing how Modal containers retire after the local entrypoint completes
2. once teardown is clean enough to trust operationally, run the full 288-row Phase 04 recapture with `prompt_eos`
3. only after the full recapture finishes, read probe quality under grouped evaluation before making any stronger representational claim

The main methodological point is unchanged: this smoke result earns a full rerun, not a mechanism claim.

#v(2em)
#line(length: 100%, stroke: 0.5pt + rgb("#CCC"))
#v(0.3em)
#text(size: 8pt, fill: rgb("#999"))[
  Prompt Confusion Phase 04 Smoke Findings - accuracy summary from capture run `86287b7c78a3` and analysis run `f844453d2c91`, with stability follow-up from capture runs `acf299a8c4ae` and `c8911ffa6b50`.
]
