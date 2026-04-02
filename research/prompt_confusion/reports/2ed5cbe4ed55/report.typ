#set page(
  paper: "us-letter",
  margin: (x: 0.72in, y: 0.78in),
)

#set par(justify: false, leading: 0.62em)
#set text(font: "Libertinus Serif", size: 10pt, fill: rgb("#16202A"))

#let navy = rgb("#16324F")
#let teal = rgb("#2E6A69")
#let muted = rgb("#5E6F82")
#let divider = rgb("#D6DEE3")

#show heading.where(level: 1): it => block(
  above: 1.1em,
  below: 0.35em,
  text(17pt, weight: "bold", fill: navy)[#it.body],
)

#show heading.where(level: 2): it => block(
  above: 0.75em,
  below: 0.25em,
  text(12pt, weight: "bold", fill: teal)[#it.body],
)

#let stat(label, value) = block(
  stroke: (paint: divider, thickness: 0.7pt),
  radius: 10pt,
  inset: 12pt,
  width: 100%,
)[
  #text(size: 8pt, fill: muted, weight: "bold")[#label]
  #v(4pt)
  #text(size: 17pt, fill: navy, weight: "bold")[#value]
]

#align(left)[
  #text(size: 9pt, fill: rgb("#B33A2A"), tracking: 0.08em, weight: "medium")[XENON INTERPRETABILITY]
  #v(0.3em)
  #text(size: 22pt, weight: "bold")[Conflict Probe v0]
  #v(0.4em)
  #text(size: 11pt, fill: muted)[
    Workflow report generated 2026-04-02T19:46:45.991026+00:00. This report summarizes the latest successful dataset, capture, and analysis chain for the selected workflow spec.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 8pt,
    [#text(size: 7.5pt, fill: muted, weight: "bold")[SPEC]\ #text(size: 9pt)[conflict_probe_v0]],
    [#text(size: 7.5pt, fill: muted, weight: "bold")[PUBLICATION]\ #text(size: 9pt)[workflow_dataset_conflict_probe_v0_v1]],
    [#text(size: 7.5pt, fill: muted, weight: "bold")[ROWS]\ #text(size: 9pt)[375]],
    [#text(size: 7.5pt, fill: muted, weight: "bold")[ANALYSIS RUN]\ #text(size: 9pt)[2ed5cbe4ed55]],
  )
]

#v(12pt)

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 10pt,
  stat("Spec", "Conflict Probe v0"),
  stat("Publication", "workflow_dataset_conflict_probe_v0_v1"),
  stat("Rows", "375"),
)

= Workflow

- Spec id: `conflict_probe_v0`
- Version: `1`
- Dataset run: `1bdf3bed5b62`
- Capture run: `9c0467b2ea48`
- Analysis run: `2ed5cbe4ed55`

= Analysis Setup

- Mode: `probe`
- Target: `workflow_label`
- Data source: `residual`
- Pooling: `last_token`
- Labels parquet: `data/workflows/conflict_probe_v0/labels/workflow_dataset_conflict_probe_v0_v1.parquet`
- Output dir: `research/prompt_confusion/outputs/conflict_probe_v0_analysis/2ed5cbe4ed55`

= Executive Read

#block(
  width: 100%,
  inset: (left: 14pt, top: 12pt, bottom: 12pt, right: 12pt),
  stroke: (left: 3pt + rgb("#B56662"), top: none, right: none, bottom: none),
  fill: rgb("#FAF5F3"),
)[
  #text(size: 7.5pt, fill: rgb("#B56662"), weight: "bold", tracking: 0.08em)[MAIN READ]
  #v(0.3em)
  #text(size: 12.5pt, weight: "medium")[The best probe layer does not beat the majority baseline, so this run does not support a useful linear readout for the workflow label.]
]

= Main Quantitative Findings

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 8pt,
  stat("Best Layer", "34"),
  stat("Best Accuracy", "0.576"),
  stat("Majority Baseline", "0.627"),
  stat("Lift vs Baseline", "-0.051"),
)

#v(8pt)

Why this matters:

- best-layer balanced accuracy is `0.524`
- run covers `48` probed layers
- best-layer fold std is `0.051`

= Top Layers

#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr),
  align: (left, right, right, right, right),
  inset: 6pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 { rgb("#DDEBF0") } else if calc.odd(y) { rgb("#F8FBFC") } else { white },

  [*Layer*], [*Accuracy*], [*Std*], [*Balanced*], [*Selectivity*],
  [`34`],
 [`0.576`],
 [`0.051`],
 [`0.524`],
 [`0.035`],
  [`6`],
 [`0.525`],
 [`0.073`],
 [`0.500`],
 [`0.037`],
  [`31`],
 [`0.523`],
 [`0.040`],
 [`0.472`],
 [`0.016`],
  [`39`],
 [`0.517`],
 [`0.046`],
 [`0.486`],
 [`-0.048`],
  [`41`],
 [`0.515`],
 [`0.035`],
 [`0.471`],
 [`-0.016`],
  [`5`],
 [`0.509`],
 [`0.059`],
 [`0.507`],
 [`0.024`],
  [`4`],
 [`0.504`],
 [`0.039`],
 [`0.493`],
 [`0.005`],
  [`38`],
 [`0.504`],
 [`0.058`],
 [`0.470`],
 [`0.032`],
)

= Artifacts

- Results JSON: `research/prompt_confusion/outputs/conflict_probe_v0_analysis/2ed5cbe4ed55/results.json`
- Primary parquet: `research/prompt_confusion/outputs/conflict_probe_v0_analysis/2ed5cbe4ed55/probe_workflow_label_residual.parquet`
- Remote output: `/data/analysis_results/workflows/conflict_probe_v0/2ed5cbe4ed55`
