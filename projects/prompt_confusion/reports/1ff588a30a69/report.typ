#set page(
  paper: "us-letter",
  margin: (x: 0.72in, y: 0.78in),
)

#set par(justify: false, leading: 0.62em)
#set text(font: "Libertinus Serif", size: 10pt, fill: rgb("#16202A"))

#let navy = rgb("#16324F")
#let teal = rgb("#2E6A69")
#let muted = rgb("#5E6F82")
#let line = rgb("#D6DEE3")

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
  stroke: (paint: line, thickness: 0.7pt),
  radius: 10pt,
  inset: 12pt,
  width: 100%,
)[
  #text(size: 8pt, fill: muted, weight: "bold")[#label]
  #v(4pt)
  #text(size: 17pt, fill: navy, weight: "bold")[#value]
]

#heading(level: 1)[Workflow Report]

#text(size: 11pt, fill: muted)[
  Generated 2026-04-02T19:34:44.374893+00:00
]

#v(10pt)

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 10pt,
  stat("Spec", "Conflict Probe v0"),
  stat("Publication", "workflow_dataset_conflict_probe_v0_v1"),
  stat("Rows", "375"),
)

#heading(level: 2)[Workflow]

- Spec id: `conflict_probe_v0`
- Version: `1`
- Dataset run: `1bdf3bed5b62`
- Capture run: `9c0467b2ea48`
- Analysis run: `1ff588a30a69`

#heading(level: 2)[Analysis]

- Mode: `probe`
- Target: `workflow_label`
- Data source: `residual`
- Pooling: `last_token`
- Labels parquet: `data/workflows/conflict_probe_v0/labels/workflow_dataset_conflict_probe_v0_v1.parquet`
- Output dir: `projects/prompt_confusion/outputs/conflict_probe_v0_analysis`

#heading(level: 2)[Result JSON]

```json
{}
```
