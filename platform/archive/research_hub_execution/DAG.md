# Complete task dependency graph

Solid edges are start prerequisites; dashed edges are integration prerequisites. Both must be satisfied for acceptance. Shared-file locks are listed in graph.json and EXECUTOR_GUIDE.md.

```mermaid
flowchart TD
  T00["T00 Capture the baseline and freeze integration contracts"]
  T24["T24 Stabilize extension registration and legacy serialization"]
  T01["T01 Version provenance and scientific reuse"]
  T02["T02 Lock runtimes and implement staged admission"]
  T05["T05 Make completion atomic and long operations resumable"]
  T07["T07 Create reusable experiment packs and evidence records"]
  T25["T25 Define image, latent and spatial artifact contracts"]
  T03["T03 Enforce bounded offline local scratch"]
  T04["T04 Implement a narrow differentiable PyTorch engine"]
  T09["T09 Publish eligible local artifacts with lineage"]
  T10["T10 Integrate the official MAttr harness in isolation"]
  T06["T06 Gate fanout with canaries and reuse qualified sessions"]
  T08["T08 Deliver the first real offline experiment"]
  T12["T12 Add verified native neuron, head and expert coordinates"]
  T11["T11 Run MAttr reference and cross-model comparisons"]
  T13["T13 Build the neuron microscope and Goodfire test pack"]
  T15["T15 Add the recognition-model microscope"]
  T16["T16 Qualify SDXL generation and internal steering hooks"]
  T18["T18 Import pretrained SAEs and measure intervention fidelity"]
  T14["T14 Implement typed graphs and causal route validation"]
  T17["T17 Deliver a controlled image-steering experiment"]
  T19["T19 Fit and test one aligned crosscoder"]
  T20["T20 Integrate pretrained transcoder circuit tracing"]
  T23["T23 Attribute a benign compatible checkpoint weight delta"]
  T21["T21 Build evidence retrieval and the compact workbench"]
  T22["T22 Qualify the integrated release and write operator recipes"]
  T00 --> T24
  T24 --> T01
  T01 --> T02
  T01 --> T05
  T01 --> T07
  T01 --> T25
  T24 --> T25
  T01 --> T03
  T02 -.-> T03
  T01 --> T04
  T02 --> T04
  T03 -.-> T04
  T03 --> T09
  T05 --> T09
  T07 --> T09
  T02 --> T10
  T03 --> T10
  T05 --> T10
  T07 --> T10
  T02 --> T06
  T05 --> T06
  T04 -.-> T06
  T03 --> T08
  T04 --> T08
  T07 --> T08
  T05 -.-> T08
  T04 --> T12
  T05 -.-> T12
  T10 --> T11
  T06 --> T11
  T07 --> T11
  T12 --> T13
  T07 --> T13
  T06 -.-> T13
  T12 --> T15
  T07 --> T15
  T25 --> T15
  T06 -.-> T15
  T02 --> T16
  T05 --> T16
  T25 --> T16
  T06 -.-> T16
  T04 --> T18
  T07 --> T18
  T12 -.-> T18
  T06 -.-> T18
  T07 --> T14
  T12 --> T14
  T10 -.-> T14
  T11 -.-> T14
  T16 --> T17
  T07 --> T17
  T18 --> T19
  T05 --> T19
  T06 -.-> T19
  T10 --> T20
  T14 --> T20
  T06 -.-> T20
  T18 -.-> T20
  T11 --> T23
  T14 --> T23
  T05 --> T23
  T06 -.-> T23
  T07 --> T21
  T13 -.-> T21
  T14 -.-> T21
  T17 -.-> T21
  T18 -.-> T21
  T20 -.-> T21
  T08 --> T22
  T09 --> T22
  T11 --> T22
  T13 --> T22
  T14 --> T22
  T15 --> T22
  T17 --> T22
  T18 --> T22
  T19 --> T22
  T20 --> T22
  T21 --> T22
```

The adjacency lists in [graph.json](graph.json) are authoritative. T23 is conditional and outside the release gate.
