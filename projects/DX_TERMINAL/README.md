# DX_TERMINAL

DX_TERMINAL is the umbrella project for Terminal Markets interpretability work.

This directory holds:

- `project_spec.json`
  Umbrella project metadata and defaults.
- `phases/`
  Phase-level subprojects such as prompt confusion, counterfactuals, decision
  structure, reruns, and synthetic market studies.

The executable runtime contract still lives in Neon-backed `workflow_specs`,
with checked-in phase-local snapshots under:

```text
projects/DX_TERMINAL/phases/<phase>/specs/workflow.json
```

Use `pipelines.cli` to register and run those workflow specs.
