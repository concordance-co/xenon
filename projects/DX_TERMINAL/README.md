# DX_TERMINAL

DX_TERMINAL is the umbrella project for Terminal Markets interpretability work.

This directory holds:

- `project_spec.json`
  Umbrella project metadata and defaults.
- `phases/`
  Phase-level subprojects such as prompt confusion, counterfactuals, decision
  structure, and reruns.
- `synthetic_market/`
  A synthetic-market subproject with shared code plus phase folders such as
  `dimension_exploration`, `restoration`, and `path_validation`.

The executable runtime contract still lives in Neon-backed `workflow_specs`,
with checked-in phase-local snapshots under:

```text
projects/DX_TERMINAL/<subproject>/<phase>/specs/workflow.json
```

Use `pipelines.cli` to register and run those workflow specs.
