# MOREBENCH

This project is the canonical home for all MoReBench-specific work.

## Layout

- `phase_00` through `phase_03`
  Benchmark-first MoReBench work:
  validation, latent labels, augmentation, and analysis planning/execution.
  Each phase may contain:
  - `docs/` for canonical skill artifacts
  - `outputs/` for project-local materialized data
  - `reports/` for local summaries
  - `specs/` and `scripts/` when the phase has executable workflow or build logic
- `benchmark_context.md`
  Frozen MoReBench sidecar context shared across the benchmark-first phases
- `procedural_probe/`
  Separate executable subproject for rubric-profile and prompt-vs-generation probing work
- `shared/`
  Reusable project-local helpers for the procedural-probe workflows
- `scripts/`
  Project-level builders used to materialize benchmark-first canonical artifacts

## Conventions

- MoReBench-specific code and artifacts should live under `projects/MOREBENCH`
- Benchmark-first artifacts are not a separate subproject; they live directly in the root phase folders
- `procedural_probe` is a real subproject with its own `phase_spec.json` and checked-in workflows
- When a benchmark-first phase has executable code, prefer:
  - `projects/MOREBENCH/phase_<nn>/specs/...`
  - `projects/MOREBENCH/phase_<nn>/scripts/...`

## Current Split

- Use root `phase_00` to `phase_03` for the benchmark-first skill flow
- Use `procedural_probe` for the rubric-oriented pipelines_v2 experiments
