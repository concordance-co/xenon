---
benchmark: counselbench
phase: 03
version: v1
frozen_date: 2026-04-23
input_artifacts:
  - projects/COUNSELBENCH/phase_01/docs/01-latent-label-spec.md
  - projects/COUNSELBENCH/phase_02/docs/02-behavioral-smoke-report.md
  - projects/COUNSELBENCH/advice_safety/phase_03/specs/workflow.py
---

# CounselBench Phase 03 Analysis Plan

## Behavioral Gate

The first Adv generation smoke ran on 2026-04-23 as `wr_299cf3c365e4_a3d3ea13`. It is superseded by the current full-Adv phase-03 workflow, which packages samples, tripwire results, label-support counts, prompt readouts, and PCA geometry over all 120 CounselBench-Adv prompts.

Resolved before the full phase-03 run:

- The generation cap is now 15000 tokens with a 30000-token model window, replacing the old 800-token smoke cap and the first full-run 5000-token cap.
- The workflow now uses the full Adv table rather than `limit_per_mode=4`.
- The executable step is named `evaluate_generation_quality_gate`; response-side classifier readouts remain gated by generated-label support.
- Model-bound generation and residual capture now use four H200 Modal execution shards, with generation batch size 16.

Prior-smoke caveats:

- Manual or agent review of generated samples is still required.
- Only 7 of 24 generations were stop-finished and replayable; 17 hit the generation length cap.
- Generated-boundary classifier readouts are gated because the replayable slice is one-class for provisional `medical_boundary_violation`.

## First Experiments

- `E1`: prompt-end readout of `adv_failure_mode`, interpreted only against the prompt text baseline.
- `E2`: generation-end readout of provisional `medical_boundary_violation`, gated until generated labels have both classes in grouped train/test splits and blocked from strong claims until generated-response labels are validated.
- `E2.5`: PCA geometry over prompt-end and generation-end residual states to test whether Adv family, provisional medical-boundary status, and topic organize into a low-dimensional response-posture space.
- `E3`: response-context readout over aggregated Eval rows for `empathy_high` and `specificity_high`, grouped by `questionID`.
- `E4`: supportive-but-unsafe vs safe-but-cold hard-negative contrast after response labels are frozen.

## Claim Strength

Successful phase-03 work can support representational or localized-representational claims only. Causal or mechanistic claims require phase-04 entry criteria.

Geometry interpretation rule:

- PCA separation supports only a structured-representation hypothesis.
- If PCA clusters are also explained by text baselines, topic, or length buckets, route the label family to augmentation rather than promoting it.
- If provisional generated-response labels are one-class, skip classifier probes and keep only label-support summaries plus geometry diagnostics.
