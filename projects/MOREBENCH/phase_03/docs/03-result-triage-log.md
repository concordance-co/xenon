---
benchmark: morebench
phase: 03
version: v1
frozen_date: 2026-04-23
input_artifacts:
  - projects/MOREBENCH/phase_03/docs/03-analysis-plan.md
  - projects/MOREBENCH/phase_03/docs/03-experiment-specs.md
  - projects/MOREBENCH/phase_03/docs/03-controls-and-splits.md
  - projects/MOREBENCH/phase_03/docs/03-execution-targets.md
---

# MoReBench 03 Result Triage Log

## Experiment 1: Theory-Identity Prompt Readout

- execution date:
  `2026-04-23`
- workflow:
  `morebench_phase03_experiment01_theory_identity`
- execution note:
  the auxiliary `anchor_clause` span-capture branch stalled operationally during a reuse run, but the core readouts needed for triage completed and are sufficient for the verdict below
- completed evidence used for triage:
  - `text_baseline_1_f2ae3c29`
  - `probe_1_7282017a`
  - `transfer_probe_1_213eabe1`
  - `probe_1_e4e36c77`

### Metrics

- prompt-EOS theory-vs-control readout:
  best balanced accuracy `1.0` at layer `20`
- direct / wording-variant / anchor-only transfer:
  best balanced accuracy `1.0` across all three families at layers `20` and `44`
- named-theory clause localization:
  balanced accuracy `1.0` at every captured layer
- cheap semantic baseline:
  `anchor_text` bag-of-words logistic baseline balanced accuracy `1.0`

### Interpretation

The result is strong in a narrow readout sense and weak in a mechanistic-discovery sense.

What it establishes:

- the current phase-02 theory prompt family very strongly encodes `theory_identity`
- that encoding survives direct-vs-wording transfer
- the supposed `anchor_only` control was not a credible anti-shortcut family because the fixed per-theory anchor sentence remained
- the signal is localizable to the named theory clause on named rows

What it does **not** establish:

- a nontrivial framework-conditioned prompt state beyond explicit prompt semantics
- a compelling target for deeper mechanistic follow-up in the current prompt design

The decisive issue is the cheap baseline.
The `anchor_text` baseline alone classifies theory identity perfectly, so the current result is fully explainable by explicit semantic content in the prompt family.

### Verdict

- verdict:
  `AUGMENTATION_NEEDED`
- routing:
  hand the current `theory_identity` prompt family back to phase 02 for anti-shortcut repair before any prompt-side retry
- why this verdict rather than promotion:
  the result does **not** beat the cheap surface-semantic baseline named in the control philosophy, so this is a repair signal about the dataset family rather than a usable phase-03 finding

### Follow-on Action

Because `theory_identity` remains strategically important, it should now stay in the phase-02 repair loop:

- treat the legacy explicit-theory family as known-broken
- materialize harder factorial, alias-based, and description-based theory families
- run stronger prompt-side baseline preflight before any phase-03 retry
- treat `alias_only` as the best current prompt-side diagnostic family, but keep the retry gate closed until its held-out text baselines fall further
- treat `description_only` as a theory-priming family for generation-time persistence work rather than as a clean prompt-side retry family

If the goal is the strongest next phase-03 execution target, move to the response-side pilot and freeze path for:

- `theory_conditioned_generation_persistence`
- `tradeoff_engagement`
- `commitment_style`
- `helpfulness_invoked`
- `harm_avoidance_invoked`
