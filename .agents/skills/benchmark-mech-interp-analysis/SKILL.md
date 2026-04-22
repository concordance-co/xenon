---
name: benchmark-mech-interp-analysis
description: Use when a benchmark already has a validated latent label spec and the next job is to plan or review the actual mechanistic analysis program. Covers feature hypotheses, readout and localization strategy, probe choice, control design, evidence ladder, and first experiments.
---

# Benchmark Mech Interp Analysis

Use this skill when the user wants to:

- turn validated latent labels into a concrete mech-interp analysis plan
- choose methods per label family
- plan evidence from readout through localization and causal follow-up
- define first experiments for a benchmark program
- review whether a benchmark analysis plan is scientifically credible

This skill is the **analysis planning** phase of the benchmark-flow pipeline.

It assumes the benchmark has already gone through:

- validation
- latent-label formation
- augmentation if needed

This skill should not be the place where native labels are first interpreted.

## Core rule

`Do not jump from labels to mechanism without an evidence ladder.`

The right progression is usually:

1. confirm behavioral sanity
2. global readout
3. localization
4. causal testing
5. mechanism-focused follow-up
6. claim shaping

## Required gate

Before using this skill, complete the behavioral-sanity check in:

- [mech-interp-principles.md](../../../docs/mech-interp/mech-interp-principles.md)

In particular:

- inspect real examples from each label family
- run the probe-target model on a small slice
- verify output parseability
- inspect failures manually

If the task is not behaviorally sane for the target model, stop and repair it first.

## What this skill owns

### 1. Feature hypotheses from latent labels

For each refined label, ask:

- what internal representation or control state might exist?
- what would a readable version of that signal look like?
- what would a causally useful version look like?

### 2. Method selection per label family

Choose methods per label, not per benchmark.

Common options:

- linear probes
- difference-in-means directions
- PCA / geometry checks
- span-local localization
- transfer tests
- steering or erasure
- patching follow-up

### 3. Evidence ladder planning

Decide what level of claim is realistic:

- indicator
- representational
- localized representational
- causal
- mechanistic

Use the canonical ladder in:

- [mech-interp-principles.md](../../../docs/mech-interp/mech-interp-principles.md)

### 4. Split and control planning

Carry forward the confound plan into:

- split design
- baselines
- matched comparisons
- same-label controls where relevant

### 5. First experiment specs

For the highest-value label families, define:

- target model
- prompt / generation protocol
- capture location
- first-pass readout
- stronger follow-up
- failure criteria

### 6. Claim shaping

At the end of analysis planning, explicitly state what each planned method would support if it succeeds.

Useful pattern:

- behavior establishes the task is real
- readout establishes representation
- localization establishes where the signal emerges
- intervention establishes causal leverage
- later mechanism follow-up may support stronger computation claims

## References

Primary references:

- [mech-interp-principles.md](../../../docs/mech-interp/mech-interp-principles.md)
- [constructing-llm-probes](../constructing-llm-probes/SKILL.md)
- [activation-patching-causal-evals](../activation-patching-causal-evals/SKILL.md)

Use `constructing-llm-probes` for probe implementation details.

## Required artifacts

At minimum, leave behind:

- `docs/mech-interp/benchmarks/<benchmark>/03-analysis-plan.md`
  the main human-readable analysis plan, including:
  - methodology choices by label family
  - evidence ladder
  - behavioral-sanity gate status
  - expected artifacts
  - key risks
- `docs/mech-interp/benchmarks/<benchmark>/03-feature-hypotheses.md`
  label-family -> candidate representation/control-state hypotheses
- `docs/mech-interp/benchmarks/<benchmark>/03-experiment-specs.md`
  2-5 concrete first experiments with method, split, and success criteria
- `docs/mech-interp/benchmarks/<benchmark>/03-controls-and-splits.md`
  the control plan carried into actual experiments
- `docs/mech-interp/benchmarks/<benchmark>/03-analysis-summary.json`
  compact structured summary of chosen label families, methods, and next steps

Use simple frontmatter on markdown artifacts:

- `benchmark`
- `phase`
- `version`
- `frozen_date`
- `input_artifacts`

Someone inspecting this phase should be able to answer:

- what internal features we think might exist
- which methods we plan to use first
- what level of claim we think is realistic
- what the first concrete experiments are

## Handoff

If the next step is readout or localization work:

- hand off to [constructing-llm-probes](../constructing-llm-probes/SKILL.md)

If the next step is causal intervention planning:

- hand off to [mechanistic-interventions](../mechanistic-interventions/SKILL.md)

Update the benchmark sidecar with:

- strongest candidate feature hypotheses
- methods that look promising
- methods or comparisons to be careful about
