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

The preferred canonical input artifact for this gate is:

- `projects/<BENCHMARK_PROJECT>/phase_02/docs/02-behavioral-smoke-report.md`

If an equivalent artifact is used instead, name it explicitly and explain why.

For response-side work, "behaviorally sane" should be interpreted at the level of proposed labels, not just answer existence.
Ask:

- are the generated responses actually labelable for the planned response-side labels?
- do the responses contain enough substantive content to support annotation and validation?
- are the apparent passes only tripwire-level, or do they survive close inspection?

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

If an experiment depends on fresh generations, the spec should also define:

- generation batch
- annotation / validation step
- label freeze prerequisite
- the point at which probing is allowed to begin

If the target variable is explicitly named in the prompt, the spec should also define:

- the main shortcut risk from explicit naming
- the anti-shortcut control required before a strong success claim is allowed

### 6. Claim shaping

At the end of analysis planning, explicitly state what each planned method would support if it succeeds.

Useful pattern:

- behavior establishes the task is real
- readout establishes representation
- localization establishes where the signal emerges
- intervention establishes causal leverage
- later mechanism follow-up may support stronger computation claims

### 7. Result triage and routing

After experiments run, record what the result means for the project, not just the metric.

For each experiment, explicitly decide whether the outcome means:

- stay in phase 03 and tighten controls
- hand back to phase 02 for augmentation
- hand back to phase 01 for label refinement
- promote to phase 04 causal follow-up
- or stop / demote the label family

Use this routing logic:

1. Did a named cheap baseline in `03-controls-and-splits.md` achieve baseline-dominant performance on the same target?
   - yes -> `AUGMENTATION_NEEDED`
   - no -> continue
   - for imported benchmark labels, treat this as the default posture under the shared principle that benchmarks leak by construction
2. Did the result beat the cheap baselines named in `03-controls-and-splits.md`?
   - no -> `TRIVIAL_OR_NULL`
   - yes -> continue
3. Did the result survive the planned controls?
   - no -> `CONTROL_INSUFFICIENT`
   - yes -> continue
4. Did the result transfer across the planned splits?
   - no, because the label definition is wrong or unstable -> `LABEL_REFINEMENT_NEEDED`
   - no, because the data still entangles the target with a confound -> `AUGMENTATION_NEEDED`
   - yes -> continue
5. Is the result localized to a meaningful span, position, or layer band?
   - no -> `REPRESENTATIONAL_ONLY`
   - yes -> continue
6. Does the localized result satisfy the phase-04 entry criteria?
   - no -> `LOCALIZED_REPRESENTATIONAL`
   - yes -> `PROMOTE_TO_CAUSAL`

`Baseline-dominant performance` means a cheap lexical, semantic, template, or positional baseline matches the main signal closely enough that the experiment remains shortcut-satisfiable.

Default rule:

- if a named cheap baseline matches the main signal closely enough that the result is still plausibly explained by surface recoverability, do not accept a thin representational win by default
- route to `AUGMENTATION_NEEDED` unless the experiment was explicitly designed only to measure prompt-family recoverability
- for active benchmark targets, leakage-dominant wins should default to repair, not quiet demotion
- any override should be written explicitly in the triage log, including:
  - what the neural readout captures that the baseline does not
  - why the baseline match is acceptable for that experiment

Verdicts and routing:

- `TRIVIAL_OR_NULL`
  stop, demote, or park the label family
- `CONTROL_INSUFFICIENT`
  stay in phase 03 and strengthen controls / baselines
- `LABEL_REFINEMENT_NEEDED`
  hand back to phase 01 and re-open the latent label spec and labeling functions
- `AUGMENTATION_NEEDED`
  hand back to phase 02 and add the missing repair to the augmentation plan and gap resolution
  this is the default verdict for shortcut-dominated families whose main signal is matched by a cheap baseline
- `REPRESENTATIONAL_ONLY`
  cap claims at level 2
- `LOCALIZED_REPRESENTATIONAL`
  cap claims at level 3
- `PROMOTE_TO_CAUSAL`
  hand off to mechanistic-interventions

## References

Primary references:

- [mech-interp-principles.md](../../../docs/mech-interp/mech-interp-principles.md)
- [constructing-llm-probes](../constructing-llm-probes/SKILL.md)
- [activation-patching-causal-evals](../activation-patching-causal-evals/SKILL.md)

Use `constructing-llm-probes` for probe implementation details.

## Required artifacts

At minimum, leave behind:

- `projects/<BENCHMARK_PROJECT>/phase_03/docs/03-analysis-plan.md`
  the main human-readable analysis plan, including:
  - methodology choices by label family
  - evidence ladder
  - behavioral-sanity gate status
  - expected artifacts
  - key risks
- `projects/<BENCHMARK_PROJECT>/phase_03/docs/03-feature-hypotheses.md`
  label-family -> candidate representation/control-state hypotheses
- `projects/<BENCHMARK_PROJECT>/phase_03/docs/03-experiment-specs.md`
  2-5 concrete first experiments with method, split, and success criteria
- `projects/<BENCHMARK_PROJECT>/phase_03/docs/03-controls-and-splits.md`
  the control plan carried into actual experiments
- `projects/<BENCHMARK_PROJECT>/phase_03/docs/03-analysis-summary.json`
  compact structured summary of chosen label families, methods, and next steps
- `projects/<BENCHMARK_PROJECT>/phase_03/docs/03-result-triage-log.md`
  one entry per executed experiment, including verdict and routing

Also leave behind:

- `projects/<BENCHMARK_PROJECT>/phase_03/docs/03-execution-targets.md`
  execution-facing model and inference config artifact, including:
  - primary execution model
  - any secondary comparison models if planned
  - which model currently satisfies the behavioral gate
  - exact inference config and system prompt
- `projects/<BENCHMARK_PROJECT>/phase_03/docs/03-phase-04-entry-criteria.md`
  explicit gate for promotion from phase 03 readout/localization work to phase 04 causal follow-up

If any phase-03 experiment depends on response-side labels from fresh generations, also leave behind:

- `projects/<BENCHMARK_PROJECT>/phase_03/docs/03-response-label-pilot.md`
  generation -> annotate -> validate -> freeze plan for the response-side labels

If a label family is blocked on gold-slice validation, leave behind a concrete prerequisite artifact such as:

- `projects/<BENCHMARK_PROJECT>/phase_03/docs/03-<label>-gold-slice-spec.md`
  including sample size, labeler identity or procedure, and the validation threshold required before probing

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
- what each executed experiment implies about the next phase of work

## Phase Done Criteria

This phase is done when:

- the behavioral gate is satisfied and named explicitly
- the required planning artifacts exist
- each executed experiment has a triage-log entry with:
  - baseline comparison
  - control survival
  - split-transfer judgment
  - verdict
  - routing
- phase-04 escalation is not proposed without a `PROMOTE_TO_CAUSAL` triage verdict
- whether the work is only planning-ready or actually execution-ready

The analysis summary should make execution readiness explicit.
Useful statuses include:

- `planning_ready`
- `execution_blocked_on_target_model`
- `execution_blocked_on_response_label_freeze`
- `execution_ready`

Each experiment spec should include, either inline or by direct reference:

- primary shortcut risk
- required baseline or control
- claim ceiling
- label-freeze prerequisite when applicable

## Handoff

If the next step is readout or localization work:

- hand off to [constructing-llm-probes](../constructing-llm-probes/SKILL.md)

If the next step is causal intervention planning:

- hand off to [mechanistic-interventions](../mechanistic-interventions/SKILL.md)

Update the benchmark sidecar with:

- strongest candidate feature hypotheses
- methods that look promising
- methods or comparisons to be careful about
- triage decisions that changed the benchmark plan materially
