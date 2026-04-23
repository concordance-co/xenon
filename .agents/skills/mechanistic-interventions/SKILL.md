---
name: mechanistic-interventions
description: Use when moving from benchmark-driven analysis into causal or mechanism-oriented interventions. Covers activation patching, interchange, control design, read-vs-write distinctions, and intervention-specific success criteria, while pointing to future attention and routing follow-up work.
---

# Mechanistic Interventions

Use this skill when the user wants to:

- design causal intervention experiments after readout/localization work
- plan activation patching or interchange tests
- decide whether a readable signal is also writable
- structure intervention controls and success criteria
- review whether an intervention claim is too strong for the evidence

This skill is the **interventions** phase of the benchmark-flow pipeline.

It begins after:

- latent labels are validated
- the benchmark analysis has a site hypothesis
- readout/localization evidence exists

This skill owns:

- intervention framing
- site-choice hypothesis
- donor-target pairing and control design
- success and failure criteria
- planning-time interpretation heuristics

It does not own:

- patch operator choice
- layer sweep execution
- flip-rate computation
- execution-time interpretation of specific patching results

Those belong to:

- [activation-patching-causal-evals](../activation-patching-causal-evals/SKILL.md)

## Core rule

`A readable signal is not automatically a writable signal.`

Do not treat a high-AUROC probe or clean direction as sufficient evidence for causal importance.

## What this skill owns

### 1. Intervention framing

Define:

- exact behavior to change
- success criteria
- malformed-output criteria
- intended direction of change

### 2. Site choice from the computation story

Choose intervention sites because:

- localization suggests them
- timing suggests them
- the computation is still live there

Not because:

- they are easy to patch
- they are the best probe layer

### 3. Paired design and controls

Prefer:

- matched donor-target pairs
- single-layer first tests
- same-label controls
- random or unmatched controls when useful

### 4. Interpretation discipline

Track:

- intended-direction flips
- reverse-direction flips
- malformed outputs
- same-label instability

Separate:

- causal evidence
- broad destabilization
- late summary-state corruption

Planning-time heuristics to keep in mind:

- if the same tiny set of rows always flips, suspect borderline sensitivity rather than broad causal control
- if late-layer patching produces garbage, suspect compressed summary-state corruption
- if a mean-swap patch does almost nothing, suspect a lossy operator before abandoning the hypothesis

### 5. Follow-on mechanism work

This skill should also point to likely next steps:

- attention follow-up
- routing / MoE follow-up
- narrower span decomposition
- read-vs-write comparisons

Those may later justify dedicated skills, but this phase should at least surface them.

## References

Primary reference:

- [mech-interp-principles.md](../../../docs/mech-interp/mech-interp-principles.md)
- [activation-patching-causal-evals](../activation-patching-causal-evals/SKILL.md)

Related references:

- [benchmark-mech-interp-analysis](../benchmark-mech-interp-analysis/SKILL.md)

Use `activation-patching-causal-evals` as the main methodology library here.

Use the shared principles doc for:

- read layer vs write layer
- evidence ladder
- claim shaping discipline

## Required artifacts

At minimum, leave behind:

- `projects/<BENCHMARK_PROJECT>/phase_04/docs/04-intervention-plan.md`
  the main human-readable intervention plan, including:
  - intervention target behavior
  - success metrics
  - interpretation caveats
  - likely next mechanism-focused analyses
- `projects/<BENCHMARK_PROJECT>/phase_04/docs/04-site-hypothesis.md`
  why these intervention sites are plausible
- `projects/<BENCHMARK_PROJECT>/phase_04/docs/04-pairing-and-controls.md`
  donor-target design, same-label controls, and failure criteria
- `projects/<BENCHMARK_PROJECT>/phase_04/docs/04-intervention-summary.json`
  compact structured summary of intervention target, sites, controls, and next actions

Use simple frontmatter on markdown artifacts:

- `benchmark`
- `phase`
- `version`
- `frozen_date`
- `input_artifacts`

Someone inspecting this phase should be able to answer:

- what behavior the intervention is supposed to change
- why the chosen site is plausible
- what counts as success vs destabilization
- what technique skill should be invoked next

## Handoff

If the intervention design is weak because the benchmark lacks proper contrasts:

- hand back to [latent-label-data-augmentation](../latent-label-data-augmentation/SKILL.md)

If the intervention plan is sound:

- proceed with [activation-patching-causal-evals](../activation-patching-causal-evals/SKILL.md) for detailed causal-eval execution

Update the benchmark sidecar with:

- intervention candidates
- planning-time failure-mode cautions
- what mechanism follow-up looks most justified next
