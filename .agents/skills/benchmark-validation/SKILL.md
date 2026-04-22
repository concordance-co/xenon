---
name: benchmark-validation
description: Use when deciding whether a benchmark is worth deeper benchmark-first mechanistic interpretability work. Covers public availability, runnable access, label richness, product relevance, likely mechanistic question richness, scale, and obvious confounds before investing in latent-label work.
---

# Benchmark Validation

Use this skill when the user wants to:

- decide whether a benchmark is worth mech-interp investment
- compare candidate benchmarks before deeper work
- sanity-check whether a benchmark is public, runnable, and label-rich enough
- identify product relevance and likely mechanistic question richness
- reject weak benchmark ideas cheaply before ontology or probe work

This skill is the **front door** of the benchmark-flow pipeline.

It should answer:

- is this benchmark worth deeper work?
- what kinds of mechanistic questions does it likely support?
- what are the most obvious reasons it might fail?

It should **not** yet do full latent-label conversion or experiment design.

## Core rule

`A benchmark being interesting is not the same as a benchmark being ready.`

Benchmark existence is evidence of value:

- someone thought the task was worth building
- the design likely encodes useful domain judgment

But validation still needs to ask:

- is the data accessible?
- is the label structure rich enough?
- is there enough scale?
- are there plausible mechanistic questions here?
- are the confounds so strong that deeper work should wait?

## What to validate

### 1. Access and usability

Check:

- public availability
- license / usage constraints
- dataset or repo access
- runnable harness or legible schema
- prompt and response text availability
- whether activations could plausibly be captured on the real task
- whether the probe-target model is capable enough to produce usable outputs on the benchmark

If the benchmark is not accessible enough to inspect, run, or derive labels from, flag that early.

### 2. Label richness

Look for:

- rubric dimensions
- theory or framework labels
- span or claim labels
- role or source labels
- trajectory or turn-level annotations
- outcome labels that separate multiple failure modes

Prefer benchmarks where labels are rich enough to support decomposition.

Weak starting points:

- leaderboard-only accuracy
- one opaque preference score
- no access to raw prompts or responses

### 3. Product usefulness

Ask:

- does this benchmark resemble a product-relevant task or failure mode?
- would a resulting monitor, probe, or intervention matter?
- is this benchmark a plausible warm-start for later product transfer?

This does not have to be the only criterion, but it should be explicit.

### 4. Likely mechanistic question richness

Ask what internal distinctions the benchmark seems to invite.

Good signs:

- explicit conflict structure
- role or policy variation
- multi-turn or trajectory structure
- contrastive pairs
- theory or framework overlays
- multiple rubric axes that may map to distinct internal objectives

Useful prompt:

- what internal representation, process, or control-state question seems to be hiding inside this benchmark?

### 5. Scale and sliceability

Check:

- total dataset size
- size of key subgroups
- whether the benchmark will collapse into tiny cells after stratification
- whether paired or contrastive structure already exists

Small datasets can still be useful, but note when they are likely seed sets rather than end-state datasets.

### 6. Obvious confound burden

Look early for:

- source-template aliasing
- domain/topic leakage
- length differences
- role-token leakage
- grading labels that are really evaluator instructions
- competence confounds
- theory/style confounds

If the confounds are severe, note whether they look repairable or fundamental.

## Required artifacts

At minimum, leave behind:

- `docs/mech-interp/benchmarks/<benchmark>/00-validation-memo.md`
  the human-readable decision memo with:
  - benchmark identity and links
  - access / usability status
  - label richness summary
  - product relevance summary
  - 2-5 plausible mechanistic questions
  - major confounds and risks
  - recommendation:
    - proceed to `benchmark-to-latent-labels`
    - proceed only with augmentation
    - defer / reject
- `docs/mech-interp/benchmarks/<benchmark>/00-validation-summary.json`
  compact structured summary of the memo
- `docs/mech-interp/benchmarks/<benchmark>/00-validation-notes.md`
  optional working notes, edge cases, and unresolved concerns

Use simple frontmatter on markdown artifacts:

- `benchmark`
- `phase`
- `version`
- `frozen_date`
- `input_artifacts`

The main thing is that someone should be able to inspect:

- what was reviewed
- what was concluded
- why the benchmark was advanced, deferred, or rejected

## References

Read these before making a validation call:

- [mech-interp-principles.md](../../../docs/mech-interp/mech-interp-principles.md)
- [benchmark-to-mech-interp.md](../../../docs/mech-interp/benchmark-to-mech-interp.md)
- [benchmark-first-mech-interp-failure-modes.md](../../../docs/mech-interp/benchmark-first-mech-interp-failure-modes.md)
- [benchmark-context-template.md](../../../docs/mech-interp/benchmark-context-template.md)

## Handoff

If the benchmark passes:

- hand off to [benchmark-to-latent-labels](../benchmark-to-latent-labels/SKILL.md)
- initialize the benchmark sidecar from [benchmark-context-template.md](../../../docs/mech-interp/benchmark-context-template.md)

If the benchmark fails only because of missing controls or structure:

- hand off to [latent-label-data-augmentation](../latent-label-data-augmentation/SKILL.md)

If the benchmark is fundamentally weak for mechanistic work:

- say so directly and stop.
