---
name: latent-label-data-augmentation
description: Use when a benchmark cannot support the desired latent labels cleanly and needs rewrites, matched pairs, counterbalancing, response generations, or synthetic augmentation. Covers benchmark repair for confounds, split construction, framing variants, and contrast-set design for mechanistic interpretability.
---

# Latent Label Data Augmentation

Use this skill when the user wants to:

- repair a benchmark so refined latent labels become usable
- generate matched pairs or minimal contrasts
- create counterbalanced rewrites
- collect target-model responses for response-side labels
- remove or reduce confounds in a benchmark slice
- expand a benchmark with synthetic or semi-synthetic support data

This skill is the **data augmentation and benchmark repair** phase of the benchmark-flow pipeline.

It is broader than "synthetic data generation."
Often the right move is:

- rewriting
- pairing
- counterbalancing
- reframing

not whole-cloth synthesis.

## Core rule

`Augment to repair the experiment, not to make the dataset bigger.`

The job is to produce data that makes the latent-label question cleaner:

- fewer shortcuts
- clearer contrasts
- better controls
- better slice balance
- cleaner response-side supervision

## Start from the gap list

Do not begin augmentation from vague dissatisfaction.

This skill should usually consume a gap list from:

- [benchmark-to-latent-labels](../benchmark-to-latent-labels/SKILL.md)

Typical gaps:

- no matched pairs for causal questions
- response-side labels need new generations
- theory or framing overlays are unpaired
- target label is too correlated with source or length
- a good label exists only in a tiny subset

## Preferred repair moves

### 1. Rewrite before inventing

Prefer:

- prompt rewrites
- carrier rewrites
- matched framing variants
- length-normalized versions
- source-balanced or domain-balanced subsets

before:

- fully new synthetic dilemmas

Why:

- lower artifact risk
- closer to original benchmark semantics
- easier to justify in writeups

### 2. Build matched pairs when the question is causal

If the downstream question is causal, try to create:

- matched donor-target pairs
- same-label controls
- cross-label contrasts with nuisance dimensions held fixed

### 3. Generate model responses when labels are response-side

If the label lives in:

- deliberative process
- commitment
- hedging
- explicit objective invocation

then benchmark text alone is not enough.
Generate responses under the intended protocol and label those.

### 4. Preserve the real decision bottleneck

Simplify toward the core computation, not toward convenience.

The augmented data should still force the intended reasoning problem.

## Confound-focused design checklist

Always consider:

- lexical leakage
- template aliasing
- source-family confounds
- length imbalance
- role-token leakage
- domain/topic leakage
- annotator-family leakage
- theory/style leakage
- competence vs target-variable confounds

If possible, construct data that lets these be:

- balanced
- stratified
- held out
- explicitly contrasted

## Required outputs

Produce:

- augmented dataset or benchmark slice
- description of what was rewritten, generated, or paired
- rationale for each repair move
- expected confounds reduced
- residual confounds
- mapping back to the original gap list

## Required artifacts

At minimum, leave behind:

- `docs/mech-interp/benchmarks/<benchmark>/02-augmentation-plan.md`
  what is being repaired and why
- `docs/mech-interp/benchmarks/<benchmark>/02-gap-list-resolution.md`
  direct mapping from each gap-list item to the chosen repair move
- `docs/mech-interp/benchmarks/<benchmark>/02-augmented-data-manifest.json`
  where rewritten, paired, generated, or counterbalanced data now lives
- `docs/mech-interp/benchmarks/<benchmark>/02-augmentation-report.md`
  what was changed, what improved, and what residual confounds remain

If new rows or slices were created, also leave behind:

- dataset location(s)
- generation protocol details if responses were generated
- pairing or rewrite rules used

Use simple frontmatter on markdown artifacts:

- `benchmark`
- `phase`
- `version`
- `frozen_date`
- `input_artifacts`

Someone inspecting this phase should be able to answer:

- what defect in the original benchmark was being repaired
- what data was added or changed
- whether the repair actually reduced the intended confound

## References

Primary references:

- [mech-interp-principles.md](../../../docs/mech-interp/mech-interp-principles.md)
- [synthetic-data-generation](../synthetic-data-generation/SKILL.md)
- [benchmark-to-latent-labels](../benchmark-to-latent-labels/SKILL.md)
- [activation-patching-causal-evals](../activation-patching-causal-evals/SKILL.md)

Use the synthetic-data-generation skill as the main methodology library for:

- lexical controls
- benchmark repair
- split schemes
- prompt role placement
- behavior-validating smoke tests

## Handoff

Once the data is repaired enough to support the latent labels:

- hand back to [benchmark-to-latent-labels](../benchmark-to-latent-labels/SKILL.md) if the ontology needs refreezing
- or hand forward to [benchmark-mech-interp-analysis](../benchmark-mech-interp-analysis/SKILL.md) if the label spec is already stable
- update the benchmark sidecar with:
  - what augmentation was performed
  - which confounds were reduced
  - what residual gaps remain
