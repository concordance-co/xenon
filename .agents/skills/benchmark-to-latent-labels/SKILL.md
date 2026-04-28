---
name: benchmark-to-latent-labels
description: Use when converting a benchmark's native labels into a benchmark-specific latent label spec for mechanistic interpretability. Covers prompt-side vs response-side separation, label-type classification, direct vs derived targets, derivability checks, confounds, ontology freeze, and gap-list handoff.
---

# Benchmark To Latent Labels

Use this skill when the user wants to:

- convert benchmark labels into mechanistically useful target labels
- separate prompt-side structure from response-side behavior
- decide which native labels are direct targets, derived targets, validation signals, or nuisance variables
- build a benchmark-specific latent label ontology
- freeze a first-pass label spec before probing

This skill is the **ingestion and labeling** phase of the benchmark-flow pipeline.

It turns:

- benchmark fields
- rubric dimensions
- metadata
- response annotations

into:

- a latent label spec
- labeling functions
- confound plan
- frozen first-pass ontology
- gap list for augmentation

## Core rules

`Grader-designed labels are not automatically latent-designed labels.`

`Benchmark existence is evidence of value, but not proof of mechanistic tractability.`

Start from the benchmark as a crystallized domain artifact.
Then recover the implicit mechanistic questions it seems to ask.
Then audit whether the benchmark's labels really support those questions.

## Required inputs

Before starting, record:

- probe-target model(s)
- generation protocol
- activation capture regime
- research mode:
  correlational readout / causal intervention / both
- sampling parameters and seeds if response-side generations are in scope

These choices affect which labels are even meaningful.

## Workflow

Use this workflow as the canonical benchmark-to-latent-labels process. Older
long-form planning notes have been archived in the knowledge-base repo and
should not be treated as active dependencies.

### 1. Benchmark intake

Record:

- size
- splits and configs
- prompt/response structure
- label surfaces
- harness availability
- whether prompt-only or generation-time work is possible

### 2. Extract the benchmark's implicit mechanistic questions

Do not start from an external wishlist.

Ask:

- what internal distinction does this benchmark seem built to surface?
- what evidence in the benchmark design supports that reading?

### 3. Run a tractability check

Ask:

- is the extracted question actually mechanistic?
- what likely signal location does it imply?
- are there enough examples after likely stratification?
- would matched pairs or augmentation already be required?

### 4. Inventory native labels

For each label, record:

- where it lives
- what it was designed to do
- assignment method
- granularity
- value type

### 5. Classify label types

Useful buckets:

- prompt-side structure
- response-side process
- response-side objective orientation
- outcome
- rubric score
- metadata / nuisance variable

### 6. Rate match to latent usefulness

Use:

- direct
- derived
- validation-only
- nuisance-only
- not useful

### 7. Select a minimal first-pass ontology

Prefer a small first pass:

- 1-2 prompt-side labels
- 1-2 response-side labels
- 1 objective-orientation contrast
- tracked nuisance set

### 8. Specify labeling functions

For each surviving label, define:

- human, LLM, regex, classifier, or other
- version
- storage location
- hash / freeze plan

### 9. Validate derivability

Use:

- gold set checks
- agreement checks
- edge-case review

If a label cannot be assigned consistently, drop or narrow it.

### 10. Audit baselines and confounds

Always check:

- surface-feature baselines
- inter-label correlations
- label-vs-nuisance correlations
- template, source, length, role, domain, annotator-family leakage

### 11. Freeze the ontology

Commit the label spec before activation work.

This is the wall against post-hoc relabeling.

If some labels are still honestly blocked:

- freeze the usable subset anyway
- mark blocked labels explicitly in the frozen label set
- record the blocker and the next phase that must resolve it

Do not silently skip the freeze because one label family still needs augmentation or generations.

### 12. Produce the gap list

Explicitly note what the benchmark still cannot support without augmentation.

## Required artifacts

At minimum, leave behind:

- `projects/<BENCHMARK_PROJECT>/phase_01/docs/01-latent-label-spec.md`
  canonical human-readable spec for the benchmark's refined ontology
- `projects/<BENCHMARK_PROJECT>/phase_01/docs/01-label-inventory.csv` or `.parquet`
  native labels, their types, and latent-usefulness ratings
- `projects/<BENCHMARK_PROJECT>/phase_01/docs/01-labeling-functions.md`
  extracted mechanistic questions, surviving labels, and how each is assigned and validated
- `projects/<BENCHMARK_PROJECT>/phase_01/docs/01-confound-audit.md`
  target-vs-nuisance concerns and planned controls
- `projects/<BENCHMARK_PROJECT>/phase_01/docs/01-frozen-label-set.csv` or `.parquet`
  row-level first-pass labels with version/freeze metadata
  - partial frozen label sets are allowed when some labels are still blocked
  - blocked labels must carry explicit status / blocker metadata, not disappear from the artifact
- `projects/<BENCHMARK_PROJECT>/phase_01/docs/01-gap-list.md`
  what must be repaired or augmented before deeper work
- `projects/<BENCHMARK_PROJECT>/phase_01/docs/01-latent-label-summary.json`
  compact structured summary of the refined ontology, labels, and next action

Optional but useful:

- `projects/<BENCHMARK_PROJECT>/phase_01/docs/01-derivability-report.md`
- `projects/<BENCHMARK_PROJECT>/phase_01/docs/01-baseline-report.md`

Use simple frontmatter on markdown artifacts:

- `benchmark`
- `phase`
- `version`
- `frozen_date`
- `input_artifacts`

Someone reviewing this phase should be able to see:

- what the original labels were
- what the refined labels are
- why they were chosen
- what remains missing

## Phase Done Criteria

This phase is done when:

- required inputs are recorded explicitly
- the canonical label spec, labeling-functions doc, confound audit, gap list, and summary exist
- a real frozen label set exists at the canonical path
- if the frozen label set is partial, each blocked label family has explicit status and blocker metadata
- response-side labels are not frozen as if they were direct prompt-side labels when generations are still missing
- the benchmark sidecar is updated with the refined ontology, key confounds, and the current gap list

## References

Primary references:

- [methodology/PRINCIPLES.md](../../../methodology/PRINCIPLES.md)
- [methodology/CHECKS.md](../../../methodology/CHECKS.md)
- [methodology/FLYWHEEL.md](../../../methodology/FLYWHEEL.md)
- [methodology/templates/REAL_DATA.md](../../../methodology/templates/REAL_DATA.md)

Related skills:

- [benchmark-validation](../benchmark-validation/SKILL.md)
- [latent-label-data-augmentation](../latent-label-data-augmentation/SKILL.md)
- [benchmark-mech-interp-analysis](../benchmark-mech-interp-analysis/SKILL.md)

## Handoff

When the ontology is frozen:

- if the data is sufficient, hand off to [benchmark-mech-interp-analysis](../benchmark-mech-interp-analysis/SKILL.md)
- if matched pairs, rewrites, response generations, or counterbalancing are required first, hand off to [latent-label-data-augmentation](../latent-label-data-augmentation/SKILL.md)
- update the benchmark sidecar with:
  - refined latent labels
  - known confounds
  - current data gap list
