# Prompt Confusion Pipeline Notes

This note captures the workflow and infra updates surfaced by the Phase 03
mechanistic slice work.

## Desired Operator Flow

The synthetic-data workflow should be simple, inspectable, and fast to iterate:

1. Talk with an agent that can inspect Neon-backed source data directly.
2. Extract the relevant abstractions from real prompts.
3. Produce a human-inspectable Markdown artifact that shows:
   - dataset families
   - row variants
   - lexical or prompt splits
   - expected outputs
4. Review that Markdown artifact before any large build step.
5. Generate a deterministic builder script from the approved artifact.
6. Run the builder and inspect outputs across the important slices.
7. Publish the final dataset to a Neon relation.
8. Run capture on Modal.
9. Run analysis on Modal.
10. Build local reports from analysis outputs.

The important point is that the review artifact comes before the dataset build,
and that iteration on dataset design should happen against small, readable
artifacts rather than large opaque tables.

## Immediate Lessons From Phase 03

### 1. Local-first analysis needs to end

The canonical path is:

`spec -> publication -> capture -> analysis -> report`

For this project:

- dataset publication is Neon-backed
- capture runs on Modal
- analysis runs on Modal
- reports are built locally

Agents still tend to drift into local ad hoc analysis scripts. That should be
treated as a fallback for debugging only, not the default operator path.

### 2. Slice definitions should be first-class artifacts

We often do not want to analyze an entire publication. We want a smaller,
behaviorally sane slice, for example:

- aligned vs strong_conflict
- strong_conflict rows with usable strategy vs setting readout
- one or two promising families only

The clean pattern is:

1. define the slice in Neon
2. give it a stable relation/view name
3. run capture-analysis tooling against that slice

This is better than encoding slice logic implicitly in one-off local scripts.

### 3. Compaction must be slice-aware

Before the recent patch, analysis compaction read the full capture even when the
publication slice was much smaller. That made narrow analyses far too slow and
cancellation-prone.

The better rule is:

- compact only rows present in the current exported label slice
- persist a manifest of which rows were compacted
- only reuse compact files when they actually cover the requested slice

This should be the default behavior for workflow-driven analysis.

### 4. Probe splits must respect dependence structure

Row-level random CV is not enough for these datasets. Related rows share too
much structure.

At minimum:

- keep matched pairs together

Often better:

- leave out a broader family or template group

The general rule is that the split unit should match the real independence unit,
not the row.

### 5. “Grouped” only helps if the grouping key is real

One Phase 03 rerun showed an important failure mode:

- the source probe used `matched_pair_id`
- but the slice had one row per pair
- so grouped CV was not actually stricter

Infra should make this visible instead of silently accepting it. The analysis
surface should report:

- number of rows
- number of groups
- average rows per group

If `rows == groups`, the user should immediately know the grouping key is not
doing meaningful work.

### 6. Reports should explicitly record evaluation conditions

Probe reports need to say:

- publication relation used
- families included
- conflict buckets included
- grouping key used, if any
- number of rows and groups
- whether compaction was full-capture or slice-local

Without that, strong numbers are too easy to misread.

## Recommended Infra Improvements

### A. Synthetic dataset authoring flow

Add a clearer “dataset design” surface that produces:

- a reviewed Markdown design artifact
- a deterministic generator script
- a small preview output

This should feel like a standard phase step, not an improvised process.

### B. Stronger agent onboarding

Agents should be able to read:

- top-level docs
- `.claude` skills
- project README

and immediately infer:

- which phase they are in
- the canonical flow
- that capture and analysis are Modal-first
- where project-local scripts and notes belong

The onboarding should strongly bias toward phase-local workflow execution, not
generic local experimentation.

### C. Better slice tooling

We should have a standard way to create and track analysis slices:

- stable Neon views or relations
- optional slice manifests checked into the phase folder
- explicit labels and grouping keys

This should be easy enough that “make a slice” is a routine step.

### D. Better evaluation controls in the generic analysis runner

The generic runner should support:

- grouped CV
- leave-one-group-out style evaluation
- explicit train/test/val splits when needed
- reporting of split metadata

Grouped CV is the immediate baseline. Harder held-out generalization controls
should be easy to add when the benchmark demands them.

### E. Better run metadata

Workflow runs should preserve enough information to reconstruct exactly what was
evaluated:

- publication relation
- group column
- effective row count
- effective group count
- compact manifest
- output artifact paths

This makes later review and report generation much easier.

## Practical Near-Term Changes

1. Keep Phase 03 slice views as the canonical analysis surface for this work.
2. Prefer grouped or held-out evaluation over random row-level CV.
3. Continue moving project READMEs and notes away from local-first analysis.
4. Treat local scripts as debugging aids unless they are intentionally part of
   the phase workflow.
5. Add stronger reporting around slice definitions and split conditions.
