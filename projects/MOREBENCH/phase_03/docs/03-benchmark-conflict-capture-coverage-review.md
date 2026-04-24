# Benchmark Conflict Capture Coverage Review

This note explains why the old benchmark capture is not a full `30 x 6 = 180` benchmark-conflict activation set, even though the underlying benchmark generations existed.

## Headline

The old benchmark activation artifact is the **filtered main Experiment 2 capture**, not a replay capture over all benchmark rows.

That means:

- source generation existed for all `180` benchmark prompt-condition rows
- but the main capture dataset removed rows flagged as direct theory/cue copy before capture
- so the capture artifact is incomplete for several of the benchmark conflict groups

## Core files to review

1. Filter logic:
- [experiment_02_workflow.py](/Users/trentelmore/Projects/concordance/xenon/projects/MOREBENCH/phase_03/specs/experiment_02_workflow.py)

Relevant places:
- copy metrics and thresholds around `cue_overlap_fraction`, `cue_longest_run`, and repeated theory-name checks
- the main dataset builder where `direct_theory_copy_flag` rows are excluded from capture

2. Old benchmark main-capture summary:
- [summary.json](/Users/trentelmore/Projects/concordance/xenon/projects/MOREBENCH/phase_03/reports/experiment_02_manual_analysis/summary.json)

Relevant numbers:
- `source_row_count = 180`
- `flagged_direct_copy_count = 46`
- `kept_capture_example_count = 134`

So the old benchmark activation artifact was never a full `180`-row capture.

3. Manual judged benchmark split groups:
- [report.md](/Users/trentelmore/Projects/concordance/xenon/projects/MOREBENCH/phase_03/reports/experiment_02_behavior_broad_llm_judged/report.md)

This identifies the benchmark groups that are behaviorally contested and therefore most relevant for the new target:
- `theory_group_005`
- `theory_group_007`
- `theory_group_009`
- `theory_group_011`
- `theory_group_013`
- `theory_group_015`
- `theory_group_022`

## Coverage of the benchmark split groups inside the old capture dataset

This is the actual prime coverage for those `7` benchmark split groups inside the old capture dataset artifact `transform_1_4a60e2ca` / `capture_1_34cdfd7923d9`.

### `theory_group_005`
- present: `utilitarian`, `virtue_ethics`, `contractarianism`, `deontology`, `generic_ethics_control`
- missing: `contractualism`

### `theory_group_007`
- present: `utilitarian`, `virtue_ethics`, `contractarianism`, `deontology`, `contractualism`, `generic_ethics_control`
- missing: none

### `theory_group_009`
- present: `utilitarian`, `virtue_ethics`, `contractarianism`, `deontology`, `generic_ethics_control`
- missing: `contractualism`

### `theory_group_011`
- present: `virtue_ethics`, `contractarianism`, `deontology`, `generic_ethics_control`
- missing: `utilitarian`, `contractualism`

### `theory_group_013`
- present: `utilitarian`, `virtue_ethics`, `contractarianism`, `deontology`, `generic_ethics_control`
- missing: `contractualism`

### `theory_group_015`
- present: `utilitarian`, `contractarianism`, `deontology`, `generic_ethics_control`
- missing: `virtue_ethics`, `contractualism`

### `theory_group_022`
- present: `contractarianism`, `deontology`, `generic_ethics_control`
- missing: `utilitarian`, `virtue_ethics`, `contractualism`

## Why this happened

The old benchmark capture came from the main Experiment 2 path, which intentionally filtered out rows that looked like direct theory/cue copying before capture.

The relevant logic is in [experiment_02_workflow.py](/Users/trentelmore/Projects/concordance/xenon/projects/MOREBENCH/phase_03/specs/experiment_02_workflow.py):

- repeated theory-name mention can trigger `theory_name_copy_flag`
- near-verbatim cue overlap can trigger `cue_overlap_copy_flag`
- either can set `direct_theory_copy_flag`
- rows with direct-copy flags are removed from the main capture dataset

So the benchmark generation existed, but the activation artifact was not designed as a full conflict replay capture.

## Practical implication

For contested-case activation analysis:

- the new public conflict capture is clean and complete for its targeted groups
- the old benchmark capture is only partially usable for the benchmark split groups

So if we want a true full benchmark contested-case activation set, we would need a benchmark replay capture over the judged benchmark conflict groups rather than relying on the filtered main Experiment 2 capture.
