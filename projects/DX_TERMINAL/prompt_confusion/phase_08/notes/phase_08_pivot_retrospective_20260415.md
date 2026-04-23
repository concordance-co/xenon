# Phase 08 Pivot Retrospective

Date: 2026-04-15
Status: archived design pivot

## Why Phase 08 Matters

Phase 08 is worth preserving because it is the point where the
`prompt_confusion` project changed its core question.

Before Phase 08, a large part of the effort was trying to rescue
`family` as an interpretable benchmark variable.

The lesson from Phase 05 and Phase 07 was:

- `family` was not failing because the shell wording was sloppy
- `family` was failing because it was structurally tied to semantic
  polarity on the target dimension
- a prompt that faithfully stated the strategy and the active setting
  would always tend to make `family` decodable from raw text

Phase 08 is the point where we stopped asking:

- can the model represent `family`?

and started asking:

- can the model represent whether `STRATEGY` and `ACTIVE SETTINGS` agree
  or disagree?

That shift is the foundation for everything Phase 09 later did well.

## What Changed In The Dataset Design

Phase 08 introduced several important design changes relative to the old
family-based benchmark.

### 1. Family was removed as the main organizing variable

Old design:
- rows belonged to fixed strategy families such as
  `trade_size_force_large`, `trade_size_force_small`,
  `activity_force_trade`, `activity_force_observe`
- strategy polarity was locked inside those families

Phase 08 design:
- `strategy_direction` varies within the dataset
- the same direction words appear across both aligned and conflict rows

Motivation:
- if strategy polarity is locked to a family label, the semantic content
  of the prompt and the family label become almost the same variable
- once that happens, lexical control can only help at the shell level,
  not at the content level

### 2. Conflict became relational by construction

Old design:
- labels often implicitly tracked one side's semantic identity

Phase 08 design:
- `conflict_present` is defined from the relation between
  `strategy_direction` and `setting_implied_direction`
- the same tokens should appear in both classes

Example:
- size strategy says `large`
- settings imply `large`
  - aligned
- size strategy says `large`
- settings imply `small`
  - conflict

Motivation:
- make the benchmark target a relation between prompt spans, not a
  unigram-level property of one span

### 3. Two policy dimensions were kept, but under one crossed design

Phase 08 kept:
- `trade_size`
- `trading_activity`

Motivation:
- preserve replication across policy dimensions
- avoid treating dimension as "family"
- make it possible to compare a simpler conflict type (`trade_size`)
  with a more threshold-sensitive one (`trading_activity`)

### 4. Full settings blocks were retained

Phase 08 kept the fuller settings structure introduced in the Phase 07
cleanup:
- `Trading Activity`
- `Trade Size`
- `Risk`
- `Holding`
- `Diversification`

Motivation:
- avoid a toy prompt where one single setting line behaves like the
  entire world
- keep prompts closer to the realistic prompt geometry the model sees
- support future extensions to other conflict dimensions

### 5. Nuisance settings were allowed to vary

Rather than pinning all non-target settings to a fixed middle value,
Phase 08 explicitly kept nuisance variation in scope.

Motivation:
- reduce structural shortcuts
- make prompts look less synthetic
- avoid making the target dimension trivially identifiable from the
  settings block

### 6. Edge cases were represented explicitly

Phase 08 introduced explicit handling for middle / ambiguous values,
especially around `setting_value=3`.

Motivation:
- retain graded structure in the dataset
- but avoid polluting the primary binary task with easy token-label
  shortcuts like `3/5`
- keep a distinction between:
  - canonical aligned/conflict rows
  - edge / boundary rows

### 7. Lexical gating became a pre-capture requirement

Phase 08 formalized the idea that raw-text baselines should be run
before capture.

Motivation:
- avoid another expensive cycle where a dataset looks interesting only to
  discover later that the target is mostly text-decodable
- make lexical confound control part of the benchmark lifecycle, not an
  after-the-fact critique

## What Phase 08 Did Not Yet Solve

Phase 08 was the right structural pivot, but not the final prompt
system.

The main things it still struggled with were:

- activity thresholds still carried too much semantic ambiguity
- the benchmark still used a richer target-setting scale than we
  ultimately wanted
- behavior under conflict was not yet clean enough for a final capture
  story

That is why Phase 09 later made additional changes:

- cleaner prompt wording
- clearer descriptive market language
- extremes-only target settings
- removal of `medium` action size
- tighter behavior-smoke iteration before capture

## How To Read Phase 08 Relative To Phase 09

The right mental model is:

- Phase 08 = design pivot
- Phase 09 = operationalized benchmark

Phase 08 is still useful because it records the project-level decision
that the old family-based framing was structurally wrong.

Phase 09 is still the place to look for:

- the final cleaned prompt system
- the implemented workflow
- the actual capture results
- the current interpretation

## Bottom Line

Phase 08 should be understood as the point where the project made the
correct strategic move:

- stop trying to rescue `family`
- focus directly on relational conflict
- make lexical control a first-class requirement

That move was later validated by Phase 09, which absorbed the core Phase
08 design ideas and turned them into the strongest empirical result in
the project so far.
