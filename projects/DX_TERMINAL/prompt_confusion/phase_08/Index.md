# Prompt Confusion Phase 08

Status: **design pivot archived; implemented historically in local worktrees, superseded by Phase 09**

Phase 08 is preserved in-tree because it marks the moment the project
stopped trying to rescue `family` and instead pivoted to relational
conflict as the main benchmark target.

If you are reading the project cold:

- Phase 05 explains why the old family/arbitration story was not clean
- Phase 07 shows that additional wording cleanup still did not rescue
  `family`
- Phase 08 is the design pivot
- Phase 09 is the benchmark that absorbed this design and produced the
  strongest empirical result

See also:

- [Pivot Retrospective](./notes/phase_08_pivot_retrospective_20260415.md)
- [Phase 09 Index](../phase_09/Index.md)

Phase 08 is a relational rebuild of the prompt-confusion benchmark.

The core lesson from Phase 05 and Phase 07 is:

- `family` is not a clean variable
- the surviving result is `conflict detection`
- the next benchmark should directly test that relational variable

## What changes in Phase 08

1. No family-locked strategy polarity
   - `strategy_direction` varies within the dataset
   - example: `small` and `large` both appear across aligned and conflict rows

2. Conflict is relational by construction
   - `conflict_present = (strategy_direction != setting_implied_direction)`
   - the same direction tokens appear in both labels
   - the primary binary task uses canonical aligned/conflict rows only;
     middle `edge` rows stay in the dataset but are excluded from the
     headline binary probe

3. Two dimensions, same crossed design
   - `trade_size`
   - `trading_activity`

4. Keep the good Phase 07 structure
   - full settings block
   - nuisance settings vary
   - shared context pools
   - prompt-only capture default
   - grouped CV over 5-row sweeps with explicit canonical pairs

## Primary question

Can we build a benchmark where raw text is near-chance for
`conflict_present`, while the model still behaviorally respects the
settings-constrained output?

If yes, Phase 08 becomes the new base benchmark for conflict-detection
mechanistic work.
