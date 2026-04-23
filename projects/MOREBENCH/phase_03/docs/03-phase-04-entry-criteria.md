---
benchmark: morebench
phase: 03
version: v1
frozen_date: 2026-04-23
input_artifacts:
  - projects/MOREBENCH/phase_03/docs/03-analysis-plan.md
  - projects/MOREBENCH/phase_03/docs/03-experiment-specs.md
  - projects/MOREBENCH/phase_03/docs/03-controls-and-splits.md
---

# MoReBench 03 Phase-04 Entry Criteria

Promotion from phase 03 to phase 04 causal follow-up should require an explicit yes on all relevant items below.

## Core Gate

- behavioral gate satisfied for the execution model used in the candidate result
- readout beats the cheap baselines named in `03-controls-and-splits.md`
- signal survives at least one nuisance-aware split or holdout
- localization narrows the candidate site beyond a whole-prompt or whole-response claim
- claim ceiling remains representational or localized representational at the moment of promotion

## Additional Gate For Response-Side Labels

- the result was built on a frozen validated response-side label slice
- conclusion-span-only explanations have been ruled out where relevant
- label-shuffled sanity checks are negative

## Additional Gate For Prompt-Side Explicitly Named Variables

- at least one anti-shortcut control has been run
- success does not depend only on the named token or its fixed position

## Promotion Rule

If any required item above is missing, the family stays in phase 03.
