# Prompt Confusion Phase 12 Design

## Goal

Phase 12 adds a third prompt-local conflict family:

- `diversification_preference`

The design target is a clean portfolio-conditioned asset-selection
benchmark.

## Core idea

Unlike `risk_preference`, which varies the asset choice based on risk
profile, this phase varies the asset choice based on how the new trade
should relate to the existing book.

The portfolio is fixed so the mapping remains prompt-local:

- the current book already has meaningful overlap with `ALPHA`
- `BETA` is a distinct but still viable sleeve

That means:

- `concentrated` resolution -> `ALPHA`
- `diversified` resolution -> `BETA`

## Why this family is useful

This family tests a new kind of conflict:

- not output-size conflict
- not risk-profile conflict
- but portfolio-allocation conflict

It also forces `PORTFOLIO` to do real semantic work. The benchmark is
not answerable from `STRATEGY` and `ACTIVE SETTINGS` alone; the model has
to integrate how the candidate assets relate to the existing book.

## Strategy vs settings structure

### Strategy directions

- `diversified`
- `concentrated`

### Setting values

- `1/5` -> diversify the book
- `5/5` -> concentrate into the strongest overlapping sleeve

### Resulting label

`conflict_present = (strategy_direction != setting_implied_direction)`

This preserves the same XOR-style relational structure used in the
stronger earlier phases.

## Prompt philosophy

The prompt remains local and single-tick:

- `STRATEGY` gives a defeasible portfolio-allocation preference
- `ACTIVE SETTINGS` gives a binding diversification posture
- `PORTFOLIO` states the current book and overlap structure
- `MARKET` makes both `ALPHA` and `BETA` viable while making their book
  relationship different

The expected output is always:

- `action = buy`
- `size = large`
- `asset` determined by the resolved diversification posture

## Asset semantics

`ALPHA`

- stronger immediate setup
- overlaps heavily with the existing book
- increases concentration if selected

`BETA`

- still strong enough to buy
- distinct from the current sleeve
- broadens exposure if selected

The design risk is obvious:

- the model may still prefer `ALPHA` on diversified rows because it reads
  as the highest-conviction asset

That is acceptable for a first scaffold. The first behavior smoke should
specifically test whether the portfolio-conditioned diversification logic
is strong enough to override that default.

## What would count as success

Phase 12 is promising if it shows:

1. text gates stay near chance
2. aligned rows are behaviorally clean
3. conflict rows show some meaningful follow-setting behavior
4. the diversification conflict probe is strongly readable

## Why this is a better third family than trading_activity

The relevant comparison is not that diversification is "easy." The
important point is that it should remain prompt-local:

- the whole conflict is defined by current prompt text
- no hidden trading history is required
- no temporal threshold needs to be reconstructed from missing state

That keeps it in the family of benchmarks that are most likely to
support clean geometry comparisons later.
