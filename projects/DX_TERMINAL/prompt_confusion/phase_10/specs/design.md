# Prompt Confusion Phase 10 Design

## Goal

Phase 10 is a clean expansion beyond `trade_size`:

- `risk_preference` should stay prompt-local
- conflict should be relational and binary
- the market should not need hidden trade history

This phase is explicitly meant to avoid the structural problem we hit in
`trading_activity`, where the policy variable was partly stateful and only
weakly specified by a single-tick prompt snapshot.

## Core design

### Target dimension

- `target_dimension = risk_preference`
- `strategy_direction ∈ {conservative, aggressive}`
- `setting_value ∈ {1, 5}`
- `setting_implied_direction ∈ {conservative, aggressive}`
- `conflict_present = (strategy_direction != setting_implied_direction)`

### Why risk is asset selection, not entry gating

Risk is represented as a choice between two viable assets:

- `ALPHA`
  - lower-variance
  - steadier follow-through
  - tighter downside
- `BETA`
  - higher-beta / more convex upside
  - wider swings
  - larger adverse excursions if wrong

This keeps the task prompt-local:

- the question is not whether to trade
- the question is not what size to use
- the question is which risk profile wins

That makes it much closer to `trade_size` than to `trading_activity`.

## Market portrayal

The main market-design requirement is:

- both `ALPHA` and `BETA` must be tradeable
- the prompt should not make one of them globally "correct"
- the conflict variable should decide which one to prefer

So each market context should present:

- one stable / bounded / smoother candidate (`ALPHA`)
- one faster / wider / more explosive candidate (`BETA`)
- two weak distractors (`DELTA`, `GAMMA`)

The context tier can change how compelling both opportunities are, but should
preserve the risk contrast:

- `solid`
  - both are viable
  - `ALPHA` looks cleaner
  - `BETA` offers more upside but more path risk
- `exceptional`
  - both are unusually attractive
  - `ALPHA` remains the safer expression
  - `BETA` remains the more aggressive expression

## Prompt philosophy

Phase 10 reuses the successful Phase 09 structure:

- `STRATEGY`
  - defeasible preference
- `ACTIVE SETTINGS`
  - binding constraint
- `MARKET`
  - descriptive evidence

For this phase:

- `Trading Activity` is fixed permissive
- `Trade Size` is fixed large
- `Risk Preference` is the target setting
- holding / diversification remain nuisance settings

## Expected output logic

- if the resolved risk direction is `conservative`
  - expected output is `buy ALPHA large`
- if the resolved risk direction is `aggressive`
  - expected output is `buy BETA large`

There are no observe rows in the initial scaffold.

## What would count as success

This dimension is promising if it shows:

1. weak text-only decodability of `conflict_present`
2. strong behavioral compliance
3. cleaner lexical holdout than `trading_activity`
4. either:
   - a clean dimension-specific conflict direction
   - or an interesting partial relation to `trade_size`
