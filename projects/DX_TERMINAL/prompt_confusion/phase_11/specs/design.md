# Prompt Confusion Phase 11 Design

## Goal

Phase 11 is the first multi-conflict phase.

It combines the two prompt-local dimensions that currently look robust:
- `trade_size`
- `risk_preference`

The design goal is to test whether both conflict families can be
represented simultaneously in a single prompt.

## Core design

### Joint strategy state

Each prompt contains:
- one size preference:
  - `small` or `large`
- one risk preference:
  - `conservative` or `aggressive`

These are both expressed in `STRATEGY`.

### Joint settings state

Each prompt also contains:
- one target size setting:
  - `1/5` or `5/5`
- one target risk setting:
  - `1/5` or `5/5`

These are both binding fields in `ACTIVE SETTINGS`.

### Resulting labels

Each row has:
- `size_conflict_present`
- `risk_conflict_present`
- `conflict_count ∈ {0, 1, 2}`
- `any_conflict_present`
- `double_conflict_present`

The primary expectation is:
- size and risk conflict should both be independently decodable from the
  same capture

## Why this phase exists

Phase 10 showed:
- `trade_size` and `risk_preference` have meaningful shared structure
- but also dimension-specific specialization

The natural next question is:
- what happens when the model has to represent both at once?

This phase is the simplest way to test that without waiting on causal
infrastructure.

## Prompt philosophy

Phase 11 preserves the successful prompt-local structure:
- `STRATEGY` gives defeasible preferences
- `ACTIVE SETTINGS` provide binding execution constraints
- `MARKET` provides descriptive evidence only

The expected output is always a buy:
- asset is determined by resolved risk
- size is determined by resolved size

This keeps the benchmark local and compositional.

## Expected-output logic

Resolved size:
- `setting_value = 1` -> `small`
- `setting_value = 5` -> `large`

Resolved risk:
- `setting_value = 1` -> `ALPHA`
- `setting_value = 5` -> `BETA`

Expected output:
- `{"action":"buy","asset":resolved_asset,"size":resolved_size}`

## What would count as success

Phase 11 is promising if it shows:

1. text gates stay near chance for:
   - `size_conflict_present`
   - `risk_conflict_present`
2. both conflict families remain independently readable from one forward
   pass
3. double-conflict rows do not collapse the benchmark semantics
4. the joint representation helps us distinguish:
   - additive conflict structure
   - from genuinely interactive conflict structure
