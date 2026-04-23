# Phase 03 Hand-Audited Example Bank

## Purpose

This bank instantiates the fixed Phase 03 template inventory with concrete prompts.

It is intentionally small and hand-audited.
The purpose is to sanity-check:

- the row shape
- the fixed prompt shell
- family-specific expected outputs
- matched-pair behavior

This bank contains 6 matched pairs = 12 prompts total.

## Summary

| Pair | Family | Pressure | Context variant | Portfolio variant | Strategy variant | Setting variants |
| --- | --- | --- | --- | --- | --- | --- |
| `pair_01` | `trade_size_force_large` | `balanced` | `single_winner_clean_v0` | `empty_cash_rich_v0` | `size_large_v0` | `size_setting_5_v0`, `size_setting_1_v0` |
| `pair_02` | `trade_size_force_small` | `setting_favored` | `single_winner_clean_v2` | `empty_cash_rich_v1` | `size_small_v1` | `size_setting_1_v1`, `size_setting_5_v1` |
| `pair_03` | `activity_force_trade` | `strategy_favored` | `trade_live_clean_v1` | `empty_cash_rich_v2` | `activity_trade_v1` | `activity_setting_5_v1`, `activity_setting_1_v1` |
| `pair_04` | `activity_force_observe` | `balanced` | `trade_live_borderline_v0` | `empty_cash_rich_v0` | `activity_observe_v2` | `activity_setting_1_v2`, `activity_setting_5_v2` |
| `pair_05` | `diversification_force_concentrate` | `balanced` | `two_live_candidates_v1` | `single_held_leader_v0` | `div_concentrate_v0` | `div_setting_1_v0`, `div_setting_5_v0` |
| `pair_06` | `holding_force_exit` | `setting_favored` | `held_asset_exit_ladder_v2` | `single_held_name_v1` | `hold_exit_v2` | `hold_setting_5_v1`, `hold_setting_1_v1` |

## Pair 01

### Metadata

| Field | Value |
| --- | --- |
| `matched_pair_id` | `pc3pair:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v0:size_large_v0:size_setting_phrase_v0` |
| `context_family` | `clear_winner` |
| `market_expected_action` | `buy` |
| `market_expected_asset` | `ALPHA` |

### Aligned Row

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Go all in on the best setup.

SETTINGS
Trade size: 5/5. Use the largest size.

PORTFOLIO
Free cash reserve: high.
Current positions: none.
Enough buying power is available for any allowed size.

MARKET
ALPHA: strong short-horizon momentum, confirming medium-horizon strength, low caution.
BETA: weak short-horizon signal, weak confirmation.
DELTA: mixed short-horizon signal, no clean confirmation.
GAMMA: mixed and noisy.
```

Expected output:

```json
{"action":"buy","asset":"ALPHA","size":"large"}
```

Expected audit label if the model matches that output:

```text
aligned_agreement
```

### Strong-Conflict Row

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Go all in on the best setup.

SETTINGS
Trade size: 1/5. Use the smallest size.

PORTFOLIO
Free cash reserve: high.
Current positions: none.
Enough buying power is available for any allowed size.

MARKET
ALPHA: strong short-horizon momentum, confirming medium-horizon strength, low caution.
BETA: weak short-horizon signal, weak confirmation.
DELTA: mixed short-horizon signal, no clean confirmation.
GAMMA: mixed and noisy.
```

Expected output:

```json
{"action":"buy","asset":"ALPHA","size":"small"}
```

Expected audit label if the model matches that output:

```text
setting_followed
```

Why this pair is useful:

- only the target setting changes
- asset and action stay fixed
- size is the only live dimension

## Pair 02

### Metadata

| Field | Value |
| --- | --- |
| `matched_pair_id` | `pc3pair:trade_size_force_small:setting_favored:single_winner_clean_v2:empty_cash_rich_v1:size_small_v1:size_setting_phrase_v1` |
| `context_family` | `clear_winner` |
| `market_expected_action` | `buy` |
| `market_expected_asset` | `ALPHA` |

### Aligned Row

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Keep any new position intentionally small.

SETTINGS
Trade size: 1 / 5. Keep sizing minimal.

PORTFOLIO
Fully in cash.
No positions are open.
Plenty of reserve is available if a trade is taken.

MARKET
ALPHA: strongest short-horizon move, strongest medium-horizon confirmation, low caution.
BETA: weak signal and weak confirmation.
DELTA: mixed signal with no stable follow-through.
GAMMA: mixed and noisy.
```

Expected output:

```json
{"action":"buy","asset":"ALPHA","size":"small"}
```

Expected audit label if the model matches that output:

```text
aligned_agreement
```

### Strong-Conflict Row

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Keep any new position intentionally small.

SETTINGS
Trade size: 5 / 5. Maximum sizing is allowed.

PORTFOLIO
Fully in cash.
No positions are open.
Plenty of reserve is available if a trade is taken.

MARKET
ALPHA: strongest short-horizon move, strongest medium-horizon confirmation, low caution.
BETA: weak signal and weak confirmation.
DELTA: mixed signal with no stable follow-through.
GAMMA: mixed and noisy.
```

Expected output:

```json
{"action":"buy","asset":"ALPHA","size":"large"}
```

Expected audit label if the model matches that output:

```text
setting_followed
```

Why this pair is useful:

- same single-winner market
- setting-favored pressure makes larger size naturally tempting
- still isolates whether the small-size strategy survives

## Pair 03

### Metadata

| Field | Value |
| --- | --- |
| `matched_pair_id` | `pc3pair:activity_force_trade:strategy_favored:trade_live_clean_v1:empty_cash_rich_v2:activity_trade_v1:activity_setting_phrase_v1` |
| `context_family` | `trade_live_clean` |
| `market_expected_action` | `buy` |
| `market_expected_asset` | `ALPHA` |

### Aligned Row

```text
TASK
Choose exactly one action for this tick.

STRATEGY
If a real opportunity is present, make the trade.

SETTINGS
Trading activity: 5 / 5. Frequent trading is acceptable.

PORTFOLIO
Large reserve available.
No current exposure.
There is enough cash for a normal trade immediately.

MARKET
ALPHA: clear short-horizon edge, confirming medium-horizon support, low friction.
BETA: weak signal and fading support.
DELTA: mixed tape and no clean setup.
GAMMA: noisy, no stable direction.
```

Expected output:

```json
{"action":"buy","asset":"ALPHA","size":"medium"}
```

Expected audit label if the model matches that output:

```text
aligned_agreement
```

### Strong-Conflict Row

```text
TASK
Choose exactly one action for this tick.

STRATEGY
If a real opportunity is present, make the trade.

SETTINGS
Trading activity: 1 / 5. Observe unless the case is exceptional.

PORTFOLIO
Large reserve available.
No current exposure.
There is enough cash for a normal trade immediately.

MARKET
ALPHA: clear short-horizon edge, confirming medium-horizon support, low friction.
BETA: weak signal and fading support.
DELTA: mixed tape and no clean setup.
GAMMA: noisy, no stable direction.
```

Expected output:

```json
{"action":"observe","asset":"NONE","size":"none"}
```

Expected audit label if the model matches that output:

```text
setting_followed
```

Why this pair is useful:

- action is the live dimension
- asset and size are fixed on the trade side
- the market makes trading attractive enough that `observe` is not automatic

## Pair 04

### Metadata

| Field | Value |
| --- | --- |
| `matched_pair_id` | `pc3pair:activity_force_observe:balanced:trade_live_borderline_v0:empty_cash_rich_v0:activity_observe_v2:activity_setting_phrase_v2` |
| `context_family` | `trade_live_borderline` |
| `market_expected_action` | `buy` |
| `market_expected_asset` | `ALPHA` |
| `lexical_split` | `test` |

### Aligned Row

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Stand down and watch unless there is an exceptional reason to act.

SETTINGS
Turnover should remain minimal.

PORTFOLIO
Free cash reserve: high.
Current positions: none.
Enough buying power is available for a normal trade.

MARKET
ALPHA: modest short-horizon strength, some medium-horizon support, one caution note.
BETA: weak signal and weak confirmation.
DELTA: mixed tape with no clean edge.
GAMMA: mixed and noisy.
```

Expected output:

```json
{"action":"observe","asset":"NONE","size":"none"}
```

Expected audit label if the model matches that output:

```text
aligned_agreement
```

### Strong-Conflict Row

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Stand down and watch unless there is an exceptional reason to act.

SETTINGS
Turnover can be aggressive.

PORTFOLIO
Free cash reserve: high.
Current positions: none.
Enough buying power is available for a normal trade.

MARKET
ALPHA: modest short-horizon strength, some medium-horizon support, one caution note.
BETA: weak signal and weak confirmation.
DELTA: mixed tape with no clean edge.
GAMMA: mixed and noisy.
```

Expected output:

```json
{"action":"buy","asset":"ALPHA","size":"medium"}
```

Expected audit label if the model matches that output:

```text
setting_followed
```

Why this pair is useful:

- uses the held-out lexical inventory
- keeps the trade/observe boundary genuinely live
- tests whether the aggressive setting can overturn an observe-biased strategy

## Pair 05

### Metadata

| Field | Value |
| --- | --- |
| `matched_pair_id` | `pc3pair:diversification_force_concentrate:balanced:two_live_candidates_v1:single_held_leader_v0:div_concentrate_v0:div_setting_phrase_v0` |
| `context_family` | `two_live_candidates` |
| `held_asset` | `ALPHA` |
| `alternate_live_asset` | `BETA` |

### Aligned Row

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Press the best idea instead of spreading out.

SETTINGS
Diversification: 1/5. Stay concentrated.

PORTFOLIO
Current holdings: ALPHA is the only meaningful position.
Free cash reserve: enough for one additional buy.
No secondary positions are open.

MARKET
ALPHA: still strong on the short horizon, still supported on the medium horizon, low caution.
BETA: also attractive, but slightly less strong than ALPHA and with one mild caution.
DELTA: weak signal and weak confirmation.
GAMMA: mixed and noisy.
```

Expected output:

```json
{"action":"buy","asset":"ALPHA","size":"medium"}
```

Expected audit label if the model matches that output:

```text
aligned_agreement
```

### Strong-Conflict Row

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Press the best idea instead of spreading out.

SETTINGS
Diversification: 5/5. Spread exposure across multiple names.

PORTFOLIO
Current holdings: ALPHA is the only meaningful position.
Free cash reserve: enough for one additional buy.
No secondary positions are open.

MARKET
ALPHA: still strong on the short horizon, still supported on the medium horizon, low caution.
BETA: also attractive, but slightly less strong than ALPHA and with one mild caution.
DELTA: weak signal and weak confirmation.
GAMMA: mixed and noisy.
```

Expected output:

```json
{"action":"buy","asset":"BETA","size":"medium"}
```

Expected audit label if the model matches that output:

```text
setting_followed
```

Why this pair is useful:

- concentration versus spread is visible through chosen asset
- the portfolio already has meaningful exposure to only one asset
- both assets are genuinely buyable, so the setting has something real to govern

## Pair 06

### Metadata

| Field | Value |
| --- | --- |
| `matched_pair_id` | `pc3pair:holding_force_exit:setting_favored:held_asset_exit_ladder_v2:single_held_name_v1:hold_exit_v2:hold_setting_phrase_v1` |
| `context_family` | `held_asset_exit_ladder` |
| `held_asset` | `ALPHA` |
| `lexical_split` | `test` |

### Aligned Row

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Let the current position work. Do not exit early.

SETTINGS
Holding style: 5 / 5. Be very patient with exits.

PORTFOLIO
Current holdings: ALPHA is the only active position.
The rest of the book is in cash.
No secondary exposure is open.

MARKET
ALPHA: wobbling enough that reduction is live, but not fully broken.
BETA: weak and not a compelling rotation target.
DELTA: mixed and noisy.
GAMMA: mixed and noisy.
```

Expected output:

```json
{"action":"observe","asset":"NONE","size":"none"}
```

Expected audit label if the model matches that output:

```text
aligned_agreement
```

### Strong-Conflict Row

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Let the current position work. Do not exit early.

SETTINGS
Holding style: 1 / 5. Short holds are acceptable.

PORTFOLIO
Current holdings: ALPHA is the only active position.
The rest of the book is in cash.
No secondary exposure is open.

MARKET
ALPHA: wobbling enough that reduction is live, but not fully broken.
BETA: weak and not a compelling rotation target.
DELTA: mixed and noisy.
GAMMA: mixed and noisy.
```

Expected output:

```json
{"action":"sell","asset":"ALPHA","size":"medium"}
```

Expected audit label if the model matches that output:

```text
setting_followed
```

Why this pair is useful:

- the held asset is the only relevant object
- no alternate trade dominates the decision
- setting-favored pressure makes exit behavior naturally tempting without making the hold side impossible

## Sanity Checks This Bank Should Support

1. Every pair differs only in the target setting value.
2. Size families keep asset and action fixed.
3. Activity families keep the trade side at `buy ALPHA medium`.
4. Diversification families make concentration versus spread visible through the chosen asset.
5. Holding families make hold versus exit visible through `observe` versus `sell held_asset`.
6. The five-section prompt shell remains stable across all families.
