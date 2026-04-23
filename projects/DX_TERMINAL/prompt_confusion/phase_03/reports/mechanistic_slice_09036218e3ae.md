# Phase 03 Mechanistic Slice

Run: `09036218e3ae`
Families: `trade_size_force_large, activity_force_observe`

## Summary

- Filtered matched pairs: `601`
- `trade_size_force_large`: `563` pairs, `294` strong-conflict exact matches, readout counts `{"neither": 269, "setting": 294}`
- `activity_force_observe`: `38` pairs, `0` strong-conflict exact matches, readout counts `{"setting": 38}`

## trade_size_force_large

### `pc3pair:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v0:size_large_v0:size_setting_phrase_v0`

- Family: `trade_size_force_large`
- Pressure: `balanced`
- Context: `single_winner_clean_v0`
- Changed output: `true`
- Strong conflict readout side: `setting`
- Strong conflict exact expected: `true`

**Aligned**

- `example_id`: `pc3:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v0:size_large_v0:size_setting_5_v0:aligned`
- generated: `{"action":"buy","asset":"ALPHA","size":"large"}`
- expected: `{"action": "buy", "asset": "ALPHA", "size": "large"}`

**Strong Conflict**

- `example_id`: `pc3:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v0:size_large_v0:size_setting_1_v0:strong_conflict`
- generated: `{"action":"buy","asset":"ALPHA","size":"small"}`
- expected: `{"action": "buy", "asset": "ALPHA", "size": "small"}`

**Prompt**

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

### `pc3pair:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v0:size_large_v0:size_setting_phrase_v1`

- Family: `trade_size_force_large`
- Pressure: `balanced`
- Context: `single_winner_clean_v0`
- Changed output: `true`
- Strong conflict readout side: `setting`
- Strong conflict exact expected: `true`

**Aligned**

- `example_id`: `pc3:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v0:size_large_v0:size_setting_5_v1:aligned`
- generated: `{"action":"buy","asset":"ALPHA","size":"large"}`
- expected: `{"action": "buy", "asset": "ALPHA", "size": "large"}`

**Strong Conflict**

- `example_id`: `pc3:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v0:size_large_v0:size_setting_1_v1:strong_conflict`
- generated: `{"action":"buy","asset":"ALPHA","size":"small"}`
- expected: `{"action": "buy", "asset": "ALPHA", "size": "small"}`

**Prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Go all in on the best setup.

SETTINGS
Trade size: 5 / 5. Maximum sizing is allowed.

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

### `pc3pair:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v0:size_large_v0:size_setting_phrase_v2`

- Family: `trade_size_force_large`
- Pressure: `balanced`
- Context: `single_winner_clean_v0`
- Changed output: `true`
- Strong conflict readout side: `setting`
- Strong conflict exact expected: `true`

**Aligned**

- `example_id`: `pc3:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v0:size_large_v0:size_setting_5_v2:aligned`
- generated: `{"action":"buy","asset":"ALPHA","size":"large"}`
- expected: `{"action": "buy", "asset": "ALPHA", "size": "large"}`

**Strong Conflict**

- `example_id`: `pc3:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v0:size_large_v0:size_setting_1_v2:strong_conflict`
- generated: `{"action":"buy","asset":"ALPHA","size":"small"}`
- expected: `{"action": "buy", "asset": "ALPHA", "size": "small"}`

**Prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Go all in on the best setup.

SETTINGS
Position sizing can be aggressive.

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

### `pc3pair:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v0:size_large_v1:size_setting_phrase_v0`

- Family: `trade_size_force_large`
- Pressure: `balanced`
- Context: `single_winner_clean_v0`
- Changed output: `true`
- Strong conflict readout side: `setting`
- Strong conflict exact expected: `true`

**Aligned**

- `example_id`: `pc3:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v0:size_large_v1:size_setting_5_v0:aligned`
- generated: `{"action":"buy","asset":"ALPHA","size":"large"}`
- expected: `{"action": "buy", "asset": "ALPHA", "size": "large"}`

**Strong Conflict**

- `example_id`: `pc3:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v0:size_large_v1:size_setting_1_v0:strong_conflict`
- generated: `{"action":"buy","asset":"ALPHA","size":"small"}`
- expected: `{"action": "buy", "asset": "ALPHA", "size": "small"}`

**Prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Deploy maximum size on the strongest opportunity.

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

### `pc3pair:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v0:size_large_v1:size_setting_phrase_v2`

- Family: `trade_size_force_large`
- Pressure: `balanced`
- Context: `single_winner_clean_v0`
- Changed output: `true`
- Strong conflict readout side: `setting`
- Strong conflict exact expected: `true`

**Aligned**

- `example_id`: `pc3:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v0:size_large_v1:size_setting_5_v2:aligned`
- generated: `{"action":"buy","asset":"ALPHA","size":"large"}`
- expected: `{"action": "buy", "asset": "ALPHA", "size": "large"}`

**Strong Conflict**

- `example_id`: `pc3:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v0:size_large_v1:size_setting_1_v2:strong_conflict`
- generated: `{"action":"buy","asset":"ALPHA","size":"small"}`
- expected: `{"action": "buy", "asset": "ALPHA", "size": "small"}`

**Prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Deploy maximum size on the strongest opportunity.

SETTINGS
Position sizing can be aggressive.

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

### `pc3pair:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v1:size_large_v0:size_setting_phrase_v0`

- Family: `trade_size_force_large`
- Pressure: `balanced`
- Context: `single_winner_clean_v0`
- Changed output: `true`
- Strong conflict readout side: `setting`
- Strong conflict exact expected: `true`

**Aligned**

- `example_id`: `pc3:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v1:size_large_v0:size_setting_5_v0:aligned`
- generated: `{"action":"buy","asset":"ALPHA","size":"large"}`
- expected: `{"action": "buy", "asset": "ALPHA", "size": "large"}`

**Strong Conflict**

- `example_id`: `pc3:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v1:size_large_v0:size_setting_1_v0:strong_conflict`
- generated: `{"action":"buy","asset":"ALPHA","size":"small"}`
- expected: `{"action": "buy", "asset": "ALPHA", "size": "small"}`

**Prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Go all in on the best setup.

SETTINGS
Trade size: 5/5. Use the largest size.

PORTFOLIO
Fully in cash.
No positions are open.
Plenty of reserve is available if a trade is taken.

MARKET
ALPHA: strong short-horizon momentum, confirming medium-horizon strength, low caution.
BETA: weak short-horizon signal, weak confirmation.
DELTA: mixed short-horizon signal, no clean confirmation.
GAMMA: mixed and noisy.
```

### `pc3pair:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v1:size_large_v0:size_setting_phrase_v1`

- Family: `trade_size_force_large`
- Pressure: `balanced`
- Context: `single_winner_clean_v0`
- Changed output: `true`
- Strong conflict readout side: `setting`
- Strong conflict exact expected: `true`

**Aligned**

- `example_id`: `pc3:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v1:size_large_v0:size_setting_5_v1:aligned`
- generated: `{"action":"buy","asset":"ALPHA","size":"large"}`
- expected: `{"action": "buy", "asset": "ALPHA", "size": "large"}`

**Strong Conflict**

- `example_id`: `pc3:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v1:size_large_v0:size_setting_1_v1:strong_conflict`
- generated: `{"action":"buy","asset":"ALPHA","size":"small"}`
- expected: `{"action": "buy", "asset": "ALPHA", "size": "small"}`

**Prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Go all in on the best setup.

SETTINGS
Trade size: 5 / 5. Maximum sizing is allowed.

PORTFOLIO
Fully in cash.
No positions are open.
Plenty of reserve is available if a trade is taken.

MARKET
ALPHA: strong short-horizon momentum, confirming medium-horizon strength, low caution.
BETA: weak short-horizon signal, weak confirmation.
DELTA: mixed short-horizon signal, no clean confirmation.
GAMMA: mixed and noisy.
```

### `pc3pair:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v1:size_large_v0:size_setting_phrase_v2`

- Family: `trade_size_force_large`
- Pressure: `balanced`
- Context: `single_winner_clean_v0`
- Changed output: `true`
- Strong conflict readout side: `setting`
- Strong conflict exact expected: `true`

**Aligned**

- `example_id`: `pc3:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v1:size_large_v0:size_setting_5_v2:aligned`
- generated: `{"action":"buy","asset":"ALPHA","size":"large"}`
- expected: `{"action": "buy", "asset": "ALPHA", "size": "large"}`

**Strong Conflict**

- `example_id`: `pc3:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v1:size_large_v0:size_setting_1_v2:strong_conflict`
- generated: `{"action":"buy","asset":"ALPHA","size":"small"}`
- expected: `{"action": "buy", "asset": "ALPHA", "size": "small"}`

**Prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Go all in on the best setup.

SETTINGS
Position sizing can be aggressive.

PORTFOLIO
Fully in cash.
No positions are open.
Plenty of reserve is available if a trade is taken.

MARKET
ALPHA: strong short-horizon momentum, confirming medium-horizon strength, low caution.
BETA: weak short-horizon signal, weak confirmation.
DELTA: mixed short-horizon signal, no clean confirmation.
GAMMA: mixed and noisy.
```


## activity_force_observe

### `pc3pair:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v0:activity_observe_v1:activity_setting_phrase_v1`

- Family: `activity_force_observe`
- Pressure: `strategy_favored`
- Context: `trade_live_borderline_v0`
- Changed output: `true`
- Strong conflict readout side: `setting`
- Strong conflict exact expected: `false`

**Aligned**

- `example_id`: `pc3:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v0:activity_observe_v1:activity_setting_1_v1:aligned`
- generated: `{"action":"observe","asset":"NONE","size":"none"}`
- expected: `{"action": "observe", "asset": "NONE", "size": "none"}`

**Strong Conflict**

- `example_id`: `pc3:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v0:activity_observe_v1:activity_setting_5_v1:strong_conflict`
- generated: `{"action":"buy","asset":"ALPHA","size":"small"}`
- expected: `{"action": "buy", "asset": "ALPHA", "size": "medium"}`

**Prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Default to no action unless one setup is unusually strong.

SETTINGS
Trading activity: 1 / 5. Observe unless the case is exceptional.

PORTFOLIO
Free cash reserve: high.
Current positions: none.
Enough buying power is available for any allowed size.

MARKET
ALPHA: modest but usable short-horizon strength, enough support to justify acting.
BETA: weak signal and weak confirmation.
DELTA: mixed tape with no clean edge.
GAMMA: mixed and noisy.
```

### `pc3pair:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v0:activity_observe_v1:activity_setting_phrase_v2`

- Family: `activity_force_observe`
- Pressure: `strategy_favored`
- Context: `trade_live_borderline_v0`
- Changed output: `true`
- Strong conflict readout side: `setting`
- Strong conflict exact expected: `false`

**Aligned**

- `example_id`: `pc3:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v0:activity_observe_v1:activity_setting_1_v2:aligned`
- generated: `{"action":"observe","asset":"NONE","size":"none"}`
- expected: `{"action": "observe", "asset": "NONE", "size": "none"}`

**Strong Conflict**

- `example_id`: `pc3:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v0:activity_observe_v1:activity_setting_5_v2:strong_conflict`
- generated: `{"action":"buy","asset":"ALPHA","size":"small"}`
- expected: `{"action": "buy", "asset": "ALPHA", "size": "medium"}`

**Prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Default to no action unless one setup is unusually strong.

SETTINGS
Turnover should remain minimal.

PORTFOLIO
Free cash reserve: high.
Current positions: none.
Enough buying power is available for any allowed size.

MARKET
ALPHA: modest but usable short-horizon strength, enough support to justify acting.
BETA: weak signal and weak confirmation.
DELTA: mixed tape with no clean edge.
GAMMA: mixed and noisy.
```

### `pc3pair:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v2:activity_observe_v0:activity_setting_phrase_v2`

- Family: `activity_force_observe`
- Pressure: `strategy_favored`
- Context: `trade_live_borderline_v0`
- Changed output: `true`
- Strong conflict readout side: `setting`
- Strong conflict exact expected: `false`

**Aligned**

- `example_id`: `pc3:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v2:activity_observe_v0:activity_setting_1_v2:aligned`
- generated: `{"action":"observe","asset":"NONE","size":"none"}`
- expected: `{"action": "observe", "asset": "NONE", "size": "none"}`

**Strong Conflict**

- `example_id`: `pc3:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v2:activity_observe_v0:activity_setting_5_v2:strong_conflict`
- generated: `{"action":"buy","asset":"ALPHA","size":"small"}`
- expected: `{"action": "buy", "asset": "ALPHA", "size": "medium"}`

**Prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Capital preservation comes first. Do not trade unless the edge is clearly exceptional.

SETTINGS
Turnover should remain minimal.

PORTFOLIO
Large reserve available.
No current exposure.
There is enough cash for a normal trade immediately.

MARKET
ALPHA: modest but usable short-horizon strength, enough support to justify acting.
BETA: weak signal and weak confirmation.
DELTA: mixed tape with no clean edge.
GAMMA: mixed and noisy.
```

### `pc3pair:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v2:activity_observe_v1:activity_setting_phrase_v0`

- Family: `activity_force_observe`
- Pressure: `strategy_favored`
- Context: `trade_live_borderline_v0`
- Changed output: `true`
- Strong conflict readout side: `setting`
- Strong conflict exact expected: `false`

**Aligned**

- `example_id`: `pc3:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v2:activity_observe_v1:activity_setting_1_v0:aligned`
- generated: `{"action":"observe","asset":"NONE","size":"none"}`
- expected: `{"action": "observe", "asset": "NONE", "size": "none"}`

**Strong Conflict**

- `example_id`: `pc3:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v2:activity_observe_v1:activity_setting_5_v0:strong_conflict`
- generated: `{"action":"buy","asset":"ALPHA","size":"small"}`
- expected: `{"action": "buy", "asset": "ALPHA", "size": "medium"}`

**Prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Default to no action unless one setup is unusually strong.

SETTINGS
Trading activity: 1/5. Stay patient and avoid churn.

PORTFOLIO
Large reserve available.
No current exposure.
There is enough cash for a normal trade immediately.

MARKET
ALPHA: modest but usable short-horizon strength, enough support to justify acting.
BETA: weak signal and weak confirmation.
DELTA: mixed tape with no clean edge.
GAMMA: mixed and noisy.
```

### `pc3pair:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v2:activity_observe_v1:activity_setting_phrase_v1`

- Family: `activity_force_observe`
- Pressure: `strategy_favored`
- Context: `trade_live_borderline_v0`
- Changed output: `true`
- Strong conflict readout side: `setting`
- Strong conflict exact expected: `false`

**Aligned**

- `example_id`: `pc3:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v2:activity_observe_v1:activity_setting_1_v1:aligned`
- generated: `{"action":"observe","asset":"NONE","size":"none"}`
- expected: `{"action": "observe", "asset": "NONE", "size": "none"}`

**Strong Conflict**

- `example_id`: `pc3:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v2:activity_observe_v1:activity_setting_5_v1:strong_conflict`
- generated: `{"action":"buy","asset":"ALPHA","size":"small"}`
- expected: `{"action": "buy", "asset": "ALPHA", "size": "medium"}`

**Prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Default to no action unless one setup is unusually strong.

SETTINGS
Trading activity: 1 / 5. Observe unless the case is exceptional.

PORTFOLIO
Large reserve available.
No current exposure.
There is enough cash for a normal trade immediately.

MARKET
ALPHA: modest but usable short-horizon strength, enough support to justify acting.
BETA: weak signal and weak confirmation.
DELTA: mixed tape with no clean edge.
GAMMA: mixed and noisy.
```

### `pc3pair:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v2:activity_observe_v2:activity_setting_phrase_v0`

- Family: `activity_force_observe`
- Pressure: `strategy_favored`
- Context: `trade_live_borderline_v0`
- Changed output: `true`
- Strong conflict readout side: `setting`
- Strong conflict exact expected: `false`

**Aligned**

- `example_id`: `pc3:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v2:activity_observe_v2:activity_setting_1_v0:aligned`
- generated: `{"action":"observe","asset":"NONE","size":"none"}`
- expected: `{"action": "observe", "asset": "NONE", "size": "none"}`

**Strong Conflict**

- `example_id`: `pc3:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v2:activity_observe_v2:activity_setting_5_v0:strong_conflict`
- generated: `{"action":"buy","asset":"ALPHA","size":"small"}`
- expected: `{"action": "buy", "asset": "ALPHA", "size": "medium"}`

**Prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Stay disciplined and avoid forcing trades on marginal signals.

SETTINGS
Trading activity: 1/5. Stay patient and avoid churn.

PORTFOLIO
Large reserve available.
No current exposure.
There is enough cash for a normal trade immediately.

MARKET
ALPHA: modest but usable short-horizon strength, enough support to justify acting.
BETA: weak signal and weak confirmation.
DELTA: mixed tape with no clean edge.
GAMMA: mixed and noisy.
```

### `pc3pair:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v2:activity_observe_v2:activity_setting_phrase_v1`

- Family: `activity_force_observe`
- Pressure: `strategy_favored`
- Context: `trade_live_borderline_v0`
- Changed output: `true`
- Strong conflict readout side: `setting`
- Strong conflict exact expected: `false`

**Aligned**

- `example_id`: `pc3:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v2:activity_observe_v2:activity_setting_1_v1:aligned`
- generated: `{"action":"observe","asset":"NONE","size":"none"}`
- expected: `{"action": "observe", "asset": "NONE", "size": "none"}`

**Strong Conflict**

- `example_id`: `pc3:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v2:activity_observe_v2:activity_setting_5_v1:strong_conflict`
- generated: `{"action":"buy","asset":"ALPHA","size":"small"}`
- expected: `{"action": "buy", "asset": "ALPHA", "size": "medium"}`

**Prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Stay disciplined and avoid forcing trades on marginal signals.

SETTINGS
Trading activity: 1 / 5. Observe unless the case is exceptional.

PORTFOLIO
Large reserve available.
No current exposure.
There is enough cash for a normal trade immediately.

MARKET
ALPHA: modest but usable short-horizon strength, enough support to justify acting.
BETA: weak signal and weak confirmation.
DELTA: mixed tape with no clean edge.
GAMMA: mixed and noisy.
```

### `pc3pair:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v2:activity_observe_v2:activity_setting_phrase_v2`

- Family: `activity_force_observe`
- Pressure: `strategy_favored`
- Context: `trade_live_borderline_v0`
- Changed output: `true`
- Strong conflict readout side: `setting`
- Strong conflict exact expected: `false`

**Aligned**

- `example_id`: `pc3:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v2:activity_observe_v2:activity_setting_1_v2:aligned`
- generated: `{"action":"observe","asset":"NONE","size":"none"}`
- expected: `{"action": "observe", "asset": "NONE", "size": "none"}`

**Strong Conflict**

- `example_id`: `pc3:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v2:activity_observe_v2:activity_setting_5_v2:strong_conflict`
- generated: `{"action":"buy","asset":"ALPHA","size":"small"}`
- expected: `{"action": "buy", "asset": "ALPHA", "size": "medium"}`

**Prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Stay disciplined and avoid forcing trades on marginal signals.

SETTINGS
Turnover should remain minimal.

PORTFOLIO
Large reserve available.
No current exposure.
There is enough cash for a normal trade immediately.

MARKET
ALPHA: modest but usable short-horizon strength, enough support to justify acting.
BETA: weak signal and weak confirmation.
DELTA: mixed tape with no clean edge.
GAMMA: mixed and noisy.
```
