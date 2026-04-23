# Phase 03 Full-Run Slice

Run: `09036218e3ae`  
Publication: `workflow_dataset_conflict_probe_v2_v1`

## System Prompt

```text
You are a trading agent. Choose exactly one action each turn. Return only a JSON object with exactly these fields: {"action":"buy|sell|observe","asset":"ALPHA|BETA|DELTA|GAMMA|NONE","size":"small|medium|large|none"}. Do not return any other keys or any other text. If no trade should be made, return {"action":"observe","asset":"NONE","size":"none"}.
```

## What We Have

- `7290` rows total
- `7290` valid JSON outputs
- `3063` exact expected matches
- Strict `behavior_side` counts:
  - `strategy`: `3996`
  - `setting`: `742`
  - `neither`: `2552`
- Activations:
  - residual + router captured for all `7290` rows
  - layers: `0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44`
  - pooling: `last_token`
  - hidden dim: `2048`
  - remote path: `/data/activations/workflows/conflict_probe_v2/09036218e3ae`

## Setting Chosen, Conflict Cases

### Example A

**User prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
If a real opportunity is present, make the trade.

SETTINGS
Trading activity: 1/5. Stay patient and avoid churn.

PORTFOLIO
Free cash reserve: high.
Current positions: none.
Enough buying power is available for any allowed size.

MARKET
ALPHA: clear short-horizon edge, confirming medium-horizon support, low friction.
BETA: weak signal and weak confirmation.
DELTA: mixed tape with no clean edge.
GAMMA: mixed and noisy.
```

**Expected**

```json
{"size": "none", "asset": "NONE", "action": "observe"}
```

**Conflict**

```text
true
```

**Generated**

```json
{"action":"observe","asset":"NONE","size":"none"}
```

- `log_id`: `739`
- `family`: `activity_force_trade`
- `conflict_strength`: `2`
- `example_id`: `pc3:activity_force_trade:balanced:trade_live_clean_v0:empty_cash_rich_v0:activity_trade_v1:activity_setting_1_v0:strong_conflict`

### Example B

**User prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Stay engaged and take executable setups.

SETTINGS
Trading activity: 1/5. Stay patient and avoid churn.

PORTFOLIO
Free cash reserve: high.
Current positions: none.
Enough buying power is available for any allowed size.

MARKET
ALPHA: clear short-horizon edge, confirming medium-horizon support, low friction.
BETA: weak signal and weak confirmation.
DELTA: mixed tape with no clean edge.
GAMMA: mixed and noisy.
```

**Expected**

```json
{"size": "none", "asset": "NONE", "action": "observe"}
```

**Conflict**

```text
true
```

**Generated**

```json
{"action":"observe","asset":"NONE","size":"none"}
```

- `log_id`: `748`
- `family`: `activity_force_trade`
- `conflict_strength`: `2`
- `example_id`: `pc3:activity_force_trade:balanced:trade_live_clean_v0:empty_cash_rich_v0:activity_trade_v2:activity_setting_1_v0:strong_conflict`

## Setting Chosen, Non-Conflict Cases

### Example A

**User prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
If a real opportunity is present, make the trade.

SETTINGS
Trading activity: 5/5. Stay highly active when setups are available.

PORTFOLIO
Free cash reserve: high.
Current positions: none.
Enough buying power is available for any allowed size.

MARKET
ALPHA: good short-horizon edge, but not so strong that waiting would be absurd.
BETA: weak signal and weak confirmation.
DELTA: mixed tape with no clean edge.
GAMMA: mixed and noisy.
```

**Expected**

```json
{"size": "medium", "asset": "ALPHA", "action": "buy"}
```

**Conflict**

```text
false
```

**Generated**

```json
{"action":"observe","asset":"NONE","size":"none"}
```

- `log_id`: `988`
- `family`: `activity_force_trade`
- `conflict_strength`: `0`
- `example_id`: `pc3:activity_force_trade:setting_favored:trade_live_clean_v0:empty_cash_rich_v0:activity_trade_v1:activity_setting_5_v0:aligned`

### Example B

**User prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
If a real opportunity is present, make the trade.

SETTINGS
Trading activity: 5 / 5. Frequent trading is acceptable.

PORTFOLIO
Free cash reserve: high.
Current positions: none.
Enough buying power is available for any allowed size.

MARKET
ALPHA: good short-horizon edge, but not so strong that waiting would be absurd.
BETA: weak signal and weak confirmation.
DELTA: mixed tape with no clean edge.
GAMMA: mixed and noisy.
```

**Expected**

```json
{"size": "medium", "asset": "ALPHA", "action": "buy"}
```

**Conflict**

```text
false
```

**Generated**

```json
{"action":"observe","asset":"NONE","size":"none"}
```

- `log_id`: `989`
- `family`: `activity_force_trade`
- `conflict_strength`: `0`
- `example_id`: `pc3:activity_force_trade:setting_favored:trade_live_clean_v0:empty_cash_rich_v0:activity_trade_v1:activity_setting_5_v1:aligned`

## Strategy Chosen, Conflict Cases

### Example A

**User prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Capital preservation comes first. Do not trade unless the edge is clearly exceptional.

SETTINGS
Trading activity: 3/5. Use normal turnover.

PORTFOLIO
Free cash reserve: high.
Current positions: none.
Enough buying power is available for any allowed size.

MARKET
ALPHA: modest short-horizon strength, some medium-horizon support, one caution note.
BETA: weak signal and weak confirmation.
DELTA: mixed tape with no clean edge.
GAMMA: mixed and noisy.
```

**Expected**

```json
{"size": "medium", "asset": "ALPHA", "action": "buy"}
```

**Conflict**

```text
true
```

**Generated**

```json
{"action":"observe","asset":"NONE","size":"none"}
```

- `log_id`: `4`
- `family`: `activity_force_observe`
- `conflict_strength`: `1`
- `example_id`: `pc3:activity_force_observe:balanced:trade_live_borderline_v0:empty_cash_rich_v0:activity_observe_v0:activity_setting_3_v0:middle`

### Example B

**User prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Capital preservation comes first. Do not trade unless the edge is clearly exceptional.

SETTINGS
Trading activity: 3 / 5. Trade when the edge is clear.

PORTFOLIO
Free cash reserve: high.
Current positions: none.
Enough buying power is available for any allowed size.

MARKET
ALPHA: modest short-horizon strength, some medium-horizon support, one caution note.
BETA: weak signal and weak confirmation.
DELTA: mixed tape with no clean edge.
GAMMA: mixed and noisy.
```

**Expected**

```json
{"size": "medium", "asset": "ALPHA", "action": "buy"}
```

**Conflict**

```text
true
```

**Generated**

```json
{"action":"observe","asset":"NONE","size":"none"}
```

- `log_id`: `5`
- `family`: `activity_force_observe`
- `conflict_strength`: `1`
- `example_id`: `pc3:activity_force_observe:balanced:trade_live_borderline_v0:empty_cash_rich_v0:activity_observe_v0:activity_setting_3_v1:middle`

## Strategy Chosen, Non-Conflict Cases

### Example A

**User prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Capital preservation comes first. Do not trade unless the edge is clearly exceptional.

SETTINGS
Trading activity: 1/5. Stay patient and avoid churn.

PORTFOLIO
Free cash reserve: high.
Current positions: none.
Enough buying power is available for any allowed size.

MARKET
ALPHA: modest short-horizon strength, some medium-horizon support, one caution note.
BETA: weak signal and weak confirmation.
DELTA: mixed tape with no clean edge.
GAMMA: mixed and noisy.
```

**Expected**

```json
{"size": "none", "asset": "NONE", "action": "observe"}
```

**Conflict**

```text
false
```

**Generated**

```json
{"action":"observe","asset":"NONE","size":"none"}
```

- `log_id`: `1`
- `family`: `activity_force_observe`
- `conflict_strength`: `0`
- `example_id`: `pc3:activity_force_observe:balanced:trade_live_borderline_v0:empty_cash_rich_v0:activity_observe_v0:activity_setting_1_v0:aligned`

### Example B

**User prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Capital preservation comes first. Do not trade unless the edge is clearly exceptional.

SETTINGS
Trading activity: 1 / 5. Observe unless the case is exceptional.

PORTFOLIO
Free cash reserve: high.
Current positions: none.
Enough buying power is available for any allowed size.

MARKET
ALPHA: modest short-horizon strength, some medium-horizon support, one caution note.
BETA: weak signal and weak confirmation.
DELTA: mixed tape with no clean edge.
GAMMA: mixed and noisy.
```

**Expected**

```json
{"size": "none", "asset": "NONE", "action": "observe"}
```

**Conflict**

```text
false
```

**Generated**

```json
{"action":"observe","asset":"NONE","size":"none"}
```

- `log_id`: `2`
- `family`: `activity_force_observe`
- `conflict_strength`: `0`
- `example_id`: `pc3:activity_force_observe:balanced:trade_live_borderline_v0:empty_cash_rich_v0:activity_observe_v0:activity_setting_1_v1:aligned`

## Neither

These are strict full-triplet `neither` rows. In these sampled examples the model still clearly leans setting-side behaviorally, but misses the exact setting triplet on size.

### Example A

**User prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Default to no action unless one setup is unusually strong.

SETTINGS
Trading activity: 3/5. Use normal turnover.

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

**Expected**

```json
{"size": "medium", "asset": "ALPHA", "action": "buy"}
```

**Conflict**

```text
true
```

**Generated**

```json
{"action":"buy","asset":"ALPHA","size":"small"}
```

- `log_id`: `499`
- `family`: `activity_force_observe`
- `conflict_strength`: `1`
- `readout_side`: `setting`
- `example_id`: `pc3:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v0:activity_observe_v1:activity_setting_3_v0:middle`

### Example B

**User prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Default to no action unless one setup is unusually strong.

SETTINGS
Trading activity: 3 / 5. Trade when the edge is clear.

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

**Expected**

```json
{"size": "medium", "asset": "ALPHA", "action": "buy"}
```

**Conflict**

```text
true
```

**Generated**

```json
{"action":"buy","asset":"ALPHA","size":"small"}
```

- `log_id`: `500`
- `family`: `activity_force_observe`
- `conflict_strength`: `1`
- `readout_side`: `setting`
- `example_id`: `pc3:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v0:activity_observe_v1:activity_setting_3_v1:middle`

### Example C

**User prompt**

```text
TASK
Choose exactly one action for this tick.

STRATEGY
Default to no action unless one setup is unusually strong.

SETTINGS
Turnover should stay balanced.

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

**Expected**

```json
{"size": "medium", "asset": "ALPHA", "action": "buy"}
```

**Conflict**

```text
true
```

**Generated**

```json
{"action":"buy","asset":"ALPHA","size":"small"}
```

- `log_id`: `501`
- `family`: `activity_force_observe`
- `conflict_strength`: `1`
- `readout_side`: `setting`
- `example_id`: `pc3:activity_force_observe:strategy_favored:trade_live_borderline_v0:empty_cash_rich_v0:activity_observe_v1:activity_setting_3_v2:middle`
