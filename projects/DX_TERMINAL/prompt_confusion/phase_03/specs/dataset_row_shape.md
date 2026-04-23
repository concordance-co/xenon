# Phase 03 Dataset Row Shape

## Purpose

This document defines the exact row shape for Phase 03 generator output and the later post-inference enrichment pass.

The intended on-disk generator format is JSONL, one row per prompt.

## Row Lifecycle

Phase 03 rows have two stages:

1. generator output
2. post-inference enrichment

Generator output contains prompt content, template ids, expected strategy-side behavior, and expected setting-side behavior.

Post-inference enrichment adds model outputs and the derived audit labels.

## ID Conventions

### `example_id`

Deterministic row id for one prompt.

Recommended format:

```text
pc3:{strategy_family}:{environment_pressure_bucket}:{context_variant_id}:{portfolio_variant_id}:{strategy_variant_id}:{setting_variant_id}:{setting_bucket}
```

Example:

```text
pc3:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v0:size_large_v0:size_setting_5_v0:aligned
```

### `matched_pair_id`

Stable id shared by the aligned and strong-conflict versions of the same template realization.

Recommended format:

```text
pc3pair:{strategy_family}:{environment_pressure_bucket}:{context_variant_id}:{portfolio_variant_id}:{strategy_variant_id}:{setting_lexical_family_id}
```

### `pair_member`

Enum:

- `aligned`
- `middle`
- `strong_conflict`

## Generator Output Fields

| Field | Type | Allowed values / shape | Purpose |
| --- | --- | --- | --- |
| `example_id` | `str` | deterministic id | unique row key |
| `matched_pair_id` | `str` | deterministic id | pair grouping key |
| `pair_member` | `str` | `aligned|middle|strong_conflict` | identifies which setting bucket this row instantiates |
| `strategy_family` | `str` | fixed family ids | top-level conflict family |
| `strategy_variant_id` | `str` | fixed inventory id | exact strategy wording id |
| `setting_lexical_family_id` | `str` | fixed wording-family id | ties aligned/middle/conflict rows to the same setting phrasing family |
| `setting_family` | `str` | `trade_size|trading_activity|diversification|holding_style` | target setting family |
| `setting_variant_id` | `str` | fixed inventory id | exact setting wording id |
| `setting_value` | `int` | `1|3|5` | coarse slider value |
| `setting_bucket` | `str` | `aligned|middle|strong_conflict` | semantic bucket for the setting value |
| `conflict_present` | `bool` | `true|false` | whether the row contains an explicit conflict |
| `conflict_strength` | `int` | `0|1|2` | ordinal strength derived from `setting_bucket` |
| `environment_pressure_bucket` | `str` | `balanced|strategy_favored|setting_favored` | which side the environment naturally leans toward |
| `context_family` | `str` | semantic context family | human-readable environment family |
| `context_variant_id` | `str` | fixed market-template variant id | exact market template realization |
| `portfolio_state_family` | `str` | fixed portfolio family | high-level portfolio setup |
| `portfolio_variant_id` | `str` | fixed portfolio-template variant id | exact portfolio wording/setup |
| `lexical_split` | `str` | `train|test` | held-out lexical split marker |
| `strategy_lexical_split` | `str` | `train|test` | strategy-side lexical split marker |
| `setting_lexical_split` | `str` | `train|test` | setting-side lexical split marker |
| `system_text` | `str` | prompt text | system message |
| `user_text` | `str` | prompt text | rendered five-section user message |
| `prompt_messages_json` | `json` | list of two messages | canonical prompt payload |
| `strategy_snapshot_json` | `json` | object | structured strategy metadata |
| `settings_snapshot_json` | `json` | object | structured setting metadata |
| `portfolio_snapshot_json` | `json` | object | structured portfolio metadata |
| `market_snapshot_json` | `json` | object | structured market metadata |
| `market_expected_action` | `str` | `buy|sell|observe` | what the environment alone makes natural |
| `market_expected_asset` | `str` | asset id or `NONE` | the asset favored by the market alone |
| `strategy_expected_action` | `str` | `buy|sell|observe` | expected action if strategy side governs |
| `strategy_expected_asset` | `str` | asset id or `NONE` | expected asset if strategy side governs |
| `strategy_expected_size` | `str` | `small|medium|large|none` | expected size if strategy side governs |
| `setting_expected_action` | `str` | `buy|sell|observe` | expected action if setting side governs |
| `setting_expected_asset` | `str` | asset id or `NONE` | expected asset if setting side governs |
| `setting_expected_size` | `str` | `small|medium|large|none` | expected size if setting side governs |
| `expected_output_json` | `json` | object with `action/asset/size` | row-level expected output for the instantiated bucket |

## Post-Inference Enrichment Fields

| Field | Type | Allowed values / shape | Purpose |
| --- | --- | --- | --- |
| `model_action` | `str` | `buy|sell|observe` | parsed model action |
| `model_asset` | `str` | asset id or `NONE` | parsed model asset |
| `model_size` | `str` | `small|medium|large|none` | parsed model size |
| `audit_label` | `str` | `aligned_agreement|strategy_followed|setting_followed|mixed_or_neither` | audit label |
| `binary_label` | `str|null` | `strategy_followed|setting_followed|null` | derived binary target |

## Bucket Semantics

Use the same bucket semantics for every family:

| `setting_bucket` | `conflict_present` | `conflict_strength` |
| --- | --- | --- |
| `aligned` | `false` | `0` |
| `middle` | `true` | `1` |
| `strong_conflict` | `true` | `2` |

## Family-Specific Expected Output Shape

| Family | Strategy-side expected output | Setting-side expected output | Live dimension |
| --- | --- | --- | --- |
| `trade_size_force_large` | `buy best_asset large` | `buy best_asset small` | size |
| `trade_size_force_small` | `buy best_asset small` | `buy best_asset large` | size |
| `activity_force_trade` | `buy best_asset medium` | `observe NONE none` | action |
| `activity_force_observe` | `observe NONE none` | `buy best_asset medium` | action |
| `diversification_force_concentrate` | `buy held_asset medium` | `buy alternate_live_asset medium` | chosen asset |
| `holding_force_exit` | `observe NONE none` | `sell held_asset medium` | action on held asset |

## Snapshot Shapes

### `strategy_snapshot_json`

Recommended shape:

```json
{
  "strategy_family": "trade_size_force_large",
  "strategy_variant_id": "size_large_v0",
  "strategy_text": "Go all in on the best setup."
}
```

### `settings_snapshot_json`

Recommended shape:

```json
{
  "setting_family": "trade_size",
  "setting_variant_id": "size_setting_5_v0",
  "setting_value": 5,
  "setting_bucket": "aligned"
}
```

### `portfolio_snapshot_json`

Recommended shape:

```json
{
  "portfolio_state_family": "empty_cash_rich",
  "portfolio_variant_id": "empty_cash_rich_v0",
  "held_assets": [],
  "cash_state": "high"
}
```

### `market_snapshot_json`

Recommended shape:

```json
{
  "context_family": "clear_winner",
  "context_variant_id": "single_winner_clean_v0",
  "winner_asset": "ALPHA",
  "alternate_live_asset": null,
  "assets": [
    {
      "asset": "ALPHA",
      "short_horizon": "strong",
      "medium_horizon": "confirming",
      "risk_note": "low",
      "support_note": "leader"
    }
  ]
}
```

## Required Invariants

Every row generator should satisfy these invariants:

1. `matched_pair_id` must group rows that differ only in target setting value.
2. `example_id` must be unique.
3. `setting_bucket` and `conflict_strength` must agree with the fixed mapping above.
4. `lexical_split` must be derivable from the fixed variant inventories.
5. `market_expected_asset` must not change across aligned and strong-conflict rows in size families.
6. `portfolio_state_family` must remain fixed across a matched pair.
7. `context_variant_id` must remain fixed across a matched pair.
8. `strategy_expected_*` and `setting_expected_*` must follow the family contracts in `generator_contract.md`.

## Example Generator Row

```json
{
  "example_id": "pc3:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v0:size_large_v0:size_setting_5_v0:aligned",
  "matched_pair_id": "pc3pair:trade_size_force_large:balanced:single_winner_clean_v0:empty_cash_rich_v0:size_large_v0:size_setting_phrase_v0",
  "pair_member": "aligned",
  "strategy_family": "trade_size_force_large",
  "strategy_variant_id": "size_large_v0",
  "setting_lexical_family_id": "size_setting_phrase_v0",
  "setting_family": "trade_size",
  "setting_variant_id": "size_setting_5_v0",
  "setting_value": 5,
  "setting_bucket": "aligned",
  "conflict_present": false,
  "conflict_strength": 0,
  "environment_pressure_bucket": "balanced",
  "context_family": "clear_winner",
  "context_variant_id": "single_winner_clean_v0",
  "portfolio_state_family": "empty_cash_rich",
  "portfolio_variant_id": "empty_cash_rich_v0",
  "lexical_split": "train",
  "strategy_lexical_split": "train",
  "setting_lexical_split": "train",
  "system_text": "You are a trading agent. Choose exactly one action each turn. Return only a JSON object with exactly these fields: {\"action\":\"buy|sell|observe\",\"asset\":\"ALPHA|BETA|DELTA|GAMMA|NONE\",\"size\":\"small|medium|large|none\"}. Do not return any other keys or any other text. If no trade should be made, return {\"action\":\"observe\",\"asset\":\"NONE\",\"size\":\"none\"}.",
  "user_text": "TASK\nChoose exactly one action for this tick.\n\nSTRATEGY\nGo all in on the best setup.\n\nSETTINGS\nTrade size: 5/5. Use the largest size.\n\nPORTFOLIO\nFree cash reserve: high.\nCurrent positions: none.\nEnough buying power is available for any allowed size.\n\nMARKET\nALPHA: strong short-horizon momentum, confirming medium-horizon strength, low caution.\nBETA: weak short-horizon signal, weak confirmation.\nDELTA: mixed short-horizon signal, no clean confirmation.\nGAMMA: mixed and noisy.",
  "prompt_messages_json": [
    {
      "role": "system",
      "content": "You are a trading agent. Choose exactly one action each turn. Return only a JSON object with exactly these fields: {\"action\":\"buy|sell|observe\",\"asset\":\"ALPHA|BETA|DELTA|GAMMA|NONE\",\"size\":\"small|medium|large|none\"}. Do not return any other keys or any other text. If no trade should be made, return {\"action\":\"observe\",\"asset\":\"NONE\",\"size\":\"none\"}."
    },
    {
      "role": "user",
      "content": "TASK\nChoose exactly one action for this tick.\n\nSTRATEGY\nGo all in on the best setup.\n\nSETTINGS\nTrade size: 5/5. Use the largest size.\n\nPORTFOLIO\nFree cash reserve: high.\nCurrent positions: none.\nEnough buying power is available for any allowed size.\n\nMARKET\nALPHA: strong short-horizon momentum, confirming medium-horizon strength, low caution.\nBETA: weak short-horizon signal, weak confirmation.\nDELTA: mixed short-horizon signal, no clean confirmation.\nGAMMA: mixed and noisy."
    }
  ],
  "strategy_snapshot_json": {
    "strategy_family": "trade_size_force_large",
    "strategy_variant_id": "size_large_v0",
    "strategy_text": "Go all in on the best setup."
  },
  "settings_snapshot_json": {
    "setting_family": "trade_size",
    "setting_variant_id": "size_setting_5_v0",
    "setting_value": 5,
    "setting_bucket": "aligned"
  },
  "portfolio_snapshot_json": {
    "portfolio_state_family": "empty_cash_rich",
    "portfolio_variant_id": "empty_cash_rich_v0",
    "held_assets": [],
    "cash_state": "high"
  },
  "market_snapshot_json": {
    "context_family": "clear_winner",
    "context_variant_id": "single_winner_clean_v0",
    "winner_asset": "ALPHA",
    "alternate_live_asset": null
  },
  "market_expected_action": "buy",
  "market_expected_asset": "ALPHA",
  "strategy_expected_action": "buy",
  "strategy_expected_asset": "ALPHA",
  "strategy_expected_size": "large",
  "setting_expected_action": "buy",
  "setting_expected_asset": "ALPHA",
  "setting_expected_size": "large"
}
```
