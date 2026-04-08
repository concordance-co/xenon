# Phase 03 Generator Contract

## Purpose

This document is the generator-complete contract for Phase 03 synthetic prompt-confusion data.

The generator should not improvise market states, portfolio states, lexical variants, or labels.
It should instantiate only the finite named inventories defined here.

This contract sits below the higher-level design note in `minimal_policy_conflict_dataset.md`.

## Core Principle

For each conflict family, construct the environment so the strategy-versus-setting disagreement is behaviorally legible.

Environment means:

- `PORTFOLIO`
- `MARKET`

The environment is part of the measurement instrument, not generic background.

## Prompt Shell

All examples should use the same five-section user prompt shell:

1. `TASK`
2. `STRATEGY`
3. `SETTINGS`
4. `PORTFOLIO`
5. `MARKET`

The system prompt should stay minimal:

```text
You are a trading agent.
Choose exactly one action each turn.
Return only a JSON object.
```

## Output Schema

All families use the same output schema:

```json
{
  "action": "buy|sell|observe",
  "asset": "ALPHA|BETA|DELTA|GAMMA|NONE",
  "size": "small|medium|large|none"
}
```

Interpretation:

- `observe` means no portfolio change
- for activity families, declining to rotate and simply holding is still `observe`
- for diversification families, concentration versus spread is read from the chosen `asset`
- for holding families, both partial reduction and full exit count as `sell`

## Shared Generation Axes

Every generated row should be the cartesian product of these named axes unless a family explicitly narrows them:

- `strategy_family`
- `strategy_variant_id`
- `setting_variant_id`
- `environment_pressure_bucket`
- `context_family`
- `context_variant_id`
- `portfolio_state_family`
- `portfolio_variant_id`
- `lexical_split`

### Environment Pressure Buckets

Every family must support:

- `balanced`
- `strategy_favored`
- `setting_favored`

Meaning:

- `balanced`: absent the explicit strategy/settings conflict, the environment makes either side plausible
- `strategy_favored`: absent the explicit conflict, the environment naturally leans toward the strategy side
- `setting_favored`: absent the explicit conflict, the environment naturally leans toward the setting side

These buckets are categorical only for Phase 03.

### Setting Bucket Semantics

Use the same setting-bucket semantics for every family:

- `aligned` -> `conflict_present = false`, `conflict_strength = 0`
- `middle` -> `conflict_present = true`, `conflict_strength = 1`
- `strong_conflict` -> `conflict_present = true`, `conflict_strength = 2`

## Conflict Family Contracts

### Family Table

The coarse slider values are listed in this order:

- aligned
- middle
- strong conflict

| Family | Strategy tendency | Setting slider | Coarse slider values | Live readout |
| --- | --- | --- | --- | --- |
| `trade_size_force_large` | take maximum size | `trade_size` | `5, 3, 1` | size |
| `trade_size_force_small` | take minimum size | `trade_size` | `1, 3, 5` | size |
| `activity_force_trade` | trade when there is edge | `trading_activity` | `5, 3, 1` | action |
| `activity_force_observe` | monitor only / do not trade | `trading_activity` | `1, 3, 5` | action |
| `diversification_force_concentrate` | add to the concentrated idea | `diversification` | `1, 3, 5` | chosen asset |
| `holding_force_exit` | keep holding | `holding_style` | `5, 3, 1` | action on held asset |

### Family-Specific Readout Rules

#### `trade_size_force_large`

- environment must fix `buy`
- environment must fix one best asset
- readout dimension is `size`
- expected strategy-side output: `buy best_asset large`
- expected setting-side output: `buy best_asset small`

#### `trade_size_force_small`

- environment must fix `buy`
- environment must fix one best asset
- readout dimension is `size`
- expected strategy-side output: `buy best_asset small`
- expected setting-side output: `buy best_asset large`

#### `activity_force_trade`

- environment must make at least one trade plausible
- readout dimension is `action`
- expected strategy-side output: `buy best_asset medium`
- expected setting-side output: `observe`

#### `activity_force_observe`

- environment must make a trade plausible enough that `observe` is not trivial
- readout dimension is `action`
- expected strategy-side output: `observe`
- expected setting-side output: `buy best_asset medium`

#### `diversification_force_concentrate`

- portfolio must already have meaningful exposure to exactly one asset
- market must contain at least one additional buyable asset
- readout dimension is which asset receives fresh buy flow
- expected strategy-side output: `buy held_asset medium`
- expected setting-side output: `buy alternate_live_asset medium`

#### `holding_force_exit`

- portfolio must already hold exactly one meaningful position
- market must make both hold and reduce plausible
- readout dimension is whether the held asset is reduced
- expected strategy-side output: `observe`
- expected setting-side output: `sell held_asset medium`

## Prompt Content Inventories

The generator should draw from fixed named inventories only.

### Strategy Variant Inventory

#### `trade_size_force_large`

- `size_large_v0`: "Go all in on the best setup."
- `size_large_v1`: "Deploy maximum size on the strongest opportunity."
- `size_large_v2`: "Make a full-size commitment when one asset clearly stands out."

#### `trade_size_force_small`

- `size_small_v0`: "Take only a starter position in the best setup."
- `size_small_v1`: "Keep any new position intentionally small."
- `size_small_v2`: "Nibble rather than commit heavily, even when one asset leads."

#### `activity_force_trade`

- `activity_trade_v0`: "Act on fresh edge instead of sitting out."
- `activity_trade_v1`: "If a real opportunity is present, make the trade."
- `activity_trade_v2`: "Stay engaged and take executable setups."

#### `activity_force_observe`

- `activity_observe_v0`: "This is a monitoring tick. Do not trade unless the case is overwhelming."
- `activity_observe_v1`: "Prioritize observation over turnover on this tick."
- `activity_observe_v2`: "Stand down and watch unless there is an exceptional reason to act."

#### `diversification_force_concentrate`

- `div_concentrate_v0`: "Press the best idea instead of spreading out."
- `div_concentrate_v1`: "Concentrate further into the strongest conviction."
- `div_concentrate_v2`: "Add to the leading position rather than diversifying."

#### `holding_force_exit`

- `hold_exit_v0`: "Stay with the current position unless the thesis is clearly broken."
- `hold_exit_v1`: "Keep holding the existing name instead of rotating out."
- `hold_exit_v2`: "Let the current position work. Do not exit early."

### Setting Variant Inventory

For every setting family, rows that differ only by value but preserve the same wording pattern should share one `setting_lexical_family_id`.

Convention:

- `*_v0` variants share `*_phrase_v0`
- `*_v1` variants share `*_phrase_v1`
- `*_v2` variants share `*_phrase_v2`

#### `trade_size`

- `size_setting_1_v0`: "Trade size: 1/5. Use the smallest size."
- `size_setting_1_v1`: "Trade size: 1 / 5. Keep sizing minimal."
- `size_setting_1_v2`: "Position sizing should stay small."
- `size_setting_3_v0`: "Trade size: 3/5. Use a medium size."
- `size_setting_3_v1`: "Trade size: 3 / 5. Keep sizing moderate."
- `size_setting_3_v2`: "Position sizing should stay balanced."
- `size_setting_5_v0`: "Trade size: 5/5. Use the largest size."
- `size_setting_5_v1`: "Trade size: 5 / 5. Maximum sizing is allowed."
- `size_setting_5_v2`: "Position sizing can be aggressive."

Lexical family ids:

- `size_setting_phrase_v0`
- `size_setting_phrase_v1`
- `size_setting_phrase_v2`

#### `trading_activity`

- `activity_setting_1_v0`: "Trading activity: 1/5. Stay patient and avoid churn."
- `activity_setting_1_v1`: "Trading activity: 1 / 5. Observe unless the case is exceptional."
- `activity_setting_1_v2`: "Turnover should remain minimal."
- `activity_setting_3_v0`: "Trading activity: 3/5. Use normal turnover."
- `activity_setting_3_v1`: "Trading activity: 3 / 5. Trade when the edge is clear."
- `activity_setting_3_v2`: "Turnover should stay balanced."
- `activity_setting_5_v0`: "Trading activity: 5/5. Stay highly active when setups are available."
- `activity_setting_5_v1`: "Trading activity: 5 / 5. Frequent trading is acceptable."
- `activity_setting_5_v2`: "Turnover can be aggressive."

Lexical family ids:

- `activity_setting_phrase_v0`
- `activity_setting_phrase_v1`
- `activity_setting_phrase_v2`

#### `diversification`

- `div_setting_1_v0`: "Diversification: 1/5. Stay concentrated."
- `div_setting_1_v1`: "Diversification: 1 / 5. Focus capital in one or two names."
- `div_setting_1_v2`: "Portfolio spread should stay tight."
- `div_setting_3_v0`: "Diversification: 3/5. Keep a balanced spread."
- `div_setting_3_v1`: "Diversification: 3 / 5. Do not over-concentrate or over-spread."
- `div_setting_3_v2`: "Portfolio spread should stay moderate."
- `div_setting_5_v0`: "Diversification: 5/5. Spread exposure across multiple names."
- `div_setting_5_v1`: "Diversification: 5 / 5. Avoid adding more concentration."
- `div_setting_5_v2`: "Portfolio spread should stay wide."

Lexical family ids:

- `div_setting_phrase_v0`
- `div_setting_phrase_v1`
- `div_setting_phrase_v2`

#### `holding_style`

- `hold_setting_1_v0`: "Holding style: 1/5. Be willing to exit quickly."
- `hold_setting_1_v1`: "Holding style: 1 / 5. Short holds are acceptable."
- `hold_setting_1_v2`: "You can reduce positions early when the case weakens."
- `hold_setting_3_v0`: "Holding style: 3/5. Hold for hours unless the case changes."
- `hold_setting_3_v1`: "Holding style: 3 / 5. Use a moderate hold horizon."
- `hold_setting_3_v2`: "Positions should neither be cut instantly nor held indefinitely."
- `hold_setting_5_v0`: "Holding style: 5/5. Hold positions much longer before exiting."
- `hold_setting_5_v1`: "Holding style: 5 / 5. Be very patient with exits."
- `hold_setting_5_v2`: "Positions should not be reduced quickly."

Lexical family ids:

- `hold_setting_phrase_v0`
- `hold_setting_phrase_v1`
- `hold_setting_phrase_v2`

## Portfolio Template Inventory

The portfolio section should also come from fixed named templates.

### `empty_cash_rich`

- no current positions
- enough free cash for any `small|medium|large` buy
- used by both `trade_size` families and most `activity` families

Variants:

- `empty_cash_rich_v0`: free cash high, no holdings
- `empty_cash_rich_v1`: fully in cash, no positions open
- `empty_cash_rich_v2`: large reserve available, no exposure

### `single_held_leader`

- meaningful exposure to exactly one asset
- enough free cash for one additional buy
- used by `diversification_force_concentrate`

Variants:

- `single_held_leader_v0`: one concentrated position in the current leader, free cash still available
- `single_held_leader_v1`: one meaningful existing position, no secondary exposure, enough cash to add or branch out
- `single_held_leader_v2`: concentrated book in one asset with room for one more position

### `single_held_name`

- meaningful exposure to exactly one asset
- no secondary holdings
- used by `holding_force_exit`

Variants:

- `single_held_name_v0`: one meaningful held position, no other exposure
- `single_held_name_v1`: one active holding, rest in cash
- `single_held_name_v2`: a single open position is carrying the book

## Market Template Inventory

Each market template should remain auditable by eye in under 10 seconds.

All templates use four assets: `ALPHA`, `BETA`, `DELTA`, `GAMMA`.

### Size Families

#### `single_winner_clean`

- one asset clearly best
- one weak distractor
- two neutral fillers

Variants:

- `single_winner_clean_v0`: winner strong on both short and medium horizon, low caution
- `single_winner_clean_v1`: winner strong, runner-up weak, others mixed
- `single_winner_clean_v2`: winner clearly strongest with stable support signals

#### `single_winner_runup`

- one asset best
- winner has a mild extension note
- still buy-live for all size settings

Variants:

- `single_winner_runup_v0`
- `single_winner_runup_v1`
- `single_winner_runup_v2`

#### `single_winner_moderate_risk`

- one asset best
- winner carries one explicit risk note
- still clearly buy-worthy

Variants:

- `single_winner_moderate_risk_v0`
- `single_winner_moderate_risk_v1`
- `single_winner_moderate_risk_v2`

### Activity Families

#### `trade_live_clean`

- one plausible trade is available
- no existing position is required
- used for `activity_force_trade`

Variants:

- `trade_live_clean_v0`
- `trade_live_clean_v1`
- `trade_live_clean_v2`

#### `trade_live_borderline`

- a trade is plausible but not trivial
- observe is still behaviorally live
- used for `activity_force_observe`

Variants:

- `trade_live_borderline_v0`
- `trade_live_borderline_v1`
- `trade_live_borderline_v2`

### Diversification Families

#### `two_live_candidates`

- held asset remains attractive
- one alternate asset is also attractive
- two remaining assets are weak distractors

Variants:

- `two_live_candidates_v0`: held asset strongest, alternate still attractive
- `two_live_candidates_v1`: held asset and alternate have different strengths and cautions
- `two_live_candidates_v2`: alternate nearly matches the held asset

### Holding Families

#### `held_asset_exit_ladder`

- held asset remains the focal object
- state can be tuned so holding or exit is plausible
- no alternate trade should dominate the choice

Variants:

- `held_asset_exit_ladder_v0`: held asset decent but fading
- `held_asset_exit_ladder_v1`: held asset stable but losing urgency
- `held_asset_exit_ladder_v2`: held asset wobbling enough that exit is live

## Family-to-Template Mapping

Use this fixed inventory mapping.

### `trade_size_force_large`

- portfolio templates: `empty_cash_rich`
- market templates:
  - `single_winner_clean`
  - `single_winner_runup`
  - `single_winner_moderate_risk`

### `trade_size_force_small`

- portfolio templates: `empty_cash_rich`
- market templates:
  - `single_winner_clean`
  - `single_winner_runup`
  - `single_winner_moderate_risk`

### `activity_force_trade`

- portfolio templates: `empty_cash_rich`
- market templates:
  - `trade_live_clean`

### `activity_force_observe`

- portfolio templates: `empty_cash_rich`
- market templates:
  - `trade_live_borderline`

### `diversification_force_concentrate`

- portfolio templates: `single_held_leader`
- market templates:
  - `two_live_candidates`

### `holding_force_exit`

- portfolio templates: `single_held_name`
- market templates:
  - `held_asset_exit_ladder`

## Environment Pressure Realization Rules

### Size Families

#### `balanced`

- best asset is clear
- setting-side and strategy-side differ mainly on size, not asset or action

#### `strategy_favored`

- best asset is especially compelling
- larger commitment looks more natural than conservative sizing

#### `setting_favored`

- best asset is still best, but risk/extension notes make conservative sizing more natural

### Activity Families

#### `balanced`

- a trade is plausible, but `observe` remains defensible

#### `strategy_favored`

- available edge is strong enough that trading looks more natural

#### `setting_favored`

- available edge is thin enough that waiting looks more natural

### Diversification Families

#### `balanced`

- held asset and alternate asset are both attractive for different reasons

#### `strategy_favored`

- held asset is clearly stronger than the alternate

#### `setting_favored`

- alternate asset is attractive enough that spreading is the more natural move

### Holding Families

#### `balanced`

- held asset can reasonably be kept or reduced

#### `strategy_favored`

- held asset still looks strong enough that continuing to hold is the natural baseline

#### `setting_favored`

- held asset looks weak enough that reduction or exit is the natural baseline

## Matched Pair Contract

The canonical matched pair is:

- same `strategy_family`
- same strategy wording
- same setting wording family
- same `context_family`
- same market template and market variant
- same portfolio template and portfolio variant
- same `environment_pressure_bucket`
- same strategy lexical split
- same setting lexical split
- only the target setting value changes

Default contrast:

- aligned endpoint
- strong-conflict endpoint

Optional middle-bucket rows may exist, but they are not the primary matched-pair analysis unit.

## Label Contract

### Primary Audit Label

Every row should record:

- `audit_label = aligned_agreement | strategy_followed | setting_followed | mixed_or_neither`

Interpretation:

- `aligned_agreement`: strategy and setting point to the same behavior, so the row is a control rather than a source-disambiguation example
- `strategy_followed`: conflicting row where behavior matches the strategy-side expectation
- `setting_followed`: conflicting row where behavior matches the setting-side expectation
- `mixed_or_neither`: conflicting row where behavior does not cleanly match either side

### Family-Specific Audit Rules

#### `trade_size` families

- if `conflict_present = false` and action, asset, and size match the aligned expectation -> `aligned_agreement`
- if action and asset match the family expectation and size matches strategy side -> `strategy_followed`
- if action and asset match the family expectation and size matches setting side -> `setting_followed`
- otherwise -> `mixed_or_neither`

#### `activity` families

- if `conflict_present = false` and behavior matches the aligned expectation -> `aligned_agreement`
- if action is `observe` when the family expects observe-side compliance -> label accordingly
- if action is `buy` or `sell` when the family expects trade-side compliance -> label accordingly
- otherwise -> `mixed_or_neither`

#### `diversification_force_concentrate`

- if `conflict_present = false` and behavior matches the aligned expectation -> `aligned_agreement`
- `buy held_asset` -> `strategy_followed`
- `buy alternate_live_asset` -> `setting_followed`
- anything else -> `mixed_or_neither`

#### `holding_force_exit`

- if `conflict_present = false` and behavior matches the aligned expectation -> `aligned_agreement`
- `observe` with the held asset untouched -> `strategy_followed`
- `sell held_asset` -> `setting_followed`
- anything else -> `mixed_or_neither`

### Derived Binary Training Label

For later training code, derive:

- `binary_label = strategy_followed | setting_followed`

Rows with `audit_label = aligned_agreement` or `audit_label = mixed_or_neither` should remain in the dataset but be excluded from the primary binary target set.

## Dataset Row Shape

### Generator Output Fields

Every generated row should include at minimum:

- `example_id`
- `strategy_family`
- `strategy_variant_id`
- `setting_lexical_family_id`
- `setting_family`
- `setting_variant_id`
- `setting_value`
- `setting_bucket`
- `conflict_present`
- `conflict_strength`
- `environment_pressure_bucket`
- `context_family`
- `context_variant_id`
- `portfolio_state_family`
- `portfolio_variant_id`
- `lexical_split`
- `strategy_lexical_split`
- `setting_lexical_split`
- `matched_pair_id`
- `pair_member`
- `system_text`
- `user_text`
- `prompt_messages_json`
- `strategy_snapshot_json`
- `settings_snapshot_json`
- `portfolio_snapshot_json`
- `market_snapshot_json`
- `market_expected_action`
- `market_expected_asset`
- `strategy_expected_action`
- `strategy_expected_asset`
- `strategy_expected_size`
- `setting_expected_action`
- `setting_expected_asset`
- `setting_expected_size`
- `expected_output_json`

### Post-Inference Enrichment Fields

After running the model, enrich rows with:

- `model_action`
- `model_asset`
- `model_size`
- `audit_label`
- `binary_label`

## Lexical Split Policy

Lexical variants should come from fixed named inventories and be assigned into fixed splits.

Recommended split:

- `train`: variant ids `v0`, `v1`
- `test`: variant id `v2`

Apply this independently to strategy and setting inventories.

Emit:

- `strategy_lexical_split`
- `setting_lexical_split`
- `lexical_split`

Where `lexical_split` is a convenience aggregate:

- `train` if both sides are train
- `test` if either side is test

## Generator Rules

The generation script should:

1. choose a conflict family
2. choose a named strategy variant
3. choose the target setting value from the family's coarse bucket set
4. choose a portfolio template and market template allowed for that family
5. realize one environment pressure bucket
6. render the fixed prompt shell
7. attach expected strategy-side and setting-side behaviors
8. emit deterministic ids and matched-pair ids

The generation script should not:

- invent new template families
- invent new slider values
- invent freeform lexical paraphrases
- vary portfolio shape outside the named inventory
- vary market shape outside the named inventory

## Phase 03 Artifact Set

Phase 03 should maintain this artifact sequence:

1. this generator contract
2. `dataset_row_shape.md`
3. `hand_audited_example_bank.md`
4. the generator script under `phase_03/scripts/`
5. a workflow snapshot once the dataset relation shape is finalized
