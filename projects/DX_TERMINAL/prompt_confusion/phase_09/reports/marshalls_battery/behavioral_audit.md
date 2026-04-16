# Phase 09 Behavioral Audit (Marshall's Battery)

Total parsed generations: **864**

## Overall outcome distribution

| Outcome | Count | Share |
|---|---|---|
| `aligned_match` | 415 | 48.0% |
| `aligned_mismatch` | 17 | 2.0% |
| `follow_strategy` | 96 | 11.1% |
| `follow_setting` | 332 | 38.4% |
| `refuse` | 4 | 0.5% |

## By (target_dimension, conflict_band)

| Dimension | Band | Outcome | Count | Share of cell |
|---|---|---|---|---|
| trade_size | aligned | `aligned_match` | 192 | 100.0% |
| trade_size | strong_conflict | `follow_strategy` | 1 | 0.5% |
| trade_size | strong_conflict | `follow_setting` | 187 | 97.4% |
| trade_size | strong_conflict | `refuse` | 4 | 2.1% |
| trading_activity | aligned | `aligned_match` | 223 | 92.9% |
| trading_activity | aligned | `aligned_mismatch` | 17 | 7.1% |
| trading_activity | strong_conflict | `follow_strategy` | 95 | 39.6% |
| trading_activity | strong_conflict | `follow_setting` | 145 | 60.4% |

## Aligned correctness by (dimension, strategy_direction, setting_value)

If an aligned cell shows <90% aligned_match, the row design is muddy there.

| Dimension | strategy_direction | setting_value | aligned_match / total | Rate |
|---|---|---|---|---|
| trade_size | large | 5 | 96/96 | 100% |
| trade_size | small | 1 | 96/96 | 100% |
| trading_activity | observe | 1 | 96/96 | 100% |
| trading_activity | trade | 1 | 32/48 | 67% |
| trading_activity | trade | 5 | 95/96 | 99% |

## Variant x variant resolution (conflict rows, follow_setting rate)

If one variant pair dominates with follow_setting ~= 100%, that's the Phase 06 v4 pattern and signals variant-wording authority asymmetry.

### trade_size

| strategy_variant | settings_variant | follow_setting / non-refuse | rate | refuse |
|---|---|---|---|---|
| policy_v0 | settings_v0 | 12/12 | 100% | 0 |
| policy_v0 | settings_v1 | 12/12 | 100% | 0 |
| policy_v0 | settings_v2 | 12/12 | 100% | 0 |
| policy_v0 | settings_v3 | 12/12 | 100% | 0 |
| policy_v1 | settings_v0 | 12/12 | 100% | 0 |
| policy_v1 | settings_v1 | 12/12 | 100% | 0 |
| policy_v1 | settings_v2 | 12/12 | 100% | 0 |
| policy_v1 | settings_v3 | 12/12 | 100% | 0 |
| policy_v2 | settings_v0 | 12/12 | 100% | 0 |
| policy_v2 | settings_v1 | 12/12 | 100% | 0 |
| policy_v2 | settings_v2 | 11/12 | 92% | 0 |
| policy_v2 | settings_v3 | 12/12 | 100% | 0 |
| policy_v3 | settings_v0 | 11/11 | 100% | 1 |
| policy_v3 | settings_v1 | 11/11 | 100% | 1 |
| policy_v3 | settings_v2 | 11/11 | 100% | 1 |
| policy_v3 | settings_v3 | 11/11 | 100% | 1 |

### trading_activity

| strategy_variant | settings_variant | follow_setting / non-refuse | rate | refuse |
|---|---|---|---|---|
| policy_v0 | settings_v0 | 10/15 | 67% | 0 |
| policy_v0 | settings_v1 | 10/15 | 67% | 0 |
| policy_v0 | settings_v2 | 10/15 | 67% | 0 |
| policy_v0 | settings_v3 | 9/15 | 60% | 0 |
| policy_v1 | settings_v0 | 8/15 | 53% | 0 |
| policy_v1 | settings_v1 | 9/15 | 60% | 0 |
| policy_v1 | settings_v2 | 7/15 | 47% | 0 |
| policy_v1 | settings_v3 | 10/15 | 67% | 0 |
| policy_v2 | settings_v0 | 10/15 | 67% | 0 |
| policy_v2 | settings_v1 | 8/15 | 53% | 0 |
| policy_v2 | settings_v2 | 9/15 | 60% | 0 |
| policy_v2 | settings_v3 | 10/15 | 67% | 0 |
| policy_v3 | settings_v0 | 7/15 | 47% | 0 |
| policy_v3 | settings_v1 | 10/15 | 67% | 0 |
| policy_v3 | settings_v2 | 9/15 | 60% | 0 |
| policy_v3 | settings_v3 | 9/15 | 60% | 0 |
