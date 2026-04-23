# Complaint → Root Cause Labeled Dataset

Derived from the `complaint_trace_analysis` pipeline (Mar 25, 2026 run).
Each row is a single user complaint from PostHog chat, matched to the agent's
actual behavior from Athena inference logs, and labeled by Claude Sonnet for
root cause.

Total rows: **1090**

## Label distribution

Binary-ish `label` field (the primary target):
```
  user_confusion                   577  ( 52.9%)
  true_confusion                   480  ( 44.0%)
  market                            33  (  3.0%)
```

`fault` field (the Sonnet fault assignment, 3-way):
```
  user                             577  ( 52.9%)
  system                           480  ( 44.0%)
  market                            33  (  3.0%)
```

## Label mapping

| root_cause                 | fault  | label           |
|----------------------------|--------|-----------------|
| RULE_FABRICATION           | system | true_confusion  |
| PROMPT_FAILURE             | system | true_confusion  |
| STRATEGY_SLIDER_LOCKOUT    | system | true_confusion  |
| OVERTRADING                | system | true_confusion  |
| HOLDING_VIOLATION          | system | true_confusion  |
| CHAT_AI_FABRICATION        | system | true_confusion  |
| USER_CONFIG_CONFLICT       | user   | user_confusion  |
| USER_EXPECTATION_MISMATCH  | user   | user_confusion  |
| CORRECT_BEHAVIOR           | user   | user_confusion  |
| MARKET_LEGITIMATE          | market | market          |
| STRATEGY_IGNORED           | market | market          |

- **true_confusion**: the model failed. Fabricated a rule, got locked out by its
  own frequency gates, hallucinated a cooldown, etc.
- **user_confusion**: the user set up something that couldn't work, or expected
  a feature that doesn't exist, or complained about behavior that was actually
  correct per their own config.
- **market**: legitimate memecoin loss, or an edge case outside the above.

## Raw counts

Root cause:
```
  RULE_FABRICATION                 325  ( 29.8%)
  USER_CONFIG_CONFLICT             251  ( 23.0%)
  USER_EXPECTATION_MISMATCH        191  ( 17.5%)
  CORRECT_BEHAVIOR                 135  ( 12.4%)
  PROMPT_FAILURE                   106  (  9.7%)
  MARKET_LEGITIMATE                 29  (  2.7%)
  STRATEGY_SLIDER_LOCKOUT           28  (  2.6%)
  OVERTRADING                       16  (  1.5%)
  STRATEGY_IGNORED                   4  (  0.4%)
  HOLDING_VIOLATION                  3  (  0.3%)
  CHAT_AI_FABRICATION                2  (  0.2%)
```

Complaint type (what the user was upset about, before diagnosis):
```
  GENERAL_PERFORMANCE              336  ( 30.8%)
  NOT_TRADING                      256  ( 23.5%)
  STRATEGY_IGNORED                 143  ( 13.1%)
  UNWANTED_SELL                    102  (  9.4%)
  CONFUSION                         96  (  8.8%)
  UNWANTED_BUY                      67  (  6.1%)
  WRONG_SIZE                        34  (  3.1%)
  FEATURE_EXPECTATION               31  (  2.8%)
  HOLDING_VIOLATION                 14  (  1.3%)
  OVERTRADING                       10  (  0.9%)
  UNWANTED_HOLD                      1  (  0.1%)
```

## Schema

| Column                | Type     | Description |
|-----------------------|----------|-------------|
| trace_id              | str      | Unique key (person_id:complaint_idx) |
| person_id             | str      | PostHog person UUID |
| vault_address         | str      | Agent wallet on Base |
| label                 | str      | `true_confusion` / `user_confusion` / `market` |
| fault                 | str      | `system` / `user` / `market` |
| root_cause            | str      | Fine-grained Sonnet diagnosis |
| agent_was_correct     | bool     | Sonnet's call on whether the agent did the right thing |
| severity              | int 1-5  | How bad the failure is (5 = worst) |
| confidence            | float    | Sonnet's confidence in the label (0-1) |
| urgency               | int 1-5  | How upset the user sounded |
| complaint_text        | str      | Raw user message from chat |
| complaint_type        | str      | Taxonomy of the complaint (NOT_TRADING, UNWANTED_BUY, …) |
| referenced_tokens     | list[str]| Tokens the user mentioned |
| slider_ta/arp/ts/hs/div | int 1-5 | Agent config sliders |
| has_strategy          | bool     | Whether user has written strategies |
| strategies_text       | str      | User's strategy definitions (priority-tagged) |
| n_relevant_ticks      | int      | How many inference ticks were pulled as context |
| vault_summary_json    | str      | JSON dict: total/buy/sell/observe counts, trade rate, BS ratio, top tokens |
| agent_activity        | str      | Formatted timeline of the agent's recent BUY/SELL/OBSERVE + reasoning |
| evidence_summary      | str      | Why Sonnet made this diagnosis |
| contributing_factors  | list[str]| Additional root-cause tags that played a role |
| recommended_fix       | str      | Suggested intervention |

## Formats

- `complaint_dataset.jsonl` — one JSON object per line, easiest for LLM fine-tuning or streaming
- `complaint_dataset.csv` — flat CSV; list columns stored as JSON strings
- `complaint_dataset.parquet` — zstd-compressed, preserves lists natively, fastest for pandas/polars

## Quick-start

```python
import pandas as pd
df = pd.read_parquet("complaint_dataset.parquet")
print(df["label"].value_counts())

# model failures only
true_conf = df[df["label"] == "true_confusion"]

# highest-severity rule fabrications
df[(df["root_cause"] == "RULE_FABRICATION") & (df["severity"] >= 4)]
```
