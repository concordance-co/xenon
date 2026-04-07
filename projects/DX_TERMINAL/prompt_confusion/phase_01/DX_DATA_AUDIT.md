# Xenon Data Audit — 2026-03-31

Comprehensive schema, data, and activation status report for experimental protocol design.

---

## 1. Schema & Structure

### Table Inventory (exact counts)

| Table | Rows | Purpose |
|-------|------|---------|
| `inference_logs` | 203,292 | Every LLM decision — tool call, args, status, timing |
| `full_logs` | 202,867 | Complete prompt + completion + token counts + raw JSONB payload |
| `interp_examples_v0` | 195,192 | Denormalized prep output — parsed prompts, decisions, slider snapshots, trade outcomes |
| `vaults` | 1,663 | Agent configs — sliders, persona, PnL |
| `strategies` | 12,928 | Per-vault strategy text with priority levels |
| `swaps` | 34,180 | On-chain trade execution records |
| `trade_outcomes` | 33,709 | Forward-looking PnL (1h, 4h, 1d) keyed by log_id + tx_hash |
| `capture_metadata` | 11,579 | Activation capture metadata — all overlap with interp_examples |
| `synthetic_market_examples_v0` | 7,964 | Synthetic market experiment prompts |
| `synthetic_capture_metadata` | 7,154 | Activation captures on synthetic experiments |
| `research_rerun_examples` | 288 | Research rerun cohort examples |
| `counterfactual_captures` | 750 | Counterfactual experiment activation captures |
| `decision_capture_manifests` | 5 | Balanced capture manifests (cohort sampling plans) |

Additionally, there are ~50 `synthetic_market_phase*` tables (capture + context_ladder pairs) covering phases 1–16 plus actionability/policy algebra experiments.

### `inference_logs` schema

| Column | Type | Notes |
|--------|------|-------|
| `id` | BIGINT PK | |
| `vault_address` | TEXT NOT NULL | |
| `tool` | TEXT | `buy_token`, `sell_token`, `record_observation`, `completion` |
| `tool_args_json` | TEXT | JSON: `{token, spend_pct, content, strategy}` |
| `strategy_id` | TEXT | Strategy label if action was strategy-driven |
| `status` | TEXT | e.g. `EXECUTED` |
| `inference_duration_ms` | INT | |
| `transaction_hash` | TEXT | On-chain tx if executed |
| `created_at` | TEXT | ISO timestamp |
| `cursor`, `request_id`, `execution_key`, `error`, `completed_at`, `fetched_at` | TEXT | Metadata |

### `full_logs` schema

| Column | Type | Notes |
|--------|------|-------|
| `log_id` | BIGINT PK | FK to inference_logs.id |
| `vault_address` | TEXT | |
| `prompt_text` | TEXT | Full rendered prompt |
| `completion_text` | TEXT | Model output |
| `reasoning_content` | TEXT | Chain-of-thought (Qwen3 thinking tokens) — present on 169,001 records |
| `tool_calls_json` | TEXT | Structured tool call output |
| `llm_model` | TEXT | Model identifier |
| `prompt_tokens` | INT | |
| `completion_tokens` | INT | |
| `reasoning_tokens` | INT | |
| `total_tokens` | INT | |
| `raw_payload` | JSONB | Complete raw API response payload |
| `parse_error` | TEXT | NULL for all 202,867 records (zero parse failures) |
| `fetched_at` | TEXT | |

### `interp_examples_v0` schema (58 columns — key fields below)

**Identity**: `example_id` (TEXT PK, format `{vault_address}:{log_id}`), `log_id`, `vault_address`, `created_at`

**Prompt context** (extracted from full_logs payload):
- `system_text` — system prompt (~1,426 chars)
- `user_text` — full rendered user prompt (avg 29,165 chars, range 14,878–38,888)
- `prompt_messages_json`, `tools_available_json`
- `market_snapshot_json`, `portfolio_snapshot_json`, `strategy_snapshot_json`, `config_snapshot_json`, `memory_snapshot_json`

**Decision output**:
- `decision_type` — `record_observation` (173,088) or `trade` (22,104)
- `trade_side` — `buy` or `sell` (NULL for observations)
- `asset` — token symbol
- `size` — spend_pct as TEXT
- `observation_text` — reasoning for observations
- `assistant_content`, `reasoning_content`, `tool_calls_json`
- `model_source` — e.g. `qwen/Qwen3/Qwen3-235B-A22B-Thinking-2507-FP8`

**Slider snapshots** (INT, 1–5 each):
- `vault_trade_size`, `vault_trading_activity`, `vault_holding_style`, `vault_diversification`, `vault_risk_preference`

**Trade outcomes** (joined from swaps + trade_outcomes):
- `joined_swap` (BOOLEAN), `swap_side`, `swap_token_symbol`, `swap_price_usd`
- `pnl_1h_pct`, `pnl_4h_pct`, `pnl_1d_pct`, `was_profitable_1h`
- `entry_price_usd`, `entry_price_eth`

**Quality flags**:
- `parse_ok`, `context_complete` (BOOLEAN)
- `has_messages`, `has_tools`, `has_market`, `has_portfolio`, `has_strategy`, `has_config`, `has_memory` (BOOLEAN)
- `label_quality` (`high`/`low`), `label_confidence` (`high`/`low`)

### Raw inference record example

```
id:                    1139359
vault_address:         0x069A2B63aFf81Fe9c1ABF73e60B61eE4A27063CE
tool:                  record_observation
tool_args_json:        {"content":"TA=1 requires high conviction; POOPCOIN's 1h +2.49%
                        barely covers buy fee but lacks volume confirmation. Other tokens
                        still in downtrends with weak net flow. No actionable edge versus
                        4.6% round-trip cost.","strategy":""}
status:                EXECUTED
inference_duration_ms: 16542
created_at:            2026-03-01T13:28:26.807723Z
```

### Strategy text & slider storage

**Separate fields, not embedded in prompt.** In `interp_examples_v0`:
- `strategy_snapshot_json` stores the full strategy state at inference time as JSON:
  ```json
  {"vault_strategies": [
     {"strategy_id": "18",
      "content": "Take partial profits: sell 20% of position at 2% unrealized gain...",
      "enabled": true, "strategy_priority": "high",
      "expiry": 1805315241, "created_block": 43494950, ...},
     ...
   ]}
  ```
- `config_snapshot_json` stores slider values + constraints:
  ```json
  {"trade_size": 1, "slippage_bps": 300, "holding_style": 1,
   "diversification": 3, "max_trade_amount": 2500,
   "trading_activity": 2, "max_price_impact_bps": 300,
   "asset_risk_preference": 2}
  ```

Both are **also rendered into the prompt** — strategies appear under `## ACTIVE STRATEGIES (CURRENT ONLY)` and sliders under `## ACTIVE SETTINGS`.

### Agent configuration schema (`vaults` table)

| Slider | Column | Type | Range | Semantics |
|--------|--------|------|-------|-----------|
| Trade Size | `trade_size` | INT | 1–5 | Position sizing (1=5-15%, 5=60-90%) |
| Trading Activity | `trading_activity` | INT | 1–5 | Frequency ceiling (1=couple/day, 5=highly active) |
| Holding Style | `holding_style` | INT | 1–5 | Min hold duration (1=~30min, 5=days) |
| Diversification | `diversification` | INT | 1–5 | Portfolio spread (1=concentrated, 5=spread) |
| Risk Preference | `asset_risk_preference` | INT | 1–5 | Token volatility preference (1=least volatile, 5=embrace vol) |

**Slider priority hierarchy** (enforced in prompt):
1. Trading Activity — hard ceiling on frequency
2. Holding Style — hard floor on hold duration
3. Risk Preference — which tokens to consider
4. Trade Size — position sizing within constraints
5. Diversification — portfolio spread

---

## 2. Prompt Anatomy

### Full prompt template structure

The prompt is a Go template (`DX_Confusion/user.md`, 248 lines) with these sections in order:

1. **`## OPERATING RULES (READ ONCE)`** — trading costs (2.3% per side), decision hierarchy, slider priority, frequency governor (TA calibration table), market scanning heuristics, reap mechanics, decision integrity rules
2. **`## MARKET SNAPSHOT`** — per-token: price, % changes (1m/5m/1h/6h/24h/7d/all), volume, net flow, holders, unique traders, top-20 concentration
3. **`## REAPS`** — next reap time, loser/target candidates
4. **`## ACTIVE STRATEGIES (CURRENT ONLY)`** — `[priority] content` per strategy
5. **`## ACTIVE SETTINGS`** — all 5 sliders with values and semantic explanations + conditional warnings for conflicting slider combos (e.g., Hold=1+TA>=3 warning, Active+patient warning, etc.)
6. **`## PORTFOLIO CONTEXT`** — ETH balance + token positions with entry price, unrealized PnL, time held
7. **`## CONSTRAINTS / SPECIAL RULES`** — fees, max trade amount
8. **`## PRICE IMPACT LIMITS`** — per-token max buy/sell %
9. **`## PREVIOUS DECISIONS`** — recent action history (~5min per entry)
10. **`## CURRENT STATE`** — current timestamp

**System prompt** (`DX_Confusion/system.md`, 18 lines): Agent identity, one-tool-per-tick rule, decision hierarchy summary, reasoning format instructions. Two variants exist (by hash), used 140,246 and 54,946 times respectively.

### Token count distribution (from `full_logs`, n=202,867)

| Metric | Value |
|--------|-------|
| **Mean prompt tokens** | 9,578 (FP8 variant), 10,192 (other variants) |
| **p10** | 8,828 |
| **p25** | 9,223 |
| **Median (p50)** | 9,652 |
| **p75** | 10,224 |
| **p90** | 10,763 |
| **p99** | 11,846 |
| **Min** | 5,085 |
| **Max** | 12,933 |
| **Mean completion tokens** | 995 |

Prompts are **tightly distributed** — 80% fall between 9,223 and 10,224 tokens. The long tail down to ~5k is likely early-game prompts with fewer previous decisions / smaller portfolios.

### Injected context

Yes — every prompt includes:
- **Market data snapshot** (real-time prices, volumes, flows, holders)
- **Portfolio state** (ETH balance, token positions, entry prices, unrealized PnL, hold durations)
- **Strategy text** (user-authored strategy directives with priority levels)
- **Previous decisions** (recent action history as context window)
- **Reap state** (next reap countdown, candidates)
- **Price impact limits** (per-token max trade sizes)
- **Conditional slider-combo warnings** (e.g., if Hold=1 + TA>=3, warning about not selling on a timer)

---

## 3. Labels & Outcomes

### No LLM judge labels exist in the core tables

The columns `label_quality` and `label_confidence` in `interp_examples_v0` are **data completeness indicators** (high = all context blocks parsed successfully), NOT judge scores.

However, the decision capture system has **cohort labels** in several tables:
- `decision_trade_candidates_v1`, `decision_sell_candidates_v1`, `decision_blocked_observe_candidates_v1`, `decision_policy_tension_candidates_v1` — all have `cohort_label` and `label_quality` columns
- `decision_capture_manifests` has 5 balanced manifests (v1–v5) with cohort breakdowns: `buy`, `sell`, `blocked_observe`, `policy_tension_observe`
- Best manifest (`balanced_v1`) has 959 records: 300 buy, 300 sell, 159 blocked_observe, 200 policy_tension_observe across 497 unique vaults

The `research_rerun_examples` table (288 rows) has a `labels` column. The `counterfactual_snapshots` table also has `labels`.

### Label breakdown (interp_examples_v0)

| Quality | Count |
|---------|-------|
| `high` (context_complete=true) | 167,961 |
| `low` (context_complete=false) | 27,231 |

### Decision fields

| Field | Table | Description |
|-------|-------|-------------|
| `tool` | `inference_logs` | Action type: `buy_token`, `sell_token`, `record_observation` |
| `tool_args_json` | `inference_logs` | `{token, spend_pct, content, strategy}` |
| `decision_type` | `interp_examples_v0` | `trade` or `record_observation` |
| `trade_side` | `interp_examples_v0` | `buy` or `sell` |
| `asset` | `interp_examples_v0` | Token symbol chosen |
| `size` | `interp_examples_v0` | spend_pct (as TEXT) |
| `observation_text` | `interp_examples_v0` | Reasoning text for observations |

**Tool distribution across all 203,292 inference logs:**

| Tool | Count | % |
|------|-------|---|
| `record_observation` | 180,542 | 88.8% |
| `buy_token` | 13,743 | 6.8% |
| `sell_token` | 8,668 | 4.3% |
| `completion` | 339 | 0.2% |

### Confidence / metadata from generation

- **`reasoning_content`** — Qwen3's chain-of-thought (thinking tokens). Present on 169,001 of 202,867 full_logs. Contains the model's internal deliberation before tool call.
- **No explicit confidence score.** No logprobs, no self-rated confidence field.
- **`inference_duration_ms`** — wall-clock inference time.
- **`reasoning_tokens`** — count of thinking tokens (0 in FP8 variant records, likely a parsing artifact).

### Trade outcomes

**In `trade_outcomes` table**: 33,709 records with forward-looking PnL:
- All have `pnl_1h_pct`, `pnl_4h_pct`, `pnl_1d_pct`, and `was_profitable_1h`
- Average 1h PnL: +0.73%, Average 1d PnL: +0.83%

**In `interp_examples_v0`**: Of 22,104 trade decisions:
- 15,448 (69.9%) matched to on-chain swaps (`joined_swap = true`)
- All 15,448 matched trades have `pnl_1h_pct` populated
- 6,656 trades (30.1%) did NOT match to swaps — these are decisions the model made but that weren't executed on-chain

---

## 4. Activation Capture Status

### Activations DO exist — 11,579 captures on real inference data

| Metric | Value |
|--------|-------|
| Total captures | 11,579 |
| Distinct log_ids | 11,579 |
| Overlap with interp_examples | **11,579 (100%)** |
| Has router logits | 11,487 (99.2%) |
| Capture date range | 2026-03-17 to 2026-03-20 |
| Avg sequence length | 8,927 tokens |
| Avg capture time | 1.86s |
| Hidden dim | 2,048 |
| Num experts | 128 |

### Capture configurations

| Pooling | Layers Captured | Count | Notes |
|---------|----------------|-------|-------|
| `last_token` | 4 layers | 10,570 | Majority — efficient for probing |
| `none` (full seq) | 48 layers | 917 | Full residual stream |
| `none` (full seq, no router) | 48 layers | 92 | Early captures without router |

### Additional captures

| Dataset | Count |
|---------|-------|
| Synthetic market captures | 7,154 |
| Counterfactual captures | 750 |
| Research rerun examples | 288 |

### Model

**All 203,292 inference logs were generated by Qwen3-235B-A22B** (the full 235B MoE model, 22B active parameters):

| Model Variant | Count |
|---------------|-------|
| `qwen/Qwen3/Qwen3-235B-A22B-Thinking-2507-FP8` | 169,001 |
| `qwen/qwen3-235b-a22b-thinking-2507` | 17,088 |
| `Qwen/Qwen3-235B-A22B-Thinking-2507` | 16,778 |

These are the same model with different casing — all Qwen3-235B-A22B.

**For activation capture**, the codebase uses:
- **Local**: Qwen3-8B (Apple Silicon MPS)
- **Modal**: Qwen3-30B-A3B (A100-80GB) — hidden_dim=2048, 128 experts matches this model

The 11,579 captures were done on **Qwen3-30B-A3B**, not the original 235B. This is a different model's representations run on the same prompts.

---

## 5. Conflict Characteristics

### Distinct strategy texts

- **2,265 distinct strategy texts** across 12,928 strategy records (3,033 enabled)
- **3,909 distinct strategy snapshots** in `interp_examples_v0`
- 791 of 812 vaults in interp_examples have strategies (97.4%)
- 91,555 records have a non-empty `strategy_id` (46.9%)

### Strategy priority distribution (enabled strategies)

| Priority | Count |
|----------|-------|
| `high` | 2,672 |
| `med` | 326 |
| `low` | 35 |

### Top strategy texts (by vault count)

| Strategy Text | Priority | Vault Count |
|---------------|----------|-------------|
| "Trade only FEET. Never sell any position..." | high | 122 |
| "[HIGH] Observe only. No trade." | high | 74 |
| "Sell 50% of FEET. Don't buy anything ever..." | high | 41 |
| "Sell 100% of my FEET tokens..." | high | 41 |
| "Sell 50% of my Feet token..." | high | 30 |
| "Sell all of my FEET tokens..." | high | 25 |
| "Sell 20% of my FEET tokens." | high | 20 |
| "Trade 100% of my feet tokens now..." | high | 20 |
| ... (2,245+ more unique texts) | | |

Strategies are **highly diverse** — 2,265 distinct texts. FEET-related strategies dominate the top by vault count, but the long tail is enormous.

### Slider distribution (across 1,663 vaults)

| Slider | Val=1 | Val=2 | Val=3 | Val=4 | Val=5 |
|--------|-------|-------|-------|-------|-------|
| Trade Size | 615 | 144 | 512 | 124 | 268 |
| Trading Activity | 524 | 123 | 467 | 150 | 399 |
| Holding Style | 654 | 161 | 559 | 106 | 183 |
| Diversification | 786 | 171 | 536 | 85 | 85 |
| Risk Preference | 639 | 107 | 502 | 195 | 220 |

**Skewed toward 1 and 3.** Value=1 is the most common for holding_style, diversification, and risk_preference. Val=2 and Val=4 are underrepresented across all sliders.

### Slider distribution in interp_examples_v0 (195,192 rows)

All 5 values present for all 5 sliders. Good coverage:

| Slider | Val=1 | Val=2 | Val=3 | Val=4 | Val=5 |
|--------|-------|-------|-------|-------|-------|
| Trade Size | 40,773 | 38,637 | 54,913 | 24,999 | 35,870 |
| Trading Activity | 47,529 | 30,913 | 44,628 | 26,515 | 45,607 |
| Holding Style | 27,281 | 37,832 | 67,249 | 18,833 | 43,997 |
| Diversification | 48,748 | 48,536 | 63,211 | 13,939 | 20,758 |
| Risk Preference | 41,781 | 44,831 | 44,547 | 42,128 | 21,905 |

### Top slider combos in interp_examples

| Risk | Size | TA | Hold | Div | Count |
|------|------|----|------|-----|-------|
| 3 | 3 | 3 | 3 | 3 | 20,543 |
| 1 | 1 | 1 | 1 | 1 | 9,845 |
| 5 | 5 | 5 | 5 | 5 | 6,364 |
| 2 | 2 | 2 | 2 | 2 | 6,005 |
| 2 | 2 | 2 | 4 | 2 | 6,002 |
| 4 | 5 | 2 | 5 | 2 | 5,982 |
| 4 | 2 | 3 | 3 | 1 | 5,965 |
| ... | | | | | |

The `(3,3,3,3,3)` combo dominates (10.5%), but the distribution across combos is much healthier than the old DB — many vaults have mixed slider settings.

### Detected conflicts

| Conflict Type | Count | % of Trades |
|---------------|-------|-------------|
| Strategy says "don't buy" but agent traded | 10,368 | 46.9% |
| Strategy says "don't sell" but agent traded | 11,236 | 50.8% |
| Strategy says "hold all" but agent sold | 1,636 | 7.4% |
| Low trade_size (1-2) but spend_pct >= 50% | 2,077 | 9.4% |
| Strategy says "observe only" but agent traded | 1,743 | 7.9% |

**Important caveat**: These are surface-level text matches on `strategy_snapshot_json`. Many are likely **correct priority resolution** — e.g., a vault has both "don't sell X" and "sell all Y" as HIGH strategies, and the agent correctly executed the latter. The `strategy_snapshot_json` contains ALL strategies for the vault, so a "don't sell" match doesn't mean the agent violated that specific strategy. Needs deeper analysis per-strategy-id to separate true conflicts from correct multi-strategy resolution.

### Records per agent

| Metric | Value |
|--------|-------|
| Distinct vaults in interp_examples | 812 |
| Avg records per vault | 240.4 |
| Median records per vault | 13 |
| Min records per vault | 1 |
| Max records per vault | 6,018 |

**Highly skewed** — median is only 13 vs mean of 240. A small number of vaults have thousands of records (top vault: 6,018), while many have <100. The top 20 vaults each have 3,600–6,000 records.

Across all 203,292 inference_logs (819 vaults): avg 248.2 records per vault. Date range: 2026-02-26 to 2026-03-19 (21 days — full tournament).

---

## 6. Synthetic Test Feasibility

### Can you generate new inference runs with controlled strategy/slider combos?

**Yes.** The infrastructure exists and has been extensively used:

1. **Synthetic market pipeline** (`pipelines/interp/synthetic_market.py`) — already generates controlled prompts with neutral asset symbols (A, B, C, D) and parameterized market conditions. Has been run through **16+ phases** of experiments with 7,964 examples and 7,154 activation captures.

2. **Synthetic policy pipeline** (`pipelines/interp/synthetic_policy.py`) — explicitly tests preference vs permission using policy algebra. Multiple versions (v1–v4) with both capture and context_ladder tables.

3. **Counterfactual pipeline** (`pipelines/interp/counterfactual/core.py`) — generates prompt variants with swapped preambles, modified settings, and controlled strategy edits from real prompts. 750 captures completed.

4. **Research rerun pipeline** (`pipelines/interp/research_rerun/`) — re-runs cohorts with modified prompts/settings. 288 examples completed.

5. **Decision capture manifests** — 5 balanced manifests (v1–v5) with structured cohort sampling: buy/sell/blocked_observe/policy_tension_observe, max 4 per vault, grouped by asset or block_reason.

For your specific ask (fix a strategy, sweep trade_size 1→5):
- Use the counterfactual pipeline to take a real prompt, lock the strategy text, and modify `config_snapshot_json` to sweep trade_size
- Or use synthetic_market.py to generate from scratch with full control over all variables
- Both approaches are supported, tested, and have been used in prior experiments

### Latency / cost per inference

**For activation capture (Qwen3-30B-A3B on Modal):**

| Setup | Model | Hardware | Latency/Example | Cost |
|-------|-------|----------|-----------------|------|
| Modal (last_token, 4 layers) | Qwen3-30B-A3B | A100-80GB | ~1.86s | ~$0.001/example |
| Modal (full seq, 48 layers) | Qwen3-30B-A3B | A100-80GB | ~5-15s | ~$0.003/example |
| Local | Qwen3-8B | Apple Silicon (MPS) | ~10-30s | Free |

**For re-running through the original Qwen3-235B-A22B:**
- Requires multi-GPU setup (4-8x A100/H100) or API access
- The original inference used a hosted vLLM endpoint
- Estimated ~$0.01-0.03 per inference at ~10k input tokens
- At 203k records: ~$2,000-6,000 for full re-run

**For controlled sweep experiments** (e.g., 100 strategies x 5 trade_size values x 10 market conditions = 5,000 prompts):
- On Modal with Qwen3-30B-A3B: ~$5-15, ~3-8 hours
- Locally with Qwen3-8B: free, ~15-40 hours

### Key caveat

The 11,579 existing captures are from **Qwen3-30B-A3B**, not the original 235B that generated the decisions. Same prompts, different model representations. Fine for studying prompt structure effects, but the internal computations differ. If you need the original model's activations, you need 235B serving with hook access — not currently set up.

---

## Summary for Protocol Design

| Dimension | Status | Action Needed |
|-----------|--------|---------------|
| Raw inference logs | 203,292 complete with prompts + completions | Ready |
| Parsed interp examples | 195,192 (96% of total) | Ready |
| PnL labels | 15,448 trades matched with full 1h/4h/1d PnL | Ready |
| Cohort labels | 5 balanced manifests, best has 959 examples (buy/sell/blocked/tension) | Ready |
| Judge/alignment labels | None — no LLM judge has been run | Need to build or outsource |
| Activations on real data | 11,579 captures (Qwen3-30B-A3B, not original 235B) | Available; expand coverage or re-capture on 235B |
| Synthetic experiments | 7,964 examples + 7,154 captures across 16+ phases | Infrastructure ready for new experiments |
| Strategy diversity | 2,265 unique texts across 12,928 records | Rich and diverse |
| Slider coverage | All values 1-5 present, skewed toward 1 and 3 | Workable; consider balancing for new experiments |
| Conflict detection | ~10k+ flagged records | Needs per-strategy-id analysis to separate real conflicts from multi-strategy resolution |
