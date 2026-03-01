# Xenon Interp Data Prep Runbook

## Purpose
This runbook documents how `pipelines.interp.prepare` builds an interp-ready dataset from Xenon ingest data, with a focus on:

- `trade` decisions
- `record_observation` decisions
- full context blocks (messages, market, portfolio, strategy, config, memory)

The goal is to produce stable, auditable tables before any inference/probing work.

## Entry Point
Run from the project root:

```bash
uv run -m pipelines.interp.prepare --db-path data/terminal_ingest.db
```

Main module:
- `pipelines/interp/prepare.py`

## Source Tables
- `inference_logs`
- `full_logs`
- optional: `swaps` (if present)

## Output Tables

### 1) `interp_examples_v0`
Primary row-level dataset for interp planning and smoke tests.

Row key:
- `example_id = "{vault_address}:{log_id}"`

Important columns:
- identity/time: `example_id, log_id, vault_address, created_at, strategy_id, transaction_hash`
- decision: `decision_type, trade_side, action_name, asset, size, observation_text, is_trade`
- model output: `assistant_content, reasoning_content, tool_calls_json, model_source`
- context: `prompt_messages_json, tools_available_json, market_snapshot_json, portfolio_snapshot_json, strategy_snapshot_json, config_snapshot_json, memory_snapshot_json`
- joins: `joined_swap, swap_side, swap_token_address, swap_token_symbol, swap_price_usd`
- quality: `parse_ok, parse_error, has_* flags, context_complete, missing_blocks_json, label_quality, label_confidence`
- provenance: `ingest_version, transform_version, built_at`

### 2) `interp_context_gaps_v0`
Subset of `interp_examples_v0` where:
- `parse_ok = 0` OR `context_complete = 0`

Used to diagnose extraction and missing context issues.

### 3) Sample tables
- `interp_sample_trade_v0` (target size default 150)
- `interp_sample_observation_v0` (target size default 150)
- `interp_sample_paired_v0` (target size default 100)

Sampling rules:
- trade sample prioritizes high-quality buys + sells
- observation sample prioritizes high-quality observation rows
- paired sample keeps rows with opposite decision type nearby in same vault (`abs(log_id delta) <= 100`)

## Context Extraction Rules

For each full-log payload (`full_logs.payload_path`):

1. `prompt_messages_json`
- `llm_request_payload.llm_input.messages`
- fallback: `llm_request_payload.messages`

2. `tools_available_json`
- `llm_request_payload.llm_input.tools`
- fallback: `llm_request_payload.tools`

3. `market_snapshot_json`
- `snapshot.Market`
- fallback: `llm_request_payload.llm_input.snapshot.Market`

4. `portfolio_snapshot_json`
- `snapshot.Portfolio`
- fallback: `snapshot.Vault`
- fallback: `llm_request_payload.llm_input.snapshot.Portfolio`

5. `strategy_snapshot_json`
- `snapshot.Strategies`
- fallback: `llm_request_payload.llm_input.strategies`

6. `config_snapshot_json`
- `snapshot.Config`
- fallback: `snapshot.VaultConfig`
- fallback: `snapshot.Agent.Options`
- fallback: root `options`

7. `memory_snapshot_json`
- `snapshot.Memories`
- fallback: `llm_request_payload.llm_input.memories`

8. completion output
- assistant content / reasoning / tool calls from:
  - `llm_completion_payload.choices[0].message.*`
- fallback to parsed `full_logs` columns where needed

## Decision Normalization Rules

Order of precedence:
1. `tool_calls_json` from completion payload
2. fallback to `inference_logs.tool` + `inference_logs.tool_args_json`

Mapping:
- `buy_token` => `decision_type=trade`, `trade_side=buy`
- `sell_token` => `decision_type=trade`, `trade_side=sell`
- `record_observation` => `decision_type=record_observation`
- anything else => `decision_type=other`

Focus mode default:
- keep only `trade` and `record_observation`

## Join Rules (`swaps`)

If table exists:
1. match by `swaps.logId = inference_logs.id`
2. fallback `swaps.transactionHash = inference_logs.transaction_hash`

Populate:
- `joined_swap`
- `swap_side`
- `swap_token_address`
- `swap_token_symbol`
- `swap_price_usd`

## Quality Rules

Critical context blocks:
- messages
- market
- portfolio
- strategy
- config

Flags:
- `context_complete = has_messages && has_market && has_portfolio && has_strategy && has_config`
- `parse_ok = (parse_error is null/empty)`

Quality tier:
- `high`: `parse_ok && context_complete`
- `medium`: parse ok + only missing `memory` and/or `tools`
- `low`: otherwise

## Operational Notes

- Script is idempotent:
  - upserts on `example_id`
  - rebuilds gaps + sample tables each run
- Uses gzip payload files on disk; if payload file is missing/corrupt, row downgrades quality and missing blocks increase.

## Useful Commands

Build dataset:
```bash
uv run -m pipelines.interp.prepare --db-path data/terminal_ingest.db
```

Build + export JSONL files:
```bash
uv run -m pipelines.interp.prepare \
  --db-path data/terminal_ingest.db \
  --export-jsonl \
  --export-dir data/interp_exports
```

Custom sample sizes:
```bash
uv run -m pipelines.interp.prepare \
  --db-path data/terminal_ingest.db \
  --trade-sample-size 200 \
  --observation-sample-size 200 \
  --paired-sample-size 120
```

Include all decision types:
```bash
uv run -m pipelines.interp.prepare --db-path data/terminal_ingest.db --include-all-decisions
```

Quick SQL checks:
```sql
SELECT decision_type, COUNT(*) FROM interp_examples_v0 GROUP BY 1;
SELECT label_quality, COUNT(*) FROM interp_examples_v0 GROUP BY 1;
SELECT COUNT(*) FROM interp_context_gaps_v0;
```

JSONL export outputs (when `--export-jsonl` is enabled):
- `interp_examples_v0_high_quality.jsonl`
- `interp_sample_trade_v0.jsonl`
- `interp_sample_observation_v0.jsonl`
- `interp_sample_paired_v0.jsonl`
- `interp_context_gaps_v0.jsonl`

## Handoff for Future Agents

If you continue this pipeline:
1. Never remove raw context columns from `interp_examples_v0`.
2. Increment `transform_version` when extraction logic changes.
3. Keep quality rules stable; if changed, document reason in this runbook.
4. Before inference experiments, require:
   - high-quality rows only
   - explicit report of `context_complete` and `parse_ok` rates.
