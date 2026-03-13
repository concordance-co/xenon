# Xenon Master Context

Single-file system brief for agent handoff, analysis design, and probe-planning work.

This file is intentionally optimized for:
- fast orientation by another agent
- exact current data structures and capabilities
- clear separation between `current` and `planned`

Use this as the first file to read. Use [ARCHITECTURE.md](/Users/marshallvyletel/repos/concordance/xenon/ARCHITECTURE.md) for deeper ops detail and [PROBE_DATASET_BUILDER_PLAN.md](/Users/marshallvyletel/repos/concordance/xenon/PROBE_DATASET_BUILDER_PLAN.md) for the next major probe-dataset workflow.

## 1. Project purpose

Xenon is a mech-interp pipeline built around Terminal Markets inference logs.

Core idea:
- ingest real production trading-model decisions and payloads
- transform them into probe-ready structured examples
- replay prompts through smaller Qwen3 models
- capture residual stream and MoE router activations
- analyze whether internal activations linearly encode trading-relevant concepts

Competition model:
- `Qwen3-235B-A22B`
- MoE
- 128 experts
- top-8 routing

Primary capture model today:
- `Qwen3-30B-A3B`
- MoE
- 48 layers
- hidden dim `2048`
- 128 experts
- top-8 routing

Local validation model:
- `Qwen3-8B`
- dense
- 36 layers
- hidden dim `4096`

## 2. Canonical environment and truth sources

Production truth:
- Modal volume `xenon-data`
- SQLite DB at `/data/ingest/terminal_ingest.db`
- backend API in [pipelines/backend/app.py](/Users/marshallvyletel/repos/concordance/xenon/pipelines/backend/app.py)

Local copies:
- `data/terminal_ingest.db`
- `data/full_logs/`
- `data/interp_exports/`
- `data/activations/`

Rule:
- for production counts and live inspection, prefer the deployed backend over local SQLite
- local DB is valid for rebuild/recovery workflows and local experimentation before upload

## 3. System topology

Three layers:
1. Pipeline scripts and Python modules
2. Local dashboard server + React UI
3. Modal backend API for read-only inspection of the production DB

Current UI surfaces:
- `Pipeline`
- `Explorer`

Current Explorer workspaces:
- `Query Lab`
- `Label Lab`
- `Probe Prep`

Important safety rule already in effect:
- Explorer/backend SQL paths are read-only
- prep-target persistence writes to JSON metadata, not SQLite

## 4. Data stores and exact paths

Modal volume paths:
- `/data/ingest/terminal_ingest.db`
- `/data/ingest/full_logs/.../*.json.gz`
- `/data/interp_exports/*.parquet`
- `/data/activations/`
- `/data/dashboard_stats.json`
- `/data/explorer/prep_target_specs.json`

Local analogues:
- `data/terminal_ingest.db`
- `data/full_logs/`
- `data/interp_exports/`
- `data/activations/`
- `data/dashboard_stats.json`

Backup-related Modal paths:
- `/data/ingest/db_backups/`
- `/data/ingest/repair_backups/`
- `/data/ingest/rebuild_backups/`

`upload-db` semantics:
- overwrites only `ingest/terminal_ingest.db`
- does not modify backup directories

## 5. Raw ingest database schema

Primary ingest tables:
- `vaults`
- `strategies`
- `inference_logs`
- `full_logs`
- `swaps`
- `trade_outcomes`
- `ingest_cursors`

### `vaults`

Key fields:
- `vault_address` primary key
- `owner_address`
- `nft_id`
- `nft_name`
- `persona_json`
- `trade_size`
- `trading_activity`
- `holding_style`
- `diversification`
- `asset_risk_preference`
- `max_trade_amount`
- `slippage_bps`
- `paused`
- `state`
- `leaderboard_rank`
- `total_pnl_usd`
- `realized_pnl_usd`
- `unrealized_pnl_usd`
- `fetched_at`

### `strategies`

Key fields:
- `vault_address`
- `strategy_id`
- `content`
- `enabled`
- `strategy_priority`

Composite identity:
- `vault_address`
- `strategy_id`

### `inference_logs`

One row per model decision.

Key fields:
- `id`
- `vault_address`
- `tool`
- `tool_args_json`
- `strategy_id`
- `status`
- `inference_duration_ms`
- `transaction_hash`
- `created_at`

Important semantics:
- `tool` is usually one of `buy_token`, `sell_token`, `record_observation`
- this table does not itself contain the full prompt/completion payload

### `full_logs`

One row per inference log with parsed full payload metadata.

Key fields:
- `log_id`
- `vault_address`
- `payload_path`
- `payload_sha256`
- `payload_size_bytes`
- `prompt_text`
- `completion_text`
- `reasoning_content`
- `tool_calls_json`
- `llm_model`
- `prompt_tokens`
- `completion_tokens`
- `reasoning_tokens`
- `total_tokens`
- `parse_error`

Important semantics:
- `payload_path` points to gzipped JSON under `full_logs/`
- full payload is stored as file, not necessarily inline in SQLite

### `swaps`

On-chain executions associated to decisions when available.

Key fields:
- `transaction_hash`
- `log_index`
- `vault_address`
- `side`
- `token_address`
- `token_symbol`
- `effective_price_usd`
- `log_id`
- `timestamp`

Important semantics:
- `log_id` can be null or can point to a log not present in current `inference_logs`
- this is why `swaps` counts can exceed the subset of probe-usable inference examples

### `trade_outcomes`

Forward-looking label enrichment from candle data.

Key fields:
- `log_id`
- `side`
- `token_address`
- `entry_price_usd`
- `pnl_1h_pct`
- `pnl_4h_pct`
- `pnl_1d_pct`
- `was_profitable_1h`

Important semantics:
- outcomes are enrichment over swaps
- outcomes runs are resumable and process unlabeled swaps

## 6. Full-log payload structure

Each gzipped JSON payload in `full_logs/{shard}/{log_id}.json.gz` contains roughly:

```json
{
  "snapshot": {
    "Agent": {},
    "Portfolio": {},
    "Market": {},
    "AllowedTools": [],
    "Memories": []
  },
  "llm_request_payload": {
    "llm_input": {
      "messages": [],
      "tools": []
    }
  },
  "llm_completion_payload": {
    "choices": [
      {
        "message": {
          "content": "...",
          "reasoning_content": "...",
          "tool_calls": []
        }
      }
    ],
    "usage": {
      "prompt_tokens": 0,
      "completion_tokens": 0,
      "reasoning_tokens": 0,
      "total_tokens": 0
    }
  }
}
```

This is the source of truth for:
- prompt messages
- tool definitions
- market snapshot
- portfolio snapshot
- strategy snapshot
- config snapshot
- memory snapshot
- assistant content
- reasoning content
- tool calls

## 7. Prepared probe dataset schema

The main prep table is `interp_examples_v0`, built by [pipelines/interp/prepare.py](/Users/marshallvyletel/repos/concordance/xenon/pipelines/interp/prepare.py).

This is the most important table for probe design.

Exact stored fields:
- `example_id`
- `log_id`
- `vault_address`
- `created_at`
- `strategy_id`
- `transaction_hash`
- `is_trade`
- `prompt_messages_json`
- `system_text`
- `user_text`
- `tools_available_json`
- `market_snapshot_json`
- `portfolio_snapshot_json`
- `strategy_snapshot_json`
- `config_snapshot_json`
- `memory_snapshot_json`
- `model_source`
- `assistant_content`
- `reasoning_content`
- `tool_calls_json`
- `action_name`
- `decision_type`
- `trade_side`
- `asset`
- `size`
- `observation_text`
- `joined_swap`
- `swap_side`
- `swap_token_address`
- `swap_token_symbol`
- `swap_price_usd`
- `pnl_1h_pct`
- `pnl_4h_pct`
- `pnl_1d_pct`
- `was_profitable_1h`
- `entry_price_usd`
- `entry_price_eth`
- `vault_trade_size`
- `vault_trading_activity`
- `vault_holding_style`
- `vault_diversification`
- `vault_risk_preference`
- `parse_ok`
- `parse_error`
- `has_messages`
- `has_tools`
- `has_market`
- `has_portfolio`
- `has_strategy`
- `has_config`
- `has_memory`
- `context_complete`
- `missing_blocks_json`
- `label_quality`
- `label_confidence`
- `ingest_version`
- `transform_version`
- `built_at`

Important derived semantics:
- `decision_type` is normalized to `trade` or `record_observation`
- `trade_side` is normalized to `buy` or `sell` for trade rows
- `label_quality` is one of `high`, `medium`, `low`
- `context_complete` is the key probe-readiness gate
- `vault_*` fields expose stable control variables from vault config

Current prep outputs:
- `interp_examples_v0`
- `interp_context_gaps_v0`
- `interp_sample_trade_v0`
- `interp_sample_observation_v0`
- `interp_sample_paired_v0`

Current main parquet:
- `data/interp_exports/interp_examples_v0_high_quality.parquet`

Important note:
- default probe/analysis flow still expects a labels parquet path, not a versioned probe-dataset artifact flow

## 8. Quality model

Prep quality tiers:
- `high`: parse OK and critical context present
- `medium`: parse OK but memory/tools may be missing
- `low`: parse failure or critical missing context

Critical blocks for `context_complete`:
- messages
- market
- portfolio
- strategy
- config

Non-critical but tracked:
- memory
- tools

## 9. Activation capture artifacts

Capture output root:
- `data/activations/`

Subdirs:
- `residual_stream/`
- `router_logits/`
- `compact/`

Metadata file:
- `data/activations/metadata.parquet`

Metadata columns currently written:
- `log_id`
- `seq_len`
- `num_layers_captured`
- `hidden_dim`
- `has_router`
- `num_experts`
- `prompt_hash`
- `file_size_bytes`
- `elapsed_s`
- `captured_layers`
- `pooling`
- `capture_timestamp`

Residual tensor format:
- file: `residual_stream/{log_id}.safetensors`
- key: `residual_stream`
- shape: `(layers, seq_len, hidden_dim)` or `(layers, hidden_dim)` if pooled on capture
- dtype: `float16`

Router tensor format:
- file: `router_logits/{log_id}.safetensors`
- key: `router_logits`
- shape: `(layers, seq_len, 128)` or `(layers, 128)` if pooled on capture
- dtype: `float32` in classic capture path, configurable in vLLM path
- key: `router_indices`
- shape: `(layers, seq_len, 8)` or pooled equivalent
- dtype: `int16`

Current supported pooling modes:
- `last_token`
- `mean_pool`

Current supported sources:
- `router`
- `residual`

## 10. Current analysis capabilities

Implemented analysis entrypoint:
- [pipelines/interp/analysis.py](/Users/marshallvyletel/repos/concordance/xenon/pipelines/interp/analysis.py)

Supported modes:
- `probe`
- `experts`
- `pca`
- `all`
- `compact`

Supported probe targets:
- `decision_type`
- `trade_side`
- `was_profitable_1h`
- `risk_tolerance`
- `asset`

Current target semantics:
- `decision_type`: binary `record_observation` vs `trade`
- `trade_side`: binary `sell` vs `buy` on trade rows only
- `was_profitable_1h`: binary on trade rows with outcomes only
- `risk_tolerance`: 3-bin label from `vault_risk_preference`
  - `1,2 -> low`
  - `3 -> mid`
  - `4,5 -> high`
- `asset`: multiclass over non-null `asset`

Current analysis algorithm details:
- probe mode uses linear `SGDClassifier`
- evaluation is cross-validation, not explicit train/val/test split artifacts
- baseline metrics include majority baseline and shuffled-label baseline
- outputs include `accuracy_mean`, `accuracy_std`, `balanced_accuracy`, `selectivity`

Probe result file:
- `data/analysis_results/probe_{target}_{data_source}.parquet`

Probe result columns:
- `layer`
- `accuracy_mean`
- `accuracy_std`
- `balanced_accuracy`
- `baseline_majority`
- `baseline_shuffled`
- `selectivity`
- `n_examples`
- `n_classes`

Expert analysis:
- uses `router_indices`
- computes per-class expert frequency
- binary targets use Cohen's d
- multiclass targets use max pairwise frequency gap
- saves top 10 discriminative experts per layer

Expert result file:
- `data/analysis_results/expert_specialization.parquet`

Expert result columns:
- `layer`
- `expert_id`
- `rank`
- `discriminative_score`
- plus `freq_class_*` columns

PCA outputs:
- `pca_layer{layer}_{target}.png`
- `lda_layer{layer}_{target}.png`
- `dim_layer{layer}_{target}.png` for binary targets
- `centroid_pca_layer{layer}_{target}.png` for multiclass targets

Compaction outputs:
- `data/activations/compact/{source}_{pooling}_layer{N}.safetensors`

Compact tensors:
- `features`
- `log_ids`
- optional `router_indices`

Important limitations of current analysis:
- no explicit split-aware train/val/test workflow yet
- no versioned probe-dataset artifact manager yet
- no built-in controlled-variable filtering UI for analysis runs
- no continuous regression targets yet
- no dataset manifest system yet

## 11. Current backend API capabilities

Implemented in [pipelines/backend/app.py](/Users/marshallvyletel/repos/concordance/xenon/pipelines/backend/app.py).

Read-only exploration endpoints:
- `POST /query`
- `GET /schema`
- `GET /tables`
- `GET /sample/{table}`
- `GET /stats`
- `GET /parquet/list`
- `GET /parquet/info/{name}`
- `GET /parquet/sample/{name}`
- `GET /activations/meta`

Explorer 2.0 endpoints already present:
- `POST /profile/dataset`
- `POST /label/preview`
- `GET /prep-targets`
- `GET /prep-targets/{spec_id}`
- `POST /prep-targets`
- `DELETE /prep-targets/{spec_id}`

Backend safety behavior:
- SQL must start with `SELECT`, `PRAGMA`, `EXPLAIN`, or `WITH`
- mutating SQL fragments are rejected
- DB connections are read-only
- prep-target persistence writes to `/data/explorer/prep_target_specs.json`

## 12. Current shared prep-target spec shape

Frontend/backend contract is represented in [pipelines/dashboard-ui/src/types/api.ts](/Users/marshallvyletel/repos/concordance/xenon/pipelines/dashboard-ui/src/types/api.ts).

```ts
type PrepTargetSpec = {
  id?: string
  name: string
  description?: string
  source: { mode: "table" | "sql"; table?: string; sql?: string }
  filters?: { sql_where?: string }
  label: {
    mode: "direct" | "binary_rule" | "bucket"
    expression_sql: string
    classes?: string[]
    buckets?: { name: string; min?: number; max?: number }[]
  }
  split: {
    mode: "random_stratified" | "time_based" | "group_holdout"
    train_pct: number
    val_pct: number
    test_pct: number
    group_key?: string
    time_key?: string
  }
  probe_defaults?: {
    data_source: "router" | "residual"
    pooling: "last_token" | "mean_pool"
    n_folds: number
    layers?: string
    limit?: number
  }
  created_at?: string
  updated_at?: string
}
```

Important current limitation:
- saved prep targets are metadata only
- they are not yet wired into dataset materialization or prep execution

## 13. Current dashboard behavior

Implemented in [pipelines/dashboard.py](/Users/marshallvyletel/repos/concordance/xenon/pipelines/dashboard.py).

Key behavior:
- serves UI on `localhost:8800`
- runs pipeline commands as subprocess jobs
- uses cached stats snapshot with 5-minute TTL

Refresh behavior:
- background status reads use cached `dashboard_stats.json`
- forced refresh endpoints:
  - `GET /api/status?refresh=1`
  - `POST /api/status/refresh`
- forced refresh recomputes snapshot through `modal-snapshot`

Current common confusion:
- `modal-stats` downloads existing snapshot only
- `modal-snapshot` recomputes and then downloads

## 14. Script surfaces that matter most

Primary command wrapper:
- [scripts/modal_capture.sh](/Users/marshallvyletel/repos/concordance/xenon/scripts/modal_capture.sh)

### Available scripts

#### `scripts/modal_capture.sh`

Primary operational wrapper for Modal-backed ingest, prep, outcomes, capture, analysis, DB inspection, backup, restore, and snapshot refresh.

Current commands:
- `download`
- `smoke`
- `router`
- `full`
- `inspect`
- `meta`
- `compact`
- `analyze`
- `upload-db`
- `download-db`
- `modal-ingest`
- `modal-prep`
- `modal-outcomes`
- `modal-repair-db`
- `modal-inspect-db`
- `modal-inspect-full-logs`
- `modal-rebuild-from-files`
- `modal-backup-db`
- `modal-list-db-backups`
- `modal-restore-db`
- `backfill-payloads`
- `modal-snapshot`
- `modal-stats`
- `download-activations`
- `download-results`

Current high-value usage patterns:
- production ingest: `./scripts/modal_capture.sh modal-ingest`
- outcomes continuation: `./scripts/modal_capture.sh modal-outcomes --outcomes-limit -1 --concurrency 5 --timeout-s 30 --retry-max-attempts 6`
- prep refresh: `./scripts/modal_capture.sh modal-prep`
- forced dashboard stats refresh: `./scripts/modal_capture.sh modal-snapshot`
- download cached stats only: `./scripts/modal_capture.sh modal-stats`
- backup before risky ops: `./scripts/modal_capture.sh modal-backup-db <reason>`
- inspect live DB state: `./scripts/modal_capture.sh modal-inspect-db ingest/terminal_ingest.db`
- restore a backup: `./scripts/modal_capture.sh modal-restore-db <backup_name>`
- upload rebuilt local DB: `./scripts/modal_capture.sh upload-db`
- router-only capture: `./scripts/modal_capture.sh router --limit 10`
- analysis run: `./scripts/modal_capture.sh analyze --mode probe --target decision_type`

#### `scripts/modal_restore_db.sh`

Thin safety wrapper around backup listing and restore.

Current usage:
- list backups: `./scripts/modal_restore_db.sh --list 20`
- restore named backup: `./scripts/modal_restore_db.sh 20260312T172948Z_abort-slow-rebuild`
- restore latest backup: `./scripts/modal_restore_db.sh`

#### `scripts/rebuild_db_from_full_logs_local.py`

Local heavy-lift rebuild path from downloaded `full_logs/*.json.gz` into a local SQLite DB, following the ingest data model without calling Terminal APIs.

Current usage:
```bash
uv run --extra interp python scripts/rebuild_db_from_full_logs_local.py \
  --input-dir data/full_logs \
  --db-path data/terminal_ingest.db \
  --batch-size 2000
```

Typical workflow:
- download or verify `data/full_logs/`
- rebuild locally
- run `sqlite3 data/terminal_ingest.db "PRAGMA integrity_check;"`
- backup live Modal DB
- `upload-db`

#### `scripts/xenon_backend.sh`

Wrapper for backend deploy/dev and read-only backend inspection.

Current commands:
- `serve`
- `deploy`
- `query`
- `q`
- `stats`
- `schema`
- `tables`
- `sample`
- `parquet-list`
- `pql`
- `parquet-info`
- `pqi`
- `parquet-sample`
- `pqs`
- `activations`
- `act`
- `health`
- `reload`

Current usage:
- inspect production tables: `./scripts/xenon_backend.sh tables`
- run read-only SQL: `./scripts/xenon_backend.sh query "SELECT COUNT(*) FROM trade_outcomes"`
- inspect schema: `./scripts/xenon_backend.sh schema swaps`
- fetch stats JSON: `./scripts/xenon_backend.sh stats`
- deploy backend: `./scripts/xenon_backend.sh deploy`

Most important commands:
- `modal-ingest`
- `modal-outcomes`
- `modal-prep`
- `modal-inspect-db`
- `modal-inspect-full-logs`
- `modal-backup-db`
- `modal-list-db-backups`
- `modal-restore-db`
- `modal-snapshot`
- `modal-stats`
- `upload-db`
- `download-db`
- `router`
- `full`
- `analyze`
- `compact`

Operational references:
- [scripts/README.md](/Users/marshallvyletel/repos/concordance/xenon/scripts/README.md)

## 15. Current known constraints

Facts about the current system:
- production DB writes happen through ingest/prep/outcomes pipelines, not Explorer
- Explorer should remain siloed from core pipeline behavior
- analysis still assumes an input labels parquet, not a materialized dataset registry
- current analysis is CV-based, not split-manifest-based
- there is no finished "save prep target -> run prep -> create new probe dataset artifact" path yet

Important data caveats:
- `swaps` can exceed the subset of `inference_logs` relevant for probe work
- `trade_outcomes` coverage is partial until outcomes backfill completes
- full logs on disk are the recovery backbone if SQLite gets corrupted

## 16. Planned but not yet implemented

These are plans, not shipped functionality:
- versioned probe dataset materialization under `interp_exports/probe_runs`
- controlled-variable dataset builder for things like `risk 1 vs 5` within `record_observation_only`, `buy_only`, `sell_only`, or `trade_only`
- split-aware train/val/test analysis flow
- dataset manifests and reusable probe dataset artifacts

Primary plan doc:
- [PROBE_DATASET_BUILDER_PLAN.md](/Users/marshallvyletel/repos/concordance/xenon/PROBE_DATASET_BUILDER_PLAN.md)

## 17. Best starting points for future agents

If an agent needs to design probe datasets:
- start from `interp_examples_v0` schema in this file
- use `vault_*`, `decision_type`, `trade_side`, `asset`, `was_profitable_1h`, `context_complete`, and `created_at` as the main controllable axes

If an agent needs to design new analysis modes:
- start from [pipelines/interp/analysis.py](/Users/marshallvyletel/repos/concordance/xenon/pipelines/interp/analysis.py)
- preserve backward compatibility with `labels_path`
- keep output artifacts file-based

If an agent needs to build Explorer-side workflow:
- start from [pipelines/backend/app.py](/Users/marshallvyletel/repos/concordance/xenon/pipelines/backend/app.py)
- maintain strict read-only SQL
- write shared metadata only to non-DB JSON files

## 18. One-line system summary

Xenon currently has a working ingest -> prep -> capture -> analysis pipeline with read-only exploration tools and shared prep-target metadata, but it does not yet have a first-class, versioned, controlled-variable probe-dataset builder wired into analysis runs.
