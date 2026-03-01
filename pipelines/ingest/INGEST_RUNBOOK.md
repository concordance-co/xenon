# Xenon Ingest Pipeline Runbook

## Purpose

The ingest pipeline collects data from the Terminal Markets API into a local SQLite database and gzipped payload files. It discovers top-performing vaults, fetches their inference logs (LLM prompt + completion payloads), strategies, and trade executions (swaps).

## Entry Point

```bash
uv run -m pipelines.ingest --top-n 3
```

Module: `pipelines/ingest/cli.py` → `pipelines/ingest/pipeline.py`

## Architecture

```
Terminal Markets API
        │
        ▼
 TerminalMarketsApiClient   (api.py — async, retry, semaphore)
        │
        ▼
 TerminalBackfillIngestor   (pipeline.py — 3-phase orchestration)
        │
   ┌────┴────┐
   ▼         ▼
IngestDatabase    RawPayloadStore
(SQLite)          (gzip JSON)
   │                  │
   ▼                  ▼
5 tables         data/full_logs/
                   000000-NNNNNN/
                     {log_id}.json.gz
```

## Database Schema

Database file: `data/terminal_ingest.db` (SQLite, WAL mode)

### `vaults`

Vault configuration and leaderboard ranking.

| Column | Type | Notes |
|--------|------|-------|
| `vault_address` | TEXT | **PK** |
| `owner_address` | TEXT | |
| `nft_id` | TEXT | |
| `nft_name` | TEXT | |
| `persona_json` | TEXT | JSON-serialized persona object |
| `trade_size` | INTEGER | 1-5 scale |
| `trading_activity` | INTEGER | 1-5 scale |
| `holding_style` | INTEGER | 1-5 scale |
| `diversification` | INTEGER | 1-5 scale |
| `asset_risk_preference` | INTEGER | 1-5 scale |
| `max_trade_amount` | TEXT | Wei string |
| `slippage_bps` | TEXT | |
| `paused` | INTEGER | 0/1 boolean |
| `state` | TEXT | e.g. "active" |
| `leaderboard_rank` | INTEGER | |
| `total_pnl_usd` | REAL | |
| `realized_pnl_usd` | REAL | |
| `unrealized_pnl_usd` | REAL | |
| `created_block` | INTEGER | |
| `updated_block` | INTEGER | |
| `fetched_at` | TEXT | ISO timestamp, NOT NULL |

### `strategies`

User-written trading strategies attached to vaults.

| Column | Type | Notes |
|--------|------|-------|
| `vault_address` | TEXT | **PK (composite)**, FK → vaults |
| `strategy_id` | TEXT | **PK (composite)** |
| `vault_owner_address` | TEXT | |
| `content` | TEXT | Strategy text |
| `expiry` | INTEGER | |
| `enabled` | INTEGER | 0/1 boolean |
| `strategy_priority` | TEXT | |
| `created_block` | INTEGER | |
| `updated_block` | INTEGER | |
| `fetched_at` | TEXT | ISO timestamp, NOT NULL |

### `inference_logs`

LLM inference metadata (one row per agent decision).

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER | **PK** (from API) |
| `cursor` | TEXT | Pagination cursor |
| `vault_address` | TEXT | FK → vaults, NOT NULL |
| `request_id` | TEXT | |
| `execution_key` | TEXT | |
| `tool` | TEXT | Action name: buy_token, sell_token, record_observation |
| `tool_args_json` | TEXT | JSON-serialized arguments |
| `strategy_id` | TEXT | |
| `status` | TEXT | success/failure |
| `inference_duration_ms` | INTEGER | |
| `error` | TEXT | |
| `transaction_hash` | TEXT | Links to swaps |
| `created_at` | TEXT | ISO timestamp |
| `completed_at` | TEXT | ISO timestamp |
| `fetched_at` | TEXT | ISO timestamp, NOT NULL |

**Index:** `idx_inference_logs_vault` on `(vault_address, id)`

### `full_logs`

Complete LLM prompt + completion payloads with parsed fields.

| Column | Type | Notes |
|--------|------|-------|
| `log_id` | INTEGER | **PK**, FK → inference_logs.id |
| `vault_address` | TEXT | |
| `payload_path` | TEXT | Absolute path to .json.gz, NOT NULL |
| `payload_sha256` | TEXT | Hash of uncompressed payload, NOT NULL |
| `payload_size_bytes` | INTEGER | Size of uncompressed payload, NOT NULL |
| `prompt_text` | TEXT | Concatenated `[role]\ncontent` blocks |
| `completion_text` | TEXT | Model output |
| `reasoning_content` | TEXT | Chain-of-thought (optional) |
| `tool_calls_json` | TEXT | JSON array of tool call objects |
| `llm_model` | TEXT | Model identifier |
| `prompt_tokens` | INTEGER | |
| `completion_tokens` | INTEGER | |
| `reasoning_tokens` | INTEGER | |
| `total_tokens` | INTEGER | |
| `parse_error` | TEXT | Non-null if extraction failed |
| `fetched_at` | TEXT | ISO timestamp, NOT NULL |

### `swaps`

On-chain trade execution records.

| Column | Type | Notes |
|--------|------|-------|
| `transaction_hash` | TEXT | **PK (composite)**, NOT NULL |
| `block_number` | INTEGER | NOT NULL |
| `log_index` | INTEGER | **PK (composite)**, NOT NULL |
| `timestamp` | INTEGER | Unix seconds |
| `pool_id` | TEXT | |
| `token_address` | TEXT | |
| `token_name` | TEXT | |
| `token_symbol` | TEXT | |
| `vault_address` | TEXT | FK → vaults, NOT NULL |
| `is_reap_twap` | INTEGER | 0/1 boolean |
| `side` | TEXT | "buy" or "sell" |
| `token_amount` | TEXT | Raw amount |
| `eth_amount` | TEXT | Raw amount |
| `eth_price_usd` | TEXT | |
| `effective_price_eth` | TEXT | |
| `effective_price_usd` | TEXT | |
| `log_id` | INTEGER | FK → inference_logs.id (nullable) |
| `strategy_id` | TEXT | |
| `fetched_at` | TEXT | ISO timestamp, NOT NULL |

**Indexes:**
- `idx_swaps_log_id` on `(log_id)`
- `idx_swaps_vault` on `(vault_address, timestamp)`
- `idx_swaps_token_timestamp` on `(token_address, timestamp)`

## Schema Verification

After an ingest run, verify tables and row counts:

```sql
-- List all tables
SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;

-- Expected: full_logs, inference_logs, strategies, swaps, vaults

-- List all indexes
SELECT name, tbl_name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY name;

-- Expected:
--   idx_inference_logs_vault  → inference_logs
--   idx_swaps_log_id          → swaps
--   idx_swaps_vault           → swaps
--   idx_swaps_token_timestamp → swaps

-- Row counts
SELECT 'vaults' AS tbl, COUNT(*) AS n FROM vaults
UNION ALL SELECT 'strategies', COUNT(*) FROM strategies
UNION ALL SELECT 'inference_logs', COUNT(*) FROM inference_logs
UNION ALL SELECT 'full_logs', COUNT(*) FROM full_logs
UNION ALL SELECT 'swaps', COUNT(*) FROM swaps;

-- Full log coverage (should be close to 100% of inference_logs)
SELECT
    (SELECT COUNT(*) FROM inference_logs) AS total_logs,
    (SELECT COUNT(*) FROM full_logs) AS total_full_logs,
    ROUND(100.0 * (SELECT COUNT(*) FROM full_logs) / NULLIF((SELECT COUNT(*) FROM inference_logs), 0), 1) AS coverage_pct;

-- Parse errors (should be low)
SELECT COUNT(*) AS parse_errors FROM full_logs WHERE parse_error IS NOT NULL;

-- Tool distribution
SELECT tool, COUNT(*) AS n FROM inference_logs GROUP BY tool ORDER BY n DESC;

-- Vault summary
SELECT vault_address, leaderboard_rank, total_pnl_usd,
       (SELECT COUNT(*) FROM inference_logs il WHERE il.vault_address = v.vault_address) AS log_count,
       (SELECT COUNT(*) FROM swaps s WHERE s.vault_address = v.vault_address) AS swap_count
FROM vaults v
ORDER BY leaderboard_rank;
```

Run these from the command line:

```bash
sqlite3 data/terminal_ingest.db < verification_query.sql
# or inline:
sqlite3 data/terminal_ingest.db "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
```

## Foreign Key Relationships

```
vaults.vault_address
  ├── strategies.vault_address
  ├── inference_logs.vault_address
  └── swaps.vault_address

inference_logs.id
  ├── full_logs.log_id
  └── swaps.log_id (nullable)
```

## Payload Storage

Raw API responses are stored as gzipped JSON in `data/full_logs/`:

```
data/full_logs/
├── 000000/           # log_ids 0-999
│   ├── 42.json.gz
│   └── 999.json.gz
├── 000001/           # log_ids 1000-1999
│   ├── 1001.json.gz
│   └── ...
└── 000005/           # log_ids 5000-5999
    └── 5432.json.gz
```

Shard formula: `shard = f"{log_id // 1000:06d}"`

Each payload file contains the full API response including:
- `llm_request_payload` → prompt messages, model config, tools
- `llm_completion_payload` → completion, reasoning, tool calls, usage
- `snapshot` → market data, portfolio, strategy at time of inference

Read a payload:

```python
import gzip, json
with gzip.open("data/full_logs/000001/1001.json.gz", "rt") as f:
    payload = json.load(f)
print(json.dumps(payload, indent=2)[:2000])
```

## API Client

`TerminalMarketsApiClient` (async context manager):

| Method | Endpoint | Returns |
|--------|----------|---------|
| `get_leaderboard_page()` | `/leaderboard` | Paginated vault rankings |
| `get_vault(addr)` | `/vault?vaultAddress=` | Single vault config |
| `get_strategies(addr)` | `/strategies/{addr}` | List of strategies |
| `get_logs_page(addr)` | `/logs/{addr}` | Paginated inference logs |
| `get_full_log(id)` | `/full-log/{id}` | Complete LLM payload |
| `get_swaps_page(addr)` | `/swaps?vaultAddress=` | Paginated swaps |
| `get_candles(token)` | `/candles/{token}` | OHLCV price candles |

Retry policy: exponential backoff (1s → 60s), retries on 429/5xx, max 6 attempts.

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--base-url` | `https://api.terminal.markets/api/v1` | API base URL |
| `--db-path` | `data/terminal_ingest.db` | SQLite database path |
| `--raw-payload-dir` | `data/full_logs` | Gzipped payload storage |
| `--top-n` | 3 | Number of leaderboard vaults |
| `--leaderboard-sort-by` | `total_pnl_usd` | Sort metric |
| `--request-limit` | 50 | Page size (max 50) |
| `--request-concurrency` | 10 | Concurrent API requests |
| `--timeout-s` | 30 | Request timeout |
| `--retry-max-attempts` | 6 | Max retries per request |
| `--max-logs-per-vault` | unlimited | Cap inference logs per vault |
| `--max-full-logs-per-vault` | unlimited | Cap full logs per vault |
| `--max-swaps-per-vault` | unlimited | Cap swaps per vault |
| `--exclude-reasoning` | off | Omit reasoning_content |

## Pipeline Phases

1. **Vault discovery** — paginate leaderboard, upsert vault configs + strategies
2. **Log collection** — for each vault, paginate inference logs, fetch full log payloads for new IDs, parse and store
3. **Swap collection** — for each vault, paginate swap records

All upserts use `ON CONFLICT ... DO UPDATE` so the pipeline is safe to re-run.

## Data Explorer (Web UI)

```bash
uv run -m pipelines.ingest.explorer --db-path data/terminal_ingest.db --port 8765
```

Opens a read-only web UI at `http://127.0.0.1:8765` with:

- Vault leaderboard browser
- Inference log + full-log payload viewer
- Tool distribution analysis
- Dataset readiness metrics for downstream interp work

## Tests

```bash
uv run --extra dev -m pytest tests/test_ingest.py -v
```

Covers:
- `parse_full_log`: complete payloads, empty payloads, missing choices, reasoning toggle, model fallback, tool calls
- `RawPayloadStore`: round-trip write/read, sharding, overwrite, deterministic hashing
- `IngestDatabase`: schema creation (tables + indexes), idempotency, all upsert operations (vaults, strategies, inference_logs, swaps, full_logs), `fetch_existing_full_log_ids`
- CLI argument parsing: defaults and all flags
- API types: `RetryPolicy` defaults, `TerminalApiError` inheritance

## Handoff Notes

- All upserts are idempotent. Re-running the pipeline updates existing rows.
- `full_logs.parse_error` captures extraction failures without blocking the pipeline.
- Payload files are stored atomically (temp file → rename) to prevent partial writes.
- Foreign keys are enforced (`PRAGMA foreign_keys=ON`).
- The `interp.prepare` module downstream reads from these tables to build interp-ready datasets.
