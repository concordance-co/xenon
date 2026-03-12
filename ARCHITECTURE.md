# Xenon Architecture

Comprehensive technical context for the Xenon project. Read this before making any changes.

## What this project is

Xenon is a mechanistic interpretability research pipeline built around Terminal Markets — a live AI trading competition (Feb 26 – Mar 19, 2026) where ~hundreds of AI vaults trade 16 meme tokens on Base (Ethereum L2) 24/7. Every vault runs **Qwen3-235B-A22B** (MoE: 235B total params, 22B active per token, 128 experts, top-8 routing).

The competition exposes full inference logs via API: every prompt, every completion, every chain-of-thought, every trade execution. This is a rare dataset — thousands of real-world LLM decisions with ground-truth outcomes (did the trade make money?).

Xenon captures this data, cleans it, and then replays prompts through smaller Qwen3 models to capture internal activations — specifically MoE router logits — for interpretability research. The core research question: **can we decode what the model "knows" about its trading decisions from its internal routing patterns?**

## System overview

The system has three layers:

1. **Pipeline** — Python scripts that ingest data, prepare it, and capture activations. Run locally or on Modal (remote GPU cloud).
2. **Dashboard** — Local Python server (`dashboard.py`) that serves the React UI and executes pipeline commands as subprocesses.
3. **Backend API** — Modal FastAPI app that provides read-only SQL access to the remote database on the Modal volume.

```
┌─────────────────────────────────────────────────────────────────┐
│  LOCAL MACHINE                                                  │
│                                                                 │
│  Dashboard (localhost:8800)                                     │
│  ├── Serves React UI (dashboard-ui/dist/)                      │
│  ├── Command runner (subprocess execution via SSE)              │
│  ├── Job registry (reconnectable streaming jobs)                │
│  ├── Pipeline status APIs (/api/ingest, /api/prep, etc.)       │
│  └── Modal stats cache (dashboard_stats.json, 5-min TTL)       │
│                                                                 │
│  React UI (browser)                                             │
│  ├── Pipeline view: Ingest → Prep → Capture → Analysis         │
│  ├── Data Explorer: Query Lab, Label Lab, Probe Prep            │
│  └── Calls dashboard APIs + backend API directly (CORS)        │
│                                                                 │
└──────────────────────┬──────────────────────────────────────────┘
                       │
              Modal (remote)
                       │
┌──────────────────────┴──────────────────────────────────────────┐
│  Modal Volume: xenon-data                                       │
│  ├── /data/ingest/terminal_ingest.db  (SQLite, production DB)  │
│  ├── /data/interp_exports/*.parquet                            │
│  ├── /data/activations/ (safetensors + metadata.parquet)       │
│  └── /data/dashboard_stats.json (snapshot for dashboard)       │
│                                                                 │
│  Modal Apps:                                                    │
│  ├── xenon-backend (FastAPI, read-only SQL, always-on)         │
│  ├── xenon-ingest (ingest, prep, outcomes pipelines)           │
│  └── xenon-activation-capture (GPU capture on A100-80GB)       │
└─────────────────────────────────────────────────────────────────┘
```

### Important: Production validation vs local recovery workflows

For production truth, use the deployed backend API over the Modal DB.  
The local SQLite DB (`data/terminal_ingest.db`) can be stale and is not authoritative for production checks.

Local DB queries are still valid for explicit local workflows (recovery/rebuild/sanity checks) before upload to Modal.

Use backend queries for production investigation:

```bash
./scripts/xenon_backend.sh query "SELECT COUNT(*) FROM trade_outcomes"
./scripts/xenon_backend.sh tables
./scripts/xenon_backend.sh schema swaps
./scripts/xenon_backend.sh sample vaults 5
```

## Models

| Model | Role | Type | Params | Active | Layers | hidden_dim | Experts | Top-k | Where |
|-------|------|------|--------|--------|--------|------------|---------|-------|-------|
| Qwen3-235B-A22B | Competition model (what vaults actually run) | MoE | 235B | 22B | 94 | 4096 | 128 | 8 | Terminal Markets infra |
| Qwen3-30B-A3B | Capture model (production) | MoE | 30B | 3B | 48 | 2048 | 128 | 8 | Modal A100-80GB |
| Qwen3-8B | Capture model (dev/validation) | Dense | 8B | 8B | 36 | 4096 | — | — | Local M4 Max / MPS |

We capture from 30B-A3B (not 235B) because:
- Same MoE architecture (128 experts, top-8) — routing patterns are structurally comparable
- Fits on a single A100-80GB
- 2s/example vs minutes for 235B
- hidden_dim is only 2048 (smaller residual stream), making storage practical

## Pipeline phases

The pipeline has 4 phases. Outcomes (PnL labels) is a sub-step of Ingest, not a separate phase.

```
Terminal Markets API
        │
        ▼
Phase 1: INGEST ──────────────────────────────────────────
        │  pipelines/ingest/
        │  API → SQLite (terminal_ingest.db) + gzipped JSON payloads
        │  Tables: vaults, strategies, inference_logs, full_logs, swaps
        │  Sub-step: Outcomes (trade_outcomes table, PnL from candle data)
        │
        ▼
Phase 2: INTERP DATA PREP ────────────────────────────────
        │  pipelines/interp/prepare.py
        │  SQLite → interp_examples_v0 table → parquet exports
        │  Joins logs + payloads, extracts context, normalizes decisions, quality tiers
        │
        ▼
Phase 3: ACTIVATION CAPTURE ──────────────────────────────
        │  pipelines/interp/capture.py (local) or modal_capture.py (Modal)
        │  Parquet → forward pass → safetensors (residual stream + router logits)
        │
        ▼
Phase 4: ANALYSIS ────────────────────────────────────────
           Router logit exploration, linear probes, expert specialization
```

### Phase 1: Ingest

**Entry point:** `uv run -m pipelines.ingest --top-n 3` (local) or `./scripts/modal_capture.sh modal-ingest` (Modal)

Three-phase pipeline via `TerminalBackfillIngestor`:
1. **Vault discovery** — paginate leaderboard, upsert vault configs + strategies
2. **Log collection** — per vault, paginate inference logs, fetch full-log payloads (complete LLM prompt + completion + context), parse and store
3. **Swap collection** — per vault, paginate on-chain trade execution records

**Outcomes sub-step:** `pipelines/interp/outcomes.py` computes forward-looking PnL for swaps using Terminal Markets candle data. Writes to `trade_outcomes` table. Run via `./scripts/modal_capture.sh modal-outcomes` or from the dashboard Ingest tab. Outcomes are enrichment data for ingest — they don't warrant a separate pipeline phase.

### Phase 2: Interp Data Prep

**Entry point:** `uv run -m pipelines.interp.prepare --db-path data/terminal_ingest.db --export-parquet`

1. Joins `inference_logs` + `full_logs` (decompresses gzipped payloads)
2. Extracts structured context blocks from each payload:
   - `prompt_messages_json` — the LLM message array
   - `market_snapshot_json` — token prices, volumes, price changes
   - `portfolio_snapshot_json` — vault holdings, entry prices, unrealized PnL
   - `strategy_snapshot_json` — user-written trading instructions
   - `config_snapshot_json` — vault parameters (trade_size 1-5, risk_preference 1-5, etc.)
   - `memory_snapshot_json` — recent action history
   - `tools_available_json` — available tool definitions
3. Normalizes decisions:
   - `buy_token` → decision_type=trade, trade_side=buy
   - `sell_token` → decision_type=trade, trade_side=sell
   - `record_observation` → decision_type=record_observation
4. Joins swap records (trade executions) and outcome labels (PnL) where available
5. Assigns quality tiers:
   - **high**: parse_ok AND context_complete (has messages + market + portfolio + strategy + config)
   - **medium**: parse_ok, only missing memory and/or tools
   - **low**: parse errors or missing critical context

**Output tables:** `interp_examples_v0`, `interp_context_gaps_v0`, `interp_sample_trade_v0`, `interp_sample_observation_v0`, `interp_sample_paired_v0`

**Key parquet export:** `data/interp_exports/interp_examples_v0_high_quality.parquet` — this is what the capture pipeline reads.

### Phase 3: Activation Capture

**What we capture:**

- **Residual stream** — Hidden state at each layer's output. Shape: `(num_layers, seq_len, hidden_dim)` fp16.
- **Router logits** (MoE only) — `router_logits`: `(num_layers, seq_len, 128)` fp32, `router_indices`: `(num_layers, seq_len, 8)` int16

**How it works:**
1. Load high-quality examples from parquet
2. Tokenize with Qwen3 chat template
3. Register PyTorch forward hooks on `model.model.layers[i]` (residual) and `model.model.layers[i].mlp.gate` (router)
4. Run forward pass (no backward, no generation — just encode the prompt)
5. Save to safetensors files + metadata parquet

**MoE auto-detection:** `_is_moe_model()` checks `hasattr(model.model.layers[0].mlp, 'gate')`. Same code runs on both dense (8B) and MoE (30B) models.

**Storage layout:**
```
data/activations/
├── residual_stream/{log_id}.safetensors   # key: "residual_stream", (L, seq_len, hidden_dim) fp16
├── router_logits/{log_id}.safetensors     # keys: "router_logits" (L, seq_len, 128) fp32
│                                          #        "router_indices" (L, seq_len, 8) int16
└── metadata.parquet                       # log_id, seq_len, has_router, num_experts, ...
```

**Modal deployment:**
- App: `xenon-activation-capture`
- Volumes: `xenon-models` (cached weights), `xenon-data` (activations)
- GPU: A100-80GB
- Secret: `huggingface` (HF_TOKEN for model downloads)
- `CaptureWorker` cls with `@modal.enter()` for warm model across batches
- Local entrypoint fans out batches via `.map()`

## Dashboard architecture

The dashboard is the primary UI for running and monitoring the pipeline.

### Dashboard server (`pipelines/dashboard.py`)

Local Python HTTP server on `localhost:8800`. Does NOT require Modal to run.

**What it does:**
- Serves the React build from `dashboard-ui/dist/` (falls back to inline SPA if not built)
- Provides pipeline status APIs that aggregate data from a Modal stats cache
- Runs pipeline commands as subprocesses, streaming output via Server-Sent Events (SSE)
- Tracks running/completed jobs for reconnection

**Key internal components:**

- **`DashboardStore`** — Aggregates pipeline stats. Uses `_ModalStatsCache` with a 5-minute TTL.
  - Background fetch path uses `./scripts/modal_capture.sh modal-stats` (download existing snapshot only).
  - Forced refresh path uses `./scripts/modal_capture.sh modal-snapshot` (recompute snapshot on Modal, then download).
  - Forced refresh is exposed via `GET /api/status?refresh=1` and `POST /api/status/refresh`.
  - Falls back gracefully if Modal is unreachable.
- **`JobRegistry`** — Thread-safe job tracker. Runs commands as subprocesses, buffers output, supports SSE streaming and reconnection. Retains up to 20 completed jobs.
- **Command whitelist** — Only predefined commands can be executed (ingest, prep, capture, outcomes, analysis, modal scripts). Prevents arbitrary command injection from the UI.

**Dashboard API endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/status` | Phase status overview (4 phases) |
| GET | `/api/status?refresh=1` | Force stats recompute/download, then return status |
| GET | `/api/ingest` | Vault/strategy counts, parse error stats |
| GET | `/api/outcomes` | PnL stats, win rates, risk breakdown |
| GET | `/api/prep` | Example counts, quality breakdown |
| GET | `/api/capture` | Activation file counts, metadata summary |
| GET | `/api/analysis` | Result file listing, probe results |
| GET | `/api/jobs` | List running/completed jobs |
| GET | `/api/backend-url` | Backend API URL for frontend to use |
| POST | `/api/status/refresh` | Force stats recompute/download, then return status |
| POST | `/api/run-stream` | Execute command, stream output as SSE |
| POST | `/api/job-reconnect` | Reconnect to running job by ID |

### React UI (`pipelines/dashboard-ui/`)

React 19 + Vite + TypeScript. Build with `cd pipelines/dashboard-ui && npm run build`.

**Top-level navigation (App.tsx):**
- **Pipeline** view — 4-phase pipeline control (Ingest, Prep, Capture, Analysis)
- **Explorer** view — Data exploration and prep design (`Query Lab`, `Label Lab`, `Probe Prep`)

**Components:**

| Component | What it does |
|-----------|-------------|
| `App.tsx` | Root layout, view switching (Pipeline/Explorer), config context |
| `PipelineFlow.tsx` | Visual pipeline diagram (4 phases with status indicators) |
| `PhaseStrip.tsx` | Phase summary cards with key metrics |
| `IngestView.tsx` | Ingest stats + outcomes panel (PnL stats, risk breakdown, run buttons) |
| `PrepView.tsx` | Data prep stats, quality distribution, config, run button |
| `CaptureView.tsx` | Capture stats, config, run button |
| `AnalysisView.tsx` | Analysis results, probe charts, expert data |
| `ExplorerView.tsx` | 3-workspace Explorer (`Query Lab`, `Label Lab`, `Probe Prep`) for read-only data workbench + prep-target specs |
| `CommandRunner.tsx` | Streaming command output panel (SSE), minimize/reconnect |
| `JobList.tsx` | Running job sidebar, polls `/api/jobs`, click to reconnect |
| `ProbeChart.tsx` | Recharts-based probe accuracy visualization |
| `Tip.tsx` | Tooltip hover component for stat explanations |

**Hooks:**

| Hook | What it does |
|------|-------------|
| `useApi.ts` | `useFetch(path)` — polling fetch wrapper for dashboard APIs |
| `useConfig.ts` | Pipeline config state (persisted to localStorage). Config per phase + command builders |
| `useBackend.ts` | Fetches backend URL from `/api/backend-url`, provides `backendFetch(path)` and `backendPost(path, body)` for direct calls to the Modal backend API |

**Key pattern — Explorer calls backend directly:** The Explorer view calls the Modal backend API directly (not through dashboard.py). The backend has CORS enabled for all origins. The backend URL is discovered via `/api/backend-url` which reads from `XENON_BACKEND_URL` env var or `~/.xenon_backend_url` file.

### Backend API (`pipelines/backend/app.py`)

Modal FastAPI app deployed on Modal. Provides read-only SQL access to the production database on the Modal volume.

**Deployment:** `./scripts/xenon_backend.sh deploy` (saves URL to `~/.xenon_backend_url`)

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | DB existence and size |
| POST | `/query` | Read-only SQL (SELECT/PRAGMA/EXPLAIN/WITH only, auto-LIMIT) |
| GET | `/schema` | List tables or get column schema for one table |
| GET | `/tables` | All table names with row counts |
| GET | `/stats` | Returns `dashboard_stats.json` from volume |
| GET | `/sample/{table}` | Random sample of N rows (max 500) |
| GET | `/parquet/list` | List parquet files on volume |
| GET | `/parquet/info/{name}` | Parquet metadata (rows, columns, schema) |
| GET | `/parquet/sample/{name}` | Sample rows from parquet (max 100) |
| GET | `/activations/meta` | Activations metadata.parquet summary |
| POST | `/profile/dataset` | Read-only dataset profile stats |
| POST | `/label/preview` | Label spec preview + split viability |
| GET | `/prep-targets` | List shared prep-target specs |
| GET | `/prep-targets/{id}` | Fetch one shared prep-target spec |
| POST | `/prep-targets` | Create/update shared prep-target spec (file metadata only) |
| DELETE | `/prep-targets/{id}` | Delete shared prep-target spec |
| POST | `/reload` | Force Modal volume refresh |

**Volume reload behavior:** `volume.reload()` refreshes the cached filesystem snapshot from Modal's storage backend. It's only called on the explicit `/reload` endpoint — NOT on every request (that was causing latency). The volume snapshot is from when the container started; `/reload` is needed to see writes from other containers (e.g., after an ingest run populates the DB).

**CLI wrapper:** `pipelines/backend/cli.py` — local CLI that calls the backend API. Used by `./scripts/xenon_backend.sh`.

### Stats snapshot (`dashboard_stats.json`)

Written by `_write_stats_snapshot()` in `pipelines/interp/modal_ingest.py` after each pipeline run. Contains pre-computed stats so the dashboard doesn't need to query the full DB:

```json
{
  "ingest": {
    "vault_count": 321, "strategy_count": 640, "log_count": 15000,
    "full_log_count": 14500, "coverage_pct": 96.7,
    "parse_error_count": 12, "tables": [{"name": "vaults", "count": 321}, ...]
  },
  "outcomes": {
    "total_outcomes": 200, "unlabeled_swaps": 798, "total_swaps": 998,
    "avg_pnl_1h": -0.5, "avg_pnl_4h": -1.2, "avg_pnl_1d": -2.1,
    "win_rate_1h": 0.42, "risk_breakdown": [{"risk": 1, "count": 50, ...}, ...]
  },
  "prep": {
    "total_examples": 5000, "high_quality": 4200,
    "medium_quality": 600, "low_quality": 200,
    "export_files": [...], "label_distribution": [...]
  }
}
```

## Database schema (SQLite, WAL mode)

**`vaults`** — Vault configuration and rankings
```
vault_address       TEXT PK
owner_address       TEXT
nft_id, nft_name    TEXT
persona_json        TEXT (JSON)
trade_size          INTEGER (1-5 scale)
trading_activity    INTEGER (1-5 scale)
holding_style       INTEGER (1-5 scale)
diversification     INTEGER (1-5 scale)
asset_risk_preference INTEGER (1-5 scale)
max_trade_amount    TEXT (wei)
slippage_bps        TEXT
paused              INTEGER (0/1)
state               TEXT
leaderboard_rank    INTEGER
total_pnl_usd       REAL
realized_pnl_usd    REAL
unrealized_pnl_usd  REAL
fetched_at          TEXT (ISO)
```

**`strategies`** — User-written trading strategies
```
vault_address       TEXT PK (composite)
strategy_id         TEXT PK (composite)
content             TEXT (the actual strategy text the LLM sees)
enabled             INTEGER (0/1)
strategy_priority   TEXT
```

**`inference_logs`** — One row per LLM decision
```
id                  INTEGER PK (from API)
vault_address       TEXT FK → vaults
tool                TEXT (buy_token, sell_token, record_observation)
tool_args_json      TEXT (JSON)
strategy_id         TEXT
status              TEXT (success/failure)
inference_duration_ms INTEGER
transaction_hash    TEXT (links to swaps)
created_at          TEXT (ISO)
```

**`full_logs`** — Complete LLM prompt + completion payloads
```
log_id              INTEGER PK, FK → inference_logs.id
vault_address       TEXT
payload_path        TEXT (absolute path to .json.gz)
payload_sha256      TEXT
payload_size_bytes  INTEGER
prompt_text         TEXT
completion_text     TEXT
reasoning_content   TEXT (chain-of-thought)
tool_calls_json     TEXT (JSON array)
llm_model           TEXT
prompt_tokens       INTEGER
completion_tokens   INTEGER
reasoning_tokens    INTEGER
total_tokens        INTEGER
parse_error         TEXT (non-null if extraction failed)
```

**`swaps`** — On-chain trade executions
```
transaction_hash    TEXT PK (composite)
log_index           INTEGER PK (composite)
vault_address       TEXT FK → vaults
side                TEXT (buy/sell)
token_address       TEXT
token_symbol        TEXT
effective_price_usd TEXT
log_id              INTEGER FK → inference_logs.id (nullable)
timestamp           INTEGER (unix)
```

**`trade_outcomes`** — Forward-looking PnL labels from candle data
```
log_id              INTEGER PK
side                TEXT (buy/sell)
token_address       TEXT
entry_price_usd     REAL
pnl_1h_pct          REAL
pnl_4h_pct          REAL
pnl_1d_pct          REAL
was_profitable_1h   INTEGER (0/1)
```

**FK relationships:**
```
vaults.vault_address
  ├── strategies.vault_address
  ├── inference_logs.vault_address
  └── swaps.vault_address

inference_logs.id
  ├── full_logs.log_id
  ├── swaps.log_id
  └── trade_outcomes.log_id

trade_outcomes.log_id → swaps.log_id (join through swaps to vaults for risk_preference)
```

## Full-log payload structure (gzipped JSON)

Each payload (~50KB) at `data/full_logs/{shard}/{log_id}.json.gz` contains:

```json
{
  "snapshot": {
    "Agent": { "vault config, persona, strategies" },
    "Portfolio": { "ETH balance, token holdings with entry prices + unrealized PnL" },
    "Market": { "ETH price, 16 tokens with prices + 1m/5m/1h/6h/24h metrics + liquidity" },
    "AllowedTools": ["buy_token", "sell_token", "record_observation"],
    "Memories": ["recent action history"]
  },
  "llm_request_payload": {
    "llm_input": {
      "messages": ["system prompt + user prompt with full context"],
      "tools": ["buy_token, sell_token, record_observation function schemas"]
    }
  },
  "llm_completion_payload": {
    "choices": [{
      "message": {
        "content": "...",
        "reasoning_content": "chain-of-thought (~2K chars)",
        "tool_calls": [{ "function": { "name": "buy_token", "arguments": "{...}" } }]
      }
    }],
    "usage": { "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0 }
  }
}
```

## File map

```
pipelines/
├── dashboard.py               # Dashboard server (localhost:8800, serves React + command runner)
├── ingest/
│   ├── api.py                 # TerminalMarketsApiClient (async, retry, semaphore)
│   ├── db.py                  # IngestDatabase (schema + upserts)
│   ├── pipeline.py            # TerminalBackfillIngestor (3-phase orchestration)
│   ├── payload_store.py       # RawPayloadStore (atomic gzip writes)
│   ├── full_log_parser.py     # Parse LLM payloads
│   ├── explorer.py            # Legacy web UI for data browsing (superseded by dashboard)
│   └── cli.py                 # CLI entry point
├── interp/
│   ├── prepare.py             # SQLite → interp_examples_v0 + parquet
│   ├── capture.py             # Activation capture (hooks, save, MoE detection)
│   ├── modal_capture.py       # Modal App, CaptureWorker, volumes, GPU capture
│   ├── modal_ingest.py        # Modal wrapper for ingest/prep/outcomes + stats snapshot
│   └── outcomes.py            # PnL outcome labels from candle data
├── backend/
│   ├── app.py                 # Modal FastAPI backend (read-only SQL on remote DB)
│   └── cli.py                 # Backend CLI (calls backend API)
└── dashboard-ui/
    ├── src/
    │   ├── App.tsx            # Root layout, Pipeline/Explorer view switching
    │   ├── components/
    │   │   ├── IngestView.tsx     # Ingest stats + outcomes panel
    │   │   ├── PrepView.tsx       # Data prep stats and config
    │   │   ├── CaptureView.tsx    # Capture stats and config
    │   │   ├── AnalysisView.tsx   # Analysis results, probe charts
    │   │   ├── ExplorerView.tsx   # Query/Label/Probe workspaces + prep-target management
    │   │   ├── PipelineFlow.tsx   # Visual pipeline diagram
    │   │   ├── PhaseStrip.tsx     # Phase summary cards
    │   │   ├── CommandRunner.tsx  # Streaming command output (SSE)
    │   │   ├── JobList.tsx        # Running jobs sidebar
    │   │   ├── ProbeChart.tsx     # Probe accuracy charts
    │   │   └── Tip.tsx            # Tooltip hover component
    │   ├── hooks/
    │   │   ├── useApi.ts          # Dashboard API fetch wrapper
    │   │   ├── useConfig.ts       # Pipeline config (localStorage)
    │   │   └── useBackend.ts      # Modal backend API client
    │   └── types/
    │       └── api.ts             # TypeScript interfaces for all API responses
    └── dist/                      # Built React app (served by dashboard.py)

scripts/
├── modal_capture.sh           # Shell wrapper for Modal operations + DB utilities
├── modal_restore_db.sh        # Focused DB restore/list helper with safe defaults
├── rebuild_db_from_full_logs_local.py  # Local rebuild helper from full_logs JSON.gz
└── xenon_backend.sh           # Shell wrapper for backend API operations

tests/
├── test_ingest.py             # Ingest pipeline tests
└── test_capture.py            # Capture + MoE tests (38 tests)

data/                          # Local data (gitignored, small dev subset)
├── terminal_ingest.db         # SQLite database (LOCAL COPY — not authoritative)
├── full_logs/                 # Gzipped JSON payloads
├── interp_exports/            # Parquet exports
└── activations/               # Safetensor capture output
```

## Shell scripts reference

### `scripts/modal_capture.sh`

| Command | What it does |
|---------|-------------|
| `download` | Cache model weights to `xenon-models` volume |
| `smoke` | Smoke test capture (1 example) |
| `router` | Capture router logits only |
| `full` | Capture residual + router |
| `inspect` | List files on `xenon-data` volume |
| `meta` | Show activations metadata |
| `compact` | Compact analysis |
| `analyze` | Run analysis pipeline |
| `upload-db` | Push local DB to Modal volume |
| `download-db` | Pull DB from Modal volume to local |
| `modal-ingest` | Run ingest pipeline on Modal |
| `modal-prep` | Run data prep on Modal |
| `modal-outcomes` | Run outcomes (PnL) pipeline on Modal |
| `modal-inspect-db` | Inspect DB integrity/table counts on Modal volume |
| `modal-inspect-full-logs` | Inspect full_logs JSON.gz coverage on Modal volume |
| `modal-rebuild-from-files` | Rebuild ingest DB from existing full logs on Modal |
| `modal-backup-db` | Snapshot DB (+wal/shm) to `ingest/db_backups` (retain 30) |
| `modal-list-db-backups` | List DB backup snapshots |
| `modal-restore-db` | Restore DB from backup snapshot with safety checks |
| `modal-snapshot` | Compute and download stats snapshot |
| `modal-stats` | Download existing stats (no recompute) |
| `download-activations` | Bulk download activation files |
| `download-results` | Bulk download analysis results |
| `backfill-payloads` | One-time migration of file payloads to DB |

### `scripts/modal_restore_db.sh`

| Command | What it does |
|---------|-------------|
| `--list [limit]` | List available DB backup snapshots via `modal-list-db-backups` |
| `[backup_name] [extra flags]` | Restore named backup (or latest when omitted) via `modal-restore-db` |

### `scripts/rebuild_db_from_full_logs_local.py`

| Command | What it does |
|---------|-------------|
| `python scripts/rebuild_db_from_full_logs_local.py --input-dir ... --db-path ...` | Rebuild local SQLite ingest DB from `full_logs/*.json.gz` without Terminal API calls |
| `... --batch-size N` | Control commit chunking for local rebuild throughput |

### DB recovery and restore safeguards

- **Backup-first workflow:** run `./scripts/modal_capture.sh modal-backup-db <reason>` before risky operations (repair, restore, upload, large outcomes runs).
- **Restore safety checks:** `modal-restore-db` validates source integrity, rejects empty-table restores by default, creates an auto pre-restore backup, and rolls back on failed post-restore validation.
- **Upload semantics:** `upload-db` overwrites only `xenon-data:/ingest/terminal_ingest.db` and does not modify:
  - `ingest/db_backups/`
  - `ingest/repair_backups/`
  - `ingest/rebuild_backups/`

### `scripts/xenon_backend.sh`

| Command | What it does |
|---------|-------------|
| `deploy` | Deploy backend to Modal, save URL |
| `serve` | Dev server with hot-reload |
| `query "SQL"` | Run read-only SQL query |
| `tables` | List tables with row counts |
| `schema TABLE` | Show table column schema |
| `sample TABLE N` | Random sample of N rows |
| `stats` | Download dashboard stats JSON |
| `parquet-list` | List parquet files on volume |
| `parquet-info FILE` | Parquet metadata |
| `parquet-sample FILE` | Sample rows from parquet |
| `activations` | Activations metadata summary |
| `health` | Backend health check |
| `reload` | Force volume refresh |

## Dependencies

```toml
# Base (always installed)
aiohttp, aiosqlite, pyarrow

# interp extra (local capture)
torch, transformers, safetensors

# modal extra (remote operations)
modal

# dev extra
pytest
```

## How to run things

```bash
# Start dashboard (serves React UI on localhost:8800)
uv run -m pipelines.dashboard

# Build React UI (required once, then dashboard serves it)
cd pipelines/dashboard-ui && npm install && npm run build

# Run pipeline phases on Modal (production)
./scripts/modal_capture.sh modal-ingest    # Ingest data
./scripts/modal_capture.sh modal-outcomes  # Compute PnL labels
./scripts/modal_capture.sh modal-prep      # Prepare interp dataset
./scripts/modal_capture.sh router          # Capture router logits
./scripts/modal_capture.sh full            # Capture all activations

# Query remote production data (authoritative checks)
./scripts/xenon_backend.sh query "SELECT COUNT(*) FROM swaps"
./scripts/xenon_backend.sh tables

# Deploy/redeploy backend API
./scripts/xenon_backend.sh deploy

# Run tests
uv run pytest tests/
```

## Key design decisions

1. **Replay through smaller model, not the competition model** — Qwen3-30B-A3B has the same MoE architecture as 235B but fits on one GPU. Research validity depends on routing patterns being structurally similar across model scales.

2. **Router logits as primary signal** — Small (128 floats/token/layer), interpretable, and capture the model's expert allocation decision. Residual stream is huge and harder to interpret directly.

3. **Safetensors per log_id** — Simple random access. One file per example, not one file per layer. Good for the current scale (~hundreds of examples). May need resharding if we go to thousands.

4. **Idempotent everything** — Ingest uses upserts, capture has `--skip-existing`, data prep rebuilds tables each run. Safe to re-run any stage.

5. **Local dev + Modal prod** — Same capture code runs both places. Local for iteration (Qwen3-8B on MPS), Modal for production captures (Qwen3-30B-A3B on A100).

6. **Stats snapshot pattern** — Pipeline writes a small JSON summary (`dashboard_stats.json`) to the volume after each run. Dashboard uses a cached snapshot (5-min TTL) instead of querying the full DB directly. Explicit refresh paths force recompute (`modal-snapshot`) before returning status.

7. **Outcomes in Ingest, not a separate phase** — Outcomes (PnL computation) is enrichment of ingest data. It doesn't warrant its own pipeline phase or UI tab. It lives as a sub-section inside the Ingest tab.

8. **Explorer calls backend directly** — The Data Explorer makes CORS requests directly to the Modal backend API, not through dashboard.py. This avoids proxying large query results through the local server.

9. **No `volume.reload()` per request** — Modal volumes cache a filesystem snapshot. `reload()` refreshes it but adds latency. Only the explicit `/reload` endpoint calls it. The snapshot from container start is sufficient for read-heavy workloads.

## Modal gotchas

- `from __future__ import annotations` breaks Modal's `parameter()` type introspection — don't use it in Modal files
- Modal CLI is Click-based, not argparse — doesn't use `--` separator
- `local_entrypoint` runs locally, not in container — needs local deps via `uv run`
- `modal.parameter()` must be used (not `__init__`) for Modal class parameters
- Modal apps use `add_local_python_source("pipelines")` to include local code
- Deployment: `uv run --extra modal modal deploy <file>` or `uv run --extra modal modal serve <file>` (dev)

## Current state (as of March 12, 2026)

### Data volumes (on Modal)
- Recovery baseline after DB restore/rebuild: ~53.9K inference logs, ~53.9K full logs, ~23.7K swaps (exact counts can change as outcomes runs continue)
- Trade outcomes backfill is resumable and may be partially complete at any moment
- Use `./scripts/xenon_backend.sh stats` (or dashboard forced refresh) for current authoritative counts
- Terminal Markets competition window is Feb 26 – Mar 19, 2026; as of March 12, 2026, this window is still active

### What's built and working
- Full ingest pipeline (local + Modal)
- Data prep pipeline with quality tiers
- Outcomes pipeline (PnL labels from candle data)
- Local capture (Qwen3-8B) validated end-to-end
- Modal capture (Qwen3-30B-A3B) smoke tested, router logits confirmed
- Dashboard with React UI (Pipeline + Explorer views)
- Backend API deployed on Modal with CLI wrapper
- 38 capture tests passing
- Shell scripts for all Modal operations

### What's not built yet
- Analysis notebooks (router logit exploration)
- Linear probes (predict decision_type, trade_side, profitability from activations)
- No CI — tests are run manually
