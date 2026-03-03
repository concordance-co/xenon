# Xenon Architecture

Comprehensive technical context for the Xenon project. Read this to understand the full system before making architectural decisions.

## What this project is

Xenon is a mechanistic interpretability research pipeline built around Terminal Markets — a live AI trading competition (Feb 26 – Mar 19, 2026) where ~hundreds of AI vaults trade 16 meme tokens on Base (Ethereum L2) 24/7. Every vault runs **Qwen3-235B-A22B** (MoE: 235B total params, 22B active per token, 128 experts, top-8 routing).

The competition exposes full inference logs via API: every prompt, every completion, every chain-of-thought, every trade execution. This is a rare dataset — thousands of real-world LLM decisions with ground-truth outcomes (did the trade make money?).

Xenon captures this data, cleans it, and then replays prompts through smaller Qwen3 models to capture internal activations — specifically MoE router logits — for interpretability research. The core research question: **can we decode what the model "knows" about its trading decisions from its internal routing patterns?**

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

## Pipeline overview

```
Terminal Markets API
        │
        ▼
Phase 1: INGEST ──────────────────────────────────────────
        │  pipelines/ingest/
        │  API → SQLite (terminal_ingest.db) + gzipped JSON payloads
        │  Tables: vaults, strategies, inference_logs, full_logs, swaps
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
Phase 4: ANALYSIS (not yet built) ────────────────────────
           Router logit exploration, linear probes, expert specialization
```

## Phase 1: Ingest

### Entry point
```bash
uv run -m pipelines.ingest --top-n 3
```

### What it does
Three-phase pipeline via `TerminalBackfillIngestor`:
1. **Vault discovery** — paginate leaderboard, upsert vault configs + strategies
2. **Log collection** — per vault, paginate inference logs, fetch full-log payloads (complete LLM prompt + completion + context), parse and store
3. **Swap collection** — per vault, paginate on-chain trade execution records

### Database schema (SQLite, WAL mode)

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

**`trade_outcomes`** — PnL labels (optional, from candle data)
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

### FK relationships
```
vaults.vault_address
  ├── strategies.vault_address
  ├── inference_logs.vault_address
  └── swaps.vault_address

inference_logs.id
  ├── full_logs.log_id
  └── swaps.log_id
```

### Full-log payload structure (gzipped JSON)

Each payload (~50KB) at `data/full_logs/{shard}/{log_id}.json.gz` contains:

```
{
  "snapshot": {
    "Agent": { vault config, persona, strategies },
    "Portfolio": { ETH balance, token holdings with entry prices + unrealized PnL },
    "Market": { ETH price, 16 tokens with prices + 1m/5m/1h/6h/24h metrics + liquidity },
    "AllowedTools": ["buy_token", "sell_token", "record_observation"],
    "Memories": [ recent action history ]
  },
  "llm_request_payload": {
    "llm_input": {
      "messages": [ system prompt + user prompt with full context ],
      "tools": [ buy_token, sell_token, record_observation function schemas ]
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
    "usage": { prompt_tokens, completion_tokens, reasoning_tokens, total_tokens }
  }
}
```

## Phase 2: Interp Data Prep

### Entry point
```bash
uv run -m pipelines.interp.prepare --db-path data/terminal_ingest.db --export-parquet
```

### What it does
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

### Output tables
- `interp_examples_v0` — primary table, one row per inference log
- `interp_context_gaps_v0` — rows with parse errors or missing context (diagnostics)
- `interp_sample_trade_v0` — balanced buy/sell sample (default 150 rows)
- `interp_sample_observation_v0` — observation sample (default 150)
- `interp_sample_paired_v0` — opposite decisions in same vault nearby (default 100)

### Key parquet export
`data/interp_exports/interp_examples_v0_high_quality.parquet` — this is what the capture pipeline reads. Columns include all context blocks, decision metadata, quality flags, and outcome labels.

## Phase 3: Activation Capture

### What we capture

**Residual stream** — The hidden state at each layer's output. Shape: `(num_layers, seq_len, hidden_dim)` in fp16. This is the "main information highway" through the transformer.

**Router logits** (MoE only) — At each layer, the MoE gate produces logits over all 128 experts, then selects the top-8. We capture:
- `router_logits`: `(num_layers, seq_len, 128)` fp32 — probability distribution over all experts
- `router_indices`: `(num_layers, seq_len, 8)` int16 — which 8 experts were selected

Router logits are the primary interpretability signal because:
- They're small (128 values per token per layer vs 2048+ for residual)
- They're interpretable (which expert was selected is a discrete, meaningful quantity)
- They capture the model's "routing decision" — how it allocates compute across specialized sub-networks

### How capture works

1. Load high-quality examples from parquet
2. Tokenize prompt_messages_json with Qwen3 chat template
3. Register PyTorch forward hooks:
   - Residual: `model.model.layers[i].register_forward_hook(...)` — captures `output[0]`
   - Router: `model.model.layers[i].mlp.gate.register_forward_hook(...)` — captures gate output tuple `(logits, scores, indices)`
4. Run forward pass (no backward, no generation — just encode the prompt)
5. Stack captured tensors, save to safetensors files
6. Write metadata parquet

### MoE auto-detection
`_is_moe_model(model)` checks `hasattr(model.model.layers[0].mlp, 'gate')`. If dense model + `capture_router=True`, prints warning and skips router hooks. This lets the same code run on both Qwen3-8B (dense) and Qwen3-30B-A3B (MoE).

### Storage layout
```
data/activations/
├── residual_stream/{log_id}.safetensors   # key: "residual_stream", (L, seq_len, hidden_dim) fp16
├── router_logits/{log_id}.safetensors     # keys: "router_logits" (L, seq_len, 128) fp32
│                                          #        "router_indices" (L, seq_len, 8) int16
└── metadata.parquet                       # log_id, seq_len, has_router, num_experts, ...
```

### Modal deployment
- App: `xenon-activation-capture`
- Volumes: `xenon-models` (cached weights), `xenon-data` (activations)
- GPU: A100-80GB
- Secret: `huggingface` (HF_TOKEN for model downloads)
- `CaptureWorker` cls with `@modal.enter()` for warm model across batches
- Local entrypoint fans out batches via `.map()`
- Wrapper script: `scripts/modal_capture.sh` (download, smoke, router, full, inspect, meta)

## Current state (as of March 2, 2026)

### Data volumes
- 3 vaults ingested
- ~121 high-quality interp examples
- 1 example fully captured on Modal (smoke test, all 48 layers, residual + router)
- Competition has been running ~4 days, ~15 days remaining

### What's built and tested
- Ingest pipeline: fully working, idempotent
- Data prep pipeline: fully working, quality tiers validated
- Local capture (Qwen3-8B): validated end-to-end
- Modal capture (Qwen3-30B-A3B): smoke tested, router logits confirmed
- 38 capture tests passing (includes MoE hooks, auto-detection, save/load round-trips)
- Shell scripts for Modal operations

### What's not built yet
- Analysis notebooks (router logit exploration)
- Linear probes (predict decision_type, trade_side, profitability from activations)
- PnL outcome labels may need candle data populated (check `trade_outcomes` table)
- No CI — tests are run manually

## File map

```
pipelines/
├── ingest/
│   ├── api.py              # TerminalMarketsApiClient (async, retry, semaphore)
│   ├── db.py               # IngestDatabase (schema + upserts)
│   ├── pipeline.py         # TerminalBackfillIngestor (3-phase orchestration)
│   ├── payload_store.py    # RawPayloadStore (atomic gzip writes)
│   ├── full_log_parser.py  # Parse LLM payloads
│   ├── explorer.py         # Web UI for data browsing
│   └── cli.py              # CLI entry point
└── interp/
    ├── prepare.py           # SQLite → interp_examples_v0 + parquet
    ├── capture.py           # Activation capture (hooks, save, MoE detection)
    ├── modal_capture.py     # Modal App, CaptureWorker, volumes
    └── outcomes.py          # PnL outcome labels from candle data

scripts/
└── modal_capture.sh         # Shell wrapper for Modal operations

tests/
├── test_ingest.py           # Ingest pipeline tests
└── test_capture.py          # Capture + MoE tests (38 tests)

data/
├── terminal_ingest.db       # SQLite database
├── full_logs/               # Gzipped JSON payloads
├── interp_exports/          # Parquet exports
└── activations/             # Safetensor capture output
```

## Dependencies

```toml
# Base (always installed)
aiohttp, aiosqlite, pyarrow

# interp extra (local capture)
torch, transformers, safetensors

# modal extra (remote capture)
modal

# dev extra
pytest
```

## Key design decisions

1. **Replay through smaller model, not the competition model** — Qwen3-30B-A3B has the same MoE architecture as 235B but fits on one GPU. Research validity depends on routing patterns being structurally similar across model scales.

2. **Router logits as primary signal** — Small (128 floats/token/layer), interpretable, and capture the model's expert allocation decision. Residual stream is huge and harder to interpret directly.

3. **Safetensors per log_id** — Simple random access. One file per example, not one file per layer. Good for the current scale (~hundreds of examples). May need resharding if we go to thousands.

4. **Idempotent everything** — Ingest uses upserts, capture has `--skip-existing`, data prep rebuilds tables each run. Safe to re-run any stage.

5. **Local dev + Modal prod** — Same capture code runs both places. Local for iteration (Qwen3-8B on MPS), Modal for production captures (Qwen3-30B-A3B on A100).
