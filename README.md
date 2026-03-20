# Xenon

Data pipeline for Terminal Markets inference logs → mechanistic interpretability.

Terminal Markets is a live AI trading competition where ~hundreds of AI vaults trade 16 meme tokens on Base 24/7, all running Qwen3-235B-A22B. Xenon captures the full decision-making pipeline — prompts, context, model outputs, trade executions — and then replays those prompts through smaller Qwen3 models to capture internal activations for interpretability research.

## How data flows

### Phase 1: Ingest (API → Neon Postgres)

The ingest pipeline pulls raw data from the Terminal Markets API: vault configs, inference logs (every LLM decision), full-log payloads (complete prompt + completion + context snapshots), and swap records (on-chain trade executions).

Everything lands in Neon Postgres. Full API payloads are stored as JSONB in the `full_logs.raw_payload` column. The pipeline is idempotent — re-running updates existing rows via upserts.

```bash
# Run on Modal (production)
./scripts/modal_capture.sh modal-ingest --top-n 10 --selection random

# Or run locally (requires XENON_NEON_DATABASE_URL in .env)
uv run -m pipelines.ingest --top-n 3
```

Verify with the Neon query tool:

```bash
uv run python scripts/neon_query.py tables
uv run python scripts/neon_query.py sample vaults
```

### Phase 2: Interp data prep (Neon → Neon)

The prepare pipeline joins inference logs with their full payloads (JSONB), extracts structured context blocks (messages, market data, portfolio, strategy, config, memory), normalizes decisions (trade vs observation), and assigns quality tiers.

Output: a clean `interp_examples_v0` table in Neon filtered to high-quality rows with complete context.

```bash
# Run on Modal (production)
./scripts/modal_capture.sh modal-prep
```

For market-manifold work, you can also export structured research tables directly from `full_logs.raw_payload`:

```bash
./scripts/modal_capture.sh manifold-export --output-dir data/interp_exports/manifolds --limit 1000
```

This writes:

- `tick_records.parquet`
- `asset_records.parquet`
- `pairwise_records.parquet`

### Phase 2b: Trade outcomes (API → Neon)

Enriches swaps with forward-looking PnL by fetching candle data from the Terminal Markets API.

```bash
./scripts/modal_capture.sh modal-outcomes
```

### Phase 3: Modal activation capture (Qwen3-30B-A3B, MoE)

Runs on Modal with A100-80GB GPUs, using Qwen3-30B-A3B which has MoE layers (128 experts, top-8 active). Captures both residual-stream activations and MoE router logits — the router logits are the primary signal for interpretability since they reveal which experts the model selects for each token.

```bash
# One-time: cache model weights to Modal volume
./scripts/modal_capture.sh download

# Smoke test: 1 example, layer 24
./scripts/modal_capture.sh smoke

# Router logits only (recommended for scale — small and information-dense)
./scripts/modal_capture.sh router

# Full capture (residual + router)
./scripts/modal_capture.sh full
```

## Data layout

```
Neon Postgres (XENON_NEON_DATABASE_URL):
├── vaults                  # vault configs, persona, risk params
├── strategies              # per-vault trading strategies
├── inference_logs          # every LLM decision (tool, args, status)
├── full_logs               # parsed fields + raw_payload JSONB
├── swaps                   # on-chain trade executions
├── trade_outcomes          # forward-looking PnL (1h, 4h, 1d)
├── interp_examples_v0      # denormalized prep output
└── ingest_cursors          # pagination state for incremental ingest

Modal Volume (xenon-data):
├── activations/
│   ├── residual_stream/
│   │   └── {log_id}.safetensors     # (L, seq_len, hidden_dim) fp16
│   ├── router_logits/
│   │   └── {log_id}.safetensors     # (L, seq_len, 128) fp32 + indices
│   └── metadata.parquet
└── dashboard_stats.json             # cached stats for dashboard
```

## Setup

```bash
uv sync                              # base deps (ingest + data prep)
uv sync --extra interp               # + torch, transformers, safetensors
uv sync --extra modal                # + modal (remote capture only)
uv sync --extra dev                  # + pytest
```

Environment:
```bash
# .env file (or set directly)
XENON_NEON_DATABASE_URL=postgresql://...   # Neon Postgres DSN
```

For Modal:
```bash
modal token new                                        # authenticate
modal secret create xenon-neon XENON_NEON_DATABASE_URL=postgresql://...
modal secret create huggingface HF_TOKEN=<your-token>  # for model downloads
```

## Tests

```bash
uv run --extra dev -m pytest tests/ -v
```
