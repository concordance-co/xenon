# Xenon

Data pipeline for Terminal Markets inference logs → mechanistic interpretability.

Terminal Markets is a live AI trading competition where ~hundreds of AI vaults trade 16 meme tokens on Base 24/7, all running Qwen3-235B-A22B. Xenon captures the full decision-making pipeline — prompts, context, model outputs, trade executions — and then replays those prompts through smaller Qwen3 models to capture internal activations for interpretability research.

## How data flows

### Phase 1: Ingest (API → SQLite + gzip)

The ingest pipeline pulls raw data from the Terminal Markets API: vault configs, inference logs (every LLM decision), full-log payloads (complete prompt + completion + context snapshots), and swap records (on-chain trade executions).

Everything lands in a local SQLite database (`data/terminal_ingest.db`) plus gzipped JSON payload files (`data/full_logs/{shard}/{log_id}.json.gz`). The pipeline is idempotent — re-running updates existing rows via upserts.

```bash
uv run -m pipelines.ingest --top-n 3
```

Verify it worked:

```bash
sqlite3 data/terminal_ingest.db "
  SELECT 'vaults' AS tbl, COUNT(*) AS n FROM vaults
  UNION ALL SELECT 'inference_logs', COUNT(*) FROM inference_logs
  UNION ALL SELECT 'full_logs', COUNT(*) FROM full_logs
  UNION ALL SELECT 'swaps', COUNT(*) FROM swaps;
"
```

Browse the data in a web UI:

```bash
uv run -m pipelines.ingest.explorer --db-path data/terminal_ingest.db --port 8765
```

See [`pipelines/ingest/INGEST_RUNBOOK.md`](pipelines/ingest/INGEST_RUNBOOK.md) for schema reference and SQL verification queries.

### Phase 2: Interp data prep (SQLite → cleaned parquet)

The prepare pipeline joins inference logs with their full payloads, extracts structured context blocks (messages, market data, portfolio, strategy, config, memory), normalizes decisions (trade vs observation), and assigns quality tiers.

Output: a clean `interp_examples_v0` table filtered to high-quality rows with complete context, plus sampled subsets for different analysis needs.

```bash
uv run -m pipelines.interp.prepare --db-path data/terminal_ingest.db --export-parquet
```

Verify it worked:

```bash
sqlite3 data/terminal_ingest.db "
  SELECT label_quality, COUNT(*) FROM interp_examples_v0 GROUP BY 1;
"
ls -la data/interp_exports/
```

The key output is `data/interp_exports/interp_examples_v0_high_quality.parquet` — this is what the activation capture pipeline reads.

See [`pipelines/interp/DATA_PREP_RUNBOOK.md`](pipelines/interp/DATA_PREP_RUNBOOK.md) for extraction rules and quality tier definitions.

### Phase 3a: Local activation capture (Qwen3-8B, dense)

For development and validation. Replays prompts through Qwen3-8B on your local machine (M4 Max / MPS), capturing residual-stream activations at each layer.

```bash
# Validate tokenization first (no model inference)
uv run --extra interp -m pipelines.interp.capture --validate-tokens

# Capture 1 example, specific layers
uv run --extra interp -m pipelines.interp.capture --limit 1 --layers 0,12,24,35
```

Verify output:

```bash
ls -la data/activations/residual_stream/
python -c "
from safetensors import safe_open
f = safe_open('data/activations/residual_stream/$(ls data/activations/residual_stream/ | head -1)', framework='pt')
t = f.get_tensor('residual_stream')
print(f'Shape: {t.shape}')  # (num_layers, seq_len, 4096) for Qwen3-8B
print(f'Dtype: {t.dtype}')  # float16
"
```

Qwen3-8B is dense (no MoE), so only residual-stream activations are captured here.

### Phase 3b: Modal activation capture (Qwen3-30B-A3B, MoE)

For production captures. Runs on Modal with A100-80GB GPUs, using Qwen3-30B-A3B which has MoE layers (128 experts, top-8 active). This captures both residual-stream activations and MoE router logits — the router logits are the primary signal for interpretability since they reveal which experts the model selects for each token.

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

Verify output:

```bash
# Check local metadata from the last run
./scripts/modal_capture.sh meta

# List files on the Modal volume
./scripts/modal_capture.sh inspect

# Inspect a specific example's tensor shapes on the volume
./scripts/modal_capture.sh inspect --log-id 463208
```

Data lives on two Modal volumes:
- `xenon-models` — cached Qwen3-30B-A3B weights (~18GB)
- `xenon-data` — captured activations (safetensors + metadata)

## Data layout

```
data/
├── terminal_ingest.db                    # SQLite: vaults, inference_logs, full_logs, swaps
├── full_logs/                            # Gzipped JSON payloads
│   └── {shard}/{log_id}.json.gz          #   shard = log_id // 1000, zero-padded
├── interp_exports/                       # Parquet exports from prepare step
│   ├── interp_examples_v0_high_quality.parquet  # ← input to capture pipeline
│   ├── interp_sample_trade_v0.parquet
│   ├── interp_sample_observation_v0.parquet
│   └── interp_sample_paired_v0.parquet
└── activations/                          # Capture output (local runs)
    ├── residual_stream/
    │   └── {log_id}.safetensors          #   key: "residual_stream", (L, seq_len, hidden_dim) fp16
    ├── router_logits/                    #   (MoE models only)
    │   └── {log_id}.safetensors          #   keys: "router_logits" (L, seq_len, 128) fp32
    │                                     #         "router_indices" (L, seq_len, 8) int16
    └── metadata.parquet                  #   log_id, seq_len, has_router, num_experts, etc.
```

Modal volume (`xenon-data`) has the same `activations/` layout at `/data/activations/`.

## Setup

```bash
uv sync                              # base deps (ingest + data prep)
uv sync --extra interp               # + torch, transformers, safetensors
uv sync --extra modal                # + modal (remote capture only)
uv sync --extra dev                  # + pytest
```

For Modal, you also need:
```bash
modal token new                      # authenticate with Modal
modal secret create huggingface HF_TOKEN=<your-token>  # for model downloads
```

## Tests

```bash
uv run --extra dev -m pytest tests/test_ingest.py -v          # ingest
uv run --extra interp --extra dev -m pytest tests/test_capture.py -v  # capture (local)
uv run --extra interp --extra dev -m pytest tests/ -v          # all
```

Capture tests use fake model/tokenizer objects (no GPU or model download required) and cover residual capture, MoE router hooks, auto-detection (dense vs MoE), safetensor round-trips, and end-to-end `run_capture` with all config combinations.

## Detailed docs

- [`pipelines/ingest/INGEST_RUNBOOK.md`](pipelines/ingest/INGEST_RUNBOOK.md) — SQLite schema, SQL verification queries, API client reference
- [`pipelines/interp/DATA_PREP_RUNBOOK.md`](pipelines/interp/DATA_PREP_RUNBOOK.md) — context extraction rules, quality tiers, sampling logic
- [`pipelines/interp/README.md`](pipelines/interp/README.md) — activation capture details (local + Modal), output formats, MoE router capture
