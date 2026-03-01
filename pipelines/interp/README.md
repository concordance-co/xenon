# Interp Pipeline

Replay tooling for activation capture and interpretability research.

## Data flow

```
terminal_ingest.db (raw inference logs + payloads)
    │
    ▼  pipelines.interp.prepare
interp_examples_v0 table → interp_examples_v0_high_quality.parquet
    │
    ▼  pipelines.interp.capture (local) or modal_capture.py (remote)
data/activations/
├── residual_stream/{log_id}.safetensors
├── router_logits/{log_id}.safetensors    (MoE only)
└── metadata.parquet
```

## Dataset Prep

Build interp-ready tables from ingested data. Filters to `trade` + `record_observation` decisions with complete context (messages, market, portfolio, strategy, config).

```bash
uv run -m pipelines.interp.prepare --db-path data/terminal_ingest.db --export-parquet
```

This creates:
- `interp_examples_v0` — primary table (all quality tiers)
- `interp_context_gaps_v0` — rows with parse errors or missing context
- `interp_sample_trade_v0` — balanced buy/sell sample (default 150)
- `interp_sample_observation_v0` — observation sample (default 150)
- `interp_sample_paired_v0` — opposite decisions in same vault nearby (default 100)

Verify:

```bash
sqlite3 data/terminal_ingest.db "SELECT label_quality, COUNT(*) FROM interp_examples_v0 GROUP BY 1;"
ls -la data/interp_exports/*.parquet
```

See [`DATA_PREP_RUNBOOK.md`](DATA_PREP_RUNBOOK.md) for extraction rules and quality tier definitions.

## Activation Capture

Two paths: local (development/validation) and Modal (production).

### Local capture (Qwen3-8B, dense)

Captures residual-stream activations on your local machine. Good for validating the pipeline and iterating on capture config. Qwen3-8B is dense — no MoE router logits.

```bash
uv sync --extra interp

# Validate tokenization (no inference)
uv run --extra interp -m pipelines.interp.capture --validate-tokens

# Single example
uv run --extra interp -m pipelines.interp.capture --limit 1

# Specific layers
uv run --extra interp -m pipelines.interp.capture --layers 0,12,24,35

# Resume (skip already-captured)
uv run --extra interp -m pipelines.interp.capture --skip-existing
```

Verify:

```bash
uv run --extra interp python -c "
from safetensors import safe_open
import glob
f = safe_open(glob.glob('data/activations/residual_stream/*.safetensors')[0], framework='pt')
t = f.get_tensor('residual_stream')
print(f'Shape: {t.shape}')  # (36, seq_len, 4096) for Qwen3-8B
print(f'Dtype: {t.dtype}')  # float16
"
```

### Modal capture (Qwen3-30B-A3B, MoE)

Captures both residual-stream activations and MoE router logits on A100-80GB. Router logits are the primary interpretability signal — they reveal which of the 128 experts the model selects (top-8) for each token at each layer.

```bash
uv sync --extra modal

# One-time setup
modal token new
modal secret create huggingface HF_TOKEN=<your-token>

# Cache model weights to volume (~18GB, ~5 min)
./scripts/modal_capture.sh download

# Smoke test: 1 example, layer 24
./scripts/modal_capture.sh smoke

# Router logits only (small + information-dense, recommended for scale)
./scripts/modal_capture.sh router

# Full capture (residual + router)
./scripts/modal_capture.sh full
```

Verify:

```bash
# Local metadata from last run
./scripts/modal_capture.sh meta

# List files on Modal volume
./scripts/modal_capture.sh inspect

# Inspect specific example on volume
./scripts/modal_capture.sh inspect --log-id 463208
```

All script commands accept extra flags:

```bash
./scripts/modal_capture.sh router --limit 10
./scripts/modal_capture.sh full --layers 0,24,47
```

### Output format

```
data/activations/
├── residual_stream/
│   └── {log_id}.safetensors
│       key: "residual_stream"
│       shape: (num_layers_captured, seq_len, hidden_dim) fp16
│
├── router_logits/                        # MoE models only
│   └── {log_id}.safetensors
│       key: "router_logits"
│       shape: (num_layers_captured, seq_len, 128) fp32
│       key: "router_indices"
│       shape: (num_layers_captured, seq_len, 8) int16
│
└── metadata.parquet
    columns: log_id, seq_len, num_layers_captured, hidden_dim,
             has_router, num_experts, prompt_hash, file_size_bytes, elapsed_s
```

Modal captures write to the `xenon-data` volume at `/data/activations/` (same layout).

### Model reference

| | Qwen3-8B (local) | Qwen3-30B-A3B (Modal) |
|---|---|---|
| Type | Dense | MoE |
| Layers | 36 | 48 |
| hidden_dim | 4096 | 2048 |
| Experts | — | 128 total, top-8 active |
| Gate class | — | `Qwen3MoeTopKRouter` at `.mlp.gate` |
| Captures | residual only | residual + router logits |
| Device | MPS / CPU | A100-80GB (CUDA) |

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--parquet-path` | `data/interp_exports/interp_examples_v0_high_quality.parquet` | Input examples |
| `--output-dir` | `data/activations` | Output directory |
| `--model-id` | `Qwen/Qwen3-8B` | HuggingFace model ID |
| `--device` | `mps` | Compute device (`mps`, `cpu`, `cuda`) |
| `--limit` | all | Process only N examples |
| `--layers` | all | Comma-separated layer indices |
| `--skip-existing` | off | Skip log_ids with existing safetensor files |
| `--validate-tokens` | off | Print tokenization details and exit |
| `--add-generation-prompt` | off | Append assistant turn start tokens |
| `--capture-router` / `--no-capture-router` | on | Capture MoE router logits (auto-skipped on dense models) |
| `--capture-residual` / `--no-capture-residual` | on | Capture residual stream |

### Operational notes

- **MoE auto-detection**: If `capture_router=True` but the model is dense (no `.mlp.gate`), router hooks are silently skipped with a warning.
- **MPS fp16**: If MPS fails with fp16, the module retries with float32 model weights (hooks still store fp16). Use `--device cpu` as last resort.
- **First local run**: Downloads ~16GB of Qwen3-8B weights to HuggingFace cache.
- **Local memory**: At 36 layers x 8K tokens x 4096 dim x 2 bytes = ~2.25GB per inference. Fine on 64GB M4 Max.
- **Modal cold start**: ~30s to load Qwen3-30B-A3B from volume into GPU memory. Model stays warm across batches within a single `modal run`.

## Tests

```bash
uv run --extra interp --extra dev -m pytest tests/test_capture.py -v
```

Tests use fake model/tokenizer objects (no GPU or model download) and cover:

- Message parsing from parquet rows
- Hook registration, capture, and cleanup
- Activation shape/dtype correctness (residual fp16, router logits fp32, router indices int16)
- MoE auto-detection (dense model skips router hooks)
- Router hook capture with correct tensor shapes
- Safetensor round-trip serialization (residual + router)
- End-to-end `run_capture` with MoE and dense models
- CLI argument parsing (including `--capture-router` / `--no-capture-residual` flags)
