## Interp Pipeline

Lightweight replay tooling for activation/SAE iteration.

## Dataset Prep (Context-Complete)

Build interp-ready tables focused on `trade` + `record_observation` decisions:

```bash
uv run -m pipelines.interp.prepare --db-path data/terminal_ingest.db
```

Export JSONL for inference jobs:

```bash
uv run -m pipelines.interp.prepare \
  --db-path data/terminal_ingest.db \
  --export-jsonl \
  --export-dir data/interp_exports
```

This materializes:

- `interp_examples_v0`
- `interp_context_gaps_v0`
- `interp_sample_trade_v0`
- `interp_sample_observation_v0`
- `interp_sample_paired_v0`

Detailed runbook:

- `/Users/trentelmore/concordance/xenon/pipelines/interp/DATA_PREP_RUNBOOK.md`

## CLI

```bash
uv run -m pipelines.interp \
  --db-path data/terminal_ingest.db \
  --api-base-url https://concordance--qwen3-openai-http-app.modal.run \
  --api-kind openai_chat \
  --endpoint-path /v1/chat/completions \
  --auth-header-value trent-alpha-test \
  --auto-try-auth-headers \
  --limit 25
```

This command reads candidate rows from `inference_logs/full_logs` and writes
results into `interp_runs`:

- `feature_timeline_json`
- `unique_feature_count`
- `timeline_position_count`
- `run_error` (if API call failed)

By default, request bodies are built from raw full-log message arrays
(`llm_request_payload.llm_input.messages`) to preserve structure for replay.
`model` is sent only when `--model-id` is provided.

## Activation Capture (Local)

Capture residual-stream activations from Qwen3-8B for all interp examples. Requires the `interp` optional dependency group (torch, transformers, safetensors).

### Install

```bash
uv sync --extra interp
```

### Validate tokenization

Inspect chat-template tokenization for 3 samples without running inference:

```bash
uv run --extra interp -m pipelines.interp.capture --validate-tokens
```

### Capture activations

```bash
# Single example
uv run --extra interp -m pipelines.interp.capture --limit 1

# Specific layers only
uv run --extra interp -m pipelines.interp.capture --layers 0,12,24,35

# Full run (all examples)
uv run --extra interp -m pipelines.interp.capture

# Resume (skip already-captured log_ids)
uv run --extra interp -m pipelines.interp.capture --skip-existing
```

### Output structure

```
data/activations/
├── residual_stream/
│   ├── {log_id}.safetensors    # (num_layers, seq_len, 4096) fp16
│   └── ...
└── metadata.parquet            # log_id, seq_len, num_layers_captured, prompt_hash, etc.
```

### Verify output

```python
from safetensors import safe_open
f = safe_open("data/activations/residual_stream/<log_id>.safetensors", framework="pt")
t = f.get_tensor("residual_stream")
print(f"Shape: {t.shape}")  # (36, seq_len, 4096) for Qwen3-8B
print(f"Dtype: {t.dtype}")  # float16
```

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

### Notes

- **MPS fp16**: If MPS fails with fp16, the module retries with float32 model weights (hooks still store fp16). Use `--device cpu` as last resort.
- **First run**: Downloads ~16GB of Qwen3-8B weights to HuggingFace cache.
- **Memory**: At 36 layers × 8K tokens × 4096 dim × 2 bytes ≈ 2.25GB per inference. Fine on 64GB M4 Max.
- **Qwen3-8B is dense**: No MoE router logits to capture. Those come with Qwen3-30B-A3B on Modal.

## Tests

```bash
uv run --extra interp --extra dev -m pytest tests/test_capture.py -v
```

Tests use fake model/tokenizer objects (no 16GB download) and cover:

- Message parsing from parquet rows
- Hook registration, capture, and cleanup
- Activation shape/dtype correctness
- Safetensor round-trip serialization
- End-to-end `run_capture` with skip-existing, limit, validate-tokens
- CLI argument parsing
