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
