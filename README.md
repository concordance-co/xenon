# Xenon

Data pipeline for Terminal Markets inference logs → mechanistic interpretability.

## Setup

```bash
uv sync                    # base deps (ingest + data prep)
uv sync --extra interp     # + torch, transformers, safetensors (activation capture)
uv sync --extra dev        # + pytest
```

## 1. Ingest

Backfill vaults, inference logs, full-log payloads, and swaps from Terminal Markets API into SQLite + gzipped JSON.

```bash
uv run -m pipelines.ingest --top-n 3
```

| Flag | Default | Description |
|------|---------|-------------|
| `--top-n` | `3` | Number of leaderboard vaults to ingest |
| `--db-path` | `data/terminal_ingest.db` | SQLite database path |
| `--raw-payload-dir` | `data/full_logs` | Gzipped payload storage |
| `--leaderboard-sort-by` | `total_pnl_usd` | Sort metric |
| `--request-limit` | `50` | API page size |
| `--request-concurrency` | `10` | Concurrent API requests |
| `--timeout-s` | `30` | Request timeout |
| `--retry-max-attempts` | `6` | Max retries per request |
| `--max-logs-per-vault` | unlimited | Cap inference logs per vault |
| `--max-full-logs-per-vault` | unlimited | Cap full-log fetches per vault |
| `--max-swaps-per-vault` | unlimited | Cap swaps per vault |
| `--exclude-reasoning` | off | Omit reasoning_content from parsing |

### Data Explorer

```bash
uv run -m pipelines.ingest.explorer --db-path data/terminal_ingest.db --port 8765
```

## 2. Interp Data Prep

Build interp-ready dataset from ingested data. Filters to high-quality `trade` + `record_observation` decisions with full context.

```bash
uv run -m pipelines.interp.prepare --db-path data/terminal_ingest.db
```

| Flag | Default | Description |
|------|---------|-------------|
| `--db-path` | `data/terminal_ingest.db` | Source database |
| `--limit` | `50000` | Max rows to scan |
| `--include-all-decisions` | off | Keep all decision types, not just trade + observation |
| `--trade-sample-size` | `150` | Trade sample table size |
| `--observation-sample-size` | `150` | Observation sample table size |
| `--paired-sample-size` | `100` | Paired sample table size |
| `--export-jsonl` | off | Export tables to JSONL |
| `--export-parquet` | off | Export tables to Parquet |
| `--export-dir` | `data/interp_exports` | Export output directory |

## 3. Activation Capture

Capture residual-stream activations from Qwen3-8B for interp examples. Requires `--extra interp` deps.

```bash
# Validate tokenization (no inference)
uv run --extra interp -m pipelines.interp.capture --validate-tokens

# Capture 1 example
uv run --extra interp -m pipelines.interp.capture --limit 1

# Full run
uv run --extra interp -m pipelines.interp.capture

# Specific layers only
uv run --extra interp -m pipelines.interp.capture --layers 0,12,24,35
```

| Flag | Default | Description |
|------|---------|-------------|
| `--parquet-path` | `data/interp_exports/interp_examples_v0_high_quality.parquet` | Input examples |
| `--output-dir` | `data/activations` | Output directory |
| `--model-id` | `Qwen/Qwen3-8B` | HuggingFace model ID |
| `--device` | `mps` | Compute device (`mps`, `cpu`, `cuda`) |
| `--limit` | all | Process only N examples |
| `--layers` | all | Comma-separated layer indices |
| `--skip-existing` | off | Skip log_ids with existing safetensor files |
| `--validate-tokens` | off | Print tokenization details for 3 samples and exit |
| `--add-generation-prompt` | off | Append assistant turn start tokens |

Output: `data/activations/residual_stream/{log_id}.safetensors` + `data/activations/metadata.parquet`

## Tests

```bash
uv run --extra dev -m pytest tests/test_ingest.py -v          # ingest only
uv run --extra interp --extra dev -m pytest tests/test_capture.py -v  # capture only
uv run --extra interp --extra dev -m pytest tests/ -v         # all
```

## Detailed Docs

- [`pipelines/ingest/INGEST_RUNBOOK.md`](pipelines/ingest/INGEST_RUNBOOK.md) — schema reference, SQL verification queries, architecture
- [`pipelines/interp/DATA_PREP_RUNBOOK.md`](pipelines/interp/DATA_PREP_RUNBOOK.md) — interp dataset extraction rules, quality tiers
- [`pipelines/interp/README.md`](pipelines/interp/README.md) — activation capture details, output format, operational notes
