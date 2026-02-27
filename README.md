## Terminal Markets Ingestion (Phase 1 + Phase 2)

This repo now includes a one-shot backfill CLI for:

- Phase 1: leaderboard vault discovery + vault config + strategies
- Phase 2: inference log metadata + full-log payload collection

### Layout

- `pipelines/ingest/` - ingestion implementation (API client, DB, parser, storage, pipeline orchestration)
- `pipelines/interp/` - reserved for replay/activation/SAE work

### Storage model

- Structured metadata is stored in SQLite (`data/terminal_ingest.db` by default).
- Raw `/full-log/{id}` payloads are stored as gzip JSON files in `data/full_logs/`.
- The `full_logs` table stores `payload_path` and `payload_sha256` so rows can be joined back to raw files by `log_id`.

### Run

```bash
uv run -m pipelines.ingest --top-n 3
```

Useful flags:

```bash
uv run -m pipelines.ingest \
  --top-n 3 \
  --db-path data/terminal_ingest.db \
  --raw-payload-dir data/full_logs \
  --request-concurrency 10 \
  --request-limit 50
```

Validation flags (optional):

```bash
uv run -m pipelines.ingest --top-n 3 --max-logs-per-vault 100 --max-full-logs-per-vault 100
```

### Explore Data (Web UI)

Start a local read-only explorer for the ingested SQLite data:

```bash
uv run -m pipelines.ingest.explorer --db-path data/terminal_ingest.db --port 8765
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765).

The explorer includes a **Dataset Readiness (Mech Interp)** section with:

- full-log coverage and token-usage coverage
- trade-link readiness (`transaction_hash`, optional swaps/outcomes joins when those tables exist)
- tool distribution and suggested next pipeline steps
- candidate rows to inspect likely dataset entries

Useful explorer flags:

```bash
uv run -m pipelines.ingest.explorer \
  --db-path data/terminal_ingest.db \
  --host 127.0.0.1 \
  --port 8765 \
  --payload-preview-chars 12000
```
