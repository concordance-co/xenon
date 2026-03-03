# Next Steps

## Where we are

- Ingest pipeline: working, data in SQLite + gzipped payloads
- Interp data prep: working, high-quality parquet exports ready
- Local capture (Qwen3-8B): validated end-to-end on M4 Max
- Modal capture (Qwen3-30B-A3B): smoke tested on A100-80GB, router logits + residual stream captured
- Tests: 38 passing (capture), ingest tests passing separately

## Immediate: Backfill first 24h of competition data

Currently have 3 vaults, ~121 high-quality examples, 1 fully captured on Modal (smoke test). Next step is to widen the data before doing a full capture run.

```bash
# Backfill top 10 vaults (or more)
uv run -m pipelines.ingest --top-n 10

# Rebuild interp dataset with new data
uv run -m pipelines.interp.prepare --db-path data/terminal_ingest.db --export-parquet

# Check what we got
sqlite3 data/terminal_ingest.db "
  SELECT label_quality, COUNT(*) FROM interp_examples_v0 GROUP BY 1;
"
```

Then run the full capture on the expanded dataset:

```bash
# Router logits only first (small, fast, primary signal)
./scripts/modal_capture.sh router

# Then full if storage/cost allows
./scripts/modal_capture.sh full
```

Monitor with `./scripts/modal_capture.sh inspect` and `./scripts/modal_capture.sh meta`.

Router-only files are small (~0.5MB/example/layer vs ~38MB for residual+router). At ~2s/example, even 500+ examples would finish in <20 min compute.

## Analysis: Router logit exploration

Once you have router captures, initial questions to answer:

- **Expert specialization**: Do certain experts consistently activate for trade vs observation decisions?
- **Layer-wise routing**: How do routing patterns change across layers? Early layers generic, later layers decision-specific?
- **Token-level patterns**: Which tokens in the prompt drive expert selection? Market data tokens vs strategy tokens vs portfolio tokens?
- **Vault consistency**: Does the same vault route similarly across decisions, or does routing vary with market context?

This probably wants a notebook. Load safetensors, compute basic stats (expert frequency histograms, cosine similarity of routing patterns across examples).

## Linear probes

Train simple linear classifiers on router logits to predict:

- `decision_type` (trade vs observation)
- `trade_side` (buy vs sell)
- `was_profitable_1h` (if outcome labels are populated)

Per-layer probes reveal where in the network these decisions crystallize.

## Data management

### Backfill more data

The competition runs through Mar 19. More vaults = more diverse routing patterns.

```bash
uv run -m pipelines.ingest --top-n 10
uv run -m pipelines.interp.prepare --db-path data/terminal_ingest.db --export-parquet
```

Then re-run capture on the expanded dataset.

### PnL outcome labels

`outcomes.py` exists but may need candle data populated. Outcome labels (`pnl_1h_pct`, `was_profitable_1h`) are essential for supervised probing. Check:

```bash
sqlite3 data/terminal_ingest.db "SELECT COUNT(*) FROM trade_outcomes;"
```

If empty, need to run the outcomes pipeline to fetch candle data and compute PnL.

### Volume data lifecycle

Modal volumes persist indefinitely but cost storage. Consider:

- Downloading final captures locally or to cloud storage after runs
- Clearing old/partial captures from volumes between iterations

```bash
# List what's on the volume
./scripts/modal_capture.sh inspect

# Download from volume if needed
modal volume get xenon-data activations/ ./data/activations/ --force
```

## Deeper testing

- **Capture regression test**: Run local capture with `--limit 1` as a CI-style check that the pipeline still works end-to-end (currently manual)
- **Modal integration smoke**: Can't easily test Modal in CI, but could add a `--dry-run` flag that validates config + parquet loading without spinning up GPU
- **Outcome pipeline tests**: `outcomes.py` doesn't have tests yet — add coverage if you're using PnL labels for probing

## Architecture considerations

### When to switch to `modal deploy`

If you find yourself running capture frequently (multiple times per day), switch to deployed mode to avoid cold starts:

```python
# In modal_capture.py, add to @app.cls:
scaledown_window=300,  # keep warm for 5 min after last call
```

Then `modal deploy pipelines/interp/modal_capture.py` and trigger from Python:

```python
import modal
worker = modal.Cls.lookup("xenon-activation-capture", "CaptureWorker")
results = worker().capture_batch.remote(rows)
```

### Larger models

If you want to capture from Qwen3-235B-A22B (the actual competition model), you'd need multi-GPU (H100x4 or similar). The capture code would work as-is — just change `model_id` and GPU config in Modal. But storage costs scale significantly.

### Activation storage format

Current format (one safetensor per log_id) is simple and good for random access. If the dataset grows large (1000+ examples × 48 layers), consider:

- Sharding by layer instead of by example (better for per-layer analysis)
- Streaming format for sequential processing
- Compression (router logits compress well since they're sparse-ish)
