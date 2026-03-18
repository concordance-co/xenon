#!/usr/bin/env bash
# Modal activation capture & pipeline commands
set -euo pipefail

CMD="${1:-help}"
shift || true

case "$CMD" in
  download)
    echo "Downloading model weights to volume..."
    uv run --extra interp --extra modal modal run pipelines/interp/modal_capture.py::download_model "$@"
    ;;
  smoke)
    echo "Smoke test: 1 example, layer 24..."
    uv run --extra interp --extra modal modal run pipelines/interp/modal_capture.py --limit 1 --layers 24 "$@"
    ;;
  router)
    echo "Router logits only (detached, no residual)..."
    uv run --extra interp --extra modal modal run --detach pipelines/interp/modal_capture.py --no-capture-residual "$@"
    ;;
  full)
    echo "Full capture (detached, residual + router)..."
    uv run --extra interp --extra modal modal run --detach pipelines/interp/modal_capture.py "$@"
    ;;
  inspect)
    echo "Inspecting volume contents..."
    uv run --extra interp --extra modal modal run pipelines/interp/modal_capture.py::inspect_volume "$@"
    ;;
  meta)
    echo "Local metadata:"
    uv run --extra interp python -c "
import pyarrow.parquet as pq, sys
p = 'data/activations/metadata.parquet'
try:
    t = pq.read_table(p)
except FileNotFoundError:
    print(f'  No metadata file at {p}'); sys.exit(0)
rows = t.to_pylist()
print(f'  {len(rows)} rows')
print(f'  columns: {t.column_names}')
for r in rows[:10]:
    parts = [f'log_id={r[\"log_id\"]}', f'seq_len={r[\"seq_len\"]}', f'{r[\"file_size_bytes\"]/1024/1024:.1f}MB', f'{r[\"elapsed_s\"]}s']
    if r.get('has_router'): parts.append(f'experts={r.get(\"num_experts\",\"?\")}')
    if r.get('num_layers_captured'): parts.append(f'layers={r[\"num_layers_captured\"]}')
    print(f'    {\"  \".join(parts)}')
if len(rows) > 10: print(f'    ... and {len(rows)-10} more')
"
    ;;
  compact)
    echo "Compacting activations on Modal (detached)..."
    uv run --extra analysis --extra modal modal run --detach pipelines/interp/modal_analysis.py --mode compact "$@"
    ;;
  analyze)
    echo "Running analysis on Modal (detached)..."
    uv run --extra analysis --extra modal modal run --detach pipelines/interp/modal_analysis.py "$@"
    echo ""
    echo "Downloading analysis results..."
    modal volume get xenon-data analysis_results/ ./data/ --force
    ;;
  download-activations)
    echo "Downloading activations from Modal volume..."
    modal volume get xenon-data activations/ ./data/activations/ --force
    ;;
  download-results)
    echo "Downloading analysis results from Modal volume..."
    modal volume get xenon-data analysis_results/ ./data/ --force
    ;;
  modal-ingest)
    echo "Running ingest on Modal (detached, writes to Neon Postgres)..."
    uv run --extra interp --extra modal modal run --detach pipelines/interp/modal_ingest.py --mode ingest "$@"
    ;;
  modal-backfill)
    echo "Backfilling missing full logs and null payloads (detached)..."
    uv run --extra interp --extra modal modal run --detach pipelines/interp/modal_ingest.py --mode ingest --selection backfill "$@"
    ;;
  modal-prep)
    echo "Running data prep on Modal (detached, reads/writes Neon Postgres)..."
    uv run --extra interp --extra modal modal run --detach pipelines/interp/modal_ingest.py --mode prep "$@"
    ;;
  modal-outcomes)
    echo "Computing trade outcomes on Modal (detached)..."
    uv run --extra interp --extra modal modal run --detach pipelines/interp/modal_ingest.py --mode outcomes "$@"
    ;;
  migrate-to-neon)
    echo "Migrating SQLite → Neon Postgres (one-time)..."
    uv run --extra interp --extra modal modal run scripts/migrate_sqlite_to_neon.py "$@"
    ;;
  reset-cursors)
    echo "Resetting ingest cursors (next run re-paginates from start)..."
    uv run --extra interp --extra modal modal run pipelines/interp/modal_ingest.py --mode reset-cursors "$@"
    ;;
  modal-snapshot)
    echo "Writing stats snapshot on Modal..."
    uv run --extra interp --extra modal modal run pipelines/interp/modal_ingest.py --mode snapshot "$@"
    echo "Downloading snapshot..."
    mkdir -p data
    modal volume get xenon-data dashboard_stats.json ./data/dashboard_stats.json --force
    ;;
  modal-stats)
    mkdir -p data
    modal volume get xenon-data dashboard_stats.json ./data/dashboard_stats.json --force 2>/dev/null
    ;;
  *)
    echo "Usage: $0 {download|smoke|router|full|inspect|meta|compact|analyze|modal-ingest|modal-backfill|modal-prep|modal-outcomes|modal-snapshot|modal-stats|download-activations|download-results} [extra flags]"
    echo ""
    echo "  download             Cache model weights to volume (one-time)"
    echo "  smoke                Single example, single layer (sanity check)"
    echo "  router               Router logits only, all examples"
    echo "  full                 Full capture (residual + router)"
    echo "  inspect              List/inspect safetensors on Modal volume"
    echo "  meta                 Show local metadata.parquet summary"
    echo "  compact              Consolidate per-example files into per-layer matrices"
    echo "  analyze              Run analysis on Modal (probe/experts/pca)"
    echo "  modal-ingest         Run ingest on Modal (fetches from Terminal API → Neon)"
    echo "  modal-prep           Run data prep on Modal (Neon → Neon)"
    echo "  modal-outcomes       Compute trade outcomes (PnL) on Modal"
    echo "  reset-cursors        Wipe ingest_cursors (next run re-paginates from start)"
    echo "  migrate-to-neon      Migrate SQLite DB → Neon Postgres (one-time)"
    echo "  modal-backfill       Fill missing full logs + null payloads, then run swaps"
    echo "  modal-snapshot       Write & download stats snapshot"
    echo "  modal-stats          Download cached stats snapshot (no Modal run)"
    echo "  download-activations Download activations from Modal volume"
    echo "  download-results     Download analysis results from Modal volume"
    echo ""
    echo "Examples:"
    echo "  $0 router --limit 10"
    echo "  $0 full --limit 50 --pool last_token"
    echo "  $0 inspect --log-id 463208"
    echo "  $0 analyze --mode probe --target decision_type"
    echo "  $0 modal-ingest --top-n 10 --selection random"
    echo "  $0 modal-prep"
    ;;
esac
