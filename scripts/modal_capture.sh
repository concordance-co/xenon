#!/usr/bin/env bash
# Modal activation capture — common commands
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
    echo "Router logits only (no residual)..."
    uv run --extra interp --extra modal modal run pipelines/interp/modal_capture.py --no-capture-residual "$@"
    ;;
  full)
    echo "Full capture (residual + router)..."
    uv run --extra interp --extra modal modal run pipelines/interp/modal_capture.py "$@"
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
    echo "Compacting activations on Modal..."
    uv run --extra analysis --extra modal modal run pipelines/interp/modal_analysis.py --mode compact "$@"
    ;;
  analyze)
    echo "Running analysis on Modal..."
    uv run --extra analysis --extra modal modal run pipelines/interp/modal_analysis.py "$@"
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
  upload-db)
    echo "Uploading local SQLite DB to Modal volume..."
    mkdir -p data
    DB_PATH="${2:-data/terminal_ingest.db}"
    if [ ! -f "$DB_PATH" ]; then
      echo "Error: DB not found at $DB_PATH"
      exit 1
    fi
    modal volume put xenon-data "$DB_PATH" ingest/terminal_ingest.db --force
    echo "Done. DB uploaded to xenon-data:/ingest/terminal_ingest.db"
    ;;
  download-db)
    echo "Downloading DB from Modal volume..."
    mkdir -p data
    modal volume get xenon-data ingest/terminal_ingest.db ./data/terminal_ingest.db --force
    echo "Done. DB saved to data/terminal_ingest.db"
    ;;
  modal-ingest)
    echo "Running ingest on Modal..."
    uv run --extra interp --extra modal modal run pipelines/interp/modal_ingest.py --mode ingest "$@"
    ;;
  modal-prep)
    echo "Running data prep on Modal..."
    uv run --extra interp --extra modal modal run pipelines/interp/modal_ingest.py --mode prep "$@"
    ;;
  backfill-payloads)
    echo "Migrating file payloads into DB (one-time)..."
    uv run --extra interp --extra modal modal run pipelines/interp/modal_ingest.py --mode backfill-payloads
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
    echo "Usage: $0 {download|smoke|router|full|inspect|meta|compact|analyze|upload-db|download-db|modal-ingest|modal-prep|modal-stats|download-activations|download-results} [extra flags]"
    echo ""
    echo "  download             Cache model weights to volume (one-time)"
    echo "  smoke                Single example, single layer (sanity check)"
    echo "  router               Router logits only, all examples"
    echo "  full                 Full capture (residual + router)"
    echo "  inspect              List/inspect safetensors on Modal volume"
    echo "  meta                 Show local metadata.parquet summary"
    echo "  compact              Consolidate per-example files into per-layer matrices"
    echo "  analyze              Run analysis on Modal (probe/experts/pca)"
    echo "  upload-db            Upload local SQLite DB to Modal volume"
    echo "  download-db          Download DB from Modal volume to local"
    echo "  modal-ingest         Run ingest on Modal (fetches from Terminal API)"
    echo "  modal-prep           Run data prep on Modal"
    echo "  backfill-payloads    Migrate file payloads into DB inline (one-time)"
    echo "  modal-snapshot       Write & download stats snapshot from Modal DB"
    echo "  modal-stats          Download cached stats snapshot (no Modal run)"
    echo "  download-activations Download activations from Modal volume"
    echo "  download-results     Download analysis results from Modal volume"
    echo ""
    echo "Examples:"
    echo "  $0 router --limit 10"
    echo "  $0 full --limit 50 --pool last_token"
    echo "  $0 inspect --log-id 463208      # inspect specific example"
    echo "  $0 analyze --mode probe --target decision_type"
    echo "  $0 analyze --mode all --target risk_tolerance"
    echo "  $0 upload-db"
    echo "  $0 modal-ingest --top-n 10 --selection random"
    echo "  $0 modal-prep --export-parquet"
    ;;
esac
