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
  analyze-local)
    echo "Running analysis locally..."
    uv run --extra analysis -m pipelines.interp.analysis "$@"
    ;;
  analyze)
    echo "Running analysis on Modal (detached)..."
    uv run --extra analysis --extra modal modal run --detach pipelines/interp/modal_analysis.py "$@"
    echo ""
    echo "Downloading analysis results..."
    modal volume get xenon-data analysis_results/ ./data/ --force
    ;;
  decision-structure-pool)
    echo "Pooling real-decision structure activations on Modal (detached)..."
    uv run --extra analysis --extra modal modal run --detach pipelines/interp/modal_analysis.py --mode decision-structure "$@"
    echo ""
    echo "Downloading decision structure activations..."
    modal volume get xenon-data activations/decision_structure/ ./data/activations/decision_structure/ --force
    ;;
  decision-structure-analyze)
    echo "Analyzing pooled real-decision structure activations on Modal (detached)..."
    uv run --extra analysis --extra modal modal run --detach pipelines/interp/modal_analysis.py --mode decision-structure-analysis "$@"
    echo ""
    echo "Downloading decision structure analysis results..."
    modal volume get xenon-data analysis_results/decision_structure/ ./data/analysis_results/decision_structure/ --force
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
  manifold-export)
    echo "Exporting market manifold tables from Neon full_logs.raw_payload..."
    uv run python -m pipelines.interp.manifold_dataset "$@"
    ;;
  modal-outcomes)
    echo "Computing trade outcomes on Modal (detached)..."
    uv run --extra interp --extra modal modal run --detach pipelines/interp/modal_ingest.py --mode outcomes "$@"
    ;;
  migrate-to-neon)
    echo "Migrating SQLite → Neon Postgres (one-time)..."
    uv run --extra interp --extra modal modal run scripts/migrate_sqlite_to_neon.py "$@"
    ;;
  migrate-metadata)
    echo "Migrating capture metadata from Modal volume → Neon..."
    uv run --extra interp --extra modal modal run scripts/migrate_metadata_to_neon.py "$@"
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
  counterfactual-build)
    echo "Building counterfactual datasets (local → Neon DB)..."
    uv run python -m pipelines.interp.counterfactual "$@"
    ;;
  counterfactual-capture)
    echo "Running counterfactual capture on Modal (vLLM, H200, detached)..."
    uv run --extra interp --extra modal modal run --detach pipelines/interp/modal_vllm_capture.py --mode counterfactual "$@"
    ;;
  counterfactual-analyze)
    echo "Running counterfactual analysis on Modal (detached)..."
    uv run --extra analysis --extra modal modal run --detach pipelines/interp/modal_analysis.py --mode counterfactual "$@"
    echo ""
    echo "Downloading counterfactual results..."
    modal volume get xenon-data analysis_results/counterfactual/ ./data/analysis_results/counterfactual/ --force
    ;;
  counterfactual-structure)
    echo "Running counterfactual structure analysis on Modal (detached)..."
    uv run --extra analysis --extra modal modal run --detach pipelines/interp/modal_analysis.py --mode counterfactual-structure "$@"
    echo ""
    echo "Downloading counterfactual structure results..."
    modal volume get xenon-data analysis_results/counterfactual_structure/ ./data/analysis_results/counterfactual_structure/ --force
    ;;
  *)
    echo "Usage: $0 {download|smoke|router|full|inspect|meta|compact|analyze-local|analyze|decision-structure-pool|decision-structure-analyze|modal-ingest|modal-backfill|modal-prep|manifold-export|modal-outcomes|modal-snapshot|modal-stats|download-activations|download-results|counterfactual-build|counterfactual-capture|counterfactual-analyze|counterfactual-structure} [extra flags]"
    echo ""
    echo "  download             Cache model weights to volume (one-time)"
    echo "  smoke                Single example, single layer (sanity check)"
    echo "  router               Router logits only, all examples"
    echo "  full                 Full capture (residual + router)"
    echo "  inspect              List/inspect safetensors on Modal volume"
    echo "  meta                 Show local metadata.parquet summary"
    echo "  compact              Consolidate per-example files into per-layer matrices"
    echo "  analyze-local        Run analysis locally against local activations/labels"
    echo "  analyze              Run analysis on Modal (probe/experts/pca)"
    echo "  decision-structure-pool  Pool real-decision captures into row/section states"
    echo "  decision-structure-analyze  Probe asset binding on pooled decision structure states"
    echo "  modal-ingest         Run ingest on Modal (fetches from Terminal API → Neon)"
    echo "  modal-prep           Run data prep on Modal (Neon → Neon)"
    echo "  manifold-export      Export tick/asset/pairwise manifold tables from Neon"
    echo "  modal-outcomes       Compute trade outcomes (PnL) on Modal"
    echo "  reset-cursors        Wipe ingest_cursors (next run re-paginates from start)"
    echo "  migrate-to-neon      Migrate SQLite DB → Neon Postgres (one-time)"
    echo "  modal-backfill       Fill missing full logs + null payloads, then run swaps"
    echo "  modal-snapshot       Write & download stats snapshot"
    echo "  modal-stats          Download cached stats snapshot (no Modal run)"
    echo "  counterfactual-build    Build counterfactual datasets (Dataset A + B)"
    echo "  counterfactual-capture  Run counterfactual captures on Modal A100"
    echo "  counterfactual-analyze  Run three-question analysis on Modal"
    echo "  counterfactual-structure  Run pre/post structure analysis on Modal"
    echo "  download-activations Download activations from Modal volume"
    echo "  download-results     Download analysis results from Modal volume"
    echo ""
    echo "Examples:"
    echo "  $0 router --limit 10"
    echo "  $0 full --limit 50 --pool last_token"
    echo "  $0 inspect --log-id 463208"
    echo "  $0 analyze-local --target executed_valence --labels-path data/interp_exports/manifolds/tick_records.parquet"
    echo "  $0 analyze --mode probe --target decision_type"
    echo "  $0 decision-structure-pool --limit 500"
    echo "  $0 decision-structure-analyze --layers 16,24,32"
    echo "  $0 modal-ingest --top-n 10 --selection random"
    echo "  $0 modal-prep"
    echo "  $0 modal-prep --full-rebuild --incremental-rebuild   # fast SQL upsert for missing/stale rows"
    echo "  $0 manifold-export --output-dir data/interp_exports/manifolds --limit 1000"
    echo "  $0 counterfactual-analyze --questions all"
    echo "  $0 counterfactual-structure --experiment-id init --layers 16,24,32"
    ;;
esac
