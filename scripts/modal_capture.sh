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
  *)
    echo "Usage: $0 {download|smoke|router|full|inspect|meta} [extra flags]"
    echo ""
    echo "  download  Cache model weights to volume (one-time)"
    echo "  smoke     Single example, single layer (sanity check)"
    echo "  router    Router logits only, all examples"
    echo "  full      Full capture (residual + router)"
    echo "  inspect   List/inspect safetensors on Modal volume"
    echo "  meta      Show local metadata.parquet summary"
    echo ""
    echo "Examples:"
    echo "  $0 router --limit 10"
    echo "  $0 inspect --log-id 463208      # inspect specific example"
    ;;
esac
