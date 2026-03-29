#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

launch_job() {
  local run_name="$1"
  local log_path="$2"
  shift 2
  nohup uv run --extra interp --extra modal modal run \
    pipelines/interp/modal_vllm_patching.py::run_synthetic_market_behavior_modal \
    "$@" >"$log_path" 2>&1 &
  printf 'launched %s pid=%s log=%s\n' "$run_name" "$!" "$log_path"
}

launch_job \
  phase18_market_behavior_baseline_stratified48_chunked_v3 \
  /tmp/phase18_market_behavior_baseline_stratified48_chunked_v3.txt \
  --phase-name phase15_market_basis_discovery_v1 \
  --run-name phase18_market_behavior_baseline_stratified48_chunked_v3 \
  --context-variant market_only \
  --selection-strategy stratified_family_variant_roster \
  --limit 48 \
  --patch-mode none \
  --target-layers 4,35 \
  --components-per-layer 4 \
  --max-tokens 15000 \
  --temperature 0.0 \
  --top-p 0.95 \
  --top-k -1 \
  --tool-schema-mode trading_v1 \
  --tool-choice required \
  --add-generation-prompt \
  --gpu-memory-utilization 0.85 \
  --batch-size 8 \
  --enable-chunked-prefill

launch_job \
  phase18_market_behavior_jointl4l35top4_projectout_stratified48_chunked_v3 \
  /tmp/phase18_market_behavior_jointl4l35top4_projectout_stratified48_chunked_v3.txt \
  --phase-name phase15_market_basis_discovery_v1 \
  --run-name phase18_market_behavior_jointl4l35top4_projectout_stratified48_chunked_v3 \
  --context-variant market_only \
  --selection-strategy stratified_family_variant_roster \
  --limit 48 \
  --patch-mode project_out \
  --target-layers 4,35 \
  --components-per-layer 4 \
  --max-tokens 15000 \
  --temperature 0.0 \
  --top-p 0.95 \
  --top-k -1 \
  --tool-schema-mode trading_v1 \
  --tool-choice required \
  --add-generation-prompt \
  --gpu-memory-utilization 0.85 \
  --batch-size 8 \
  --enable-chunked-prefill

launch_job \
  phase18_market_behavior_jointl4l35top4_randomcontrol_stratified48_chunked_v3 \
  /tmp/phase18_market_behavior_jointl4l35top4_randomcontrol_stratified48_chunked_v3.txt \
  --phase-name phase15_market_basis_discovery_v1 \
  --run-name phase18_market_behavior_jointl4l35top4_randomcontrol_stratified48_chunked_v3 \
  --context-variant market_only \
  --selection-strategy stratified_family_variant_roster \
  --limit 48 \
  --patch-mode random_control \
  --target-layers 4,35 \
  --components-per-layer 4 \
  --max-tokens 15000 \
  --temperature 0.0 \
  --top-p 0.95 \
  --top-k -1 \
  --tool-schema-mode trading_v1 \
  --tool-choice required \
  --add-generation-prompt \
  --gpu-memory-utilization 0.85 \
  --batch-size 8 \
  --enable-chunked-prefill
