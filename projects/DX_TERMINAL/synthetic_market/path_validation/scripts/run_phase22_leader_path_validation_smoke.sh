#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT_DIR"

PHASE_NAME="${PHASE22_PHASE_NAME:-phase15_market_basis_discovery_v1}"
MODEL_ID="${PHASE22_MODEL_ID:-Qwen/Qwen3-30B-A3B}"
RUN_PREFIX="${PHASE22_RUN_PREFIX:-phase22_leader_path_l40_smoke_v1}"
CONTEXT_VARIANT="${PHASE22_CONTEXT_VARIANT:-market_only}"
SELECTION_STRATEGY="${PHASE22_SELECTION_STRATEGY:-ordered}"
LIMIT="${PHASE22_LIMIT:-16}"
PAIR_METRIC="${PHASE22_PAIR_METRIC:-vol_1h_max}"
PAIR_MODE="${PHASE22_PAIR_MODE:-denoise}"
BATCH_SIZE="${PHASE22_BATCH_SIZE:-32}"
MAX_TOKENS="${PHASE22_MAX_TOKENS:-128}"
TEMPERATURE="${PHASE22_TEMPERATURE:-0.0}"
TOP_P="${PHASE22_TOP_P:-1.0}"
TOP_K="${PHASE22_TOP_K:--1}"
TOOL_SCHEMA_MODE="${PHASE22_TOOL_SCHEMA_MODE:-trading_v1}"
TOOL_CHOICE="${PHASE22_TOOL_CHOICE:-required}"
GPU_MEMORY_UTILIZATION="${PHASE22_GPU_MEMORY_UTILIZATION:-0.85}"
BASIS_STATE_KEY="${PHASE22_BASIS_STATE_KEY:-market_mean}"

LESION_LAYER="${PHASE22_LESION_LAYER:-4}"
LESION_COMPONENTS="${PHASE22_LESION_COMPONENTS:-4}"
RESCUE_LAYER="${PHASE22_RESCUE_LAYER:-40}"
RESCUE_COMPONENTS="${PHASE22_RESCUE_COMPONENTS:-4}"
LAUNCH_MODE="${PHASE22_LAUNCH_MODE:-sequential}"

DONOR_RUN_NAME="${PHASE22_DONOR_RUN_NAME:-${RUN_PREFIX}_donors}"
LESION_RUN_NAME="${PHASE22_LESION_RUN_NAME:-${RUN_PREFIX}_lesion}"
RESCUE_RUN_NAME="${PHASE22_RESCUE_RUN_NAME:-${RUN_PREFIX}_rescue}"

DONOR_LOG="/tmp/${DONOR_RUN_NAME}.txt"
LESION_LOG="/tmp/${LESION_RUN_NAME}.txt"
RESCUE_LOG="/tmp/${RESCUE_RUN_NAME}.txt"

run_donor_prep() {
  printf 'preparing donors %s log=%s\n' "$DONOR_RUN_NAME" "$DONOR_LOG"
  modal run projects/DX_TERMINAL/synthetic_market/shared/modal_app.py::prepare_synthetic_market_behavior_donors_modal \
    --phase-name "$PHASE_NAME" \
    --run-name "$DONOR_RUN_NAME" \
    --model-id "$MODEL_ID" \
    --context-variant "$CONTEXT_VARIANT" \
    --selection-strategy "$SELECTION_STRATEGY" \
    --limit "$LIMIT" \
    --batch-size "$BATCH_SIZE" \
    --pair-metric "$PAIR_METRIC" \
    --pair-mode "$PAIR_MODE" \
    --target-layers "$LESION_LAYER" \
    --secondary-patch-mode swap_components \
    --secondary-target-layers "$RESCUE_LAYER" \
    --tool-schema-mode "$TOOL_SCHEMA_MODE" \
    --tool-choice "$TOOL_CHOICE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --basis-state-key "$BASIS_STATE_KEY" \
    >"$DONOR_LOG" 2>&1
  printf 'finished donors %s\n' "$DONOR_RUN_NAME"
}

run_lesion() {
  printf 'running lesion baseline %s log=%s\n' "$LESION_RUN_NAME" "$LESION_LOG"
  modal run projects/DX_TERMINAL/synthetic_market/shared/modal_app.py::run_synthetic_market_behavior_modal \
    --phase-name "$PHASE_NAME" \
    --run-name "$LESION_RUN_NAME" \
    --model-id "$MODEL_ID" \
    --context-variant "$CONTEXT_VARIANT" \
    --selection-strategy "$SELECTION_STRATEGY" \
    --limit "$LIMIT" \
    --pair-metric "$PAIR_METRIC" \
    --pair-mode "$PAIR_MODE" \
    --generate-source-behavior \
    --batch-size "$BATCH_SIZE" \
    --patch-mode project_out \
    --target-layers "$LESION_LAYER" \
    --components-per-layer "$LESION_COMPONENTS" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --tool-schema-mode "$TOOL_SCHEMA_MODE" \
    --tool-choice "$TOOL_CHOICE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --no-enforce-eager \
    --enable-chunked-prefill \
    --basis-state-key "$BASIS_STATE_KEY" \
    >"$LESION_LOG" 2>&1
  printf 'finished lesion baseline %s\n' "$LESION_RUN_NAME"
}

run_rescue() {
  printf 'running lesion+rescue %s log=%s\n' "$RESCUE_RUN_NAME" "$RESCUE_LOG"
  modal run projects/DX_TERMINAL/synthetic_market/shared/modal_app.py::run_synthetic_market_behavior_modal \
    --phase-name "$PHASE_NAME" \
    --run-name "$RESCUE_RUN_NAME" \
    --model-id "$MODEL_ID" \
    --context-variant "$CONTEXT_VARIANT" \
    --selection-strategy "$SELECTION_STRATEGY" \
    --limit "$LIMIT" \
    --pair-metric "$PAIR_METRIC" \
    --pair-mode "$PAIR_MODE" \
    --generate-source-behavior \
    --batch-size "$BATCH_SIZE" \
    --patch-mode project_out \
    --target-layers "$LESION_LAYER" \
    --components-per-layer "$LESION_COMPONENTS" \
    --secondary-patch-mode swap_components \
    --secondary-target-layers "$RESCUE_LAYER" \
    --secondary-components-per-layer "$RESCUE_COMPONENTS" \
    --donor-means-run-name "$DONOR_RUN_NAME" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --tool-schema-mode "$TOOL_SCHEMA_MODE" \
    --tool-choice "$TOOL_CHOICE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --no-enforce-eager \
    --enable-chunked-prefill \
    --basis-state-key "$BASIS_STATE_KEY" \
    >"$RESCUE_LOG" 2>&1
  printf 'finished lesion+rescue %s\n' "$RESCUE_RUN_NAME"
}

launch_background() {
  nohup modal run "$@" >/dev/null 2>&1 &
  printf 'launched pid=%s\n' "$!"
}

run_donor_prep
if [[ "$LAUNCH_MODE" == "background" ]]; then
  printf 'launching lesion baseline in background\n'
  launch_background projects/DX_TERMINAL/synthetic_market/shared/modal_app.py::run_synthetic_market_behavior_modal \
    --phase-name "$PHASE_NAME" \
    --run-name "$LESION_RUN_NAME" \
    --model-id "$MODEL_ID" \
    --context-variant "$CONTEXT_VARIANT" \
    --selection-strategy "$SELECTION_STRATEGY" \
    --limit "$LIMIT" \
    --pair-metric "$PAIR_METRIC" \
    --pair-mode "$PAIR_MODE" \
    --generate-source-behavior \
    --batch-size "$BATCH_SIZE" \
    --patch-mode project_out \
    --target-layers "$LESION_LAYER" \
    --components-per-layer "$LESION_COMPONENTS" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --tool-schema-mode "$TOOL_SCHEMA_MODE" \
    --tool-choice "$TOOL_CHOICE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --no-enforce-eager \
    --enable-chunked-prefill \
    --basis-state-key "$BASIS_STATE_KEY"
  printf 'launching lesion+rescue in background\n'
  launch_background projects/DX_TERMINAL/synthetic_market/shared/modal_app.py::run_synthetic_market_behavior_modal \
    --phase-name "$PHASE_NAME" \
    --run-name "$RESCUE_RUN_NAME" \
    --model-id "$MODEL_ID" \
    --context-variant "$CONTEXT_VARIANT" \
    --selection-strategy "$SELECTION_STRATEGY" \
    --limit "$LIMIT" \
    --pair-metric "$PAIR_METRIC" \
    --pair-mode "$PAIR_MODE" \
    --generate-source-behavior \
    --batch-size "$BATCH_SIZE" \
    --patch-mode project_out \
    --target-layers "$LESION_LAYER" \
    --components-per-layer "$LESION_COMPONENTS" \
    --secondary-patch-mode swap_components \
    --secondary-target-layers "$RESCUE_LAYER" \
    --secondary-components-per-layer "$RESCUE_COMPONENTS" \
    --donor-means-run-name "$DONOR_RUN_NAME" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --tool-schema-mode "$TOOL_SCHEMA_MODE" \
    --tool-choice "$TOOL_CHOICE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --no-enforce-eager \
    --enable-chunked-prefill \
    --basis-state-key "$BASIS_STATE_KEY"
else
  run_lesion
  run_rescue
fi
