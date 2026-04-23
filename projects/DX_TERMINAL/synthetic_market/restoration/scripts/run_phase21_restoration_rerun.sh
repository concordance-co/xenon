#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT_DIR"

PHASE_NAME="${PHASE21_PHASE_NAME:-phase15_market_basis_discovery_v1}"
MODEL_ID="${PHASE21_MODEL_ID:-Qwen/Qwen3-30B-A3B}"
RUN_PREFIX="${PHASE21_RUN_PREFIX:-phase21_restoration_axis_rerun_v1}"
CONTEXT_VARIANT="${PHASE21_CONTEXT_VARIANT:-market_only}"
SELECTION_STRATEGY="${PHASE21_SELECTION_STRATEGY:-ordered}"
LIMIT="${PHASE21_LIMIT:-48}"
PAIR_METRIC="${PHASE21_PAIR_METRIC:-vol_1h_max}"
PAIR_MODE="${PHASE21_PAIR_MODE:-denoise}"
TARGET_LAYERS="${PHASE21_TARGET_LAYERS:-4}"
COMPONENTS_PER_LAYER="${PHASE21_COMPONENTS_PER_LAYER:-4}"
BATCH_SIZE="${PHASE21_BATCH_SIZE:-16}"
MAX_TOKENS="${PHASE21_MAX_TOKENS:-15000}"
TEMPERATURE="${PHASE21_TEMPERATURE:-0.0}"
TOP_P="${PHASE21_TOP_P:-1.0}"
TOP_K="${PHASE21_TOP_K:--1}"
TOOL_SCHEMA_MODE="${PHASE21_TOOL_SCHEMA_MODE:-trading_v1}"
TOOL_CHOICE="${PHASE21_TOOL_CHOICE:-required}"
GPU_MEMORY_UTILIZATION="${PHASE21_GPU_MEMORY_UTILIZATION:-0.85}"
PATCH_MODE="${PHASE21_PATCH_MODE:-swap_components}"
BASIS_STATE_KEY="${PHASE21_BASIS_STATE_KEY:-market_mean}"
LAUNCH_MODE="${PHASE21_LAUNCH_MODE:-background}"

DONOR_RUN_NAME="${PHASE21_DONOR_RUN_NAME:-${RUN_PREFIX}_donors}"
BASELINE_RUN_NAME="${PHASE21_BASELINE_RUN_NAME:-${RUN_PREFIX}_baseline}"
PATCH_RUN_NAME="${PHASE21_PATCH_RUN_NAME:-${RUN_PREFIX}_${PATCH_MODE}}"

DONOR_LOG="/tmp/${DONOR_RUN_NAME}.txt"
BASELINE_LOG="/tmp/${BASELINE_RUN_NAME}.txt"
PATCH_LOG="/tmp/${PATCH_RUN_NAME}.txt"

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
    --target-layers "$TARGET_LAYERS" \
    --secondary-patch-mode "$PATCH_MODE" \
    --secondary-target-layers "$TARGET_LAYERS" \
    --tool-schema-mode "$TOOL_SCHEMA_MODE" \
    --tool-choice "$TOOL_CHOICE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --basis-state-key "$BASIS_STATE_KEY" \
    >"$DONOR_LOG" 2>&1
  printf 'finished donors %s\n' "$DONOR_RUN_NAME"
}

run_baseline() {
  printf 'running baseline %s log=%s\n' "$BASELINE_RUN_NAME" "$BASELINE_LOG"
  modal run projects/DX_TERMINAL/synthetic_market/shared/modal_app.py::run_synthetic_market_behavior_modal \
    --phase-name "$PHASE_NAME" \
    --run-name "$BASELINE_RUN_NAME" \
    --model-id "$MODEL_ID" \
    --context-variant "$CONTEXT_VARIANT" \
    --selection-strategy "$SELECTION_STRATEGY" \
    --limit "$LIMIT" \
    --pair-metric "$PAIR_METRIC" \
    --pair-mode "$PAIR_MODE" \
    --generate-source-behavior \
    --batch-size "$BATCH_SIZE" \
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
    >"$BASELINE_LOG" 2>&1
  printf 'finished baseline %s\n' "$BASELINE_RUN_NAME"
}

run_patch() {
  printf 'running patch %s log=%s\n' "$PATCH_RUN_NAME" "$PATCH_LOG"
  modal run projects/DX_TERMINAL/synthetic_market/shared/modal_app.py::run_synthetic_market_behavior_modal \
    --phase-name "$PHASE_NAME" \
    --run-name "$PATCH_RUN_NAME" \
    --model-id "$MODEL_ID" \
    --context-variant "$CONTEXT_VARIANT" \
    --selection-strategy "$SELECTION_STRATEGY" \
    --limit "$LIMIT" \
    --pair-metric "$PAIR_METRIC" \
    --pair-mode "$PAIR_MODE" \
    --generate-source-behavior \
    --batch-size "$BATCH_SIZE" \
    --patch-mode "$PATCH_MODE" \
    --donor-means-run-name "$DONOR_RUN_NAME" \
    --target-layers "$TARGET_LAYERS" \
    --components-per-layer "$COMPONENTS_PER_LAYER" \
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
    >"$PATCH_LOG" 2>&1
  printf 'finished patch %s\n' "$PATCH_RUN_NAME"
}

launch_background() {
  nohup modal run "$@" >/dev/null 2>&1 &
  printf 'launched pid=%s\n' "$!"
}

run_donor_prep
if [[ "$LAUNCH_MODE" == "background" ]]; then
  printf 'launching baseline in background\n'
  launch_background projects/DX_TERMINAL/synthetic_market/shared/modal_app.py::run_synthetic_market_behavior_modal \
    --phase-name "$PHASE_NAME" \
    --run-name "$BASELINE_RUN_NAME" \
    --model-id "$MODEL_ID" \
    --context-variant "$CONTEXT_VARIANT" \
    --selection-strategy "$SELECTION_STRATEGY" \
    --limit "$LIMIT" \
    --pair-metric "$PAIR_METRIC" \
    --pair-mode "$PAIR_MODE" \
    --generate-source-behavior \
    --batch-size "$BATCH_SIZE" \
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
  printf 'launching patch run in background\n'
  launch_background projects/DX_TERMINAL/synthetic_market/shared/modal_app.py::run_synthetic_market_behavior_modal \
    --phase-name "$PHASE_NAME" \
    --run-name "$PATCH_RUN_NAME" \
    --model-id "$MODEL_ID" \
    --context-variant "$CONTEXT_VARIANT" \
    --selection-strategy "$SELECTION_STRATEGY" \
    --limit "$LIMIT" \
    --pair-metric "$PAIR_METRIC" \
    --pair-mode "$PAIR_MODE" \
    --generate-source-behavior \
    --batch-size "$BATCH_SIZE" \
    --patch-mode "$PATCH_MODE" \
    --donor-means-run-name "$DONOR_RUN_NAME" \
    --target-layers "$TARGET_LAYERS" \
    --components-per-layer "$COMPONENTS_PER_LAYER" \
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
  run_baseline
  run_patch
fi
