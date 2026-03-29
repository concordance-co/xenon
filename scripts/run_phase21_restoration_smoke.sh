#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PHASE_NAME="${PHASE21_PHASE_NAME:-phase15_market_basis_discovery_v1}"
MODEL_ID="${PHASE21_MODEL_ID:-Qwen/Qwen3-30B-A3B}"
RUN_PREFIX="${PHASE21_RUN_PREFIX:-phase21_restoration_swapcomponents_leader_denoise_smoke_v1}"
CONTEXT_VARIANT="${PHASE21_CONTEXT_VARIANT:-market_only}"
SELECTION_STRATEGY="${PHASE21_SELECTION_STRATEGY:-ordered}"
LIMIT="${PHASE21_LIMIT:-4}"
PAIR_METRIC="${PHASE21_PAIR_METRIC:-vol_1h_max}"
PAIR_MODE="${PHASE21_PAIR_MODE:-denoise}"
TARGET_LAYERS="${PHASE21_TARGET_LAYERS:-4}"
COMPONENTS_PER_LAYER="${PHASE21_COMPONENTS_PER_LAYER:-4}"
BATCH_SIZE="${PHASE21_BATCH_SIZE:-4}"
MAX_TOKENS="${PHASE21_MAX_TOKENS:-128}"
TEMPERATURE="${PHASE21_TEMPERATURE:-0.0}"
TOP_P="${PHASE21_TOP_P:-0.95}"
TOP_K="${PHASE21_TOP_K:--1}"
TOOL_SCHEMA_MODE="${PHASE21_TOOL_SCHEMA_MODE:-trading_v1}"
TOOL_CHOICE="${PHASE21_TOOL_CHOICE:-required}"
GPU_MEMORY_UTILIZATION="${PHASE21_GPU_MEMORY_UTILIZATION:-0.85}"
PATCH_MODE="${PHASE21_PATCH_MODE:-swap_components}"
DIRECTION_NAME="${PHASE21_DIRECTION_NAME:-}"
STRENGTH="${PHASE21_STRENGTH:-1.0}"
LAUNCH_MODE="${PHASE21_LAUNCH_MODE:-sequential}"

DONOR_RUN_NAME="${PHASE21_DONOR_RUN_NAME:-${RUN_PREFIX}_donors}"
GEN_RUN_NAME="${PHASE21_GEN_RUN_NAME:-${RUN_PREFIX}}"
DONOR_LOG="/tmp/${DONOR_RUN_NAME}.txt"
GEN_LOG="/tmp/${GEN_RUN_NAME}.txt"

run_donor_prep() {
  printf 'preparing donors %s log=%s\n' "$DONOR_RUN_NAME" "$DONOR_LOG"
  modal run pipelines/interp/modal_vllm_patching.py::prepare_synthetic_market_behavior_donors_modal \
    --phase-name "$PHASE_NAME" \
    --run-name "$DONOR_RUN_NAME" \
    --model-id "$MODEL_ID" \
    --context-variant "$CONTEXT_VARIANT" \
    --selection-strategy "$SELECTION_STRATEGY" \
    --limit "$LIMIT" \
    --pair-metric "$PAIR_METRIC" \
    --pair-mode "$PAIR_MODE" \
    --target-layers "$TARGET_LAYERS" \
    --tool-schema-mode "$TOOL_SCHEMA_MODE" \
    --tool-choice "$TOOL_CHOICE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    >"$DONOR_LOG" 2>&1
  printf 'finished donors %s\n' "$DONOR_RUN_NAME"
}

run_generation() {
  printf 'starting generation %s log=%s\n' "$GEN_RUN_NAME" "$GEN_LOG"
  modal run pipelines/interp/modal_vllm_patching.py::run_synthetic_market_behavior_modal \
    --phase-name "$PHASE_NAME" \
    --run-name "$GEN_RUN_NAME" \
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
    --direction-name "$DIRECTION_NAME" \
    --strength "$STRENGTH" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --tool-schema-mode "$TOOL_SCHEMA_MODE" \
    --tool-choice "$TOOL_CHOICE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    >"$GEN_LOG" 2>&1
  printf 'finished generation %s\n' "$GEN_RUN_NAME"
}

launch_generation_background() {
  printf 'launching generation %s log=%s\n' "$GEN_RUN_NAME" "$GEN_LOG"
  nohup modal run pipelines/interp/modal_vllm_patching.py::run_synthetic_market_behavior_modal \
    --phase-name "$PHASE_NAME" \
    --run-name "$GEN_RUN_NAME" \
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
    --direction-name "$DIRECTION_NAME" \
    --strength "$STRENGTH" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --tool-schema-mode "$TOOL_SCHEMA_MODE" \
    --tool-choice "$TOOL_CHOICE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    >"$GEN_LOG" 2>&1 &
  printf 'launched generation %s pid=%s\n' "$GEN_RUN_NAME" "$!"
}

run_donor_prep
if [[ "$LAUNCH_MODE" == "background" ]]; then
  launch_generation_background
else
  run_generation
fi
