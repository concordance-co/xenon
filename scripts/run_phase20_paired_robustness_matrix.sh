#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LAUNCH_MODE="${PHASE20_LAUNCH_MODE:-print}"
PHASE_NAME="${PHASE20_PHASE_NAME:-phase15_market_basis_discovery_v1}"
MODEL_ID="${PHASE20_MODEL_ID:-Qwen/Qwen3-30B-A3B}"
CONTEXT_VARIANT="${PHASE20_CONTEXT_VARIANT:-market_only}"
SELECTION_STRATEGY="${PHASE20_SELECTION_STRATEGY:-ordered}"
LIMIT="${PHASE20_LIMIT:-32}"
MIN_PAIR_GAP="${PHASE20_MIN_PAIR_GAP:-0.0}"
BATCH_SIZE="${PHASE20_BATCH_SIZE:-32}"
MAX_TOKENS="${PHASE20_MAX_TOKENS:-15000}"
TEMPERATURE="${PHASE20_TEMPERATURE:-0.0}"
TOP_P="${PHASE20_TOP_P:-0.95}"
TOP_K="${PHASE20_TOP_K:--1}"
GPU_MEMORY_UTILIZATION="${PHASE20_GPU_MEMORY_UTILIZATION:-0.85}"
TOOL_SCHEMA_MODE="${PHASE20_TOOL_SCHEMA_MODE:-trading_v1}"
TOOL_CHOICE="${PHASE20_TOOL_CHOICE:-required}"
RUN_SUFFIX="${PHASE20_RUN_SUFFIX:-v1}"
LAMBDA_SWEEP="${PHASE20_LAMBDA_SWEEP:-0.5,1.0,1.5}"
COMPONENT_OFFSETS="${PHASE20_COMPONENT_OFFSETS:-0}"
SUBSPACE_SIZES="${PHASE20_SUBSPACE_SIZES:-}"
RANDOM_CONTROL_SEEDS="${PHASE20_RANDOM_CONTROL_SEEDS:-11,17}"
PAIR_MODES="${PHASE20_PAIR_MODES:-denoise,noise}"
INCLUDE_BASELINES="${PHASE20_INCLUDE_BASELINES:-1}"
ENABLE_CHUNKED_PREFILL="${PHASE20_ENABLE_CHUNKED_PREFILL:-1}"
ENFORCE_EAGER="${PHASE20_ENFORCE_EAGER:-0}"
HYPOTHESES="${PHASE20_HYPOTHESES:-leader,dispersion}"

launch_run() {
  local run_name="$1"
  local log_path="$2"
  shift 2
  printf 'launching %s log=%s\n' "$run_name" "$log_path"
  if [[ "$LAUNCH_MODE" == "print" ]]; then
    printf 'modal run pipelines/interp/modal_vllm_patching.py::run_synthetic_market_behavior_matrix_modal'
    printf ' %q' "$@"
    printf '\n'
    return
  fi
  if [[ "$LAUNCH_MODE" == "sequential" ]]; then
    modal run pipelines/interp/modal_vllm_patching.py::run_synthetic_market_behavior_matrix_modal \
      "$@" >"$log_path" 2>&1
    printf 'finished %s\n' "$run_name"
    return
  fi
  nohup modal run pipelines/interp/modal_vllm_patching.py::run_synthetic_market_behavior_matrix_modal \
    "$@" >"$log_path" 2>&1 &
  printf 'launched %s pid=%s\n' "$run_name" "$!"
}

launch_matrix() {
  local hypothesis_name="$1"
  local pair_metric="$2"
  local target_layers="$3"
  local components_per_layer="$4"
  local run_prefix="phase20_market_behavior_${hypothesis_name}_${RUN_SUFFIX}"
  local log_path="/tmp/${run_prefix}.txt"
  local run_args=(
    --run-name "$run_prefix"
    --run-name-prefix "$run_prefix"
    --phase-name "$PHASE_NAME"
    --model-id "$MODEL_ID"
    --context-variant "$CONTEXT_VARIANT"
    --selection-strategy "$SELECTION_STRATEGY"
    --limit "$LIMIT"
    --pair-metric "$pair_metric"
    --pair-modes "$PAIR_MODES"
    --min-pair-gap "$MIN_PAIR_GAP"
    --batch-size "$BATCH_SIZE"
    --patch-mode project_out
    --target-layers "$target_layers"
    --components-per-layer "$components_per_layer"
    --max-tokens "$MAX_TOKENS"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --top-k "$TOP_K"
    --tool-schema-mode "$TOOL_SCHEMA_MODE"
    --tool-choice "$TOOL_CHOICE"
    --lambda-sweep "$LAMBDA_SWEEP"
    --neighboring-component-offsets "$COMPONENT_OFFSETS"
    --subspace-sizes "$SUBSPACE_SIZES"
    --random-control-seeds "$RANDOM_CONTROL_SEEDS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  )
  if [[ "$INCLUDE_BASELINES" != "1" ]]; then
    run_args+=(--no-include-baselines)
  fi
  if [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]]; then
    run_args+=(--enable-chunked-prefill)
  fi
  if [[ "$ENFORCE_EAGER" == "0" ]]; then
    run_args+=(--no-enforce-eager)
  fi
  launch_run "$run_prefix" "$log_path" "${run_args[@]}"
}

IFS=',' read -r -a hypothesis_list <<<"$HYPOTHESES"
for hypothesis in "${hypothesis_list[@]}"; do
  case "$hypothesis" in
    leader)
      launch_matrix "leader_l4top4" "vol_1h_max" "4" "4"
      ;;
    dispersion)
      launch_matrix "dispersion_l35top4" "pct_1h_mad" "35" "4"
      ;;
    *)
      printf 'unknown hypothesis %s\n' "$hypothesis" >&2
      exit 1
      ;;
  esac
done
