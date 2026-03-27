#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LAUNCH_MODE="${PHASE19_LAUNCH_MODE:-parallel}"

run_one() {
  local run_name="$1"
  local log_path="$2"
  shift 2
  printf 'starting %s log=%s\n' "$run_name" "$log_path"
  uv run --extra interp --extra modal modal run \
    pipelines/interp/modal_vllm_patching.py::run_synthetic_market_behavior_modal \
    "$@" >"$log_path" 2>&1
  printf 'finished %s\n' "$run_name"
}

run_donor_prep() {
  local run_name="$1"
  local log_path="$2"
  shift 2
  printf 'preparing donors %s log=%s\n' "$run_name" "$log_path"
  uv run --extra interp --extra modal modal run \
    pipelines/interp/modal_vllm_patching.py::prepare_synthetic_market_behavior_donors_modal \
    "$@" >"$log_path" 2>&1
  printf 'finished donors %s\n' "$run_name"
}

launch_job() {
  if [[ "$LAUNCH_MODE" == "sequential" ]]; then
    run_one "$@"
    return
  fi
  local run_name="$1"
  local log_path="$2"
  shift 2
  nohup uv run --extra interp --extra modal modal run \
    pipelines/interp/modal_vllm_patching.py::run_synthetic_market_behavior_modal \
    "$@" >"$log_path" 2>&1 &
  printf 'launched %s pid=%s log=%s\n' "$run_name" "$!" "$log_path"
}

run_donor_prep \
  phase19_donors_leader_denoise_v1 \
  /tmp/phase19_donors_leader_denoise_v1.txt \
  --phase-name phase15_market_basis_discovery_v1 \
  --run-name phase19_donors_leader_denoise_v1 \
  --context-variant market_only \
  --selection-strategy ordered \
  --limit 24 \
  --pair-metric vol_1h_max \
  --pair-mode denoise \
  --target-layers 4 \
  --tool-schema-mode trading_v1 \
  --tool-choice required \
  --gpu-memory-utilization 0.85

run_donor_prep \
  phase19_donors_leader_noise_v1 \
  /tmp/phase19_donors_leader_noise_v1.txt \
  --phase-name phase15_market_basis_discovery_v1 \
  --run-name phase19_donors_leader_noise_v1 \
  --context-variant market_only \
  --selection-strategy ordered \
  --limit 24 \
  --pair-metric vol_1h_max \
  --pair-mode noise \
  --target-layers 4 \
  --tool-schema-mode trading_v1 \
  --tool-choice required \
  --gpu-memory-utilization 0.85

run_donor_prep \
  phase19_donors_dispersion_denoise_v1 \
  /tmp/phase19_donors_dispersion_denoise_v1.txt \
  --phase-name phase15_market_basis_discovery_v1 \
  --run-name phase19_donors_dispersion_denoise_v1 \
  --context-variant market_only \
  --selection-strategy ordered \
  --limit 24 \
  --pair-metric pct_1h_mad \
  --pair-mode denoise \
  --target-layers 35 \
  --tool-schema-mode trading_v1 \
  --tool-choice required \
  --gpu-memory-utilization 0.85

run_donor_prep \
  phase19_donors_dispersion_noise_v1 \
  /tmp/phase19_donors_dispersion_noise_v1.txt \
  --phase-name phase15_market_basis_discovery_v1 \
  --run-name phase19_donors_dispersion_noise_v1 \
  --context-variant market_only \
  --selection-strategy ordered \
  --limit 24 \
  --pair-metric pct_1h_mad \
  --pair-mode noise \
  --target-layers 35 \
  --tool-schema-mode trading_v1 \
  --tool-choice required \
  --gpu-memory-utilization 0.85

launch_job \
  phase19_market_behavior_baseline_leader_denoise_v1 \
  /tmp/phase19_market_behavior_baseline_leader_denoise_v1.txt \
  --phase-name phase15_market_basis_discovery_v1 \
  --run-name phase19_market_behavior_baseline_leader_denoise_v1 \
  --context-variant market_only \
  --selection-strategy ordered \
  --limit 24 \
  --pair-metric vol_1h_max \
  --pair-mode denoise \
  --patch-mode none \
  --target-layers 4 \
  --components-per-layer 4 \
  --max-tokens 15000 \
  --temperature 0.0 \
  --top-p 0.95 \
  --top-k -1 \
  --tool-schema-mode trading_v1 \
  --tool-choice required \
  --gpu-memory-utilization 0.85

launch_job \
  phase19_market_behavior_leader_swapcomponents_denoise_v1 \
  /tmp/phase19_market_behavior_leader_swapcomponents_denoise_v1.txt \
  --phase-name phase15_market_basis_discovery_v1 \
  --run-name phase19_market_behavior_leader_swapcomponents_denoise_v1 \
  --context-variant market_only \
  --selection-strategy ordered \
  --limit 24 \
  --pair-metric vol_1h_max \
  --pair-mode denoise \
  --generate-source-behavior \
  --patch-mode swap_components \
  --donor-means-run-name phase19_donors_leader_denoise_v1 \
  --target-layers 4 \
  --components-per-layer 4 \
  --direction-name leader_axis \
  --strength 1.0 \
  --max-tokens 15000 \
  --temperature 0.0 \
  --top-p 0.95 \
  --top-k -1 \
  --tool-schema-mode trading_v1 \
  --tool-choice required \
  --gpu-memory-utilization 0.85

launch_job \
  phase19_market_behavior_baseline_leader_noise_v1 \
  /tmp/phase19_market_behavior_baseline_leader_noise_v1.txt \
  --phase-name phase15_market_basis_discovery_v1 \
  --run-name phase19_market_behavior_baseline_leader_noise_v1 \
  --context-variant market_only \
  --selection-strategy ordered \
  --limit 24 \
  --pair-metric vol_1h_max \
  --pair-mode noise \
  --patch-mode none \
  --target-layers 4 \
  --components-per-layer 4 \
  --max-tokens 15000 \
  --temperature 0.0 \
  --top-p 0.95 \
  --top-k -1 \
  --tool-schema-mode trading_v1 \
  --tool-choice required \
  --gpu-memory-utilization 0.85

launch_job \
  phase19_market_behavior_leader_swapcomponents_noise_v1 \
  /tmp/phase19_market_behavior_leader_swapcomponents_noise_v1.txt \
  --phase-name phase15_market_basis_discovery_v1 \
  --run-name phase19_market_behavior_leader_swapcomponents_noise_v1 \
  --context-variant market_only \
  --selection-strategy ordered \
  --limit 24 \
  --pair-metric vol_1h_max \
  --pair-mode noise \
  --generate-source-behavior \
  --patch-mode swap_components \
  --donor-means-run-name phase19_donors_leader_noise_v1 \
  --target-layers 4 \
  --components-per-layer 4 \
  --direction-name leader_axis \
  --strength 1.0 \
  --max-tokens 15000 \
  --temperature 0.0 \
  --top-p 0.95 \
  --top-k -1 \
  --tool-schema-mode trading_v1 \
  --tool-choice required \
  --gpu-memory-utilization 0.85

launch_job \
  phase19_market_behavior_baseline_dispersion_denoise_v1 \
  /tmp/phase19_market_behavior_baseline_dispersion_denoise_v1.txt \
  --phase-name phase15_market_basis_discovery_v1 \
  --run-name phase19_market_behavior_baseline_dispersion_denoise_v1 \
  --context-variant market_only \
  --selection-strategy ordered \
  --limit 24 \
  --pair-metric pct_1h_mad \
  --pair-mode denoise \
  --patch-mode none \
  --target-layers 35 \
  --components-per-layer 4 \
  --max-tokens 15000 \
  --temperature 0.0 \
  --top-p 0.95 \
  --top-k -1 \
  --tool-schema-mode trading_v1 \
  --tool-choice required \
  --gpu-memory-utilization 0.85

launch_job \
  phase19_market_behavior_dispersion_swapcomponents_denoise_v1 \
  /tmp/phase19_market_behavior_dispersion_swapcomponents_denoise_v1.txt \
  --phase-name phase15_market_basis_discovery_v1 \
  --run-name phase19_market_behavior_dispersion_swapcomponents_denoise_v1 \
  --context-variant market_only \
  --selection-strategy ordered \
  --limit 24 \
  --pair-metric pct_1h_mad \
  --pair-mode denoise \
  --generate-source-behavior \
  --patch-mode swap_components \
  --donor-means-run-name phase19_donors_dispersion_denoise_v1 \
  --target-layers 35 \
  --components-per-layer 4 \
  --direction-name dispersion_axis \
  --strength 1.0 \
  --max-tokens 15000 \
  --temperature 0.0 \
  --top-p 0.95 \
  --top-k -1 \
  --tool-schema-mode trading_v1 \
  --tool-choice required \
  --gpu-memory-utilization 0.85

launch_job \
  phase19_market_behavior_baseline_dispersion_noise_v1 \
  /tmp/phase19_market_behavior_baseline_dispersion_noise_v1.txt \
  --phase-name phase15_market_basis_discovery_v1 \
  --run-name phase19_market_behavior_baseline_dispersion_noise_v1 \
  --context-variant market_only \
  --selection-strategy ordered \
  --limit 24 \
  --pair-metric pct_1h_mad \
  --pair-mode noise \
  --patch-mode none \
  --target-layers 35 \
  --components-per-layer 4 \
  --max-tokens 15000 \
  --temperature 0.0 \
  --top-p 0.95 \
  --top-k -1 \
  --tool-schema-mode trading_v1 \
  --tool-choice required \
  --gpu-memory-utilization 0.85

launch_job \
  phase19_market_behavior_dispersion_swapcomponents_noise_v1 \
  /tmp/phase19_market_behavior_dispersion_swapcomponents_noise_v1.txt \
  --phase-name phase15_market_basis_discovery_v1 \
  --run-name phase19_market_behavior_dispersion_swapcomponents_noise_v1 \
  --context-variant market_only \
  --selection-strategy ordered \
  --limit 24 \
  --pair-metric pct_1h_mad \
  --pair-mode noise \
  --generate-source-behavior \
  --patch-mode swap_components \
  --donor-means-run-name phase19_donors_dispersion_noise_v1 \
  --target-layers 35 \
  --components-per-layer 4 \
  --direction-name dispersion_axis \
  --strength 1.0 \
  --max-tokens 15000 \
  --temperature 0.0 \
  --top-p 0.95 \
  --top-k -1 \
  --tool-schema-mode trading_v1 \
  --tool-choice required \
  --gpu-memory-utilization 0.85
