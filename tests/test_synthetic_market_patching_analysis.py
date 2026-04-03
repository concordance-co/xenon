from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors.numpy import save_file

from projects.synthetic_market.synthetic_market_patching_analysis import (
    SyntheticMarketPatchingAnalysisConfig,
    classify_state_key,
    run_synthetic_market_patching_analysis,
)


def test_classify_state_key_covers_section_row_and_terminal() -> None:
    assert classify_state_key("market_mean")["state_kind"] == "section"
    assert classify_state_key("row_eos_3")["row_index"] == 3
    assert classify_state_key("last_token")["state_kind"] == "terminal"


def test_run_synthetic_market_patching_analysis_summarizes_margin(tmp_path) -> None:
    baseline_dir = tmp_path / "baseline"
    intervention_dir = tmp_path / "intervention"
    control_dir = tmp_path / "control"
    output_dir = tmp_path / "out"
    basis_npz = tmp_path / "basis.npz"
    for root in (baseline_dir, intervention_dir, control_dir):
        (root / "residual").mkdir(parents=True)

    pooled_base = {
        "market_mean": np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        "last_token": np.array([[0.5, 0.0], [0.5, 0.0]], dtype=np.float32),
    }
    pooled_intv = {
        "market_mean": np.array([[0.0, 0.0], [1.5, 0.0]], dtype=np.float32),
        "last_token": np.array([[0.7, 0.0], [0.8, 0.0]], dtype=np.float32),
    }
    pooled_ctrl = {
        "market_mean": np.array([[0.8, 0.0], [1.1, 0.0]], dtype=np.float32),
        "last_token": np.array([[0.55, 0.0], [0.6, 0.0]], dtype=np.float32),
    }
    for root, pooled in (
        (baseline_dir, pooled_base),
        (intervention_dir, pooled_intv),
        (control_dir, pooled_ctrl),
    ):
        save_file(pooled, str(root / "residual" / "1.safetensors"))

    base_meta = [{"log_id": 1}]
    intv_meta = [{
        "log_id": 1,
        "patch_stats_json": json.dumps({"4": {"delta_norm_raw": 1.0, "selected_proj_norm_before": 2.0, "selected_coeff_after": [0.0, 0.0]}}),
    }]
    ctrl_meta = [{
        "log_id": 1,
        "patch_stats_json": json.dumps({"4": {"delta_norm_raw": 0.5, "selected_proj_norm_before": 2.0, "selected_coeff_after": [0.3, 0.1]}}),
    }]
    pq.write_table(pa.Table.from_pylist(base_meta), baseline_dir / "metadata.parquet")
    pq.write_table(pa.Table.from_pylist(intv_meta), intervention_dir / "metadata.parquet")
    pq.write_table(pa.Table.from_pylist(ctrl_meta), control_dir / "metadata.parquet")
    np.savez(
        basis_npz,
        market_mean_layer_0__mean=np.zeros((2,), dtype=np.float32),
        market_mean_layer_0__scale=np.ones((2,), dtype=np.float32),
        market_mean_layer_0__components=np.eye(2, dtype=np.float32),
        market_mean_layer_1__mean=np.zeros((2,), dtype=np.float32),
        market_mean_layer_1__scale=np.ones((2,), dtype=np.float32),
        market_mean_layer_1__components=np.eye(2, dtype=np.float32),
    )

    result = run_synthetic_market_patching_analysis(
        SyntheticMarketPatchingAnalysisConfig(
            baseline_dir=baseline_dir,
            intervention_dir=intervention_dir,
            control_dir=control_dir,
            output_dir=output_dir,
            basis_npz_path=basis_npz,
        )
    )

    assert result["num_shared_examples"] == 1
    direct = result["direct_target_market_mean_l4"]
    assert direct is None
    rows = pq.read_table(output_dir / "state_layer_summary.parquet").to_pylist()
    market_l0 = next(row for row in rows if row["state_key"] == "market_mean" and row["layer"] == 0)
    assert market_l0["delta_margin"] > 0
    assert market_l0["basis_shift_margin"] > 0
    patch = result["patch_stats_summary"][0]
    assert patch["layer"] == 4
    assert patch["intervention_selected_coeff_after_abs_max"] == 0.0
