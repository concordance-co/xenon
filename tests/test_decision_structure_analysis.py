from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors.numpy import save_file

from pipelines.interp.decision_structure.analysis import (
    DecisionStructureAnalysisConfig,
    collect_concat_groups,
    collect_pre_groups,
    run_decision_structure_analysis,
)


def _write_structure_fixture(tmp_path: Path) -> Path:
    structure_dir = tmp_path / "decision_structure"
    residual_dir = structure_dir / "residual"
    residual_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = []
    tick_rows = []
    asset_rows = []

    for log_id in range(1, 9):
        is_buy = log_id <= 4
        meta_rows.append({"log_id": log_id})
        tick_rows.append({"log_id": log_id})

        if is_buy:
            row0 = np.array([[3.0, 0.0], [3.0, 0.0]], dtype=np.float32)
            row1 = np.array([[0.0, 3.0], [0.0, 3.0]], dtype=np.float32)
            asset_rows.extend([
                {
                    "log_id": log_id,
                    "row_index": 0,
                    "is_target_asset": True,
                    "is_buy_target": True,
                    "is_sell_target": False,
                },
                {
                    "log_id": log_id,
                    "row_index": 1,
                    "is_target_asset": False,
                    "is_buy_target": False,
                    "is_sell_target": False,
                },
            ])
        else:
            row0 = np.array([[0.0, 3.0], [0.0, 3.0]], dtype=np.float32)
            row1 = np.array([[3.0, 0.0], [3.0, 0.0]], dtype=np.float32)
            asset_rows.extend([
                {
                    "log_id": log_id,
                    "row_index": 0,
                    "is_target_asset": False,
                    "is_buy_target": False,
                    "is_sell_target": False,
                },
                {
                    "log_id": log_id,
                    "row_index": 1,
                    "is_target_asset": True,
                    "is_buy_target": False,
                    "is_sell_target": True,
                },
            ])

        tensors = {
            "row_mean_0": row0,
            "row_mean_1": row1,
            "row_eos_0": row0,
            "row_eos_1": row1,
            "active_settings_eos": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
            "portfolio_eos": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
            "constraints_eos": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
            "prev_decisions_eos": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
            "last_token": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        }
        save_file(tensors, str(residual_dir / f"{log_id}.safetensors"))

    pq.write_table(pa.Table.from_pylist(meta_rows), structure_dir / "metadata.parquet")
    pq.write_table(pa.Table.from_pylist(tick_rows), structure_dir / "tick_labels.parquet")
    pq.write_table(pa.Table.from_pylist(asset_rows), structure_dir / "asset_labels.parquet")
    return structure_dir


def test_collect_pre_and_concat_groups(tmp_path: Path):
    structure_dir = _write_structure_fixture(tmp_path)
    asset_rows = pq.read_table(structure_dir / "asset_labels.parquet").to_pylist()
    asset_by_log: dict[int, list[dict]] = {}
    for row in asset_rows:
        asset_by_log.setdefault(int(row["log_id"]), []).append(row)

    pre_groups = collect_pre_groups(
        log_ids={1, 2, 3, 4},
        asset_by_log=asset_by_log,
        structure_dir=structure_dir,
        target="is_buy_target",
        layer=0,
        row_key="row_mean",
    )
    concat_groups = collect_concat_groups(
        log_ids={1, 2, 3, 4},
        asset_by_log=asset_by_log,
        structure_dir=structure_dir,
        target="is_buy_target",
        layer=0,
        row_key="row_mean",
        position_key="active_settings_eos",
    )

    assert len(pre_groups) == 4
    assert pre_groups[0]["X"].shape == (2, 2)
    assert concat_groups[0]["X"].shape == (2, 4)


def test_run_decision_structure_analysis_end_to_end(tmp_path: Path):
    structure_dir = _write_structure_fixture(tmp_path)
    output_dir = tmp_path / "results"
    config = DecisionStructureAnalysisConfig(
        structure_dir=structure_dir,
        output_dir=output_dir,
        layers=[0],
        seed=1,
        test_fraction=0.25,
    )
    results = run_decision_structure_analysis(config)

    assert "summary" in results
    assert results["summary"]["is_buy_target"]["best_pre"] is not None
    assert results["summary"]["is_sell_target"]["best_pre"] is not None
    assert (output_dir / "decision_structure_results.json").exists()
