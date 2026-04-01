from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pipelines.interp.analysis import _encode_labels, AnalysisDataset
from pipelines.reporting import build_workflow_report
from pipelines.workflows import build_publication_name, normalize_workflow_spec


def test_normalize_workflow_spec_accepts_legacy_prep_target_shape() -> None:
    spec = normalize_workflow_spec(
        {
            "id": "prep1",
            "name": "Prep Target",
            "source": {"mode": "table", "table": "interp_examples_v0"},
            "label": {"mode": "direct", "expression_sql": "decision_type"},
            "split": {"mode": "random_stratified", "train_pct": 70, "val_pct": 15, "test_pct": 15},
        }
    )
    assert spec["id"] == "prep1"
    assert spec["dataset"]["source"]["table"] == "interp_examples_v0"
    assert spec["dataset"]["label"]["expression_sql"] == "decision_type"
    assert spec["version"] == 1


def test_build_publication_name_defaults_from_spec_id_and_version() -> None:
    relation = build_publication_name(
        {
            "id": "Alpha-Spec",
            "version": 3,
            "dataset": {"source": {"mode": "table", "table": "interp_examples_v0"}, "label": {"mode": "direct", "expression_sql": "decision_type"}},
        }
    )
    assert relation == "workflow_dataset_alpha_spec_v3"


def test_analysis_encode_labels_supports_workflow_label() -> None:
    rows = [
        {"workflow_label": "buy_like"},
        {"workflow_label": "sell_like"},
        {"workflow_label": "buy_like"},
    ]
    filtered, y, class_names = _encode_labels(rows, "workflow_label")
    assert len(filtered) == 3
    assert list(y) == [0, 1, 0]
    assert class_names == ["buy_like", "sell_like"]


def test_analysis_loads_labels_from_parquet_without_label_quality(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.parquet"
    table_rows = [
        {"log_id": 1, "workflow_label": "a"},
        {"log_id": 2, "workflow_label": "b"},
    ]
    pq.write_table(pa.Table.from_pylist(table_rows), labels_path)

    rows = AnalysisDataset._load_labels_from_parquet(labels_path)
    assert rows == table_rows


def test_build_workflow_report_writes_summary_and_typst(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    class _Result:
        def __init__(self, row=None, rows=None):
            self._row = row
            self._rows = rows or []

        def fetchone(self):
            return self._row

        def fetchall(self):
            return self._rows

    class _Conn:
        def execute(self, query: str, params=None):
            if "FROM workflow_runs WHERE id = %s" in query:
                return _Result(
                    row={
                        "id": "analysis1",
                        "spec_id": "spec1",
                        "spec_version": 1,
                        "run_type": "analysis",
                        "status": "succeeded",
                        "source": "cli",
                        "spec_snapshot_json": {"id": "spec1", "name": "Demo", "version": 1, "report": {}},
                        "config_json": {"mode": "probe", "target": "workflow_label", "data_source": "router", "pooling": "last_token"},
                        "result_json": {"publication": "workflow_dataset_spec1_v1", "output_dir": "data/analysis_results/demo", "results": {"probe": [{"layer": 0, "accuracy_mean": 0.8}]}},
                        "error_text": None,
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "updated_at": "2026-01-01T00:00:00+00:00",
                        "completed_at": "2026-01-01T00:00:01+00:00",
                    }
                )
            if "FROM dataset_publications WHERE relation_name = %s" in query:
                return _Result(
                    row={
                        "id": "pub1",
                        "spec_id": "spec1",
                        "spec_version": 1,
                        "run_id": "dataset1",
                        "relation_name": "workflow_dataset_spec1_v1",
                        "publish_mode": "view",
                        "row_count": 42,
                        "publication_json": {},
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "updated_at": "2026-01-01T00:00:00+00:00",
                    }
                )
            if "run_type = 'capture'" in query:
                return _Result(
                    rows=[
                        {
                            "id": "capture1",
                            "result_json": {"publication": "workflow_dataset_spec1_v1", "activations_dir": "data/activations/demo"},
                        }
                    ]
                )
            if "run_type = 'dataset'" in query:
                return _Result(
                    rows=[
                        {
                            "id": "dataset1",
                            "result_json": {"publication": {"relation_name": "workflow_dataset_spec1_v1"}},
                        }
                    ]
                )
            raise AssertionError(query)

    result = build_workflow_report(_Conn(), analysis_run_id="analysis1")
    assert Path(result["summary_path"]).exists()
    assert Path(result["report_path"]).exists()
    summary = json.loads(Path(result["summary_path"]).read_text())
    assert summary["spec"]["id"] == "spec1"
    assert summary["publication"]["row_count"] == 42
