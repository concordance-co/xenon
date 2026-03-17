from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipelines.backend.app import (
    _activation_coverage,
    _build_label_expression,
    _split_viability,
    _validate_read_only_sql,
    _validate_sql_fragment,
)


def test_validate_read_only_sql_accepts_select_and_with() -> None:
    assert _validate_read_only_sql("SELECT * FROM vaults") == "SELECT * FROM vaults"
    assert _validate_read_only_sql("WITH x AS (SELECT 1) SELECT * FROM x;") == "WITH x AS (SELECT 1) SELECT * FROM x"


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE vaults SET state='paused'",
        "INSERT INTO vaults(vault_address) VALUES ('x')",
        "DELETE FROM swaps",
        "CREATE TABLE x(a INTEGER)",
    ],
)
def test_validate_read_only_sql_rejects_mutating(sql: str) -> None:
    with pytest.raises(ValueError):
        _validate_read_only_sql(sql)


def test_validate_sql_fragment_rejects_semicolon() -> None:
    with pytest.raises(ValueError):
        _validate_sql_fragment("1=1; DROP TABLE vaults", "filters.sql_where")


def test_build_label_expression_binary_and_bucket() -> None:
    binary = _build_label_expression(
        {
            "mode": "binary_rule",
            "expression_sql": "pnl_1h_pct > 0",
            "classes": ["loss", "win"],
        }
    )
    assert "CASE WHEN (pnl_1h_pct > 0)" in binary
    assert "'win'" in binary
    assert "'loss'" in binary

    bucket = _build_label_expression(
        {
            "mode": "bucket",
            "expression_sql": "pnl_1h_pct",
            "buckets": [
                {"name": "neg", "max": 0.0},
                {"name": "pos", "min": 0.0},
            ],
        }
    )
    assert "WHEN (pnl_1h_pct) < 0.0" in bucket
    assert "WHEN (pnl_1h_pct) >= 0.0" in bucket


def test_split_viability_detects_rare_class() -> None:
    result = _split_viability(
        {"majority": 90, "minority": 2},
        {"mode": "random_stratified", "train_pct": 70, "val_pct": 15, "test_pct": 15},
    )
    assert not result["viable"]
    assert any("minority" in reason for reason in result["reasons"])


def test_activation_coverage_uses_metadata_parquet(tmp_path: Path) -> None:
    activations_dir = tmp_path / "activations"
    activations_dir.mkdir(parents=True)

    table = pa.table({"log_id": [100, 101, 102]})
    pq.write_table(table, activations_dir / "metadata.parquet")

    rows = [
        {"log_id": 100, "__label_value": "buy"},
        {"log_id": 103, "__label_value": "sell"},
        {"log_id": None, "__label_value": "buy"},
        {"log_id": 101, "__label_value": None},
    ]

    coverage = _activation_coverage(activations_dir, rows)
    assert coverage["available"]
    assert coverage["eligible_labeled"] == 2
    assert coverage["matched"] == 1
    assert coverage["coverage"] == 0.5
