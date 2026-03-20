from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from pipelines.interp.synthetic_structure import (
    clear_synthetic_structure_shards,
    merge_synthetic_structure_shards,
    select_examples_for_shard,
    shard_output_paths,
)


def test_select_examples_for_shard_uses_deterministic_round_robin():
    examples = [{"log_id": idx} for idx in range(7)]

    shard0 = select_examples_for_shard(examples, shard_index=0, num_shards=3)
    shard1 = select_examples_for_shard(examples, shard_index=1, num_shards=3)
    shard2 = select_examples_for_shard(examples, shard_index=2, num_shards=3)

    assert [row["log_id"] for row in shard0] == [0, 3, 6]
    assert [row["log_id"] for row in shard1] == [1, 4]
    assert [row["log_id"] for row in shard2] == [2, 5]


def test_merge_synthetic_structure_shards_builds_canonical_tables(tmp_path):
    out_dir = tmp_path / "synthetic_structure"
    meta0, tick0, asset0 = shard_output_paths(out_dir, 0)
    meta1, tick1, asset1 = shard_output_paths(out_dir, 1)
    meta0.parent.mkdir(parents=True, exist_ok=True)

    pq.write_table(
        pa.Table.from_pylist([{"log_id": 10, "phase_name": "phase2_geometry", "seq_len": 100}]),
        meta0,
        compression="snappy",
    )
    pq.write_table(
        pa.Table.from_pylist([{"log_id": 10, "phase_name": "phase2_geometry", "family": "scalar_sweep_dense"}]),
        tick0,
        compression="snappy",
    )
    pq.write_table(
        pa.Table.from_pylist([
            {"log_id": 10, "row_index": 0, "symbol": "A"},
            {"log_id": 10, "row_index": 1, "symbol": "B"},
        ]),
        asset0,
        compression="snappy",
    )

    pq.write_table(
        pa.Table.from_pylist([{"log_id": 20, "phase_name": "phase2_geometry", "seq_len": 120}]),
        meta1,
        compression="snappy",
    )
    pq.write_table(
        pa.Table.from_pylist([{"log_id": 20, "phase_name": "phase2_geometry", "family": "scalar_sweep_minimal"}]),
        tick1,
        compression="snappy",
    )
    pq.write_table(
        pa.Table.from_pylist([
            {"log_id": 20, "row_index": 0, "symbol": "C"},
        ]),
        asset1,
        compression="snappy",
    )

    result = merge_synthetic_structure_shards(out_dir, num_shards=2)

    assert result["seen_shards"] == 2
    assert result["metadata_rows"] == 2
    assert result["tick_rows"] == 2
    assert result["asset_rows"] == 3

    merged_meta = pq.read_table(out_dir / "metadata.parquet").to_pylist()
    merged_tick = pq.read_table(out_dir / "tick_labels.parquet").to_pylist()
    merged_asset = pq.read_table(out_dir / "asset_labels.parquet").to_pylist()

    assert [row["log_id"] for row in merged_meta] == [10, 20]
    assert [row["log_id"] for row in merged_tick] == [10, 20]
    assert [(row["log_id"], row["row_index"]) for row in merged_asset] == [(10, 0), (10, 1), (20, 0)]


def test_clear_synthetic_structure_shards_removes_shard_and_canonical_tables(tmp_path):
    out_dir = tmp_path / "synthetic_structure"
    meta0, tick0, asset0 = shard_output_paths(out_dir, 0)
    meta0.parent.mkdir(parents=True, exist_ok=True)

    for path in (
        meta0,
        tick0,
        asset0,
        out_dir / "metadata.parquet",
        out_dir / "tick_labels.parquet",
        out_dir / "asset_labels.parquet",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")

    cleared = clear_synthetic_structure_shards(out_dir, num_shards=1, clear_canonical=True)

    assert cleared["removed"] == 6
    assert cleared["missing"] == 0
    assert not meta0.exists()
    assert not tick0.exists()
    assert not asset0.exists()
    assert not (out_dir / "metadata.parquet").exists()
