from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from pipelines.interp.synthetic.market import SyntheticAsset, _render_user_prompt
from pipelines.interp.synthetic.structure import (
    clear_synthetic_structure_shards,
    find_synthetic_section_boundaries,
    find_synthetic_row_boundaries,
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


class _FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, tools=None):
        parts = []
        if tools:
            parts.append(f"TOOLS:{len(tools)}")
        for message in messages:
            parts.append(f"{message['role'].upper()}:\n{message['content']}")
        return "\n\n".join(parts)

    def __call__(self, rendered_text, add_special_tokens=False, return_offsets_mapping=True):
        return {
            "input_ids": list(range(len(rendered_text))),
            "offset_mapping": [(idx, idx + 1) for idx in range(len(rendered_text))],
        }


def test_find_synthetic_row_boundaries_handles_compact_surface_style():
    assets = [
        SyntheticAsset("North", "stable_winner", 4.4, 8.0, 1.1, 5.0, 20.0, 18, 24.0, "mature"),
        SyntheticAsset("South", "crowded_risk", 4.5, 7.8, 1.0, 5.2, 21.0, 27, 61.0, "mid"),
        SyntheticAsset("East", "mean_reverter", -2.6, -4.0, 0.2, 2.4, 8.0, 10, 34.0, "mature"),
        SyntheticAsset("West", "illiquid_spike", 5.0, 4.5, 0.18, 1.1, 4.0, 4, 72.0, "fresh"),
    ]
    user_text = _render_user_prompt(
        "profile_inv_test",
        "market_only",
        assets,
        surface_style="compact",
    )
    rows = [
        {"row_index": idx, "symbol": asset.symbol}
        for idx, asset in enumerate(assets)
    ]

    bounds = find_synthetic_row_boundaries(
        _FakeTokenizer(),
        "system prompt",
        user_text,
        rows,
    )

    assert len(bounds) == 4
    assert [row["symbol"] for row in bounds] == ["North", "South", "East", "West"]
    assert all(row["full_start"] < row["full_end"] for row in bounds)


def test_find_synthetic_section_boundaries_trims_market_separator():
    assets = [
        SyntheticAsset("North", "stable_winner", 4.4, 8.0, 1.1, 5.0, 20.0, 18, 24.0, "mature"),
        SyntheticAsset("South", "crowded_risk", 4.5, 7.8, 1.0, 5.2, 21.0, 27, 61.0, "mid"),
    ]
    user_text = _render_user_prompt("boundary_test", "market_only", assets)
    tokenizer = _FakeTokenizer()
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": "system prompt"}, {"role": "user", "content": user_text}],
        add_generation_prompt=False,
        tokenize=False,
    )

    boundaries = find_synthetic_section_boundaries(tokenizer, "system prompt", user_text)
    market_start, market_end = boundaries["market"]
    market_text = rendered[market_start:market_end]

    assert "## MARKET SNAPSHOT" in market_text
    assert "------------------------------" not in market_text
    assert market_text.rstrip().endswith("18h") or market_text.rstrip().endswith("32h")


def test_find_synthetic_section_boundaries_handles_tool_augmented_template():
    assets = [
        SyntheticAsset("North", "stable_winner", 4.4, 8.0, 1.1, 5.0, 20.0, 18, 24.0, "mature"),
        SyntheticAsset("South", "crowded_risk", 4.5, 7.8, 1.0, 5.2, 21.0, 27, 61.0, "mid"),
    ]
    user_text = _render_user_prompt("boundary_test_tools", "market_only", assets)
    tokenizer = _FakeTokenizer()

    without_tools = find_synthetic_section_boundaries(tokenizer, "system prompt", user_text)["market"]
    with_tools = find_synthetic_section_boundaries(
        tokenizer,
        "system prompt",
        user_text,
        tools=[{"type": "function", "function": {"name": "record_observation"}}],
    )["market"]

    assert with_tools[0] > without_tools[0]
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": "system prompt"}, {"role": "user", "content": user_text}],
        add_generation_prompt=False,
        tokenize=False,
        tools=[{"type": "function", "function": {"name": "record_observation"}}],
    )
    market_text = rendered[with_tools[0]:with_tools[1]]
    assert "## MARKET SNAPSHOT" in market_text
