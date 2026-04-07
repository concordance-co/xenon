from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from projects.DX_TERMINAL.counterfactual import MarketRow
from projects.DX_TERMINAL.decision_structure import (
    build_asset_label_rows,
    build_tick_label_row,
    clear_decision_structure_shards,
    find_real_row_boundaries,
    find_real_section_boundaries,
    merge_decision_structure_shards,
    pool_decision_residual,
    select_examples_for_shard,
    shard_output_paths,
)


class DummyTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=False, return_tensors=None, tokenize=True):
        rendered = "".join(
            f"<{m['role']}>{m['content']}</{m['role']}>"
            for m in messages
        )
        if not tokenize:
            return rendered
        return [ord(ch) for ch in rendered]

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, return_tensors=None):
        input_ids = [ord(ch) for ch in text]
        if return_offsets_mapping:
            return {
                "input_ids": input_ids,
                "offset_mapping": [(idx, idx + 1) for idx in range(len(text))],
            }
        return {"input_ids": input_ids}


def test_pool_decision_residual_extracts_row_and_section_states():
    # (layers=2, seq_len=6, dim=3)
    residual = np.arange(2 * 6 * 3, dtype=np.float32).reshape(2, 6, 3)
    row_boundaries = [
        {
            "row_index": 0,
            "content_start": 1,
            "content_end": 3,
            "full_start": 0,
            "full_end": 3,
        },
        {
            "row_index": 1,
            "content_start": 3,
            "content_end": 5,
            "full_start": 3,
            "full_end": 5,
        },
    ]
    section_boundaries = {
        "preamble": (0, 1),
        "market": (1, 5),
        "active_settings": (5, 6),
    }

    pooled = pool_decision_residual(residual, row_boundaries, section_boundaries)

    assert pooled["row_mean_0"].shape == (2, 3)
    assert pooled["row_eos_1"].shape == (2, 3)
    assert pooled["market_mean"].shape == (2, 3)
    assert pooled["active_settings_eos"].shape == (2, 3)
    np.testing.assert_allclose(pooled["row_eos_0"], residual[:, 2, :])
    np.testing.assert_allclose(pooled["last_token"], residual[:, -1, :])


def test_build_asset_label_rows_marks_buy_target():
    market_rows = [
        MarketRow(symbol="AAA", name="Alpha", text_block="row1", pct_5m=1.0),
        MarketRow(symbol="BBB", name="Beta", text_block="row2", pct_5m=2.0),
    ]
    computed_labels = {"is_top_5m_gainer": [0, 1]}

    rows = build_asset_label_rows(
        log_id=7,
        market_rows=market_rows,
        computed_labels=computed_labels,
        decision_type="trade",
        trade_side="buy",
        target_asset="BBB",
    )

    assert len(rows) == 2
    assert rows[0]["asset_executed_valence"] == "neutral"
    assert rows[1]["is_target_asset"] is True
    assert rows[1]["is_buy_target"] is True
    assert rows[1]["asset_executed_valence"] == "bullish"
    assert rows[1]["is_top_5m_gainer"] == 1


def test_build_asset_label_rows_marks_sell_target():
    market_rows = [
        MarketRow(symbol="AAA", name="Alpha", text_block="row1"),
    ]
    rows = build_asset_label_rows(
        log_id=9,
        market_rows=market_rows,
        computed_labels={},
        decision_type="trade",
        trade_side="sell",
        target_asset="AAA",
    )
    assert rows[0]["is_sell_target"] is True
    assert rows[0]["asset_executed_valence"] == "bearish"


def test_build_tick_label_row_sets_executed_valence():
    tick = build_tick_label_row(
        log_id=10,
        decision_type="trade",
        trade_side="sell",
        target_asset="AAA",
        n_rows=3,
        user_text="hello",
    )
    assert tick["executed_valence"] == "bearish"
    assert tick["n_rows"] == 3


def test_find_real_section_boundaries_matches_headers_by_token_subsequence():
    tokenizer = DummyTokenizer()
    user_text = (
        "Intro\n"
        "## MARKET SNAPSHOT\n"
        "- A (A) | Price: 1\n"
        "## ACTIVE STRATEGIES\n"
        "none\n"
        "## ACTIVE SETTINGS\n"
        "risk=2\n"
        "## PORTFOLIO CONTEXT\n"
        "flat\n"
    )

    boundaries = find_real_section_boundaries(tokenizer, "sys", user_text)

    assert "preamble" in boundaries
    assert "market" in boundaries
    assert "active_strategies" in boundaries
    assert "active_settings" in boundaries
    assert "portfolio" in boundaries
    assert boundaries["preamble"][1] == boundaries["market"][0]
    assert boundaries["market"][0] < boundaries["active_strategies"][0]


def test_find_real_row_boundaries_finds_each_market_row():
    tokenizer = DummyTokenizer()
    user_text = (
        "Intro\n"
        "## MARKET SNAPSHOT\n"
        "  - Alpha (AAA) | Price: 1\n"
        "    Volume: 10\n"
        "  - Beta (BBB) | Price: 2\n"
        "    Volume: 20\n"
        "## ACTIVE SETTINGS\n"
        "risk=2\n"
    )
    market_rows = [
        MarketRow(symbol="AAA", name="Alpha", text_block="  - Alpha (AAA) | Price: 1\n    Volume: 10"),
        MarketRow(symbol="BBB", name="Beta", text_block="  - Beta (BBB) | Price: 2\n    Volume: 20"),
    ]

    row_bounds = find_real_row_boundaries(tokenizer, "sys", user_text, market_rows)

    assert [rb["symbol"] for rb in row_bounds] == ["AAA", "BBB"]
    assert row_bounds[0]["full_start"] < row_bounds[0]["content_start"] < row_bounds[0]["full_end"]
    assert row_bounds[0]["full_end"] <= row_bounds[1]["full_start"]


def test_select_examples_for_shard_uses_deterministic_round_robin():
    examples = [{"log_id": idx} for idx in range(7)]

    shard0 = select_examples_for_shard(examples, shard_index=0, num_shards=3)
    shard1 = select_examples_for_shard(examples, shard_index=1, num_shards=3)
    shard2 = select_examples_for_shard(examples, shard_index=2, num_shards=3)

    assert [row["log_id"] for row in shard0] == [0, 3, 6]
    assert [row["log_id"] for row in shard1] == [1, 4]
    assert [row["log_id"] for row in shard2] == [2, 5]


def test_merge_decision_structure_shards_builds_canonical_tables(tmp_path):
    out_dir = tmp_path / "decision_structure"
    meta0, tick0, asset0 = shard_output_paths(out_dir, 0)
    meta1, tick1, asset1 = shard_output_paths(out_dir, 1)
    meta0.parent.mkdir(parents=True, exist_ok=True)

    pq.write_table(
        pa.Table.from_pylist([{"log_id": 10, "seq_len": 100}]),
        meta0,
        compression="snappy",
    )
    pq.write_table(
        pa.Table.from_pylist([{"log_id": 10, "executed_valence": "bullish"}]),
        tick0,
        compression="snappy",
    )
    pq.write_table(
        pa.Table.from_pylist([
            {"log_id": 10, "row_index": 0, "symbol": "AAA"},
            {"log_id": 10, "row_index": 1, "symbol": "BBB"},
        ]),
        asset0,
        compression="snappy",
    )

    pq.write_table(
        pa.Table.from_pylist([{"log_id": 20, "seq_len": 120}]),
        meta1,
        compression="snappy",
    )
    pq.write_table(
        pa.Table.from_pylist([{"log_id": 20, "executed_valence": "bearish"}]),
        tick1,
        compression="snappy",
    )
    pq.write_table(
        pa.Table.from_pylist([
            {"log_id": 20, "row_index": 0, "symbol": "CCC"},
        ]),
        asset1,
        compression="snappy",
    )

    result = merge_decision_structure_shards(out_dir, num_shards=2)

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


def test_clear_decision_structure_shards_removes_shard_and_canonical_tables(tmp_path):
    out_dir = tmp_path / "decision_structure"
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

    cleared = clear_decision_structure_shards(out_dir, num_shards=1, clear_canonical=True)

    assert cleared["removed"] == 6
    assert cleared["missing"] == 0
    assert not meta0.exists()
    assert not tick0.exists()
    assert not asset0.exists()
    assert not (out_dir / "metadata.parquet").exists()
