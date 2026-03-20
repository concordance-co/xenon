from __future__ import annotations

import numpy as np

from pipelines.interp.counterfactual import MarketRow
from pipelines.interp.decision_structure import (
    build_asset_label_rows,
    build_tick_label_row,
    pool_decision_residual,
)


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
