from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from pipelines.interp.synthetic.db import (
    build_synthetic_example_query,
    capture_view_name,
    context_ladder_view_name,
    prepare_rows_for_upload,
)


def _write_tables(tmp_path) -> None:
    tick_rows = [
        {
            "log_id": 2_000_000_010,
            "example_id": "ex_b",
            "family": "archetype_family",
            "family_variant": "stable_winner",
            "context_variant": "high_risk",
            "system_prompt": "s",
            "user_prompt": "u",
            "prompt_messages_json": "[]",
            "labels_json": "{}",
            "best_asset": "A",
            "buy_any": 1,
            "observe_vs_act": "act",
            "num_assets": 4,
        },
        {
            "log_id": 2_000_000_001,
            "example_id": "ex_a",
            "family": "pairwise_tradeoff",
            "family_variant": "momentum_vs_flow_s00",
            "context_variant": "market_only",
            "system_prompt": "s",
            "user_prompt": "u",
            "prompt_messages_json": "[]",
            "labels_json": "{}",
            "best_asset": "B",
            "buy_any": 1,
            "observe_vs_act": "act",
            "num_assets": 4,
        },
    ]
    asset_rows = [
        {
            "log_id": 2_000_000_001,
            "example_id": "ex_a",
            "family": "pairwise_tradeoff",
            "family_variant": "momentum_vs_flow_s00",
            "context_variant": "market_only",
            "row_index": 0,
            "symbol": "A",
            "archetype": "momentum_burst",
            "pct_5m": 1.0,
            "pct_1h": 2.0,
            "net_flow_5m": 0.1,
            "vol_5m": 1.0,
            "vol_1h": 2.0,
            "unique_traders_5m": 4,
            "top20_holder_pct": 20.0,
            "age_bucket": "mid",
            "momentum_score": 1.0,
            "participation_score": 1.0,
            "flow_score": 1.0,
            "concentration_penalty": 0.0,
            "riskiness_score": 1.0,
            "attractiveness_score": 1.0,
            "risk_adjusted_score": 1.0,
            "edge_after_fee_score": 0.5,
            "edge_gt_fee": 1.0,
            "attractiveness_rank": 1,
            "risk_adjusted_rank": 1,
            "is_best_asset": 1,
            "buyable_if_unconstrained": 1,
            "acceptable_under_risk_setting": 1,
        },
    ]
    pairwise_rows = [
        {
            "log_id": 2_000_000_001,
            "example_id": "ex_a",
            "family": "pairwise_tradeoff",
            "family_variant": "momentum_vs_flow_s00",
            "context_variant": "market_only",
            "asset_a": "A",
            "asset_b": "B",
            "a_beats_b_on_attractiveness": 1,
            "a_beats_b_on_risk_adjusted": 1,
            "delta_pct_5m": 1.0,
            "delta_pct_1h": 1.0,
            "delta_net_flow_5m": 1.0,
            "delta_vol_5m": 1.0,
            "delta_unique_traders_5m": 1,
            "delta_top20_holder_pct": 1.0,
        },
    ]
    pq.write_table(pa.Table.from_pylist(tick_rows), tmp_path / "synthetic_market_tick_records.parquet")
    pq.write_table(pa.Table.from_pylist(asset_rows), tmp_path / "synthetic_market_asset_records.parquet")
    pq.write_table(pa.Table.from_pylist(pairwise_rows), tmp_path / "synthetic_market_pairwise_records.parquet")


def test_prepare_rows_for_upload_assigns_selection_rank(tmp_path) -> None:
    _write_tables(tmp_path)
    tick_rows, asset_rows, pairwise_rows = prepare_rows_for_upload(tmp_path, phase_name="phase1")
    tick_by_log_id = {row["log_id"]: row for row in tick_rows}
    assert tick_by_log_id[2_000_000_001]["selection_rank"] == 1
    assert tick_by_log_id[2_000_000_001]["capture_priority"] > tick_by_log_id[2_000_000_010]["capture_priority"]
    assert asset_rows[0]["phase_name"] == "phase1"
    assert pairwise_rows[0]["phase_name"] == "phase1"


def test_build_synthetic_example_query_defaults_to_phase1_view() -> None:
    query, params = build_synthetic_example_query(
        select_columns=["log_id", "prompt_messages_json"],
        cohort_view=None,
        order_mode="selection_rank_asc",
        limit=10,
    )
    assert "FROM synthetic_market_phase1_capture_v0" in query
    assert "ORDER BY selection_rank ASC NULLS LAST, log_id" in query
    assert params == [10]


def test_phase_view_names_are_sanitized() -> None:
    assert capture_view_name("Phase 2 Geometry") == "synthetic_market_phase_2_geometry_capture_v0"
    assert context_ladder_view_name("Phase 2 Geometry") == "synthetic_market_phase_2_geometry_context_ladder_v0"
