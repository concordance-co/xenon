from __future__ import annotations

import numpy as np

from projects.DX_TERMINAL.phases.synthetic_market.synthetic_market_axis_decomposition_analysis import (
    _best_pair,
    _best_pair_quadratic,
    _build_visible_prompt_features,
    _split_feature_table,
)


def test_build_visible_prompt_features_uses_prompt_visible_aggregates_only() -> None:
    tick_rows = [
        {
            "log_id": 1,
            "context_variant": "market_only",
            "family": "market_basis_scalar",
            "user_chars": 100,
            "n_rows": 2,
        }
    ]
    metadata_rows = [{"log_id": 1, "seq_len": 42}]
    asset_rows = [
        {
            "log_id": 1,
            "row_index": 0,
            "symbol": "AAA",
            "pct_5m": 1.0,
            "pct_1h": 2.0,
            "net_flow_5m": 3.0,
            "vol_5m": 4.0,
            "vol_1h": 5.0,
            "unique_traders_5m": 6.0,
            "top20_holder_pct": 7.0,
            "edge_after_fee_score": 99.0,
        },
        {
            "log_id": 1,
            "row_index": 1,
            "symbol": "BBB",
            "pct_5m": 3.0,
            "pct_1h": 6.0,
            "net_flow_5m": 9.0,
            "vol_5m": 12.0,
            "vol_1h": 15.0,
            "unique_traders_5m": 18.0,
            "top20_holder_pct": 21.0,
            "edge_after_fee_score": 101.0,
        },
    ]

    rows = _build_visible_prompt_features(
        tick_rows,
        asset_rows,
        metadata_rows,
        context_variant="market_only",
        family_allowlist=("market_basis_scalar",),
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["pct_1h_std"] == np.std([2.0, 6.0])
    assert row["pct_1h_gap"] == 4.0
    assert row["top20_holder_pct_mad"] == np.mean(np.abs(np.asarray([7.0, 21.0]) - 14.0))
    assert row["pct_1h_max_minus_rest_mean"] == 4.0
    assert row["pct_1h_top1_minus_median"] == 2.0
    assert row["pct_1h_leader_zscore"] > 0.0
    assert "edge_after_fee_mean" not in row
    assert "attractiveness_max" not in row


def test_best_pair_finds_joint_signal() -> None:
    X = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    y = X[:, 0] + X[:, 1]
    feature_names = ["a", "b", "c"]
    best = _best_pair(
        X,
        y,
        feature_names,
        splitter=__import__("sklearn.model_selection").model_selection.KFold(n_splits=3, shuffle=True, random_state=42),
    )
    assert best["features"] == ["a", "b"]


def test_best_pair_quadratic_finds_interaction_signal() -> None:
    X = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    y = X[:, 0] * X[:, 1]
    feature_names = ["a", "b", "c"]
    best = _best_pair_quadratic(
        X,
        y,
        feature_names,
        splitter=__import__("sklearn.model_selection").model_selection.KFold(n_splits=3, shuffle=True, random_state=42),
    )
    assert best["features"] == ["a", "b"]


def test_split_feature_table_separates_nuisance_from_candidates() -> None:
    rows = [
        {"log_id": 10, "seq_len": 5.0, "user_chars": 90.0, "n_rows": 6.0, "pct_1h_std": 1.2},
        {"log_id": 11, "seq_len": 6.0, "user_chars": 100.0, "n_rows": 6.0, "pct_1h_std": 1.8},
    ]
    log_ids, feature_names, X, nuisance_names, nuisance = _split_feature_table(rows)
    assert log_ids == [10, 11]
    assert feature_names == ["pct_1h_std"]
    assert nuisance_names == ["seq_len", "user_chars", "n_rows"]
    assert X.shape == (2, 1)
    assert nuisance.shape == (2, 3)
