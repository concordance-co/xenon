from __future__ import annotations

import numpy as np

from projects.DX_TERMINAL.research_rerun.postmarket import (
    build_affordance_edited_variant,
)
from projects.DX_TERMINAL.research_rerun.postmarket_geometry import (
    AFFORDANCE_GROUP,
    RISK_GROUP,
    _evaluate_geometry_predictions,
    _ordered_contexts,
)


def test_ordered_contexts_for_risk_and_affordance() -> None:
    risk_rows = [{"variant": "risk_5"}, {"variant": "risk_1"}, {"variant": "risk_3"}]
    affordance_rows = [{"variant": "affordance_3"}, {"variant": "market_only"}, {"variant": "affordance_1"}]
    assert _ordered_contexts(RISK_GROUP, risk_rows) == ["risk_1", "risk_3", "risk_5"]
    assert _ordered_contexts(AFFORDANCE_GROUP, affordance_rows) == ["market_only", "affordance_1", "affordance_3"]


def test_affordance_variant_rewrites_portfolio_and_constraints() -> None:
    user_text = """## PORTFOLIO CONTEXT

old portfolio

## CONSTRAINTS

old constraints

## PREVIOUS DECISIONS

history
"""
    edited = build_affordance_edited_variant(
        user_text,
        context_variant="affordance_4",
        roster_symbols=["A", "B", "C", "D", "E", "F"],
        selected_symbols=["A", "B", "C", "D", "E", "F"],
        holdings_by_symbol={
            "A": {
                "Symbol": "A",
                "Balance": 12_345.678,
                "AvgEntryPriceInEth": 0.00000123,
                "UnrealizedPnlPercent": 12.5,
                "TimeSinceLastSwapOrGenesisOrReap": 7200,
                "TimeHeld": 86400,
            },
            "C": {
                "Symbol": "C",
                "Balance": 999.0,
                "AvgEntryPriceInEth": 0.00000456,
                "UnrealizedPnlPercent": -3.4,
                "TimeSinceLastSwapOrGenesisOrReap": 3600,
                "TimeHeld": 5400,
            },
        },
    )
    assert "## PORTFOLIO CONTEXT" in edited
    assert "- ETH: Balance: 0.006000" in edited
    assert "- A: Balance:" in edited
    assert "## CONSTRAINTS" in edited
    assert "## PRICE IMPACT LIMITS (max 400 bps)" in edited
    assert "BUY max 0.00% of ETH" in edited
    assert "SELL max 100.00% of A" in edited


def test_evaluate_geometry_predictions_prefers_score_geometry() -> None:
    y_pred = np.asarray(
        [
            [0.0, 0.0, 1.0, 0.2, 2.0, 0.4, 3.0, 0.6],
            [0.0, 0.0, 1.0, 0.2, 2.0, 0.4, 3.0, 0.6],
        ],
        dtype=np.float32,
    )
    row_meta = [
        {
            "base_coords": np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
            "score_coords": np.asarray([[0.0, 0.0], [1.0, 0.2], [2.0, 0.4], [3.0, 0.6]], dtype=np.float32),
        },
        {
            "base_coords": np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
            "score_coords": np.asarray([[0.0, 0.0], [1.0, 0.2], [2.0, 0.4], [3.0, 0.6]], dtype=np.float32),
        },
    ]
    metrics = _evaluate_geometry_predictions(y_pred, row_meta)
    assert metrics["score_distance_spearman_mean"] is not None
    assert metrics["base_distance_spearman_mean"] is not None
    assert metrics["score_over_base_margin"] is not None
    assert metrics["score_over_base_margin"] >= 0.0


def test_evaluate_geometry_predictions_supports_six_asset_rosters() -> None:
    score_coords = np.asarray(
        [[0.0, 0.0], [0.5, 0.1], [1.0, 0.2], [1.5, 0.3], [2.0, 0.4], [2.5, 0.5]],
        dtype=np.float32,
    )
    y_pred = np.asarray([score_coords.reshape(-1)], dtype=np.float32)
    row_meta = [
        {
            "base_coords": np.asarray(
                [[0.0, 0.0], [0.5, 0.8], [1.0, 1.6], [1.5, 2.4], [2.0, 3.2], [2.5, 4.0]],
                dtype=np.float32,
            ),
            "score_coords": score_coords,
        }
    ]
    metrics = _evaluate_geometry_predictions(y_pred, row_meta)
    assert metrics["score_distance_spearman_mean"] is not None
    assert metrics["score_distance_cosine_mean"] is not None
