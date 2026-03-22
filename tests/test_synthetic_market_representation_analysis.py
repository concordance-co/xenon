from __future__ import annotations

import numpy as np
import pytest

from pipelines.interp.synthetic_market_representation_analysis import (
    _base_rank_context_variant,
    _rank_context_metrics,
    _symbol_permutation_metrics,
)


def test_base_rank_context_variant_strips_background_suffix() -> None:
    assert _base_rank_context_variant("fixed_momentum_flow_pair__bg02") == "fixed_momentum_flow_pair"
    assert _base_rank_context_variant("fixed_participation_concentration_pair") == "fixed_participation_concentration_pair"


def test_rank_context_metrics_prefers_same_symbol_neighbors() -> None:
    entries = [
        {
            "symbol": "A",
            "bg_variant": "00",
            "vec": np.asarray([1.0, 0.0], dtype=np.float32),
            "attractiveness_rank": 2,
            "risk_adjusted_rank": 2,
            "is_best_asset": 0,
        },
        {
            "symbol": "B",
            "bg_variant": "00",
            "vec": np.asarray([0.0, 1.0], dtype=np.float32),
            "attractiveness_rank": 1,
            "risk_adjusted_rank": 1,
            "is_best_asset": 1,
        },
        {
            "symbol": "A",
            "bg_variant": "01",
            "vec": np.asarray([0.98, 0.02], dtype=np.float32),
            "attractiveness_rank": 3,
            "risk_adjusted_rank": 3,
            "is_best_asset": 0,
        },
        {
            "symbol": "B",
            "bg_variant": "01",
            "vec": np.asarray([0.02, 0.98], dtype=np.float32),
            "attractiveness_rank": 2,
            "risk_adjusted_rank": 2,
            "is_best_asset": 0,
        },
    ]

    metrics = _rank_context_metrics(entries)
    assert metrics["same_symbol_margin"] is not None
    assert metrics["same_symbol_margin"] > 0.9
    assert metrics["same_symbol_nn_accuracy"] == 1.0
    assert metrics["pair_diff_cosine_mean"] == pytest.approx(1.0)


def test_symbol_permutation_metrics_can_favor_profile_over_symbol() -> None:
    entries = [
        {
            "example_id": "ex0",
            "profile_id": "p0",
            "symbol": "A",
            "row_index": 0,
            "vec": np.asarray([1.0, 0.0], dtype=np.float32),
        },
        {
            "example_id": "ex0",
            "profile_id": "p1",
            "symbol": "B",
            "row_index": 1,
            "vec": np.asarray([0.0, 1.0], dtype=np.float32),
        },
        {
            "example_id": "ex1",
            "profile_id": "p0",
            "symbol": "C",
            "row_index": 1,
            "vec": np.asarray([0.98, 0.02], dtype=np.float32),
        },
        {
            "example_id": "ex1",
            "profile_id": "p1",
            "symbol": "A",
            "row_index": 0,
            "vec": np.asarray([0.02, 0.98], dtype=np.float32),
        },
    ]

    metrics = _symbol_permutation_metrics(entries)
    assert metrics["same_profile_nn_accuracy"] == 1.0
    assert metrics["same_symbol_nn_accuracy"] == 0.0
    assert metrics["same_row_nn_accuracy"] == 0.0
    assert metrics["profile_minus_symbol_margin"] is not None
    assert metrics["profile_minus_symbol_margin"] > 0.9
