from __future__ import annotations

import numpy as np
import pytest

from pipelines.interp.synthetic_market_representation_analysis import (
    _base_rank_context_variant,
    _is_profile_control_family,
    _parse_profile_invariance_example_id,
    _profile_invariance_decomposition_metrics,
    _rank_context_metrics,
    _symbol_permutation_metrics,
)


def test_base_rank_context_variant_strips_background_suffix() -> None:
    assert _base_rank_context_variant("fixed_momentum_flow_pair__bg02") == "fixed_momentum_flow_pair"
    assert _base_rank_context_variant("fixed_participation_concentration_pair") == "fixed_participation_concentration_pair"


def test_profile_control_family_accepts_phase5_and_phase6_families() -> None:
    assert _is_profile_control_family("symbol_permutation_control")
    assert _is_profile_control_family("profile_invariance_control")
    assert not _is_profile_control_family("rank_context_tradeoff")


def test_parse_profile_invariance_example_id_extracts_style_and_perm() -> None:
    assert _parse_profile_invariance_example_id("profile_inv_01_02_03") == (1, 2, 3)
    assert _parse_profile_invariance_example_id("phase5_perm_00") is None


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
        {
            "example_id": "ex2",
            "profile_id": "p2",
            "symbol": "D",
            "row_index": 2,
            "vec": np.asarray([-1.0, 0.0], dtype=np.float32),
        },
        {
            "example_id": "ex2",
            "profile_id": "p3",
            "symbol": "E",
            "row_index": 3,
            "vec": np.asarray([0.0, -1.0], dtype=np.float32),
        },
        {
            "example_id": "ex3",
            "profile_id": "p2",
            "symbol": "F",
            "row_index": 3,
            "vec": np.asarray([-0.98, -0.02], dtype=np.float32),
        },
        {
            "example_id": "ex3",
            "profile_id": "p3",
            "symbol": "G",
            "row_index": 2,
            "vec": np.asarray([-0.02, -0.98], dtype=np.float32),
        },
    ]

    metrics = _symbol_permutation_metrics(entries)
    assert metrics["same_profile_nn_accuracy"] == 1.0
    assert metrics["same_symbol_nn_accuracy"] == 0.0
    assert metrics["same_row_nn_accuracy"] == 0.0
    assert metrics["profile_control_nn_accuracy"] == 1.0
    assert metrics["profile_minus_symbol_margin"] is not None
    assert metrics["profile_minus_symbol_margin"] > 0.9
    assert metrics["profile_control_margin"] is not None
    assert metrics["profile_control_margin"] > 0.9


def test_profile_invariance_decomposition_separates_style_and_layout_controls() -> None:
    entries = [
        {
            "example_id": "profile_inv_00_00_00",
            "profile_id": "p0",
            "symbol": "A",
            "row_index": 0,
            "style_idx": 0,
            "perm_idx": 0,
            "vec": np.asarray([1.0, 0.0], dtype=np.float32),
        },
        {
            "example_id": "profile_inv_00_00_00",
            "profile_id": "p1",
            "symbol": "B",
            "row_index": 1,
            "style_idx": 0,
            "perm_idx": 0,
            "vec": np.asarray([0.0, 1.0], dtype=np.float32),
        },
        {
            "example_id": "profile_inv_00_01_00",
            "profile_id": "p0",
            "symbol": "Alpha",
            "row_index": 0,
            "style_idx": 1,
            "perm_idx": 0,
            "vec": np.asarray([0.98, 0.02], dtype=np.float32),
        },
        {
            "example_id": "profile_inv_00_01_00",
            "profile_id": "p1",
            "symbol": "Beta",
            "row_index": 1,
            "style_idx": 1,
            "perm_idx": 0,
            "vec": np.asarray([0.02, 0.98], dtype=np.float32),
        },
        {
            "example_id": "profile_inv_00_00_01",
            "profile_id": "p0",
            "symbol": "B",
            "row_index": 1,
            "style_idx": 0,
            "perm_idx": 1,
            "vec": np.asarray([0.97, 0.03], dtype=np.float32),
        },
        {
            "example_id": "profile_inv_00_00_01",
            "profile_id": "p1",
            "symbol": "A",
            "row_index": 0,
            "style_idx": 0,
            "perm_idx": 1,
            "vec": np.asarray([0.03, 0.97], dtype=np.float32),
        },
    ]

    metrics = _profile_invariance_decomposition_metrics(entries)
    assert metrics["style_only_nn_accuracy"] == 1.0
    assert metrics["layout_only_nn_accuracy"] == 1.0
    assert metrics["style_only_margin"] is not None
    assert metrics["layout_only_margin"] is not None
    assert metrics["style_only_margin"] > 0.9
    assert metrics["layout_only_margin"] > 0.9
