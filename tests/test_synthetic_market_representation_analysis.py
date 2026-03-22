from __future__ import annotations

import numpy as np
import pytest

from pipelines.interp.synthetic_market_representation_analysis import (
    _base_rank_context_variant,
    _focal_relation_invariance_metrics,
    _is_profile_control_family,
    _pairwise_relation_invariance_metrics,
    _parse_profile_invariance_example_id,
    _parse_relation_invariance_example_id,
    _parse_set_geometry_example_id,
    _profile_invariance_decomposition_metrics,
    _relation_over_magnitude_control_metrics,
    _relation_over_rank_control_metrics,
    _set_geometry_alignment_metrics,
    _set_geometry_identity_metrics,
    _snapshot_geometry_metrics,
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


def test_parse_relation_invariance_example_id_extracts_all_axes() -> None:
    assert _parse_relation_invariance_example_id("relation_inv_01_02_03_04_05") == (1, 2, 3, 4, 5)
    assert _parse_relation_invariance_example_id("profile_inv_00_00_00") is None


def test_parse_set_geometry_example_id_extracts_all_axes() -> None:
    assert _parse_set_geometry_example_id("set_geom_01_02_03_04") == (1, 2, 3, 4)
    assert _parse_set_geometry_example_id("relation_inv_01_02_03_04_05") is None


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


def test_pairwise_relation_invariance_detects_stable_relations() -> None:
    examples = [
        {
            "example_id": "profile_inv_00_00_00",
            "style_idx": 0,
            "perm_idx": 0,
            "ordered_profiles": ("p0", "p1", "p2"),
            "profiles": {
                "p0": np.asarray([1.0, 0.0], dtype=np.float32),
                "p1": np.asarray([0.0, 1.0], dtype=np.float32),
                "p2": np.asarray([-1.0, 0.0], dtype=np.float32),
            },
        },
        {
            "example_id": "profile_inv_00_01_00",
            "style_idx": 1,
            "perm_idx": 0,
            "ordered_profiles": ("p0", "p1", "p2"),
            "profiles": {
                "p0": np.asarray([0.99, 0.01], dtype=np.float32),
                "p1": np.asarray([0.01, 0.99], dtype=np.float32),
                "p2": np.asarray([-0.99, 0.0], dtype=np.float32),
            },
        },
        {
            "example_id": "profile_inv_00_00_01",
            "style_idx": 0,
            "perm_idx": 1,
            "ordered_profiles": ("p0", "p1", "p2"),
            "profiles": {
                "p0": np.asarray([0.98, 0.02], dtype=np.float32),
                "p1": np.asarray([0.02, 0.98], dtype=np.float32),
                "p2": np.asarray([-0.98, 0.0], dtype=np.float32),
            },
        },
    ]

    style_metrics = _pairwise_relation_invariance_metrics(examples, mode="style_only")
    layout_metrics = _pairwise_relation_invariance_metrics(examples, mode="layout_only")
    assert style_metrics["nn_accuracy"] == 1.0
    assert style_metrics["relation_margin"] is not None
    assert style_metrics["relation_margin"] > 0.2
    assert layout_metrics["nn_accuracy"] == 1.0
    assert layout_metrics["relation_margin"] is not None
    assert layout_metrics["relation_margin"] > 0.2


def test_snapshot_geometry_metrics_detect_same_market() -> None:
    same_examples = [
        {
            "example_id": "profile_inv_00_00_00",
            "style_idx": 0,
            "perm_idx": 0,
            "ordered_profiles": ("p0", "p1", "p2"),
            "profiles": {
                "p0": np.asarray([1.0, 0.0], dtype=np.float32),
                "p1": np.asarray([0.0, 1.0], dtype=np.float32),
                "p2": np.asarray([-1.0, 0.0], dtype=np.float32),
            },
        },
        {
            "example_id": "profile_inv_00_01_00",
            "style_idx": 1,
            "perm_idx": 0,
            "ordered_profiles": ("p0", "p1", "p2"),
            "profiles": {
                "p0": np.asarray([0.99, 0.01], dtype=np.float32),
                "p1": np.asarray([0.01, 0.99], dtype=np.float32),
                "p2": np.asarray([-0.99, 0.01], dtype=np.float32),
            },
        },
    ]
    other_examples = [
        {
            "example_id": "profile_inv_01_01_00",
            "style_idx": 1,
            "perm_idx": 0,
            "ordered_profiles": ("q0", "q1", "q2"),
            "profiles": {
                "q0": np.asarray([1.0, 0.0], dtype=np.float32),
                "q1": np.asarray([0.7, 0.7], dtype=np.float32),
                "q2": np.asarray([0.0, 1.0], dtype=np.float32),
            },
        },
    ]

    metrics = _snapshot_geometry_metrics(same_examples, other_examples, mode="style_only")
    assert metrics["nn_accuracy"] == 1.0
    assert metrics["geometry_margin"] is not None
    assert metrics["geometry_margin"] > 0.9


def test_focal_relation_invariance_detects_same_scenario_across_roster_and_scale() -> None:
    examples = [
        {
            "example_id": "relation_inv_00_00_00_00_00",
            "scenario": "momentum_edge_near_tie",
            "style_idx": 0,
            "perm_idx": 0,
            "roster_idx": 0,
            "scale_idx": 0,
            "rank_bucket": "1v2",
            "vec": np.asarray([1.0, 0.0], dtype=np.float32),
        },
        {
            "example_id": "relation_inv_00_00_00_01_00",
            "scenario": "momentum_edge_near_tie",
            "style_idx": 0,
            "perm_idx": 0,
            "roster_idx": 1,
            "scale_idx": 0,
            "rank_bucket": "2v3",
            "vec": np.asarray([0.98, 0.02], dtype=np.float32),
        },
        {
            "example_id": "relation_inv_00_00_00_00_01",
            "scenario": "momentum_edge_near_tie",
            "style_idx": 0,
            "perm_idx": 0,
            "roster_idx": 0,
            "scale_idx": 1,
            "rank_bucket": "1v2",
            "vec": np.asarray([0.97, 0.03], dtype=np.float32),
        },
        {
            "example_id": "relation_inv_00_00_00_01_01",
            "scenario": "momentum_edge_near_tie",
            "style_idx": 0,
            "perm_idx": 0,
            "roster_idx": 1,
            "scale_idx": 1,
            "rank_bucket": "2v3",
            "vec": np.asarray([0.96, 0.04], dtype=np.float32),
        },
        {
            "example_id": "relation_inv_01_00_00_00_00",
            "scenario": "flow_edge_near_tie",
            "style_idx": 0,
            "perm_idx": 0,
            "roster_idx": 0,
            "scale_idx": 0,
            "rank_bucket": "1v2",
            "vec": np.asarray([0.0, 1.0], dtype=np.float32),
        },
        {
            "example_id": "relation_inv_01_00_00_01_00",
            "scenario": "flow_edge_near_tie",
            "style_idx": 0,
            "perm_idx": 0,
            "roster_idx": 1,
            "scale_idx": 0,
            "rank_bucket": "2v3",
            "vec": np.asarray([0.02, 0.98], dtype=np.float32),
        },
        {
            "example_id": "relation_inv_01_00_00_00_01",
            "scenario": "flow_edge_near_tie",
            "style_idx": 0,
            "perm_idx": 0,
            "roster_idx": 0,
            "scale_idx": 1,
            "rank_bucket": "1v2",
            "vec": np.asarray([0.03, 0.97], dtype=np.float32),
        },
        {
            "example_id": "relation_inv_01_00_00_01_01",
            "scenario": "flow_edge_near_tie",
            "style_idx": 0,
            "perm_idx": 0,
            "roster_idx": 1,
            "scale_idx": 1,
            "rank_bucket": "2v3",
            "vec": np.asarray([0.04, 0.96], dtype=np.float32),
        },
    ]

    roster_metrics = _focal_relation_invariance_metrics(examples, mode="roster_only")
    scale_metrics = _focal_relation_invariance_metrics(examples, mode="magnitude_only")
    assert roster_metrics["nn_accuracy"] == 1.0
    assert roster_metrics["relation_margin"] is not None
    assert roster_metrics["relation_margin"] > 0.5
    assert scale_metrics["nn_accuracy"] == 1.0
    assert scale_metrics["relation_margin"] is not None
    assert scale_metrics["relation_margin"] > 0.5


def test_relation_rank_control_prefers_same_relation_over_same_rank_bucket() -> None:
    examples = [
        {
            "example_id": "relation_inv_00_00_00_00_00",
            "scenario": "momentum_edge_near_tie",
            "style_idx": 0,
            "perm_idx": 0,
            "roster_idx": 0,
            "scale_idx": 0,
            "rank_bucket": "1v2",
            "vec": np.asarray([1.0, 0.0], dtype=np.float32),
        },
        {
            "example_id": "relation_inv_00_00_00_01_00",
            "scenario": "momentum_edge_near_tie",
            "style_idx": 0,
            "perm_idx": 0,
            "roster_idx": 1,
            "scale_idx": 0,
            "rank_bucket": "2v3",
            "vec": np.asarray([0.98, 0.02], dtype=np.float32),
        },
        {
            "example_id": "relation_inv_01_00_00_00_00",
            "scenario": "flow_edge_near_tie",
            "style_idx": 0,
            "perm_idx": 0,
            "roster_idx": 0,
            "scale_idx": 0,
            "rank_bucket": "1v2",
            "vec": np.asarray([0.2, 0.8], dtype=np.float32),
        },
        {
            "example_id": "relation_inv_01_00_00_01_00",
            "scenario": "flow_edge_near_tie",
            "style_idx": 0,
            "perm_idx": 0,
            "roster_idx": 1,
            "scale_idx": 0,
            "rank_bucket": "2v3",
            "vec": np.asarray([0.18, 0.82], dtype=np.float32),
        },
    ]

    metrics = _relation_over_rank_control_metrics(examples)
    assert metrics["nn_accuracy"] == 1.0
    assert metrics["relation_over_rank_margin"] is not None
    assert metrics["relation_over_rank_margin"] > 0.5


def test_relation_scale_control_prefers_same_relation_over_same_scale_bucket() -> None:
    examples = [
        {
            "example_id": "relation_inv_00_00_00_00_00",
            "scenario": "momentum_edge_near_tie",
            "style_idx": 0,
            "perm_idx": 0,
            "roster_idx": 0,
            "scale_idx": 0,
            "rank_bucket": "1v2",
            "vec": np.asarray([1.0, 0.0], dtype=np.float32),
        },
        {
            "example_id": "relation_inv_00_00_00_00_01",
            "scenario": "momentum_edge_near_tie",
            "style_idx": 0,
            "perm_idx": 0,
            "roster_idx": 0,
            "scale_idx": 1,
            "rank_bucket": "1v2",
            "vec": np.asarray([0.98, 0.02], dtype=np.float32),
        },
        {
            "example_id": "relation_inv_01_00_00_00_00",
            "scenario": "flow_edge_near_tie",
            "style_idx": 0,
            "perm_idx": 0,
            "roster_idx": 0,
            "scale_idx": 0,
            "rank_bucket": "1v2",
            "vec": np.asarray([0.2, 0.8], dtype=np.float32),
        },
        {
            "example_id": "relation_inv_01_00_00_00_01",
            "scenario": "flow_edge_near_tie",
            "style_idx": 0,
            "perm_idx": 0,
            "roster_idx": 0,
            "scale_idx": 1,
            "rank_bucket": "1v2",
            "vec": np.asarray([0.18, 0.82], dtype=np.float32),
        },
    ]

    metrics = _relation_over_magnitude_control_metrics(examples)
    assert metrics["nn_accuracy"] == 1.0
    assert metrics["relation_over_scale_margin"] is not None
    assert metrics["relation_over_scale_margin"] > 0.5


def test_relation_controls_can_anchor_one_scenario_against_full_pool() -> None:
    examples = [
        {
            "example_id": "relation_inv_00_00_00_00_00",
            "scenario": "momentum_edge_near_tie",
            "style_idx": 0,
            "perm_idx": 0,
            "roster_idx": 0,
            "scale_idx": 0,
            "rank_bucket": "1v2",
            "vec": np.asarray([1.0, 0.0], dtype=np.float32),
        },
        {
            "example_id": "relation_inv_00_00_00_01_00",
            "scenario": "momentum_edge_near_tie",
            "style_idx": 0,
            "perm_idx": 0,
            "roster_idx": 1,
            "scale_idx": 0,
            "rank_bucket": "2v3",
            "vec": np.asarray([0.98, 0.02], dtype=np.float32),
        },
        {
            "example_id": "relation_inv_00_00_00_00_01",
            "scenario": "momentum_edge_near_tie",
            "style_idx": 0,
            "perm_idx": 0,
            "roster_idx": 0,
            "scale_idx": 1,
            "rank_bucket": "1v2",
            "vec": np.asarray([0.97, 0.03], dtype=np.float32),
        },
        {
            "example_id": "relation_inv_00_00_00_01_01",
            "scenario": "momentum_edge_near_tie",
            "style_idx": 0,
            "perm_idx": 0,
            "roster_idx": 1,
            "scale_idx": 1,
            "rank_bucket": "2v3",
            "vec": np.asarray([0.96, 0.04], dtype=np.float32),
        },
        {
            "example_id": "relation_inv_01_00_00_00_00",
            "scenario": "flow_edge_near_tie",
            "style_idx": 0,
            "perm_idx": 0,
            "roster_idx": 0,
            "scale_idx": 0,
            "rank_bucket": "1v2",
            "vec": np.asarray([0.2, 0.8], dtype=np.float32),
        },
        {
            "example_id": "relation_inv_01_00_00_01_00",
            "scenario": "flow_edge_near_tie",
            "style_idx": 0,
            "perm_idx": 0,
            "roster_idx": 1,
            "scale_idx": 0,
            "rank_bucket": "2v3",
            "vec": np.asarray([0.18, 0.82], dtype=np.float32),
        },
        {
            "example_id": "relation_inv_01_00_00_00_01",
            "scenario": "flow_edge_near_tie",
            "style_idx": 0,
            "perm_idx": 0,
            "roster_idx": 0,
            "scale_idx": 1,
            "rank_bucket": "1v2",
            "vec": np.asarray([0.19, 0.81], dtype=np.float32),
        },
        {
            "example_id": "relation_inv_01_00_00_01_01",
            "scenario": "flow_edge_near_tie",
            "style_idx": 0,
            "perm_idx": 0,
            "roster_idx": 1,
            "scale_idx": 1,
            "rank_bucket": "2v3",
            "vec": np.asarray([0.17, 0.83], dtype=np.float32),
        },
    ]

    inv = _focal_relation_invariance_metrics(
        examples,
        mode="roster_only",
        anchor_scenario="momentum_edge_near_tie",
    )
    rank = _relation_over_rank_control_metrics(
        examples,
        anchor_scenario="momentum_edge_near_tie",
    )
    scale = _relation_over_magnitude_control_metrics(
        examples,
        anchor_scenario="momentum_edge_near_tie",
    )

    assert inv["nn_accuracy"] == 1.0
    assert inv["relation_margin"] is not None
    assert inv["relation_margin"] > 0.5
    assert rank["nn_accuracy"] == 1.0
    assert rank["relation_over_rank_margin"] is not None
    assert rank["relation_over_rank_margin"] > 0.5
    assert scale["nn_accuracy"] == 1.0
    assert scale["relation_over_scale_margin"] is not None
    assert scale["relation_over_scale_margin"] > 0.5


def test_set_geometry_alignment_metrics_detect_latent_shape() -> None:
    examples = [
        {
            "example_id": "set_geom_00_00_00_00",
            "scenario": "even_ladder",
            "geometry_vec": np.asarray([0.20, 0.49, 0.80, 0.29, 0.61, 0.41], dtype=np.float32),
            "pair_labels": (
                "geo_alpha__geo_beta",
                "geo_alpha__geo_gamma",
                "geo_alpha__geo_delta",
                "geo_beta__geo_gamma",
                "geo_beta__geo_delta",
                "geo_gamma__geo_delta",
            ),
        },
        {
            "example_id": "set_geom_00_01_00_00",
            "scenario": "even_ladder",
            "geometry_vec": np.asarray([0.19, 0.48, 0.79, 0.28, 0.60, 0.40], dtype=np.float32),
            "pair_labels": (
                "geo_alpha__geo_beta",
                "geo_alpha__geo_gamma",
                "geo_alpha__geo_delta",
                "geo_beta__geo_gamma",
                "geo_beta__geo_delta",
                "geo_gamma__geo_delta",
            ),
        },
    ]

    metrics = _set_geometry_alignment_metrics(examples)
    assert metrics["distance_spearman_mean"] is not None
    assert metrics["distance_spearman_mean"] > 0.9
    assert metrics["closest_pair_accuracy"] == 1.0
    assert metrics["farthest_pair_accuracy"] == 1.0


def test_set_geometry_identity_metrics_prefer_same_shape_over_other_same_rank_shapes() -> None:
    examples = [
        {
            "example_id": "set_geom_00_00_00_00",
            "scenario": "even_ladder",
            "style_idx": 0,
            "perm_idx": 0,
            "scale_idx": 0,
            "geometry_vec": np.asarray([0.20, 0.49, 0.80, 0.29, 0.61, 0.41], dtype=np.float32),
        },
        {
            "example_id": "set_geom_00_01_00_00",
            "scenario": "even_ladder",
            "style_idx": 1,
            "perm_idx": 0,
            "scale_idx": 0,
            "geometry_vec": np.asarray([0.19, 0.48, 0.79, 0.28, 0.60, 0.40], dtype=np.float32),
        },
        {
            "example_id": "set_geom_01_00_00_00",
            "scenario": "top_pair_cluster",
            "style_idx": 0,
            "perm_idx": 0,
            "scale_idx": 0,
            "geometry_vec": np.asarray([0.04, 0.68, 1.02, 0.64, 0.96, 0.22], dtype=np.float32),
        },
        {
            "example_id": "set_geom_01_01_00_00",
            "scenario": "top_pair_cluster",
            "style_idx": 1,
            "perm_idx": 0,
            "scale_idx": 0,
            "geometry_vec": np.asarray([0.05, 0.67, 1.01, 0.63, 0.95, 0.23], dtype=np.float32),
        },
    ]

    metrics = _set_geometry_identity_metrics(
        examples,
        anchor_scenario="even_ladder",
        mode="style_only",
    )
    assert metrics["nn_accuracy"] == 1.0
    assert metrics["geometry_identity_margin"] is not None
    assert metrics["geometry_identity_margin"] > 0.03
