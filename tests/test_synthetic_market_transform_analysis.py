from __future__ import annotations

import numpy as np

from pipelines.interp.synthetic_market_transform_analysis import (
    _compose_matrices,
    _evaluate_transform,
    _fit_diagonal,
    _fit_linear,
    _fit_orthogonal,
    _fit_similarity,
    _matrix_summary,
    _select_states,
)


def test_fit_orthogonal_recovers_rotation() -> None:
    theta = np.deg2rad(30.0)
    rotation = np.asarray(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=np.float32,
    )
    x = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5], [0.5, -0.75]], dtype=np.float32)
    y = x @ rotation
    fitted = _fit_orthogonal(x, y)
    assert np.allclose(fitted, rotation, atol=1e-4)


def test_fit_similarity_recovers_rotation_and_scale() -> None:
    theta = np.deg2rad(-20.0)
    rotation = np.asarray(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=np.float32,
    )
    matrix = 1.7 * rotation
    x = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5], [0.5, -0.75]], dtype=np.float32)
    y = x @ matrix
    fitted = _fit_similarity(x, y)
    assert np.allclose(fitted, matrix, atol=1e-4)


def test_fit_diagonal_recovers_axis_scaling() -> None:
    matrix = np.diag(np.asarray([1.5, 0.6], dtype=np.float32))
    x = np.asarray([[1.0, 2.0], [-1.0, 1.5], [0.25, -0.5], [2.0, -1.0]], dtype=np.float32)
    y = x @ matrix
    fitted = _fit_diagonal(x, y)
    assert np.allclose(fitted, matrix, atol=1e-5)


def test_fit_linear_recovers_general_map() -> None:
    matrix = np.asarray([[1.4, 0.3], [-0.2, 0.8]], dtype=np.float32)
    x = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5], [0.5, -0.75]], dtype=np.float32)
    y = x @ matrix
    fitted = _fit_linear(x, y)
    assert np.allclose(fitted, matrix, atol=1e-5)


def test_evaluate_transform_scores_perfect_map() -> None:
    matrix = np.asarray([[1.2, 0.0], [0.0, 0.7]], dtype=np.float32)
    source_coords = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    target_coords = source_coords @ matrix
    source_vec = np.asarray([1.0, 1.0, 2.0, np.sqrt(2.0), np.sqrt(2.0), 2.0], dtype=np.float32)
    target_vec = np.asarray(
        [
            np.linalg.norm(target_coords[0] - target_coords[1]),
            np.linalg.norm(target_coords[0] - target_coords[2]),
            np.linalg.norm(target_coords[0] - target_coords[3]),
            np.linalg.norm(target_coords[1] - target_coords[2]),
            np.linalg.norm(target_coords[1] - target_coords[3]),
            np.linalg.norm(target_coords[2] - target_coords[3]),
        ],
        dtype=np.float32,
    )
    pairs = [
        (
            {
                "decoded_centered": source_coords,
                "geometry_vec": source_vec,
                "score_geometry_vec": target_vec,
                "latent_geometry_vec": source_vec,
            },
            {
                "decoded_centered": target_coords,
                "geometry_vec": target_vec,
                "score_geometry_vec": target_vec,
                "latent_geometry_vec": source_vec,
            },
        )
    ]
    metrics = _evaluate_transform(pairs, matrix)
    assert metrics["coord_r2_mean"] is not None and metrics["coord_r2_mean"] > 0.999
    assert metrics["distance_spearman_mean"] is not None and metrics["distance_spearman_mean"] > 0.999


def test_matrix_summary_exposes_rotation_and_anisotropy() -> None:
    matrix = np.asarray([[0.0, -2.0], [1.0, 0.0]], dtype=np.float32)
    summary = _matrix_summary(matrix)
    assert summary["anisotropy_ratio"] is not None
    assert summary["singular_values"][0] >= summary["singular_values"][1]


def test_compose_matrices_applies_row_vector_order() -> None:
    first = np.asarray([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    second = np.asarray([[1.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    composed = _compose_matrices([first, second])
    point = np.asarray([[1.0, 1.0]], dtype=np.float32)
    assert np.allclose(point @ composed, (point @ first) @ second)


def test_select_states_prefers_best_average_transfer_and_realignment() -> None:
    fake = {
        "set_geometry_context_transfer": {
            "latent_x": {
                "market_only_to_risk_1": {
                    "row_mean": [{"r2": 0.9, "layer": 0}, {"r2": 0.8, "layer": 1}],
                    "row_eos": [{"r2": 0.4, "layer": 0}, {"r2": 0.5, "layer": 1}],
                },
                "market_only_to_risk_2": {
                    "row_mean": [{"r2": 0.88, "layer": 0}, {"r2": 0.79, "layer": 1}],
                    "row_eos": [{"r2": 0.41, "layer": 0}, {"r2": 0.52, "layer": 1}],
                },
                "market_only_to_risk_3": {
                    "row_mean": [{"r2": 0.89, "layer": 0}, {"r2": 0.81, "layer": 1}],
                    "row_eos": [{"r2": 0.42, "layer": 0}, {"r2": 0.53, "layer": 1}],
                },
                "market_only_to_risk_4": {
                    "row_mean": [{"r2": 0.87, "layer": 0}, {"r2": 0.82, "layer": 1}],
                    "row_eos": [{"r2": 0.43, "layer": 0}, {"r2": 0.54, "layer": 1}],
                },
                "market_only_to_risk_5": {
                    "row_mean": [{"r2": 0.86, "layer": 0}, {"r2": 0.83, "layer": 1}],
                    "row_eos": [{"r2": 0.44, "layer": 0}, {"r2": 0.55, "layer": 1}],
                },
            },
            "latent_y": {
                "market_only_to_risk_1": {
                    "row_mean": [{"r2": 0.91, "layer": 0}, {"r2": 0.81, "layer": 1}],
                    "row_eos": [{"r2": 0.45, "layer": 0}, {"r2": 0.56, "layer": 1}],
                },
                "market_only_to_risk_2": {
                    "row_mean": [{"r2": 0.9, "layer": 0}, {"r2": 0.8, "layer": 1}],
                    "row_eos": [{"r2": 0.46, "layer": 0}, {"r2": 0.57, "layer": 1}],
                },
                "market_only_to_risk_3": {
                    "row_mean": [{"r2": 0.89, "layer": 0}, {"r2": 0.79, "layer": 1}],
                    "row_eos": [{"r2": 0.47, "layer": 0}, {"r2": 0.58, "layer": 1}],
                },
                "market_only_to_risk_4": {
                    "row_mean": [{"r2": 0.88, "layer": 0}, {"r2": 0.78, "layer": 1}],
                    "row_eos": [{"r2": 0.48, "layer": 0}, {"r2": 0.59, "layer": 1}],
                },
                "market_only_to_risk_5": {
                    "row_mean": [{"r2": 0.87, "layer": 0}, {"r2": 0.77, "layer": 1}],
                    "row_eos": [{"r2": 0.49, "layer": 0}, {"r2": 0.6, "layer": 1}],
                },
            },
        },
        "set_geometry_context_realignment": {
            context: {
                "row_mean": [
                    {"score_over_base_margin": 0.01, "layer": 0},
                    {"score_over_base_margin": 0.02, "layer": 1},
                ],
                "row_eos": [
                    {"score_over_base_margin": 0.03, "layer": 0},
                    {"score_over_base_margin": 0.05, "layer": 1},
                ],
            }
            for context in ["market_only", "risk_1", "risk_2", "risk_3", "risk_4", "risk_5"]
        },
    }
    states = _select_states(fake)
    assert states["early"]["row_key"] == "row_mean"
    assert states["early"]["layer"] == 0
    assert states["late"]["row_key"] == "row_eos"
    assert states["late"]["layer"] == 1


def test_select_states_supports_portfolio_ladder_contexts() -> None:
    fake = {
        "set_geometry_context_transfer": {
            "latent_x": {
                "market_only_to_portfolio_1": {
                    "row_mean": [{"r2": 0.91, "layer": 0}, {"r2": 0.8, "layer": 1}],
                    "row_eos": [{"r2": 0.3, "layer": 0}, {"r2": 0.4, "layer": 1}],
                },
                "market_only_to_portfolio_2": {
                    "row_mean": [{"r2": 0.9, "layer": 0}, {"r2": 0.79, "layer": 1}],
                    "row_eos": [{"r2": 0.31, "layer": 0}, {"r2": 0.41, "layer": 1}],
                },
                "market_only_to_portfolio_3": {
                    "row_mean": [{"r2": 0.89, "layer": 0}, {"r2": 0.78, "layer": 1}],
                    "row_eos": [{"r2": 0.32, "layer": 0}, {"r2": 0.42, "layer": 1}],
                },
                "market_only_to_portfolio_4": {
                    "row_mean": [{"r2": 0.88, "layer": 0}, {"r2": 0.77, "layer": 1}],
                    "row_eos": [{"r2": 0.33, "layer": 0}, {"r2": 0.43, "layer": 1}],
                },
                "market_only_to_portfolio_5": {
                    "row_mean": [{"r2": 0.87, "layer": 0}, {"r2": 0.76, "layer": 1}],
                    "row_eos": [{"r2": 0.34, "layer": 0}, {"r2": 0.44, "layer": 1}],
                },
            },
            "latent_y": {
                "market_only_to_portfolio_1": {
                    "row_mean": [{"r2": 0.92, "layer": 0}, {"r2": 0.81, "layer": 1}],
                    "row_eos": [{"r2": 0.35, "layer": 0}, {"r2": 0.45, "layer": 1}],
                },
                "market_only_to_portfolio_2": {
                    "row_mean": [{"r2": 0.91, "layer": 0}, {"r2": 0.8, "layer": 1}],
                    "row_eos": [{"r2": 0.36, "layer": 0}, {"r2": 0.46, "layer": 1}],
                },
                "market_only_to_portfolio_3": {
                    "row_mean": [{"r2": 0.9, "layer": 0}, {"r2": 0.79, "layer": 1}],
                    "row_eos": [{"r2": 0.37, "layer": 0}, {"r2": 0.47, "layer": 1}],
                },
                "market_only_to_portfolio_4": {
                    "row_mean": [{"r2": 0.89, "layer": 0}, {"r2": 0.78, "layer": 1}],
                    "row_eos": [{"r2": 0.38, "layer": 0}, {"r2": 0.48, "layer": 1}],
                },
                "market_only_to_portfolio_5": {
                    "row_mean": [{"r2": 0.88, "layer": 0}, {"r2": 0.77, "layer": 1}],
                    "row_eos": [{"r2": 0.39, "layer": 0}, {"r2": 0.49, "layer": 1}],
                },
            },
        },
        "set_geometry_context_realignment": {
            context: {
                "row_mean": [
                    {"score_over_base_margin": 0.01, "layer": 0},
                    {"score_over_base_margin": 0.015, "layer": 1},
                ],
                "row_eos": [
                    {"score_over_base_margin": 0.02, "layer": 0},
                    {"score_over_base_margin": 0.04, "layer": 1},
                ],
            }
            for context in [
                "market_only",
                "portfolio_1",
                "portfolio_2",
                "portfolio_3",
                "portfolio_4",
                "portfolio_5",
            ]
        },
    }
    states = _select_states(fake)
    assert states["early"]["row_key"] == "row_mean"
    assert states["early"]["layer"] == 0
    assert states["late"]["row_key"] == "row_eos"
    assert states["late"]["layer"] == 1


def test_select_states_supports_affordance_ladder_contexts() -> None:
    fake = {
        "set_geometry_context_transfer": {
            "latent_x": {
                "market_only_to_affordance_1": {
                    "row_mean": [{"r2": 0.89, "layer": 0}, {"r2": 0.8, "layer": 1}],
                    "row_eos": [{"r2": 0.31, "layer": 0}, {"r2": 0.41, "layer": 1}],
                },
                "market_only_to_affordance_2": {
                    "row_mean": [{"r2": 0.88, "layer": 0}, {"r2": 0.79, "layer": 1}],
                    "row_eos": [{"r2": 0.32, "layer": 0}, {"r2": 0.42, "layer": 1}],
                },
                "market_only_to_affordance_3": {
                    "row_mean": [{"r2": 0.87, "layer": 0}, {"r2": 0.78, "layer": 1}],
                    "row_eos": [{"r2": 0.33, "layer": 0}, {"r2": 0.43, "layer": 1}],
                },
                "market_only_to_affordance_4": {
                    "row_mean": [{"r2": 0.86, "layer": 0}, {"r2": 0.77, "layer": 1}],
                    "row_eos": [{"r2": 0.34, "layer": 0}, {"r2": 0.44, "layer": 1}],
                },
                "market_only_to_affordance_5": {
                    "row_mean": [{"r2": 0.85, "layer": 0}, {"r2": 0.76, "layer": 1}],
                    "row_eos": [{"r2": 0.35, "layer": 0}, {"r2": 0.45, "layer": 1}],
                },
            },
            "latent_y": {
                "market_only_to_affordance_1": {
                    "row_mean": [{"r2": 0.91, "layer": 0}, {"r2": 0.81, "layer": 1}],
                    "row_eos": [{"r2": 0.36, "layer": 0}, {"r2": 0.46, "layer": 1}],
                },
                "market_only_to_affordance_2": {
                    "row_mean": [{"r2": 0.9, "layer": 0}, {"r2": 0.8, "layer": 1}],
                    "row_eos": [{"r2": 0.37, "layer": 0}, {"r2": 0.47, "layer": 1}],
                },
                "market_only_to_affordance_3": {
                    "row_mean": [{"r2": 0.89, "layer": 0}, {"r2": 0.79, "layer": 1}],
                    "row_eos": [{"r2": 0.38, "layer": 0}, {"r2": 0.48, "layer": 1}],
                },
                "market_only_to_affordance_4": {
                    "row_mean": [{"r2": 0.88, "layer": 0}, {"r2": 0.78, "layer": 1}],
                    "row_eos": [{"r2": 0.39, "layer": 0}, {"r2": 0.49, "layer": 1}],
                },
                "market_only_to_affordance_5": {
                    "row_mean": [{"r2": 0.87, "layer": 0}, {"r2": 0.77, "layer": 1}],
                    "row_eos": [{"r2": 0.4, "layer": 0}, {"r2": 0.5, "layer": 1}],
                },
            },
        },
        "set_geometry_context_realignment": {
            context: {
                "row_mean": [
                    {"score_over_base_margin": 0.01, "layer": 0},
                    {"score_over_base_margin": 0.015, "layer": 1},
                ],
                "row_eos": [
                    {"score_over_base_margin": 0.02, "layer": 0},
                    {"score_over_base_margin": 0.04, "layer": 1},
                ],
            }
            for context in [
                "market_only",
                "affordance_1",
                "affordance_2",
                "affordance_3",
                "affordance_4",
                "affordance_5",
            ]
        },
    }
    states = _select_states(fake)
    assert states["early"]["row_key"] == "row_mean"
    assert states["early"]["layer"] == 0
    assert states["late"]["row_key"] == "row_eos"
    assert states["late"]["layer"] == 1


def test_select_states_returns_error_when_context_transfer_missing() -> None:
    fake = {
        "set_geometry_context_transfer": {},
        "set_geometry_context_realignment": {},
    }
    states = _select_states(fake)
    assert states["early"]["error"] == "insufficient_context_transfer"
    assert states["late"]["error"] == "insufficient_context_realignment"
