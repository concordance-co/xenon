from __future__ import annotations

import numpy as np

from research.synthetic_market.synthetic_market_discovery_analysis import (
    _nuisance_matrix,
    _residualize_activations,
)


def test_nuisance_matrix_selects_requested_columns() -> None:
    feature_names = ["seq_len", "pct_5m_mean", "user_chars", "n_rows"]
    feature_matrix = np.asarray(
        [
            [10.0, 1.0, 100.0, 6.0],
            [11.0, 2.0, 120.0, 6.0],
        ],
        dtype=np.float32,
    )
    nuisance = _nuisance_matrix(feature_matrix, feature_names, nuisance_features={"seq_len", "user_chars", "n_rows"})
    assert nuisance.shape == (2, 3)
    assert np.allclose(nuisance[:, 0], feature_matrix[:, 0])
    assert np.allclose(nuisance[:, 1], feature_matrix[:, 2])
    assert np.allclose(nuisance[:, 2], feature_matrix[:, 3])


def test_residualize_activations_removes_linear_nuisance_signal() -> None:
    nuisance = np.asarray(
        [
            [1.0, 10.0, 6.0],
            [2.0, 11.0, 6.0],
            [3.0, 12.0, 6.0],
            [4.0, 13.0, 6.0],
        ],
        dtype=np.float32,
    )
    clean = np.asarray(
        [
            [0.5, -0.2],
            [-0.3, 0.4],
            [0.2, 0.1],
            [-0.4, -0.3],
        ],
        dtype=np.float32,
    )
    weights = np.asarray(
        [
            [2.0, -1.0],
            [0.5, 0.75],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    activations = nuisance @ weights + clean
    residuals = _residualize_activations(activations, nuisance)
    # The nuisance component should be removed almost perfectly from this tiny exact toy example.
    nuisance_projection = np.linalg.lstsq(nuisance, residuals, rcond=None)[0]
    assert np.max(np.abs(nuisance_projection)) < 1e-4
    assert residuals.shape == activations.shape
