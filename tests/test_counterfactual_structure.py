from __future__ import annotations

import numpy as np

from pipelines.interp.counterfactual.structure import (
    _align_three_matrices_by_id,
    analyze_position_subspace,
    collect_concat_groups,
    collect_row_groups,
    fit_market_subspace,
    subspace_energy_ratio,
)


def _make_cache() -> dict[str, dict[str, np.ndarray]]:
    # Two layers, 3-d hidden states.
    def arr(rows: list[list[float]]) -> np.ndarray:
        return np.array(rows, dtype=np.float32)

    base = {
        "row_mean_0": arr([[1, 0, 0], [2, 0, 0]]),
        "row_mean_1": arr([[0, 1, 0], [0, 2, 0]]),
        "row_eos_0": arr([[1, 0, 1], [2, 0, 1]]),
        "row_eos_1": arr([[0, 1, 1], [0, 2, 1]]),
        "settings_eos": arr([[1, 1, 0], [2, 2, 0]]),
        "portfolio_eos": arr([[0, 1, 1], [0, 2, 2]]),
        "constraints_eos": arr([[1, 0, 1], [2, 0, 2]]),
        "prev_decisions_eos": arr([[1, 1, 1], [2, 2, 2]]),
        "last_token": arr([[2, 0, 0], [3, 0, 0]]),
    }
    shifted = {
        k: (v + np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32))
        if k.endswith("_eos") or k == "last_token"
        else v.copy()
        for k, v in base.items()
    }
    return {
        "snap1_settings_all1": base,
        "snap1_settings_all5": shifted,
        "snap2_settings_all1": {
            k: v + np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32)
            for k, v in base.items()
        },
        "snap2_settings_all5": {
            k: v + np.array([[1, 0, 1], [1, 0, 1]], dtype=np.float32)
            for k, v in base.items()
        },
    }


def _make_snapshots() -> list[dict]:
    return [
        {
            "snapshot_id": "snap1",
            "n_rows": 2,
            "vault_address": "0xaaa",
            "snap_date": "2026-03-18",
            "labels": {"is_top": [1, 0]},
        },
        {
            "snapshot_id": "snap2",
            "n_rows": 2,
            "vault_address": "0xbbb",
            "snap_date": "2026-03-19",
            "labels": {"is_top": [0, 1]},
        },
    ]


def test_collect_row_groups_builds_grouped_examples():
    groups = collect_row_groups(
        _make_snapshots(),
        {"snap1", "snap2"},
        "settings_all1",
        "is_top",
        0,
        "row_mean",
        _make_cache(),
    )
    assert len(groups) == 2
    assert groups[0]["X"].shape == (2, 3)
    assert groups[0]["y"].tolist() == [1, 0]


def test_collect_concat_groups_appends_downstream_state():
    groups = collect_concat_groups(
        _make_snapshots(),
        {"snap1", "snap2"},
        "settings_all1",
        "is_top",
        0,
        "row_mean",
        "settings_eos",
        _make_cache(),
    )
    assert len(groups) == 2
    assert groups[0]["X"].shape == (2, 6)
    np.testing.assert_allclose(groups[0]["X"][0, 3:], np.array([1, 1, 0], dtype=np.float32))


def test_align_three_matrices_by_id_keeps_common_order():
    a = np.array([[1], [2], [3]], dtype=np.float32)
    b = np.array([[20], [30]], dtype=np.float32)
    c = np.array([[300], [100], [200]], dtype=np.float32)
    aligned_a, aligned_b, aligned_c, ids = _align_three_matrices_by_id(
        ["x", "y", "z"], a,
        ["y", "z"], b,
        ["z", "x", "y"], c,
    )
    assert ids == ["y", "z"]
    np.testing.assert_allclose(aligned_a[:, 0], np.array([2, 3], dtype=np.float32))
    np.testing.assert_allclose(aligned_b[:, 0], np.array([20, 30], dtype=np.float32))
    np.testing.assert_allclose(aligned_c[:, 0], np.array([200, 300], dtype=np.float32))


def test_fit_market_subspace_and_energy_ratio_capture_1d_structure():
    X = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ], dtype=np.float32)
    fitted = fit_market_subspace(X, variance_threshold=0.9)
    assert fitted["effective_dim"] == 1
    ratio = subspace_energy_ratio(X, fitted["basis"], mean=fitted["mean"])
    assert ratio > 0.99


def test_analyze_position_subspace_distinguishes_parallel_vs_orthogonal_delta():
    market = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ], dtype=np.float32)
    pos_a = market.copy()
    pos_b_parallel = market + np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    pos_b_orth = market + np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

    parallel_stats = analyze_position_subspace(
        market, pos_a, pos_b_parallel, variance_threshold=0.9,
    )
    orth_stats = analyze_position_subspace(
        market, pos_a, pos_b_orth, variance_threshold=0.9,
    )

    assert parallel_stats["delta_parallel_ratio"] > 0.99
    assert orth_stats["delta_parallel_ratio"] < 0.01
