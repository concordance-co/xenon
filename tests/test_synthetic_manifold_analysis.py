import numpy as np

from pipelines.interp.synthetic_manifold_analysis import (
    _scalar_geometry_metrics,
    _split_ids,
    _summarize_pairwise_results,
    _summarize_regression_results,
)


def test_split_ids_returns_disjoint_partitions():
    train_ids, test_ids = _split_ids(list(range(10)), seed=42, test_fraction=0.2)
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids | test_ids) == 10
    assert len(test_ids) >= 1


def test_scalar_geometry_metrics_detects_ordered_structure():
    values = np.linspace(-1.0, 1.0, 9, dtype=np.float32)
    X = np.stack([
        values,
        values**2,
        np.sin(values),
    ], axis=1)
    metrics = _scalar_geometry_metrics(X, values)
    assert metrics["n_points"] == 9
    assert metrics["distance_value_spearman"] is not None
    assert metrics["distance_value_spearman"] > 0.7
    assert metrics["participation_ratio"] is not None


def test_summarize_regression_results_picks_best_layer():
    summary = _summarize_regression_results({
        "attractiveness_score": {
            "row_mean": [{"layer": 0, "r2": 0.3}, {"layer": 1, "r2": 0.5}],
            "row_eos": [{"layer": 0, "r2": 0.4}],
        }
    })
    assert summary["attractiveness_score"]["representation"] == "row_mean"
    assert summary["attractiveness_score"]["layer"] == 1


def test_summarize_pairwise_results_picks_best_mode_and_layer():
    summary = _summarize_pairwise_results({
        "a_beats_b_on_attractiveness": {
            "diff": {
                "row_mean": [{"layer": 0, "auroc": 0.81}],
                "row_eos": [{"layer": 0, "auroc": 0.77}],
            },
            "concat": {
                "row_mean": [{"layer": 0, "auroc": 0.73}, {"layer": 4, "auroc": 0.88}],
            },
        }
    })
    assert summary["a_beats_b_on_attractiveness"]["representation"] == "concat:row_mean"
    assert summary["a_beats_b_on_attractiveness"]["layer"] == 4
