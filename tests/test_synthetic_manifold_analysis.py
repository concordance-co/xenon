import numpy as np

from research.synthetic_market.synthetic_manifold_analysis import (
    _base_coupled_variant_name,
    _coupled_geometry_metrics,
    _scalar_geometry_metrics,
    _split_ids,
    _summarize_coupled_geometry,
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


def test_coupled_variant_name_strips_background_suffix() -> None:
    assert _base_coupled_variant_name("pct_5m__net_flow_5m__bg00") == "pct_5m__net_flow_5m"
    assert _base_coupled_variant_name("pct_5m__top20_holder_pct__t01") == "pct_5m__top20_holder_pct"


def test_coupled_geometry_metrics_detects_ordered_2d_structure() -> None:
    xs = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
    ys = np.linspace(-0.5, 0.5, 5, dtype=np.float32)
    values = np.asarray([[x, y] for x in xs for y in ys], dtype=np.float32)
    X = np.stack([
        values[:, 0] + 0.2 * values[:, 1],
        values[:, 1] - 0.1 * values[:, 0],
        values[:, 0] * values[:, 1],
        values[:, 0] ** 2,
    ], axis=1)
    metrics = _coupled_geometry_metrics(X, values)
    assert metrics["n_points"] == 25
    assert metrics["distance_latent_spearman"] is not None
    assert metrics["distance_latent_spearman"] > 0.7
    assert metrics["pc12_explained_variance"] is not None
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


def test_summarize_coupled_geometry_picks_best_layer() -> None:
    summary = _summarize_coupled_geometry({
        "pct_5m__net_flow_5m": {
            "row_mean": [{"layer": 0, "distance_latent_spearman": 0.61}, {"layer": 8, "distance_latent_spearman": 0.73}],
            "row_eos": [{"layer": 4, "distance_latent_spearman": 0.69}],
        }
    })
    assert summary["pct_5m__net_flow_5m"]["representation"] == "row_mean"
    assert summary["pct_5m__net_flow_5m"]["layer"] == 8
