"""Decompose strong Phase 15 market PCs into prompt-derived market formulas.

This analysis stays in the clean Phase 15 discovery setting and asks:

1. Which prompt-visible market aggregates best explain the leader-like and
   dispersion-like discovered axes?
2. Are those axes mostly explained by a single formula, a small linear mixture,
   or a mild nonlinear interaction?
3. Do those explanations survive basic sanity checks such as nuisance controls
   and shuffled targets?

Only prompt-visible quantities and aggregates derived from prompt-visible market
rows are used. Hidden synthetic sidecar scores are intentionally excluded.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from pipelines.interp.synthetic_manifold_analysis import (
    _load_structure_tables,
    _preload_pooled_residuals,
)
from pipelines.interp.synthetic_market_discovery_analysis import (
    _group_asset_rows,
    _residualize_activations,
    _spearman,
)


VISIBLE_METRIC_NAMES = (
    "pct_5m",
    "pct_1h",
    "net_flow_5m",
    "vol_5m",
    "vol_1h",
    "unique_traders_5m",
    "top20_holder_pct",
)

NUISANCE_FEATURE_NAMES = ("seq_len", "user_chars", "n_rows")

AGGREGATE_SUFFIXES = (
    "max_minus_rest_mean",
    "top1_minus_median",
    "leader_zscore",
    "top2_mean",
    "cv_abs",
    "range",
    "mean",
    "std",
    "max",
    "min",
    "gap",
    "mad",
    "median",
)


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _safe_range(values: list[float]) -> float:
    return float(max(values) - min(values)) if values else 0.0


def _safe_median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _leader_gap(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    ordered = sorted(values, reverse=True)
    return float(ordered[0] - ordered[1])


def _top2_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values, reverse=True)
    return float(np.mean(ordered[: min(2, len(ordered))]))


def _mad(values: list[float]) -> float:
    if not values:
        return 0.0
    center = float(np.mean(values))
    return float(np.mean(np.abs(np.asarray(values, dtype=np.float32) - center)))


def _max_minus_rest_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values, reverse=True)
    return float(ordered[0] - np.mean(ordered[1:]))


def _top1_minus_median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values, reverse=True)
    return float(ordered[0] - np.median(ordered))


def _leader_zscore(values: list[float]) -> float:
    if not values:
        return 0.0
    std = float(np.std(values))
    if std == 0.0:
        return 0.0
    return float((max(values) - np.mean(values)) / std)


def _cv_abs(values: list[float]) -> float:
    if not values:
        return 0.0
    denom = float(np.mean(np.abs(values)))
    if denom == 0.0:
        return 0.0
    return float(np.std(values) / denom)


def _aggregate_metric_family(values: list[float]) -> dict[str, float]:
    return {
        "mean": _safe_mean(values),
        "std": _safe_std(values),
        "max": float(max(values)) if values else 0.0,
        "min": float(min(values)) if values else 0.0,
        "range": _safe_range(values),
        "gap": _leader_gap(values),
        "mad": _mad(values),
        "median": _safe_median(values),
        "top2_mean": _top2_mean(values),
        "max_minus_rest_mean": _max_minus_rest_mean(values),
        "top1_minus_median": _top1_minus_median(values),
        "leader_zscore": _leader_zscore(values),
        "cv_abs": _cv_abs(values),
    }


def _build_visible_prompt_features(
    tick_rows: list[dict[str, Any]],
    asset_rows: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    *,
    context_variant: str,
    family_allowlist: tuple[str, ...],
) -> list[dict[str, Any]]:
    asset_by_log = _group_asset_rows(asset_rows)
    tick_by_log = {
        int(row["log_id"]): dict(row)
        for row in tick_rows
        if str(row.get("context_variant")) == context_variant
        and (not family_allowlist or str(row.get("family")) in family_allowlist)
    }
    meta_by_log = {int(row["log_id"]): dict(row) for row in metadata_rows}

    rows: list[dict[str, Any]] = []
    for log_id, tick in sorted(tick_by_log.items()):
        asset_list = asset_by_log.get(log_id)
        if not asset_list:
            continue
        meta = meta_by_log.get(log_id, {})
        feature_row: dict[str, Any] = {
            "log_id": log_id,
            "user_chars": float(tick.get("user_chars", 0)),
            "n_rows": float(tick.get("n_rows", len(asset_list))),
            "seq_len": float(meta.get("seq_len", 0)),
        }
        for metric_name in VISIBLE_METRIC_NAMES:
            values = [float(row[metric_name]) for row in asset_list]
            for aggregate_name, aggregate_value in _aggregate_metric_family(values).items():
                feature_row[f"{metric_name}_{aggregate_name}"] = aggregate_value
        rows.append(feature_row)
    return rows


def _split_feature_table(
    rows: list[dict[str, Any]],
) -> tuple[list[int], list[str], np.ndarray, list[str], np.ndarray]:
    if not rows:
        return [], [], np.zeros((0, 0), dtype=np.float32), [], np.zeros((0, 0), dtype=np.float32)
    log_ids = [int(row["log_id"]) for row in rows]
    nuisance_names = [name for name in rows[0].keys() if name in NUISANCE_FEATURE_NAMES]
    candidate_names = [
        name
        for name in rows[0].keys()
        if name != "log_id" and name not in NUISANCE_FEATURE_NAMES
    ]
    candidate_matrix = np.asarray(
        [[float(row[name]) for name in candidate_names] for row in rows],
        dtype=np.float32,
    )
    nuisance_matrix = np.asarray(
        [[float(row[name]) for name in nuisance_names] for row in rows],
        dtype=np.float32,
    )
    return log_ids, candidate_names, candidate_matrix, nuisance_names, nuisance_matrix


def _drop_degenerate_features(
    X: np.ndarray,
    feature_names: list[str],
    *,
    std_eps: float = 1e-8,
) -> tuple[np.ndarray, list[str], list[str]]:
    if X.size == 0:
        return X, feature_names, []
    keep_indices: list[int] = []
    dropped: list[str] = []
    for idx, name in enumerate(feature_names):
        if float(np.std(X[:, idx])) <= std_eps:
            dropped.append(name)
        else:
            keep_indices.append(idx)
    if not keep_indices:
        return np.zeros((X.shape[0], 0), dtype=np.float32), [], dropped
    kept = X[:, keep_indices].astype(np.float32)
    kept_names = [feature_names[idx] for idx in keep_indices]
    return kept, kept_names, dropped


def _project_target_scores(
    *,
    state_key: str,
    layer: int,
    pc_index: int,
    log_ids: list[int],
    candidate_matrix: np.ndarray,
    nuisance_matrix: np.ndarray,
    activation_cache: dict[int, dict[str, np.ndarray]],
    basis: Any,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    kept_log_ids: list[int] = []
    act_rows: list[np.ndarray] = []
    kept_candidate_rows: list[np.ndarray] = []
    kept_nuisance_rows: list[np.ndarray] = []
    for idx, log_id in enumerate(log_ids):
        acts = activation_cache.get(log_id, {})
        tensor = acts.get(state_key)
        if tensor is None or tensor.ndim != 2 or layer >= tensor.shape[0]:
            continue
        kept_log_ids.append(log_id)
        act_rows.append(tensor[layer].astype(np.float32))
        kept_candidate_rows.append(candidate_matrix[idx])
        kept_nuisance_rows.append(nuisance_matrix[idx])
    if not act_rows:
        return (
            [],
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, candidate_matrix.shape[1]), dtype=np.float32),
            np.zeros((0, nuisance_matrix.shape[1]), dtype=np.float32),
        )

    X = np.stack(act_rows)
    candidate_subset = np.stack(kept_candidate_rows).astype(np.float32)
    nuisance_subset = np.stack(kept_nuisance_rows).astype(np.float32)
    residual = _residualize_activations(X, nuisance_subset)

    prefix = f"{state_key}_layer_{layer}"
    mean = basis[f"{prefix}__mean"]
    scale = basis[f"{prefix}__scale"]
    components = basis[f"{prefix}__components"]
    standardized = (residual - mean) / np.where(scale == 0.0, 1.0, scale)
    scores = standardized @ components.T
    return kept_log_ids, scores[:, pc_index - 1].astype(np.float32), candidate_subset, nuisance_subset


def _cross_validated_predictions(
    features: np.ndarray,
    target: np.ndarray,
    splitter: KFold,
    *,
    nonlinear_pair: bool = False,
) -> np.ndarray:
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if nonlinear_pair:
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    steps.append(("reg", LinearRegression()))
    model = Pipeline(steps)
    return cross_val_predict(model, features, target, cv=splitter).astype(np.float32)


def _cross_validated_r2(features: np.ndarray, target: np.ndarray, splitter: KFold, *, nonlinear_pair: bool = False) -> float:
    pred = _cross_validated_predictions(features, target, splitter, nonlinear_pair=nonlinear_pair)
    return float(r2_score(target, pred))


def _cross_validated_ridge_r2(features: np.ndarray, target: np.ndarray, splitter: KFold) -> float:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
    ])
    pred = cross_val_predict(model, features, target, cv=splitter).astype(np.float32)
    return float(r2_score(target, pred))


def _fit_ridge(features: np.ndarray, target: np.ndarray) -> Pipeline:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
    ])
    model.fit(features, target)
    return model


def _fit_elastic_net(features: np.ndarray, target: np.ndarray) -> Pipeline:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("enet", ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            alphas=np.logspace(-3, 1, 20),
            max_iter=50000,
            cv=5,
        )),
    ])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(features, target)
    return model


def _single_feature_rankings(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    splitter: KFold,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, feature_name in enumerate(feature_names):
        feature = X[:, idx]
        spearman = _spearman(feature, y)
        cv_r2 = _cross_validated_r2(feature[:, None], y, splitter)
        rows.append({
            "feature": feature_name,
            "spearman": None if spearman is None else float(spearman),
            "abs_spearman": 0.0 if spearman is None else abs(float(spearman)),
            "cv_r2": float(cv_r2),
        })
    rows.sort(key=lambda row: (-row["cv_r2"], -row["abs_spearman"], row["feature"]))
    return rows


def _best_pair(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    splitter: KFold,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for idx_a, idx_b in itertools.combinations(range(X.shape[1]), 2):
        pair_X = X[:, [idx_a, idx_b]]
        cv_r2 = _cross_validated_r2(pair_X, y, splitter)
        row = {
            "features": [feature_names[idx_a], feature_names[idx_b]],
            "cv_r2": float(cv_r2),
        }
        if best is None or row["cv_r2"] > best["cv_r2"]:
            best = row
    return best or {"features": [], "cv_r2": float("nan")}


def _best_pair_quadratic(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    splitter: KFold,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for idx_a, idx_b in itertools.combinations(range(X.shape[1]), 2):
        pair_X = X[:, [idx_a, idx_b]]
        cv_r2 = _cross_validated_r2(pair_X, y, splitter, nonlinear_pair=True)
        row = {
            "features": [feature_names[idx_a], feature_names[idx_b]],
            "cv_r2": float(cv_r2),
        }
        if best is None or row["cv_r2"] > best["cv_r2"]:
            best = row
    return best or {"features": [], "cv_r2": float("nan")}


def _coef_rows(coefs: np.ndarray, feature_names: list[str]) -> list[dict[str, Any]]:
    rows = [
        {
            "feature": feature_name,
            "coefficient": float(coef),
            "abs_coefficient": abs(float(coef)),
        }
        for feature_name, coef in zip(feature_names, coefs, strict=True)
        if abs(float(coef)) > 1e-8
    ]
    rows.sort(key=lambda row: (-row["abs_coefficient"], row["feature"]))
    return rows


def _shuffle_sanity(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    splitter: KFold,
    *,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(y)
    rankings = _single_feature_rankings(X, y_shuffled, feature_names, splitter)
    best_pair = _best_pair(X, y_shuffled, feature_names, splitter)
    best_pair_quadratic = _best_pair_quadratic(X, y_shuffled, feature_names, splitter)
    return {
        "best_single_feature": rankings[0] if rankings else None,
        "best_pair_linear": best_pair,
        "best_pair_quadratic": best_pair_quadratic,
    }


def _single_feature_fold_winners(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    splitter: KFold,
) -> list[dict[str, Any]]:
    winners: list[dict[str, Any]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        fold_rows: list[dict[str, Any]] = []
        y_train = y[train_idx]
        y_test = y[test_idx]
        for feature_idx, feature_name in enumerate(feature_names):
            reg = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", LinearRegression()),
            ])
            reg.fit(X[train_idx, feature_idx : feature_idx + 1], y_train)
            pred = reg.predict(X[test_idx, feature_idx : feature_idx + 1])
            fold_rows.append({
                "feature": feature_name,
                "r2": float(r2_score(y_test, pred)),
            })
        fold_rows.sort(key=lambda row: (-row["r2"], row["feature"]))
        winners.append({
            "fold": fold_idx,
            "winner": fold_rows[0]["feature"],
            "winner_r2": fold_rows[0]["r2"],
            "runner_up": fold_rows[1]["feature"] if len(fold_rows) > 1 else None,
            "runner_up_r2": fold_rows[1]["r2"] if len(fold_rows) > 1 else None,
        })
    return winners


def _parse_feature_parts(feature_name: str) -> tuple[str, str]:
    for suffix in AGGREGATE_SUFFIXES:
        token = f"_{suffix}"
        if feature_name.endswith(token):
            return feature_name[: -len(token)], suffix
    return feature_name, "raw"


def _summarize_feature_families(rows: list[dict[str, Any]], *, key: str) -> dict[str, list[dict[str, Any]]]:
    family_scores: defaultdict[str, float] = defaultdict(float)
    aggregate_scores: defaultdict[str, float] = defaultdict(float)
    for row in rows:
        feature = str(row["feature"])
        score = abs(float(row[key]))
        family, aggregate = _parse_feature_parts(feature)
        family_scores[family] += score
        aggregate_scores[aggregate] += score
    family_rows = [
        {"family": family, "score": float(score)}
        for family, score in family_scores.items()
    ]
    aggregate_rows = [
        {"aggregate": aggregate, "score": float(score)}
        for aggregate, score in aggregate_scores.items()
    ]
    family_rows.sort(key=lambda row: (-row["score"], row["family"]))
    aggregate_rows.sort(key=lambda row: (-row["score"], row["aggregate"]))
    return {
        "metric_families": family_rows,
        "aggregate_types": aggregate_rows,
    }


def _target_nuisance_correlations(
    y: np.ndarray,
    nuisance_names: list[str],
    nuisance_matrix: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(nuisance_names):
        corr = _spearman(y, nuisance_matrix[:, idx])
        rows.append({
            "feature": name,
            "spearman": None if corr is None else float(corr),
            "abs_spearman": 0.0 if corr is None else abs(float(corr)),
        })
    rows.sort(key=lambda row: (-row["abs_spearman"], row["feature"]))
    return rows


def _top_feature_redundancy(
    X: np.ndarray,
    feature_names: list[str],
    rankings: list[dict[str, Any]],
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    top_features = [row["feature"] for row in rankings[:top_k]]
    indices = [feature_names.index(name) for name in top_features]
    rows: list[dict[str, Any]] = []
    for idx_a, idx_b in itertools.combinations(indices, 2):
        corr = _spearman(X[:, idx_a], X[:, idx_b])
        rows.append({
            "features": [feature_names[idx_a], feature_names[idx_b]],
            "spearman": None if corr is None else float(corr),
            "abs_spearman": 0.0 if corr is None else abs(float(corr)),
        })
    rows.sort(key=lambda row: (-row["abs_spearman"], row["features"]))
    return rows


def _group_ridge_rankings(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    splitter: KFold,
    *,
    group_mode: str,
) -> list[dict[str, Any]]:
    grouped_indices: defaultdict[str, list[int]] = defaultdict(list)
    for idx, feature_name in enumerate(feature_names):
        family, aggregate = _parse_feature_parts(feature_name)
        key = family if group_mode == "family" else aggregate
        grouped_indices[key].append(idx)
    rows: list[dict[str, Any]] = []
    for key, indices in grouped_indices.items():
        group_X = X[:, indices]
        cv_r2 = _cross_validated_ridge_r2(group_X, y, splitter)
        rows.append({
            "group": key,
            "num_features": len(indices),
            "cv_r2": float(cv_r2),
        })
    rows.sort(key=lambda row: (-row["cv_r2"], row["group"]))
    return rows


@dataclass
class TargetAxis:
    name: str
    state_key: str
    layer: int
    pc_index: int


@dataclass
class SyntheticMarketAxisDecompositionConfig:
    structure_dir: Path = Path("data/activations/synthetic_structure/phase15_market_basis_discovery_v1")
    output_dir: Path = Path("data/analysis_results/synthetic_market_axis_decomposition/phase15_market_basis_discovery_v1/prompt_visible_v2")
    phase_name: str = "phase15_market_basis_discovery_v1"
    basis_npz_path: Path = Path(
        "data/analysis_results/synthetic_market_discovery/phase15_market_basis_discovery_v1/residualized_nuisance_v1/pca_basis.npz"
    )
    basis_results_path: Path = Path(
        "data/analysis_results/synthetic_market_discovery/phase15_market_basis_discovery_v1/residualized_nuisance_v1/results.json"
    )
    context_variant: str = "market_only"
    family_allowlist: tuple[str, ...] = ("market_basis_scalar", "market_basis_coupled")
    num_workers: int = 8
    seed: int = 42
    targets: tuple[TargetAxis, ...] = (
        TargetAxis(name="leader_axis", state_key="market_mean", layer=4, pc_index=1),
        TargetAxis(name="dispersion_axis", state_key="market_mean", layer=35, pc_index=1),
    )


def run_synthetic_market_axis_decomposition_analysis(
    config: SyntheticMarketAxisDecompositionConfig,
) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows, tick_rows, asset_rows = _load_structure_tables(config.structure_dir)
    prompt_rows = _build_visible_prompt_features(
        tick_rows,
        asset_rows,
        metadata_rows,
        context_variant=config.context_variant,
        family_allowlist=config.family_allowlist,
    )
    if not prompt_rows:
        result = {"error": "no_prompt_rows"}
        (config.output_dir / "results.json").write_text(json.dumps(result, indent=2))
        return result

    (
        log_ids,
        feature_names,
        feature_matrix,
        nuisance_names,
        nuisance_matrix,
    ) = _split_feature_table(prompt_rows)
    feature_matrix, feature_names, dropped_features = _drop_degenerate_features(feature_matrix, feature_names)

    activation_cache = _preload_pooled_residuals(config.structure_dir, log_ids, max_workers=config.num_workers)
    basis = np.load(config.basis_npz_path)
    basis_results = json.loads(config.basis_results_path.read_text())
    splitter = KFold(n_splits=5, shuffle=True, random_state=config.seed)

    results: dict[str, Any] = {
        "phase_name": config.phase_name,
        "context_variant": config.context_variant,
        "n_prompts": len(log_ids),
        "visible_feature_names": feature_names,
        "nuisance_feature_names": nuisance_names,
        "dropped_degenerate_features": dropped_features,
        "targets": {},
    }

    for target in config.targets:
        kept_log_ids, y, X, nuisance_subset = _project_target_scores(
            state_key=target.state_key,
            layer=target.layer,
            pc_index=target.pc_index,
            log_ids=log_ids,
            candidate_matrix=feature_matrix,
            nuisance_matrix=nuisance_matrix,
            activation_cache=activation_cache,
            basis=basis,
        )
        if y.size == 0:
            continue

        rankings = _single_feature_rankings(X, y, feature_names, splitter)
        best_pair = _best_pair(X, y, feature_names, splitter)
        best_pair_quadratic = _best_pair_quadratic(X, y, feature_names, splitter)
        fold_winners = _single_feature_fold_winners(X, y, feature_names, splitter)
        family_group_rankings = _group_ridge_rankings(X, y, feature_names, splitter, group_mode="family")
        aggregate_group_rankings = _group_ridge_rankings(X, y, feature_names, splitter, group_mode="aggregate")
        ridge = _fit_ridge(X, y)
        ridge_pred = ridge.predict(X)
        enet = _fit_elastic_net(X, y)
        enet_pred = enet.predict(X)
        shuffle = _shuffle_sanity(X, y, feature_names, splitter, seed=config.seed + target.layer + target.pc_index)

        basis_layer = basis_results["states"][target.state_key][str(target.layer)]["pcs"][target.pc_index - 1]
        results["targets"][target.name] = {
            "state_key": target.state_key,
            "layer": target.layer,
            "pc_index": target.pc_index,
            "n_examples": int(len(kept_log_ids)),
            "phase15_top_market_correlations": basis_layer["top_market_correlations"],
            "target_nuisance_correlations": _target_nuisance_correlations(y, nuisance_names, nuisance_subset),
            "best_single_feature": rankings[0],
            "all_single_features": rankings,
            "top_single_features": rankings[:15],
            "best_pair_linear": best_pair,
            "best_pair_quadratic": best_pair_quadratic,
            "single_feature_fold_winners": fold_winners,
            "single_feature_winner_counts": Counter(row["winner"] for row in fold_winners),
            "metric_family_group_ridge_cv_r2": family_group_rankings,
            "aggregate_group_ridge_cv_r2": aggregate_group_rankings,
            "top_feature_redundancy": _top_feature_redundancy(X, feature_names, rankings),
            "shuffle_sanity": shuffle,
            "ridge_in_sample_r2": float(r2_score(y, ridge_pred)),
            "ridge_coefficients": _coef_rows(ridge.named_steps["ridge"].coef_, feature_names)[:20],
            "ridge_family_summary": _summarize_feature_families(
                _coef_rows(ridge.named_steps["ridge"].coef_, feature_names),
                key="coefficient",
            ),
            "elastic_net_in_sample_r2": float(r2_score(y, enet_pred)),
            "elastic_net_alpha": float(enet.named_steps["enet"].alpha_),
            "elastic_net_l1_ratio": float(enet.named_steps["enet"].l1_ratio_),
            "elastic_net_coefficients": _coef_rows(enet.named_steps["enet"].coef_, feature_names)[:20],
            "elastic_net_family_summary": _summarize_feature_families(
                _coef_rows(enet.named_steps["enet"].coef_, feature_names),
                key="coefficient",
            ),
        }

    (config.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decompose Phase 15 leader/dispersion PCs into prompt-derived market formulas.")
    parser.add_argument("--structure-dir", type=Path, default=Path("data/activations/synthetic_structure/phase15_market_basis_discovery_v1"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis_results/synthetic_market_axis_decomposition/phase15_market_basis_discovery_v1/prompt_visible_v2"),
    )
    parser.add_argument("--phase-name", default="phase15_market_basis_discovery_v1")
    parser.add_argument("--context-variant", default="market_only")
    parser.add_argument("--num-workers", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result = run_synthetic_market_axis_decomposition_analysis(
        SyntheticMarketAxisDecompositionConfig(
            structure_dir=args.structure_dir,
            output_dir=args.output_dir,
            phase_name=args.phase_name,
            context_variant=args.context_variant,
            num_workers=args.num_workers,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
