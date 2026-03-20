"""First-pass manifold analysis for the synthetic market phase-1 dataset.

This module focuses on the market-only captures from the synthetic phase-1
experiment and asks three concrete questions:

1. Are clean latent asset scores linearly decodable from row activations?
2. Are pairwise preferences linearly decodable from row-difference vectors?
3. Do scalar sweep families exhibit low-dimensional ordered geometry?
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from pipelines.db import connect_neon
from pipelines.interp.counterfactual.analysis import (
    evaluate_probe_per_snapshot,
    train_probe,
)


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if y_true.size < 2:
        return None
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return None
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if y_true.size < 2:
        return None
    from scipy.stats import spearmanr

    corr = spearmanr(y_true, y_pred).correlation
    if corr is None or np.isnan(corr):
        return None
    return float(corr)


def _split_ids(ids: list[int], *, seed: int, test_fraction: float) -> tuple[set[int], set[int]]:
    ordered = list(ids)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_test = max(1, int(len(ordered) * test_fraction))
    return set(ordered[n_test:]), set(ordered[:n_test])


def _load_pooled_residual(structure_dir: Path, log_id: int) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file

    path = structure_dir / "residual" / f"{log_id}.safetensors"
    if not path.exists():
        return {}
    return load_file(str(path))


def _preload_pooled_residuals(
    structure_dir: Path,
    log_ids: list[int],
    *,
    max_workers: int,
) -> dict[int, dict[str, np.ndarray]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from safetensors.numpy import load_file

    residual_dir = structure_dir / "residual"

    def _load_one(log_id: int) -> tuple[int, dict[str, np.ndarray] | None]:
        path = residual_dir / f"{log_id}.safetensors"
        if not path.exists():
            return log_id, None
        return log_id, load_file(str(path))

    cache: dict[int, dict[str, np.ndarray]] = {}
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {pool.submit(_load_one, log_id): log_id for log_id in log_ids}
        loaded = 0
        total = len(futures)
        for fut in as_completed(futures):
            log_id, data = fut.result()
            loaded += 1
            if data is not None:
                cache[log_id] = data
            if loaded % 50 == 0 or loaded == total:
                print(f"Preloaded synthetic pooled residuals: {loaded}/{total}", flush=True)
    return cache


def _train_regression_probe(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    probe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    probe.fit(X_train, y_train)
    return probe


def _evaluate_regression_probe(probe: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import mean_squared_error, r2_score

    pred = probe.predict(X_test)
    return {
        "r2": float(r2_score(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "pearson": _pearson(y_test, pred),
        "spearman": _spearman(y_test, pred),
        "n_rows": int(len(y_test)),
    }


@dataclass
class SyntheticManifoldAnalysisConfig:
    structure_dir: Path = Path("data/activations/synthetic_structure/phase1")
    output_dir: Path = Path("data/analysis_results/synthetic_manifold/phase1")
    phase_name: str = "phase1"
    context_variant: str = "market_only"
    row_keys: tuple[str, ...] = ("row_mean", "row_eos")
    layers: list[int] | None = None
    seed: int = 42
    test_fraction: float = 0.2
    num_workers: int = 8
    regression_targets: tuple[str, ...] = (
        "attractiveness_score",
        "risk_adjusted_score",
        "edge_after_fee_score",
    )
    pairwise_targets: tuple[str, ...] = (
        "a_beats_b_on_attractiveness",
        "a_beats_b_on_risk_adjusted",
    )
    scalar_families: tuple[str, ...] = (
        "pct_5m",
        "net_flow_5m",
        "top20_holder_pct",
    )


def _load_structure_tables(structure_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_meta_rows = pq.read_table(structure_dir / "metadata.parquet").to_pylist()
    raw_tick_rows = pq.read_table(structure_dir / "tick_labels.parquet").to_pylist()
    raw_asset_rows = pq.read_table(structure_dir / "asset_labels.parquet").to_pylist()

    meta_rows: list[dict[str, Any]] = []
    seen_meta: set[int] = set()
    for row in raw_meta_rows:
        log_id = int(row["log_id"])
        if log_id in seen_meta:
            continue
        seen_meta.add(log_id)
        meta_rows.append(row)

    tick_rows: list[dict[str, Any]] = []
    seen_tick: set[int] = set()
    for row in raw_tick_rows:
        log_id = int(row["log_id"])
        if log_id in seen_tick:
            continue
        seen_tick.add(log_id)
        tick_rows.append(row)

    asset_rows: list[dict[str, Any]] = []
    seen_asset: set[tuple[int, int, str]] = set()
    for row in raw_asset_rows:
        key = (int(row["log_id"]), int(row["row_index"]), str(row["symbol"]))
        if key in seen_asset:
            continue
        seen_asset.add(key)
        asset_rows.append(row)

    return meta_rows, tick_rows, asset_rows


def _group_asset_rows(asset_rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in asset_rows:
        grouped.setdefault(int(row["log_id"]), []).append(dict(row))
    for log_id in grouped:
        grouped[log_id].sort(key=lambda item: int(item["row_index"]))
    return grouped


def _load_pairwise_rows(
    *,
    phase_name: str,
    context_variant: str,
    log_ids: list[int],
) -> list[dict[str, Any]]:
    if not log_ids:
        return []
    conn = connect_neon()
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM synthetic_market_pairs_v0
            WHERE phase_name = %s
              AND context_variant = %s
              AND log_id = ANY(%s)
            ORDER BY log_id, asset_a, asset_b
            """,
            (phase_name, context_variant, log_ids),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _collect_regression_rows(
    *,
    log_ids: set[int],
    asset_by_log: dict[int, list[dict[str, Any]]],
    activation_cache: dict[int, dict[str, np.ndarray]],
    row_key: str,
    layer: int,
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    X_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    for log_id in sorted(log_ids):
        acts = activation_cache.get(log_id)
        rows = asset_by_log.get(log_id)
        if not acts or not rows:
            continue
        for row in rows:
            key = f"{row_key}_{int(row['row_index'])}"
            if key not in acts or row.get(target) is None:
                continue
            X_rows.append(acts[key][layer].astype(np.float32))
            y_rows.append(float(row[target]))
    if not X_rows:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.stack(X_rows), np.asarray(y_rows, dtype=np.float32)


def _collect_best_asset_groups(
    *,
    log_ids: set[int],
    asset_by_log: dict[int, list[dict[str, Any]]],
    activation_cache: dict[int, dict[str, np.ndarray]],
    row_key: str,
    layer: int,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for log_id in sorted(log_ids):
        acts = activation_cache.get(log_id)
        rows = asset_by_log.get(log_id)
        if not acts or not rows or not any(int(row.get("is_best_asset", 0)) for row in rows):
            continue
        X_rows: list[np.ndarray] = []
        y_rows: list[int] = []
        for row in rows:
            key = f"{row_key}_{int(row['row_index'])}"
            if key not in acts:
                X_rows = []
                break
            X_rows.append(acts[key][layer].astype(np.float32))
            y_rows.append(int(row.get("is_best_asset", 0)))
        if not X_rows or len(set(y_rows)) < 2:
            continue
        groups.append({
            "X": np.stack(X_rows),
            "y": np.asarray(y_rows, dtype=np.int64),
            "snapshot_id": str(log_id),
        })
    return groups


def _collect_pairwise_examples(
    *,
    rows: list[dict[str, Any]],
    pairwise_rows: list[dict[str, Any]],
    activation_cache: dict[int, dict[str, np.ndarray]],
    row_key: str,
    layer: int,
    target: str,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    row_index_by_log_symbol: dict[tuple[int, str], int] = {}
    for row in rows:
        row_index_by_log_symbol[(int(row["log_id"]), str(row["symbol"]))] = int(row["row_index"])

    X: list[np.ndarray] = []
    y: list[int] = []
    families: list[str] = []
    for pair in pairwise_rows:
        log_id = int(pair["log_id"])
        row_a = row_index_by_log_symbol.get((log_id, str(pair["asset_a"])))
        row_b = row_index_by_log_symbol.get((log_id, str(pair["asset_b"])))
        acts = activation_cache.get(log_id)
        if row_a is None or row_b is None or not acts:
            continue
        key_a = f"{row_key}_{row_a}"
        key_b = f"{row_key}_{row_b}"
        if key_a not in acts or key_b not in acts:
            continue
        vec_a = acts[key_a][layer].astype(np.float32)
        vec_b = acts[key_b][layer].astype(np.float32)
        if feature_mode == "diff":
            feature = vec_a - vec_b
        else:
            feature = np.concatenate([vec_a, vec_b])
        X.append(feature)
        y.append(int(pair[target]))
        families.append(str(pair["family_variant"]))
    if not X:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64), []
    return np.stack(X), np.asarray(y, dtype=np.int64), families


def _evaluate_pairwise_probe(
    *,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    train_pairwise: list[dict[str, Any]],
    test_pairwise: list[dict[str, Any]],
    activation_cache: dict[int, dict[str, np.ndarray]],
    row_key: str,
    layer: int,
    target: str,
    feature_mode: str,
    seed: int,
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

    X_train, y_train, _ = _collect_pairwise_examples(
        rows=train_rows,
        pairwise_rows=train_pairwise,
        activation_cache=activation_cache,
        row_key=row_key,
        layer=layer,
        target=target,
        feature_mode=feature_mode,
    )
    X_test, y_test, families = _collect_pairwise_examples(
        rows=test_rows,
        pairwise_rows=test_pairwise,
        activation_cache=activation_cache,
        row_key=row_key,
        layer=layer,
        target=target,
        feature_mode=feature_mode,
    )
    if X_train.size == 0 or X_test.size == 0 or len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"error": "insufficient_data"}

    probe = train_probe(X_train, y_train, seed=seed)
    pred = probe.predict(X_test)
    prob = probe.predict_proba(X_test)[:, 1]

    by_family: dict[str, list[tuple[int, int]]] = {}
    for family, label, pred_label in zip(families, y_test, pred, strict=True):
        by_family.setdefault(family, []).append((int(label), int(pred_label)))

    family_metrics = {}
    for family, pairs in by_family.items():
        labels = np.asarray([p[0] for p in pairs], dtype=np.int64)
        preds = np.asarray([p[1] for p in pairs], dtype=np.int64)
        family_metrics[family] = {
            "accuracy": float(accuracy_score(labels, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
            "n": int(len(labels)),
        }

    return {
        "auroc": float(roc_auc_score(y_test, prob)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "n_rows": int(len(y_test)),
        "feature_mode": feature_mode,
        "by_family_variant": family_metrics,
    }


def _scalar_geometry_metrics(X: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA

    if X.shape[0] < 3:
        return {"error": "insufficient_points"}

    pca = PCA(n_components=min(5, X.shape[0], X.shape[1]))
    coords = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_
    eigvals = pca.explained_variance_
    participation_ratio = float((eigvals.sum() ** 2) / np.square(eigvals).sum()) if eigvals.size else None

    value_dists = pdist(values[:, None], metric="euclidean")
    act_dists = pdist(X, metric="euclidean")
    dist_corr = spearmanr(value_dists, act_dists).correlation

    pc1 = coords[:, 0]
    pc1_corr = _spearman(values, pc1)
    if pc1_corr is None:
        pc1_spearman = None
    else:
        pc1_spearman = float(abs(pc1_corr))

    return {
        "n_points": int(X.shape[0]),
        "explained_variance_ratio": [float(x) for x in explained[:5]],
        "participation_ratio": participation_ratio,
        "pc1_value_spearman": pc1_spearman,
        "distance_value_spearman": None if dist_corr is None or np.isnan(dist_corr) else float(dist_corr),
    }


def _summarize_regression_results(results: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for target, reps in results.items():
        best: tuple[str, float, int] | None = None
        for row_key, per_layer in reps.items():
            for layer_result in per_layer:
                score = layer_result.get("r2")
                if score is None:
                    continue
                if best is None or float(score) > best[1]:
                    best = (row_key, float(score), int(layer_result["layer"]))
        summary[target] = None if best is None else {
            "representation": best[0],
            "r2": best[1],
            "layer": best[2],
        }
    return summary


def _summarize_best_asset_results(results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    best: tuple[str, float, int] | None = None
    for row_key, per_layer in results.items():
        for layer_result in per_layer:
            score = layer_result.get("auroc")
            if score is None:
                continue
            if best is None or float(score) > best[1]:
                best = (row_key, float(score), int(layer_result["layer"]))
    return None if best is None else {
        "representation": best[0],
        "auroc": best[1],
        "layer": best[2],
    }


def _summarize_pairwise_results(results: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for target, modes in results.items():
        best: tuple[str, float, int] | None = None
        for feature_mode, per_row_key in modes.items():
            for row_key, per_layer in per_row_key.items():
                for layer_result in per_layer:
                    score = layer_result.get("auroc")
                    if score is None:
                        continue
                    if best is None or float(score) > best[1]:
                        best = (f"{feature_mode}:{row_key}", float(score), int(layer_result["layer"]))
        summary[target] = None if best is None else {
            "representation": best[0],
            "auroc": best[1],
            "layer": best[2],
        }
    return summary


def _summarize_scalar_geometry(results: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for family, reps in results.items():
        best: tuple[str, float, int] | None = None
        for row_key, per_layer in reps.items():
            for layer_result in per_layer:
                score = layer_result.get("distance_value_spearman")
                if score is None:
                    continue
                if best is None or float(score) > best[1]:
                    best = (row_key, float(score), int(layer_result["layer"]))
        summary[family] = None if best is None else {
            "representation": best[0],
            "distance_value_spearman": best[1],
            "layer": best[2],
        }
    return summary


def run_synthetic_manifold_analysis(config: SyntheticManifoldAnalysisConfig) -> dict[str, Any]:
    meta_rows, tick_rows, asset_rows = _load_structure_tables(config.structure_dir)
    if not meta_rows:
        return {"error": "no_synthetic_structure_metadata"}

    market_tick_rows = [
        row for row in tick_rows
        if str(row.get("context_variant")) == config.context_variant
    ]
    market_log_ids = sorted({int(row["log_id"]) for row in market_tick_rows})
    if not market_log_ids:
        return {"error": f"no_ticks_for_context_{config.context_variant}"}

    asset_rows = [
        row for row in asset_rows
        if int(row["log_id"]) in set(market_log_ids)
        and str(row.get("context_variant")) == config.context_variant
    ]
    asset_by_log = _group_asset_rows(asset_rows)
    print(
        f"Loaded synthetic structure labels: {len(meta_rows)} metadata rows, "
        f"{len(market_tick_rows)} {config.context_variant} ticks, "
        f"{len(asset_rows)} asset rows",
        flush=True,
    )

    activation_cache = _preload_pooled_residuals(
        config.structure_dir,
        market_log_ids,
        max_workers=config.num_workers,
    )
    if not activation_cache:
        return {"error": "no_synthetic_pooled_residuals"}

    sample_acts = next(iter(activation_cache.values()))
    if "last_token" not in sample_acts:
        return {"error": "missing_last_token_key"}
    layers = config.layers or list(range(int(sample_acts["last_token"].shape[0])))
    train_ids, test_ids = _split_ids(
        market_log_ids,
        seed=config.seed,
        test_fraction=config.test_fraction,
    )
    print(
        f"Synthetic split: train={len(train_ids)} test={len(test_ids)}; "
        f"layers={layers[0]}..{layers[-1]}",
        flush=True,
    )

    analysis: dict[str, Any] = {
        "phase_name": config.phase_name,
        "context_variant": config.context_variant,
        "n_market_ticks": len(market_log_ids),
        "layers": layers,
        "row_keys": list(config.row_keys),
        "regression": {},
        "best_asset": {},
        "pairwise": {},
        "scalar_geometry": {},
    }

    train_asset_rows = [row for row in asset_rows if int(row["log_id"]) in train_ids]
    test_asset_rows = [row for row in asset_rows if int(row["log_id"]) in test_ids]
    pairwise_rows = _load_pairwise_rows(
        phase_name=config.phase_name,
        context_variant=config.context_variant,
        log_ids=market_log_ids,
    )
    train_pairwise = [row for row in pairwise_rows if int(row["log_id"]) in train_ids and str(row["family"]) == "pairwise_tradeoff"]
    test_pairwise = [row for row in pairwise_rows if int(row["log_id"]) in test_ids and str(row["family"]) == "pairwise_tradeoff"]

    for target in config.regression_targets:
        print(f"=== Synthetic regression target: {target} ===", flush=True)
        analysis["regression"][target] = {}
        for row_key in config.row_keys:
            per_layer: list[dict[str, Any]] = []
            for layer in layers:
                X_train, y_train = _collect_regression_rows(
                    log_ids=train_ids,
                    asset_by_log=asset_by_log,
                    activation_cache=activation_cache,
                    row_key=row_key,
                    layer=layer,
                    target=target,
                )
                X_test, y_test = _collect_regression_rows(
                    log_ids=test_ids,
                    asset_by_log=asset_by_log,
                    activation_cache=activation_cache,
                    row_key=row_key,
                    layer=layer,
                    target=target,
                )
                if X_train.size == 0 or X_test.size == 0:
                    per_layer.append({"layer": layer, "error": "insufficient_data"})
                    continue
                probe = _train_regression_probe(X_train, y_train)
                metrics = _evaluate_regression_probe(probe, X_test, y_test)
                metrics["layer"] = layer
                per_layer.append(metrics)
            analysis["regression"][target][row_key] = per_layer

    print("=== Synthetic best-asset probes ===", flush=True)
    for row_key in config.row_keys:
        per_layer: list[dict[str, Any]] = []
        for layer in layers:
            train_groups = _collect_best_asset_groups(
                log_ids=train_ids,
                asset_by_log=asset_by_log,
                activation_cache=activation_cache,
                row_key=row_key,
                layer=layer,
            )
            test_groups = _collect_best_asset_groups(
                log_ids=test_ids,
                asset_by_log=asset_by_log,
                activation_cache=activation_cache,
                row_key=row_key,
                layer=layer,
            )
            if not train_groups or not test_groups:
                per_layer.append({"layer": layer, "error": "insufficient_data"})
                continue
            X_train = np.concatenate([group["X"] for group in train_groups])
            y_train = np.concatenate([group["y"] for group in train_groups])
            probe = train_probe(X_train, y_train, seed=config.seed)
            metrics = evaluate_probe_per_snapshot(probe, test_groups)
            per_layer.append({
                "layer": layer,
                "auroc": _mean(metrics["auroc"]),
                "hit_at_1": _mean(metrics["hit_at_1"]),
                "mrr": _mean(metrics["mrr"]),
                "balanced_accuracy": _mean(metrics["balanced_accuracy"]),
                "n_groups": len(test_groups),
            })
        analysis["best_asset"][row_key] = per_layer

    for target in config.pairwise_targets:
        print(f"=== Synthetic pairwise target: {target} ===", flush=True)
        analysis["pairwise"][target] = {}
        for feature_mode in ("diff", "concat"):
            analysis["pairwise"][target][feature_mode] = {}
            for row_key in config.row_keys:
                per_layer: list[dict[str, Any]] = []
                for layer in layers:
                    metrics = _evaluate_pairwise_probe(
                        train_rows=train_asset_rows,
                        test_rows=test_asset_rows,
                        train_pairwise=train_pairwise,
                        test_pairwise=test_pairwise,
                        activation_cache=activation_cache,
                        row_key=row_key,
                        layer=layer,
                        target=target,
                        feature_mode=feature_mode,
                        seed=config.seed,
                    )
                    metrics["layer"] = layer
                    per_layer.append(metrics)
                analysis["pairwise"][target][feature_mode][row_key] = per_layer

    print("=== Synthetic scalar geometry ===", flush=True)
    scalar_rows = [
        row for row in asset_rows
        if str(row.get("family")) == "scalar_sweep" and int(row.get("row_index", -1)) == 0
    ]
    for family in config.scalar_families:
        family_rows = sorted(
            [row for row in scalar_rows if str(row.get("family_variant")) == family],
            key=lambda row: float(row[family]),
        )
        analysis["scalar_geometry"][family] = {}
        log_ids = [int(row["log_id"]) for row in family_rows]
        values = np.asarray([float(row[family]) for row in family_rows], dtype=np.float32)
        for row_key in config.row_keys:
            per_layer: list[dict[str, Any]] = []
            for layer in layers:
                X_rows: list[np.ndarray] = []
                for row in family_rows:
                    acts = activation_cache.get(int(row["log_id"]))
                    key = f"{row_key}_{int(row['row_index'])}"
                    if not acts or key not in acts:
                        X_rows = []
                        break
                    X_rows.append(acts[key][layer].astype(np.float32))
                if not X_rows:
                    per_layer.append({"layer": layer, "error": "missing_activations"})
                    continue
                metrics = _scalar_geometry_metrics(np.stack(X_rows), values)
                metrics["layer"] = layer
                metrics["log_ids"] = log_ids
                per_layer.append(metrics)
            analysis["scalar_geometry"][family][row_key] = per_layer

    analysis["summary"] = {
        "regression": _summarize_regression_results(analysis["regression"]),
        "best_asset": _summarize_best_asset_results(analysis["best_asset"]),
        "pairwise": _summarize_pairwise_results(analysis["pairwise"]),
        "scalar_geometry": _summarize_scalar_geometry(analysis["scalar_geometry"]),
    }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / "results.json"
    output_path.write_text(json.dumps(analysis, indent=2))
    print(f"Wrote synthetic manifold analysis to {output_path}", flush=True)
    return analysis


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic market manifold analysis")
    parser.add_argument("--structure-dir", type=Path, default=Path("data/activations/synthetic_structure/phase1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis_results/synthetic_manifold/phase1"))
    parser.add_argument("--phase-name", default="phase1")
    parser.add_argument("--context-variant", default="market_only")
    parser.add_argument("--layers", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    layers = [int(token) for token in args.layers.split(",") if token.strip()] or None
    config = SyntheticManifoldAnalysisConfig(
        structure_dir=args.structure_dir,
        output_dir=args.output_dir,
        phase_name=args.phase_name,
        context_variant=args.context_variant,
        layers=layers,
        seed=args.seed,
        test_fraction=args.test_fraction,
        num_workers=args.num_workers,
    )
    print(json.dumps(run_synthetic_manifold_analysis(config), indent=2))


if __name__ == "__main__":
    main()
