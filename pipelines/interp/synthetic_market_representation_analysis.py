"""Representation-focused analysis for the harder synthetic market phase.

This phase is intentionally upstream of end-state behavior. The main questions are:

1. Which primitive market variables are linearly decodable from row states?
2. Are hard A-vs-B pairwise tradeoffs represented in row differences?
3. For fixed focal assets under changing backgrounds, does representation track
   asset identity or roster-relative rank more strongly?
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pipelines.db import connect_neon
from pipelines.interp.counterfactual.analysis import train_probe
from pipelines.interp.synthetic_manifold_analysis import (
    _evaluate_regression_probe,
    _load_structure_tables,
    _mean,
    _preload_pooled_residuals,
    _split_ids,
    _train_regression_probe,
)


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return None
    return float(np.dot(vec_a, vec_b) / denom)


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


@dataclass
class SyntheticMarketRepresentationConfig:
    structure_dir: Path = Path("data/activations/synthetic_structure/phase4_market_representation_v1")
    output_dir: Path = Path("data/analysis_results/synthetic_market_representation/phase4_market_representation_v1")
    phase_name: str = "phase4_market_representation_v1"
    context_variant: str = "market_only"
    row_keys: tuple[str, ...] = ("row_mean", "row_eos")
    layers: list[int] | None = None
    seed: int = 42
    test_fraction: float = 0.2
    num_workers: int = 8
    family_allowlist: tuple[str, ...] = (
        "pairwise_tradeoff_hard",
        "rank_context_tradeoff",
        "symbol_permutation_control",
    )
    regression_targets: tuple[str, ...] = (
        "pct_5m",
        "net_flow_5m",
        "unique_traders_5m",
        "top20_holder_pct",
        "attractiveness_score",
        "risk_adjusted_score",
    )


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


def _collect_focal_pairwise_examples(
    *,
    pairwise_rows: list[dict[str, Any]],
    activation_cache: dict[int, dict[str, np.ndarray]],
    row_key: str,
    layer: int,
    target: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X: list[np.ndarray] = []
    y: list[int] = []
    scenario_variants: list[str] = []
    for pair in pairwise_rows:
        if str(pair["asset_a"]) != "A" or str(pair["asset_b"]) != "B":
            continue
        log_id = int(pair["log_id"])
        acts = activation_cache.get(log_id)
        if not acts:
            continue
        key_a = f"{row_key}_0"
        key_b = f"{row_key}_1"
        if key_a not in acts or key_b not in acts:
            continue
        X.append(acts[key_a][layer].astype(np.float32) - acts[key_b][layer].astype(np.float32))
        y.append(int(pair[target]))
        scenario_variants.append(str(pair["family_variant"]))
    if not X:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64), []
    return np.stack(X), np.asarray(y, dtype=np.int64), scenario_variants


def _evaluate_focal_pairwise_probe(
    *,
    train_pairwise: list[dict[str, Any]],
    test_pairwise: list[dict[str, Any]],
    activation_cache: dict[int, dict[str, np.ndarray]],
    row_key: str,
    layer: int,
    target: str,
    seed: int,
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

    X_train, y_train, _ = _collect_focal_pairwise_examples(
        pairwise_rows=train_pairwise,
        activation_cache=activation_cache,
        row_key=row_key,
        layer=layer,
        target=target,
    )
    X_test, y_test, scenario_variants = _collect_focal_pairwise_examples(
        pairwise_rows=test_pairwise,
        activation_cache=activation_cache,
        row_key=row_key,
        layer=layer,
        target=target,
    )
    if X_train.size == 0 or X_test.size == 0 or len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"error": "insufficient_data"}

    probe = train_probe(X_train, y_train, seed=seed)
    pred = probe.predict(X_test)
    prob = probe.predict_proba(X_test)[:, 1]

    by_variant: dict[str, list[tuple[int, int]]] = {}
    for variant, label, pred_label in zip(scenario_variants, y_test, pred, strict=True):
        by_variant.setdefault(variant, []).append((int(label), int(pred_label)))

    by_variant_metrics: dict[str, Any] = {}
    for variant, rows in by_variant.items():
        labels = np.asarray([row[0] for row in rows], dtype=np.int64)
        preds = np.asarray([row[1] for row in rows], dtype=np.int64)
        by_variant_metrics[variant] = {
            "accuracy": float(accuracy_score(labels, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
            "n": int(len(labels)),
        }

    return {
        "auroc": float(roc_auc_score(y_test, prob)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "n_rows": int(len(y_test)),
        "by_family_variant": by_variant_metrics,
    }


def _base_rank_context_variant(variant: str) -> str:
    return variant.split("__bg", 1)[0]


def _collect_rank_context_entries(
    *,
    asset_rows: list[dict[str, Any]],
    activation_cache: dict[int, dict[str, np.ndarray]],
    row_key: str,
    layer: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in asset_rows:
        if str(row.get("family")) != "rank_context_tradeoff":
            continue
        symbol = str(row.get("symbol"))
        if symbol not in {"A", "B"}:
            continue
        log_id = int(row["log_id"])
        acts = activation_cache.get(log_id)
        if not acts:
            continue
        key = f"{row_key}_{int(row['row_index'])}"
        if key not in acts:
            continue
        variant = str(row["family_variant"])
        grouped.setdefault(_base_rank_context_variant(variant), []).append({
            "symbol": symbol,
            "variant": variant,
            "bg_variant": variant.split("__bg", 1)[1] if "__bg" in variant else variant,
            "vec": acts[key][layer].astype(np.float32),
            "attractiveness_rank": int(row.get("attractiveness_rank", 0)),
            "risk_adjusted_rank": int(row.get("risk_adjusted_rank", 0)),
            "is_best_asset": int(row.get("is_best_asset", 0)),
        })
    return grouped


def _rank_context_metrics(entries: list[dict[str, Any]]) -> dict[str, Any]:
    same_symbol_sims: list[float] = []
    cross_symbol_sims: list[float] = []
    nn_hits: list[int] = []

    by_bg: dict[str, dict[str, dict[str, Any]]] = {}
    for entry in entries:
        by_bg.setdefault(str(entry["bg_variant"]), {})[str(entry["symbol"])] = entry

    ordered_entries = list(entries)
    for idx, entry in enumerate(ordered_entries):
        best_sim = None
        best_symbol = None
        for other_idx, other in enumerate(ordered_entries):
            if idx == other_idx or entry["bg_variant"] == other["bg_variant"]:
                continue
            sim = _cosine_similarity(entry["vec"], other["vec"])
            if sim is None:
                continue
            if entry["symbol"] == other["symbol"]:
                same_symbol_sims.append(sim)
            else:
                cross_symbol_sims.append(sim)
            if best_sim is None or sim > best_sim:
                best_sim = sim
                best_symbol = str(other["symbol"])
        if best_symbol is not None:
            nn_hits.append(int(best_symbol == str(entry["symbol"])))

    diff_vectors: list[np.ndarray] = []
    rank_spread: list[int] = []
    for bg_variant in sorted(by_bg):
        focal = by_bg[bg_variant]
        if {"A", "B"} <= set(focal):
            diff_vectors.append(focal["A"]["vec"] - focal["B"]["vec"])
            rank_spread.append(abs(int(focal["A"]["attractiveness_rank"]) - int(focal["B"]["attractiveness_rank"])))

    diff_cosines: list[float] = []
    for i in range(len(diff_vectors)):
        for j in range(i + 1, len(diff_vectors)):
            sim = _cosine_similarity(diff_vectors[i], diff_vectors[j])
            if sim is not None:
                diff_cosines.append(sim)

    return {
        "n_entries": len(entries),
        "same_symbol_cosine_mean": _mean(same_symbol_sims),
        "cross_symbol_cosine_mean": _mean(cross_symbol_sims),
        "same_symbol_margin": None
        if not same_symbol_sims or not cross_symbol_sims
        else float(np.mean(same_symbol_sims) - np.mean(cross_symbol_sims)),
        "same_symbol_nn_accuracy": _mean(nn_hits),
        "pair_diff_cosine_mean": _mean(diff_cosines),
        "rank_spread_values": rank_spread,
    }


def _summarize_regression(results: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for target, per_row_key in results.items():
        best: tuple[str, float, int] | None = None
        for row_key, per_layer in per_row_key.items():
            for metrics in per_layer:
                score = metrics.get("r2")
                if score is None:
                    continue
                if best is None or float(score) > best[1]:
                    best = (row_key, float(score), int(metrics["layer"]))
        summary[target] = None if best is None else {
            "representation": best[0],
            "r2": best[1],
            "layer": best[2],
        }
    return summary


def _summarize_pairwise(results: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for target, per_row_key in results.items():
        best: tuple[str, float, int] | None = None
        for row_key, per_layer in per_row_key.items():
            for metrics in per_layer:
                score = metrics.get("auroc")
                if score is None:
                    continue
                if best is None or float(score) > best[1]:
                    best = (row_key, float(score), int(metrics["layer"]))
        summary[target] = None if best is None else {
            "representation": best[0],
            "auroc": best[1],
            "layer": best[2],
        }
    return summary


def _summarize_rank_context(results: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for scenario, per_row_key in results.items():
        best: tuple[str, float, int] | None = None
        for row_key, per_layer in per_row_key.items():
            for metrics in per_layer:
                score = metrics.get("same_symbol_margin")
                if score is None:
                    continue
                if best is None or float(score) > best[1]:
                    best = (row_key, float(score), int(metrics["layer"]))
        summary[scenario] = None if best is None else {
            "representation": best[0],
            "same_symbol_margin": best[1],
            "layer": best[2],
        }
    return summary


def _collect_symbol_permutation_entries(
    *,
    asset_rows: list[dict[str, Any]],
    activation_cache: dict[int, dict[str, np.ndarray]],
    row_key: str,
    layer: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in asset_rows:
        if str(row.get("family")) != "symbol_permutation_control":
            continue
        profile_id = row.get("profile_id")
        if not profile_id:
            continue
        log_id = int(row["log_id"])
        acts = activation_cache.get(log_id)
        if not acts:
            continue
        key = f"{row_key}_{int(row['row_index'])}"
        if key not in acts:
            continue
        grouped.setdefault(str(row["family_variant"]), []).append({
            "example_id": str(row["example_id"]),
            "profile_id": str(profile_id),
            "symbol": str(row["symbol"]),
            "row_index": int(row["row_index"]),
            "vec": acts[key][layer].astype(np.float32),
        })
    return grouped


def _symbol_permutation_metrics(entries: list[dict[str, Any]]) -> dict[str, Any]:
    if len(entries) < 4:
        return {"error": "insufficient_entries"}

    same_profile_hits: list[int] = []
    same_symbol_hits: list[int] = []
    same_row_hits: list[int] = []
    same_profile_sims: list[float] = []
    same_symbol_sims: list[float] = []
    same_row_sims: list[float] = []
    profile_control_hits: list[int] = []
    profile_control_same_sims: list[float] = []
    profile_control_other_sims: list[float] = []

    for idx, entry in enumerate(entries):
        best_sim = None
        best_match: dict[str, Any] | None = None
        best_control_sim = None
        best_control_match: dict[str, Any] | None = None
        best_same_profile_control = None
        best_other_profile_control = None
        for other_idx, other in enumerate(entries):
            if idx == other_idx or entry["example_id"] == other["example_id"]:
                continue
            sim = _cosine_similarity(entry["vec"], other["vec"])
            if sim is None:
                continue
            if entry["profile_id"] == other["profile_id"]:
                same_profile_sims.append(sim)
            if entry["symbol"] == other["symbol"]:
                same_symbol_sims.append(sim)
            if entry["row_index"] == other["row_index"]:
                same_row_sims.append(sim)
            if best_sim is None or sim > best_sim:
                best_sim = sim
                best_match = other
            if other["symbol"] != entry["symbol"] and other["row_index"] != entry["row_index"]:
                if best_control_sim is None or sim > best_control_sim:
                    best_control_sim = sim
                    best_control_match = other
                if other["profile_id"] == entry["profile_id"]:
                    if best_same_profile_control is None or sim > best_same_profile_control:
                        best_same_profile_control = sim
                else:
                    if best_other_profile_control is None or sim > best_other_profile_control:
                        best_other_profile_control = sim
        if best_match is not None:
            same_profile_hits.append(int(best_match["profile_id"] == entry["profile_id"]))
            same_symbol_hits.append(int(best_match["symbol"] == entry["symbol"]))
            same_row_hits.append(int(best_match["row_index"] == entry["row_index"]))
        if best_control_match is not None:
            profile_control_hits.append(int(best_control_match["profile_id"] == entry["profile_id"]))
        if best_same_profile_control is not None:
            profile_control_same_sims.append(best_same_profile_control)
        if best_other_profile_control is not None:
            profile_control_other_sims.append(best_other_profile_control)

    return {
        "n_entries": len(entries),
        "same_profile_nn_accuracy": _mean(same_profile_hits),
        "same_symbol_nn_accuracy": _mean(same_symbol_hits),
        "same_row_nn_accuracy": _mean(same_row_hits),
        "profile_control_nn_accuracy": _mean(profile_control_hits),
        "same_profile_cosine_mean": _mean(same_profile_sims),
        "same_symbol_cosine_mean": _mean(same_symbol_sims),
        "same_row_cosine_mean": _mean(same_row_sims),
        "profile_control_same_cosine_mean": _mean(profile_control_same_sims),
        "profile_control_other_cosine_mean": _mean(profile_control_other_sims),
        "profile_minus_symbol_margin": None
        if not same_profile_sims or not same_symbol_sims
        else float(np.mean(same_profile_sims) - np.mean(same_symbol_sims)),
        "profile_minus_row_margin": None
        if not same_profile_sims or not same_row_sims
        else float(np.mean(same_profile_sims) - np.mean(same_row_sims)),
        "profile_control_margin": None
        if not profile_control_same_sims or not profile_control_other_sims
        else float(np.mean(profile_control_same_sims) - np.mean(profile_control_other_sims)),
    }


def _summarize_symbol_permutation(results: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for scenario, per_row_key in results.items():
        best: tuple[str, float, int, float | None] | None = None
        for row_key, per_layer in per_row_key.items():
            for metrics in per_layer:
                score = metrics.get("profile_control_margin")
                if score is None:
                    score = metrics.get("profile_minus_symbol_margin")
                if score is None:
                    continue
                control_acc = metrics.get("profile_control_nn_accuracy")
                if best is None or float(score) > best[1]:
                    best = (row_key, float(score), int(metrics["layer"]), None if control_acc is None else float(control_acc))
        summary[scenario] = None if best is None else {
            "representation": best[0],
            "profile_control_margin": best[1],
            "layer": best[2],
            "profile_control_nn_accuracy": best[3],
        }
    return summary


def run_synthetic_market_representation_analysis(config: SyntheticMarketRepresentationConfig) -> dict[str, Any]:
    meta_rows, tick_rows, asset_rows = _load_structure_tables(config.structure_dir)
    if not meta_rows:
        return {"error": "no_synthetic_structure_metadata"}

    allowed = set(config.family_allowlist)
    tick_rows = [
        row for row in tick_rows
        if str(row.get("context_variant")) == config.context_variant
        and str(row.get("family")) in allowed
    ]
    log_ids = sorted({int(row["log_id"]) for row in tick_rows})
    if not log_ids:
        return {"error": "no_market_ticks_for_phase"}

    asset_rows = [
        row for row in asset_rows
        if int(row["log_id"]) in set(log_ids)
        and str(row.get("context_variant")) == config.context_variant
        and str(row.get("family")) in allowed
    ]
    asset_by_log = _group_asset_rows(asset_rows)

    activation_cache = _preload_pooled_residuals(
        config.structure_dir,
        log_ids,
        max_workers=config.num_workers,
    )
    if not activation_cache:
        return {"error": "no_synthetic_pooled_residuals"}

    sample_acts = next(iter(activation_cache.values()))
    layers = config.layers or list(range(int(sample_acts["last_token"].shape[0])))
    train_ids, test_ids = _split_ids(log_ids, seed=config.seed, test_fraction=config.test_fraction)
    pairwise_rows = _load_pairwise_rows(
        phase_name=config.phase_name,
        context_variant=config.context_variant,
        log_ids=log_ids,
    )
    pairwise_rows = [row for row in pairwise_rows if str(row.get("family")) in allowed]
    train_pairwise = [row for row in pairwise_rows if int(row["log_id"]) in train_ids]
    test_pairwise = [row for row in pairwise_rows if int(row["log_id"]) in test_ids]

    analysis: dict[str, Any] = {
        "phase_name": config.phase_name,
        "context_variant": config.context_variant,
        "n_market_ticks": len(log_ids),
        "layers": layers,
        "row_keys": list(config.row_keys),
        "primitive_regression": {},
        "focal_pairwise": {},
        "rank_context": {},
        "symbol_permutation": {},
    }

    for target in config.regression_targets:
        analysis["primitive_regression"][target] = {}
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
            analysis["primitive_regression"][target][row_key] = per_layer

    for target in ("a_beats_b_on_attractiveness", "a_beats_b_on_risk_adjusted"):
        analysis["focal_pairwise"][target] = {}
        for row_key in config.row_keys:
            per_layer: list[dict[str, Any]] = []
            for layer in layers:
                metrics = _evaluate_focal_pairwise_probe(
                    train_pairwise=train_pairwise,
                    test_pairwise=test_pairwise,
                    activation_cache=activation_cache,
                    row_key=row_key,
                    layer=layer,
                    target=target,
                    seed=config.seed,
                )
                metrics["layer"] = layer
                per_layer.append(metrics)
            analysis["focal_pairwise"][target][row_key] = per_layer

    rank_context_rows = [row for row in asset_rows if str(row.get("family")) == "rank_context_tradeoff"]
    if rank_context_rows:
        scenario_groups = _collect_rank_context_entries(
            asset_rows=rank_context_rows,
            activation_cache=activation_cache,
            row_key=config.row_keys[0],
            layer=layers[0],
        )
        for scenario in scenario_groups:
            analysis["rank_context"][scenario] = {}

        for row_key in config.row_keys:
            scenario_groups_by_layer = {
                scenario: [] for scenario in analysis["rank_context"]
            }
            for layer in layers:
                grouped = _collect_rank_context_entries(
                    asset_rows=rank_context_rows,
                    activation_cache=activation_cache,
                    row_key=row_key,
                    layer=layer,
                )
                for scenario, entries in grouped.items():
                    metrics = _rank_context_metrics(entries)
                    metrics["layer"] = layer
                    scenario_groups_by_layer.setdefault(scenario, []).append(metrics)
            for scenario, per_layer in scenario_groups_by_layer.items():
                analysis["rank_context"].setdefault(scenario, {})[row_key] = per_layer

    symbol_rows = [row for row in asset_rows if str(row.get("family")) == "symbol_permutation_control"]
    if symbol_rows:
        for row_key in config.row_keys:
            scenario_groups_by_layer: dict[str, list[dict[str, Any]]] = {}
            for layer in layers:
                grouped = _collect_symbol_permutation_entries(
                    asset_rows=symbol_rows,
                    activation_cache=activation_cache,
                    row_key=row_key,
                    layer=layer,
                )
                for scenario, entries in grouped.items():
                    scenario_groups_by_layer.setdefault(scenario, []).append({
                        **_symbol_permutation_metrics(entries),
                        "layer": layer,
                    })
            for scenario, per_layer in scenario_groups_by_layer.items():
                analysis["symbol_permutation"].setdefault(scenario, {})[row_key] = per_layer

    analysis["summary"] = {
        "primitive_regression": _summarize_regression(analysis["primitive_regression"]),
        "focal_pairwise": _summarize_pairwise(analysis["focal_pairwise"]),
        "rank_context": _summarize_rank_context(analysis["rank_context"]),
        "symbol_permutation": _summarize_symbol_permutation(analysis["symbol_permutation"]),
    }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / "results.json"
    output_path.write_text(json.dumps(analysis, indent=2))
    print(f"Wrote synthetic market representation analysis to {output_path}", flush=True)
    return analysis


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic market representation analysis")
    parser.add_argument("--structure-dir", type=Path, default=Path("data/activations/synthetic_structure/phase4_market_representation_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis_results/synthetic_market_representation/phase4_market_representation_v1"))
    parser.add_argument("--phase-name", default="phase4_market_representation_v1")
    parser.add_argument("--context-variant", default="market_only")
    parser.add_argument("--layers", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    layers = [int(token) for token in args.layers.split(",") if token.strip()] or None
    config = SyntheticMarketRepresentationConfig(
        structure_dir=args.structure_dir,
        output_dir=args.output_dir,
        phase_name=args.phase_name,
        context_variant=args.context_variant,
        layers=layers,
        seed=args.seed,
        test_fraction=args.test_fraction,
        num_workers=args.num_workers,
    )
    print(json.dumps(run_synthetic_market_representation_analysis(config), indent=2))


if __name__ == "__main__":
    main()
