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

PROFILE_CONTROL_FAMILIES = {
    "symbol_permutation_control",
    "profile_invariance_control",
}


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


def _is_profile_control_family(family: Any) -> bool:
    return str(family) in PROFILE_CONTROL_FAMILIES


def _parse_profile_invariance_example_id(example_id: Any) -> tuple[int, int, int] | None:
    text = str(example_id)
    if not text.startswith("profile_inv_"):
        return None
    parts = text.split("_")
    if len(parts) != 5:
        return None
    try:
        return int(parts[2]), int(parts[3]), int(parts[4])
    except ValueError:
        return None


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
        "profile_invariance_control",
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
        if not _is_profile_control_family(row.get("family")):
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


def _collect_profile_invariance_entries(
    *,
    asset_rows: list[dict[str, Any]],
    activation_cache: dict[int, dict[str, np.ndarray]],
    row_key: str,
    layer: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in asset_rows:
        if str(row.get("family")) != "profile_invariance_control":
            continue
        parsed = _parse_profile_invariance_example_id(row.get("example_id"))
        if parsed is None:
            continue
        _, style_idx, perm_idx = parsed
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
            "style_idx": style_idx,
            "perm_idx": perm_idx,
            "vec": acts[key][layer].astype(np.float32),
        })
    return grouped


def _profile_control_mode_allowed(anchor: dict[str, Any], other: dict[str, Any], *, mode: str) -> bool:
    if anchor["example_id"] == other["example_id"]:
        return False
    if mode == "full":
        return True
    if mode == "style_only":
        return other["perm_idx"] == anchor["perm_idx"] and other["style_idx"] != anchor["style_idx"]
    if mode == "layout_only":
        return other["style_idx"] == anchor["style_idx"] and other["perm_idx"] != anchor["perm_idx"]
    raise ValueError(f"unknown profile-control mode: {mode}")


def _collect_profile_invariance_examples(
    *,
    asset_rows: list[dict[str, Any]],
    activation_cache: dict[int, dict[str, np.ndarray]],
    row_key: str,
    layer: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in asset_rows:
        if str(row.get("family")) != "profile_invariance_control":
            continue
        parsed = _parse_profile_invariance_example_id(row.get("example_id"))
        if parsed is None:
            continue
        _, style_idx, perm_idx = parsed
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
        scenario = str(row["family_variant"])
        example_id = str(row["example_id"])
        scenario_group = grouped.setdefault(scenario, {})
        example = scenario_group.setdefault(
            example_id,
            {
                "example_id": example_id,
                "style_idx": style_idx,
                "perm_idx": perm_idx,
                "profiles": {},
            },
        )
        example["profiles"][str(profile_id)] = acts[key][layer].astype(np.float32)

    finalized: dict[str, list[dict[str, Any]]] = {}
    for scenario, examples in grouped.items():
        rows: list[dict[str, Any]] = []
        for example_id in sorted(examples):
            example = examples[example_id]
            ordered_profiles = tuple(sorted(example["profiles"]))
            if len(ordered_profiles) < 2:
                continue
            rows.append(
                {
                    **example,
                    "ordered_profiles": ordered_profiles,
                }
            )
        finalized[scenario] = rows
    return finalized


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


def _profile_invariance_mode_metrics(
    entries: list[dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    if len(entries) < 4:
        return {"nn_accuracy": None, "margin": None}

    hits: list[int] = []
    same_sims: list[float] = []
    other_sims: list[float] = []

    for idx, entry in enumerate(entries):
        best_match: dict[str, Any] | None = None
        best_sim = None
        best_same = None
        best_other = None
        for other_idx, other in enumerate(entries):
            if idx == other_idx or entry["example_id"] == other["example_id"]:
                continue
            if mode == "style_only":
                if other["perm_idx"] != entry["perm_idx"] or other["style_idx"] == entry["style_idx"]:
                    continue
            elif mode == "layout_only":
                if other["style_idx"] != entry["style_idx"] or other["perm_idx"] == entry["perm_idx"]:
                    continue
            else:
                raise ValueError(f"unknown profile invariance mode: {mode}")
            sim = _cosine_similarity(entry["vec"], other["vec"])
            if sim is None:
                continue
            if best_sim is None or sim > best_sim:
                best_sim = sim
                best_match = other
            if other["profile_id"] == entry["profile_id"]:
                if best_same is None or sim > best_same:
                    best_same = sim
            else:
                if best_other is None or sim > best_other:
                    best_other = sim
        if best_match is not None:
            hits.append(int(best_match["profile_id"] == entry["profile_id"]))
        if best_same is not None:
            same_sims.append(best_same)
        if best_other is not None:
            other_sims.append(best_other)

    return {
        "nn_accuracy": _mean(hits),
        "margin": None if not same_sims or not other_sims else float(np.mean(same_sims) - np.mean(other_sims)),
    }


def _profile_invariance_decomposition_metrics(entries: list[dict[str, Any]]) -> dict[str, Any]:
    style_only = _profile_invariance_mode_metrics(entries, mode="style_only")
    layout_only = _profile_invariance_mode_metrics(entries, mode="layout_only")
    return {
        "n_entries": len(entries),
        "style_only_nn_accuracy": style_only["nn_accuracy"],
        "style_only_margin": style_only["margin"],
        "layout_only_nn_accuracy": layout_only["nn_accuracy"],
        "layout_only_margin": layout_only["margin"],
    }


def _build_relation_entries(example: dict[str, Any]) -> list[dict[str, Any]]:
    ordered_profiles = tuple(example["ordered_profiles"])
    profiles = example["profiles"]
    relation_entries: list[dict[str, Any]] = []
    for left_idx in range(len(ordered_profiles)):
        for right_idx in range(left_idx + 1, len(ordered_profiles)):
            left = ordered_profiles[left_idx]
            right = ordered_profiles[right_idx]
            relation_entries.append(
                {
                    "example_id": example["example_id"],
                    "style_idx": example["style_idx"],
                    "perm_idx": example["perm_idx"],
                    "relation_id": f"{left}__minus__{right}",
                    "vec": profiles[left] - profiles[right],
                }
            )
    return relation_entries


def _pairwise_relation_invariance_metrics(examples: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    relation_entries: list[dict[str, Any]] = []
    for example in examples:
        relation_entries.extend(_build_relation_entries(example))
    if len(relation_entries) < 4:
        return {"error": "insufficient_entries"}

    hits: list[int] = []
    same_sims: list[float] = []
    other_sims: list[float] = []

    for entry in relation_entries:
        best_match: dict[str, Any] | None = None
        best_sim = None
        best_same = None
        best_other = None
        for other in relation_entries:
            if not _profile_control_mode_allowed(entry, other, mode=mode):
                continue
            sim = _cosine_similarity(entry["vec"], other["vec"])
            if sim is None:
                continue
            if best_sim is None or sim > best_sim:
                best_sim = sim
                best_match = other
            if other["relation_id"] == entry["relation_id"]:
                if best_same is None or sim > best_same:
                    best_same = sim
            else:
                if best_other is None or sim > best_other:
                    best_other = sim
        if best_match is not None:
            hits.append(int(best_match["relation_id"] == entry["relation_id"]))
        if best_same is not None:
            same_sims.append(best_same)
        if best_other is not None:
            other_sims.append(best_other)

    return {
        "n_relations": len(relation_entries),
        "nn_accuracy": _mean(hits),
        "same_relation_cosine_mean": _mean(same_sims),
        "other_relation_cosine_mean": _mean(other_sims),
        "relation_margin": None if not same_sims or not other_sims else float(np.mean(same_sims) - np.mean(other_sims)),
    }


def _snapshot_geometry_vector(example: dict[str, Any]) -> np.ndarray:
    ordered_profiles = tuple(example["ordered_profiles"])
    profiles = example["profiles"]
    values: list[float] = []
    for left_idx in range(len(ordered_profiles)):
        for right_idx in range(left_idx + 1, len(ordered_profiles)):
            left = ordered_profiles[left_idx]
            right = ordered_profiles[right_idx]
            sim = _cosine_similarity(profiles[left], profiles[right])
            values.append(0.0 if sim is None else sim)
    return np.asarray(values, dtype=np.float32)


def _snapshot_geometry_metrics(
    same_examples: list[dict[str, Any]],
    other_examples: list[dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    if len(same_examples) < 2 or not other_examples:
        return {"error": "insufficient_examples"}

    same_entries = [
        {
            "example_id": example["example_id"],
            "style_idx": example["style_idx"],
            "perm_idx": example["perm_idx"],
            "scenario": "same",
            "vec": _snapshot_geometry_vector(example),
        }
        for example in same_examples
    ]
    other_entries = [
        {
            "example_id": example["example_id"],
            "style_idx": example["style_idx"],
            "perm_idx": example["perm_idx"],
            "scenario": "other",
            "vec": _snapshot_geometry_vector(example),
        }
        for example in other_examples
    ]

    hits: list[int] = []
    same_sims: list[float] = []
    other_sims: list[float] = []

    for entry in same_entries:
        best_match: dict[str, Any] | None = None
        best_sim = None
        best_same = None
        best_other = None
        for other in same_entries:
            if not _profile_control_mode_allowed(entry, other, mode=mode):
                continue
            sim = _cosine_similarity(entry["vec"], other["vec"])
            if sim is None:
                continue
            if best_sim is None or sim > best_sim:
                best_sim = sim
                best_match = other
            if best_same is None or sim > best_same:
                best_same = sim
        for other in other_entries:
            if mode == "style_only":
                if other["perm_idx"] != entry["perm_idx"] or other["style_idx"] == entry["style_idx"]:
                    continue
            elif mode == "layout_only":
                if other["style_idx"] != entry["style_idx"] or other["perm_idx"] == entry["perm_idx"]:
                    continue
            elif mode != "full":
                raise ValueError(f"unknown snapshot-geometry mode: {mode}")
            sim = _cosine_similarity(entry["vec"], other["vec"])
            if sim is None:
                continue
            if best_sim is None or sim > best_sim:
                best_sim = sim
                best_match = other
            if best_other is None or sim > best_other:
                best_other = sim
        if best_match is not None:
            hits.append(int(best_match["scenario"] == "same"))
        if best_same is not None:
            same_sims.append(best_same)
        if best_other is not None:
            other_sims.append(best_other)

    return {
        "n_examples": len(same_examples),
        "nn_accuracy": _mean(hits),
        "same_market_cosine_mean": _mean(same_sims),
        "other_market_cosine_mean": _mean(other_sims),
        "geometry_margin": None if not same_sims or not other_sims else float(np.mean(same_sims) - np.mean(other_sims)),
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


def _summarize_profile_invariance_decomposition(results: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for scenario, per_row_key in results.items():
        best_style: tuple[str, float, int, float | None] | None = None
        best_layout: tuple[str, float, int, float | None] | None = None
        for row_key, per_layer in per_row_key.items():
            for metrics in per_layer:
                style_margin = metrics.get("style_only_margin")
                if style_margin is not None:
                    style_acc = metrics.get("style_only_nn_accuracy")
                    if best_style is None or float(style_margin) > best_style[1]:
                        best_style = (
                            row_key,
                            float(style_margin),
                            int(metrics["layer"]),
                            None if style_acc is None else float(style_acc),
                        )
                layout_margin = metrics.get("layout_only_margin")
                if layout_margin is not None:
                    layout_acc = metrics.get("layout_only_nn_accuracy")
                    if best_layout is None or float(layout_margin) > best_layout[1]:
                        best_layout = (
                            row_key,
                            float(layout_margin),
                            int(metrics["layer"]),
                            None if layout_acc is None else float(layout_acc),
                        )
        summary[scenario] = {
            "best_style_only": None if best_style is None else {
                "representation": best_style[0],
                "margin": best_style[1],
                "layer": best_style[2],
                "nn_accuracy": best_style[3],
            },
            "best_layout_only": None if best_layout is None else {
                "representation": best_layout[0],
                "margin": best_layout[1],
                "layer": best_layout[2],
                "nn_accuracy": best_layout[3],
            },
        }
    return summary


def _summarize_mode_metric(
    results: dict[str, Any],
    *,
    margin_key: str,
    acc_key: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for scenario, per_row_key in results.items():
        scenario_summary: dict[str, Any] = {}
        for mode_key in ("full", "style_only", "layout_only"):
            best: tuple[str, float, int, float | None] | None = None
            for row_key, per_layer in per_row_key.items():
                for metrics in per_layer.get(mode_key, []):
                    score = metrics.get(margin_key)
                    if score is None:
                        continue
                    acc = metrics.get(acc_key)
                    if best is None or float(score) > best[1]:
                        best = (
                            row_key,
                            float(score),
                            int(metrics["layer"]),
                            None if acc is None else float(acc),
                        )
            scenario_summary[mode_key] = None if best is None else {
                "representation": best[0],
                "margin": best[1],
                "layer": best[2],
                "nn_accuracy": best[3],
            }
        summary[scenario] = scenario_summary
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
        "profile_invariance_decomposition": {},
        "pairwise_relation_invariance": {},
        "snapshot_geometry": {},
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

    symbol_rows = [row for row in asset_rows if _is_profile_control_family(row.get("family"))]
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

    profile_invariance_rows = [row for row in asset_rows if str(row.get("family")) == "profile_invariance_control"]
    if profile_invariance_rows:
        for row_key in config.row_keys:
            scenario_groups_by_layer: dict[str, list[dict[str, Any]]] = {}
            for layer in layers:
                grouped = _collect_profile_invariance_entries(
                    asset_rows=profile_invariance_rows,
                    activation_cache=activation_cache,
                    row_key=row_key,
                    layer=layer,
                )
                for scenario, entries in grouped.items():
                    scenario_groups_by_layer.setdefault(scenario, []).append({
                        **_profile_invariance_decomposition_metrics(entries),
                        "layer": layer,
                    })
            for scenario, per_layer in scenario_groups_by_layer.items():
                analysis["profile_invariance_decomposition"].setdefault(scenario, {})[row_key] = per_layer

        scenario_names = sorted({str(row["family_variant"]) for row in profile_invariance_rows})
        for scenario in scenario_names:
            analysis["pairwise_relation_invariance"].setdefault(scenario, {})
            analysis["snapshot_geometry"].setdefault(scenario, {})

        for row_key in config.row_keys:
            relation_by_scenario_and_mode: dict[str, dict[str, list[dict[str, Any]]]] = {
                scenario: {"full": [], "style_only": [], "layout_only": []}
                for scenario in scenario_names
            }
            geometry_by_scenario_and_mode: dict[str, dict[str, list[dict[str, Any]]]] = {
                scenario: {"full": [], "style_only": [], "layout_only": []}
                for scenario in scenario_names
            }
            for layer in layers:
                grouped_examples = _collect_profile_invariance_examples(
                    asset_rows=profile_invariance_rows,
                    activation_cache=activation_cache,
                    row_key=row_key,
                    layer=layer,
                )
                for scenario in scenario_names:
                    scenario_examples = grouped_examples.get(scenario, [])
                    other_examples: list[dict[str, Any]] = []
                    for other_scenario, entries in grouped_examples.items():
                        if other_scenario != scenario:
                            other_examples.extend(entries)
                    for mode_key in ("full", "style_only", "layout_only"):
                        relation_metrics = _pairwise_relation_invariance_metrics(scenario_examples, mode=mode_key)
                        relation_metrics["layer"] = layer
                        relation_by_scenario_and_mode[scenario][mode_key].append(relation_metrics)

                        geometry_metrics = _snapshot_geometry_metrics(
                            scenario_examples,
                            other_examples,
                            mode=mode_key,
                        )
                        geometry_metrics["layer"] = layer
                        geometry_by_scenario_and_mode[scenario][mode_key].append(geometry_metrics)

            for scenario in scenario_names:
                analysis["pairwise_relation_invariance"][scenario][row_key] = relation_by_scenario_and_mode[scenario]
                analysis["snapshot_geometry"][scenario][row_key] = geometry_by_scenario_and_mode[scenario]

    analysis["summary"] = {
        "primitive_regression": _summarize_regression(analysis["primitive_regression"]),
        "focal_pairwise": _summarize_pairwise(analysis["focal_pairwise"]),
        "rank_context": _summarize_rank_context(analysis["rank_context"]),
        "symbol_permutation": _summarize_symbol_permutation(analysis["symbol_permutation"]),
        "profile_invariance_decomposition": _summarize_profile_invariance_decomposition(
            analysis["profile_invariance_decomposition"]
        ),
        "pairwise_relation_invariance": _summarize_mode_metric(
            analysis["pairwise_relation_invariance"],
            margin_key="relation_margin",
            acc_key="nn_accuracy",
        ),
        "snapshot_geometry": _summarize_mode_metric(
            analysis["snapshot_geometry"],
            margin_key="geometry_margin",
            acc_key="nn_accuracy",
        ),
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
