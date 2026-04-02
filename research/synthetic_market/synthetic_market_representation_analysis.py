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
from scipy.stats import spearmanr

from pipelines.db import connect_neon
from research.counterfactual.analysis import train_probe
from pipelines.interp.synthetic.market import SET_GEOMETRY_SCENARIOS
from research.synthetic_market.synthetic_manifold_analysis import (
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


RELATION_CONTROL_FAMILY = "relation_invariance_control"
SET_GEOMETRY_CONTROL_FAMILY = "set_geometry_control"
SET_GEOMETRY_COORDS_BY_SCENARIO = {
    str(scenario["name"]): {
        str(profile["profile_id"]): tuple(float(value) for value in profile["coords"])
        for profile in scenario["profiles"]
    }
    for scenario in SET_GEOMETRY_SCENARIOS
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


def _parse_relation_invariance_example_id(example_id: Any) -> tuple[int, int, int, int, int] | None:
    text = str(example_id)
    if not text.startswith("relation_inv_"):
        return None
    parts = text.split("_")
    if len(parts) != 7:
        return None
    try:
        return int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])
    except ValueError:
        return None


def _parse_set_geometry_example_id(example_id: Any) -> tuple[int, int, int, int] | None:
    text = str(example_id)
    if text.startswith("set_geom_aff_"):
        parts = text.split("_")
        if len(parts) != 7:
            return None
        try:
            return int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])
        except ValueError:
            return None
    if not text.startswith("set_geom_"):
        return None
    parts = text.split("_")
    if len(parts) != 6:
        return None
    try:
        return int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
    except ValueError:
        return None


def _parse_risk_ladder_context(context_variant: Any) -> int | None:
    text = str(context_variant)
    if not text.startswith("risk_"):
        return None
    suffix = text.removeprefix("risk_")
    if not suffix.isdigit():
        return None
    level = int(suffix)
    if 1 <= level <= 5:
        return level
    return None


def _parse_portfolio_ladder_context(context_variant: Any) -> int | None:
    text = str(context_variant)
    if not text.startswith("portfolio_"):
        return None
    suffix = text.removeprefix("portfolio_")
    if not suffix.isdigit():
        return None
    level = int(suffix)
    if 1 <= level <= 5:
        return level
    return None


def _parse_affordance_ladder_context(context_variant: Any) -> int | None:
    text = str(context_variant)
    if not text.startswith("affordance_"):
        return None
    suffix = text.removeprefix("affordance_")
    if not suffix.isdigit():
        return None
    level = int(suffix)
    if 1 <= level <= 5:
        return level
    return None


def _ordered_set_geometry_context_variants(context_variants: list[str]) -> list[str]:
    unique = sorted({str(context_variant) for context_variant in context_variants})
    ordered: list[str] = []
    if "market_only" in unique:
        ordered.append("market_only")
    risk_contexts = sorted(
        (context for context in unique if _parse_risk_ladder_context(context) is not None),
        key=lambda context: int(_parse_risk_ladder_context(context) or 0),
    )
    portfolio_contexts = sorted(
        (context for context in unique if _parse_portfolio_ladder_context(context) is not None),
        key=lambda context: int(_parse_portfolio_ladder_context(context) or 0),
    )
    affordance_contexts = sorted(
        (context for context in unique if _parse_affordance_ladder_context(context) is not None),
        key=lambda context: int(_parse_affordance_ladder_context(context) or 0),
    )
    if risk_contexts:
        ordered.extend(risk_contexts)
    elif portfolio_contexts:
        ordered.extend(portfolio_contexts)
    elif affordance_contexts:
        ordered.extend(affordance_contexts)
    else:
        for context in ("low_risk", "high_risk"):
            if context in unique:
                ordered.append(context)
    for context in unique:
        if context not in ordered:
            ordered.append(context)
    return ordered


def _set_geometry_context_transfer_pairs(context_variants: list[str]) -> list[tuple[str, str]]:
    ordered = _ordered_set_geometry_context_variants(context_variants)
    if not ordered:
        return []
    base_context = "market_only" if "market_only" in ordered else ordered[0]
    pairs: list[tuple[str, str]] = [(base_context, base_context)]
    for context in ordered:
        if context == base_context:
            continue
        pairs.append((base_context, context))
    return pairs


def _set_geometry_context_deformation_pairs(context_variants: list[str]) -> list[tuple[str, str]]:
    ordered = _ordered_set_geometry_context_variants(context_variants)
    if len(ordered) < 2:
        return []
    pairs: list[tuple[str, str]] = list(zip(ordered, ordered[1:], strict=False))
    long_span = (ordered[0], ordered[-1])
    if long_span not in pairs:
        pairs.append(long_span)
    return pairs


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
        "relation_invariance_control",
        "set_geometry_control",
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
        balanced_acc = None
        if len(np.unique(labels)) >= 2 and len(np.unique(preds)) >= 2:
            balanced_acc = float(balanced_accuracy_score(labels, preds))
        by_variant_metrics[variant] = {
            "accuracy": float(accuracy_score(labels, preds)),
            "balanced_accuracy": balanced_acc,
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


def _collect_relation_invariance_examples(
    *,
    asset_rows: list[dict[str, Any]],
    activation_cache: dict[int, dict[str, np.ndarray]],
    row_key: str,
    layer: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in asset_rows:
        if str(row.get("family")) != RELATION_CONTROL_FAMILY:
            continue
        parsed = _parse_relation_invariance_example_id(row.get("example_id"))
        if parsed is None:
            continue
        _, style_idx, perm_idx, roster_idx, scale_idx = parsed
        profile_id = str(row.get("profile_id") or "")
        if profile_id not in {"anchor_left", "anchor_right"}:
            continue
        log_id = int(row["log_id"])
        acts = activation_cache.get(log_id)
        if not acts:
            continue
        key = f"{row_key}_{int(row['row_index'])}"
        if key not in acts:
            continue
        example_id = str(row["example_id"])
        entry = grouped.setdefault(
            example_id,
            {
                "example_id": example_id,
                "scenario": str(row["family_variant"]),
                "style_idx": style_idx,
                "perm_idx": perm_idx,
                "roster_idx": roster_idx,
                "scale_idx": scale_idx,
                "anchors": {},
                "ranks": {},
            },
        )
        entry["anchors"][profile_id] = acts[key][layer].astype(np.float32)
        entry["ranks"][profile_id] = int(row.get("attractiveness_rank", 0))

    finalized: list[dict[str, Any]] = []
    for example_id in sorted(grouped):
        entry = grouped[example_id]
        anchors = entry["anchors"]
        if {"anchor_left", "anchor_right"} - set(anchors):
            continue
        left_rank = int(entry["ranks"].get("anchor_left", 0))
        right_rank = int(entry["ranks"].get("anchor_right", 0))
        finalized.append(
            {
                **entry,
                "vec": anchors["anchor_left"] - anchors["anchor_right"],
                "rank_bucket": f"{min(left_rank, right_rank)}v{max(left_rank, right_rank)}",
            }
        )
    return finalized


def _relation_invariance_mode_allowed(anchor: dict[str, Any], other: dict[str, Any], *, mode: str) -> bool:
    if anchor["example_id"] == other["example_id"]:
        return False
    if mode == "full":
        return True
    if mode == "style_only":
        return (
            anchor["perm_idx"] == other["perm_idx"]
            and anchor["roster_idx"] == other["roster_idx"]
            and anchor["scale_idx"] == other["scale_idx"]
            and anchor["style_idx"] != other["style_idx"]
        )
    if mode == "layout_only":
        return (
            anchor["style_idx"] == other["style_idx"]
            and anchor["roster_idx"] == other["roster_idx"]
            and anchor["scale_idx"] == other["scale_idx"]
            and anchor["perm_idx"] != other["perm_idx"]
        )
    if mode == "roster_only":
        return (
            anchor["style_idx"] == other["style_idx"]
            and anchor["perm_idx"] == other["perm_idx"]
            and anchor["scale_idx"] == other["scale_idx"]
            and anchor["roster_idx"] != other["roster_idx"]
        )
    if mode == "magnitude_only":
        return (
            anchor["style_idx"] == other["style_idx"]
            and anchor["perm_idx"] == other["perm_idx"]
            and anchor["roster_idx"] == other["roster_idx"]
            and anchor["scale_idx"] != other["scale_idx"]
        )
    raise ValueError(f"unknown relation-invariance mode: {mode}")


def _focal_relation_invariance_metrics(
    examples: list[dict[str, Any]],
    *,
    mode: str,
    anchor_scenario: str | None = None,
) -> dict[str, Any]:
    anchors = (
        [example for example in examples if example["scenario"] == anchor_scenario]
        if anchor_scenario is not None
        else list(examples)
    )
    if len(anchors) < 2 or len(examples) < 4:
        return {"error": "insufficient_examples"}

    hits: list[int] = []
    same_sims: list[float] = []
    other_sims: list[float] = []

    for entry in anchors:
        best_match: dict[str, Any] | None = None
        best_sim = None
        best_same = None
        best_other = None
        for other in examples:
            if not _relation_invariance_mode_allowed(entry, other, mode=mode):
                continue
            sim = _cosine_similarity(entry["vec"], other["vec"])
            if sim is None:
                continue
            if best_sim is None or sim > best_sim:
                best_sim = sim
                best_match = other
            if other["scenario"] == entry["scenario"]:
                if best_same is None or sim > best_same:
                    best_same = sim
            else:
                if best_other is None or sim > best_other:
                    best_other = sim
        if best_match is not None:
            hits.append(int(best_match["scenario"] == entry["scenario"]))
        if best_same is not None:
            same_sims.append(best_same)
        if best_other is not None:
            other_sims.append(best_other)

    return {
        "n_examples": len(anchors),
        "nn_accuracy": _mean(hits),
        "same_relation_cosine_mean": _mean(same_sims),
        "other_relation_cosine_mean": _mean(other_sims),
        "relation_margin": None if not same_sims or not other_sims else float(np.mean(same_sims) - np.mean(other_sims)),
    }


def _relation_over_rank_control_metrics(
    examples: list[dict[str, Any]],
    *,
    anchor_scenario: str | None = None,
) -> dict[str, Any]:
    anchors = (
        [example for example in examples if example["scenario"] == anchor_scenario]
        if anchor_scenario is not None
        else list(examples)
    )
    if len(anchors) < 2 or len(examples) < 4:
        return {"error": "insufficient_examples"}

    hits: list[int] = []
    same_sims: list[float] = []
    rank_negative_sims: list[float] = []

    for entry in anchors:
        positive_best = None
        negative_best = None
        overall_best_label: str | None = None
        overall_best_sim = None
        for other in examples:
            if entry["example_id"] == other["example_id"]:
                continue
            if not (
                entry["style_idx"] == other["style_idx"]
                and entry["perm_idx"] == other["perm_idx"]
                and entry["scale_idx"] == other["scale_idx"]
            ):
                continue
            sim = _cosine_similarity(entry["vec"], other["vec"])
            if sim is None:
                continue
            label: str | None = None
            if entry["scenario"] == other["scenario"] and entry["roster_idx"] != other["roster_idx"]:
                label = "same_relation"
                if positive_best is None or sim > positive_best:
                    positive_best = sim
            elif entry["scenario"] != other["scenario"] and entry["rank_bucket"] == other["rank_bucket"]:
                label = "same_rank"
                if negative_best is None or sim > negative_best:
                    negative_best = sim
            if label is not None and (overall_best_sim is None or sim > overall_best_sim):
                overall_best_sim = sim
                overall_best_label = label

        if positive_best is not None:
            same_sims.append(positive_best)
        if negative_best is not None:
            rank_negative_sims.append(negative_best)
        if overall_best_label is not None:
            hits.append(int(overall_best_label == "same_relation"))

    return {
        "n_examples": len(anchors),
        "nn_accuracy": _mean(hits),
        "same_relation_cosine_mean": _mean(same_sims),
        "same_rank_other_relation_cosine_mean": _mean(rank_negative_sims),
        "relation_over_rank_margin": None
        if not same_sims or not rank_negative_sims
        else float(np.mean(same_sims) - np.mean(rank_negative_sims)),
    }


def _relation_over_magnitude_control_metrics(
    examples: list[dict[str, Any]],
    *,
    anchor_scenario: str | None = None,
) -> dict[str, Any]:
    anchors = (
        [example for example in examples if example["scenario"] == anchor_scenario]
        if anchor_scenario is not None
        else list(examples)
    )
    if len(anchors) < 2 or len(examples) < 4:
        return {"error": "insufficient_examples"}

    hits: list[int] = []
    same_sims: list[float] = []
    scale_negative_sims: list[float] = []

    for entry in anchors:
        positive_best = None
        negative_best = None
        overall_best_label: str | None = None
        overall_best_sim = None
        for other in examples:
            if entry["example_id"] == other["example_id"]:
                continue
            if not (
                entry["style_idx"] == other["style_idx"]
                and entry["perm_idx"] == other["perm_idx"]
                and entry["roster_idx"] == other["roster_idx"]
            ):
                continue
            sim = _cosine_similarity(entry["vec"], other["vec"])
            if sim is None:
                continue
            label: str | None = None
            if entry["scenario"] == other["scenario"] and entry["scale_idx"] != other["scale_idx"]:
                label = "same_relation"
                if positive_best is None or sim > positive_best:
                    positive_best = sim
            elif entry["scenario"] != other["scenario"] and entry["scale_idx"] == other["scale_idx"]:
                label = "same_scale"
                if negative_best is None or sim > negative_best:
                    negative_best = sim
            if label is not None and (overall_best_sim is None or sim > overall_best_sim):
                overall_best_sim = sim
                overall_best_label = label

        if positive_best is not None:
            same_sims.append(positive_best)
        if negative_best is not None:
            scale_negative_sims.append(negative_best)
        if overall_best_label is not None:
            hits.append(int(overall_best_label == "same_relation"))

    return {
        "n_examples": len(anchors),
        "nn_accuracy": _mean(hits),
        "same_relation_cosine_mean": _mean(same_sims),
        "same_scale_other_relation_cosine_mean": _mean(scale_negative_sims),
        "relation_over_scale_margin": None
        if not same_sims or not scale_negative_sims
        else float(np.mean(same_sims) - np.mean(scale_negative_sims)),
    }


def _pairwise_distance_vector_for_profiles(
    profiles: dict[str, np.ndarray],
    *,
    ordered_profiles: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    values: list[float] = []
    labels: list[str] = []
    for left_idx in range(len(ordered_profiles)):
        for right_idx in range(left_idx + 1, len(ordered_profiles)):
            left = ordered_profiles[left_idx]
            right = ordered_profiles[right_idx]
            sim = _cosine_similarity(profiles[left], profiles[right])
            values.append(1.0 - (0.0 if sim is None else sim))
            labels.append(f"{left}__{right}")
    return np.asarray(values, dtype=np.float32), tuple(labels)


def _latent_distance_vector_for_scenario(scenario: str) -> tuple[np.ndarray, tuple[str, ...]]:
    coords_by_profile = SET_GEOMETRY_COORDS_BY_SCENARIO[str(scenario)]
    ordered_profiles = tuple(coords_by_profile)
    values: list[float] = []
    labels: list[str] = []
    for left_idx in range(len(ordered_profiles)):
        for right_idx in range(left_idx + 1, len(ordered_profiles)):
            left = ordered_profiles[left_idx]
            right = ordered_profiles[right_idx]
            left_coords = np.asarray(coords_by_profile[left], dtype=np.float32)
            right_coords = np.asarray(coords_by_profile[right], dtype=np.float32)
            values.append(float(np.linalg.norm(left_coords - right_coords)))
            labels.append(f"{left}__{right}")
    return np.asarray(values, dtype=np.float32), tuple(labels)


def _collect_set_geometry_examples(
    *,
    asset_rows: list[dict[str, Any]],
    activation_cache: dict[int, dict[str, np.ndarray]],
    row_key: str,
    layer: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in asset_rows:
        if str(row.get("family")) != SET_GEOMETRY_CONTROL_FAMILY:
            continue
        parsed = _parse_set_geometry_example_id(row.get("example_id"))
        if parsed is None:
            continue
        _, style_idx, perm_idx, scale_idx = parsed
        scenario = str(row.get("family_variant"))
        profile_id = str(row.get("profile_id") or "")
        if profile_id not in SET_GEOMETRY_COORDS_BY_SCENARIO.get(scenario, {}):
            continue
        log_id = int(row["log_id"])
        acts = activation_cache.get(log_id)
        if not acts:
            continue
        key = f"{row_key}_{int(row['row_index'])}"
        if key not in acts:
            continue
        example_id = str(row["example_id"])
        entry = grouped.setdefault(
            example_id,
            {
                "example_id": example_id,
                "scenario": scenario,
                "style_idx": style_idx,
                "perm_idx": perm_idx,
                "scale_idx": scale_idx,
                "profiles": {},
            },
        )
        entry["profiles"][profile_id] = acts[key][layer].astype(np.float32)

    finalized: list[dict[str, Any]] = []
    for example_id in sorted(grouped):
        entry = grouped[example_id]
        ordered_profiles = tuple(SET_GEOMETRY_COORDS_BY_SCENARIO[entry["scenario"]])
        if any(profile_id not in entry["profiles"] for profile_id in ordered_profiles):
            continue
        geometry_vec, pair_labels = _pairwise_distance_vector_for_profiles(
            entry["profiles"],
            ordered_profiles=ordered_profiles,
        )
        finalized.append({
            **entry,
            "ordered_profiles": ordered_profiles,
            "geometry_vec": geometry_vec,
            "pair_labels": pair_labels,
        })
    return finalized


def _collect_set_geometry_coordinate_rows(
    *,
    log_ids: set[int],
    asset_rows: list[dict[str, Any]],
    activation_cache: dict[int, dict[str, np.ndarray]],
    row_key: str,
    layer: int,
    axis_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    X_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    for row in asset_rows:
        if str(row.get("family")) != SET_GEOMETRY_CONTROL_FAMILY:
            continue
        log_id = int(row["log_id"])
        if log_id not in log_ids:
            continue
        scenario = str(row.get("family_variant"))
        profile_id = str(row.get("profile_id") or "")
        coords = SET_GEOMETRY_COORDS_BY_SCENARIO.get(scenario, {}).get(profile_id)
        if coords is None:
            continue
        acts = activation_cache.get(log_id)
        if not acts:
            continue
        key = f"{row_key}_{int(row['row_index'])}"
        if key not in acts:
            continue
        X_rows.append(acts[key][layer].astype(np.float32))
        y_rows.append(float(coords[axis_index]))
    if not X_rows:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.stack(X_rows), np.asarray(y_rows, dtype=np.float32)


def _set_geometry_alignment_metrics(examples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(examples) < 2:
        return {"error": "insufficient_examples"}

    spearmans: list[float] = []
    closest_hits: list[int] = []
    farthest_hits: list[int] = []
    for example in examples:
        latent_vec, latent_labels = _latent_distance_vector_for_scenario(example["scenario"])
        if tuple(latent_labels) != tuple(example["pair_labels"]):
            continue
        act_vec = np.asarray(example["geometry_vec"], dtype=np.float32)
        corr = spearmanr(latent_vec, act_vec).correlation
        if corr is not None and not np.isnan(corr):
            spearmans.append(float(corr))
        closest_hits.append(int(int(np.argmin(latent_vec)) == int(np.argmin(act_vec))))
        farthest_hits.append(int(int(np.argmax(latent_vec)) == int(np.argmax(act_vec))))

    return {
        "n_examples": len(examples),
        "distance_spearman_mean": _mean(spearmans),
        "closest_pair_accuracy": _mean(closest_hits),
        "farthest_pair_accuracy": _mean(farthest_hits),
    }


def _set_geometry_mode_allowed(anchor: dict[str, Any], other: dict[str, Any], *, mode: str) -> bool:
    if anchor["example_id"] == other["example_id"]:
        return False
    if mode == "full":
        return True
    if mode == "style_only":
        return (
            anchor["perm_idx"] == other["perm_idx"]
            and anchor["scale_idx"] == other["scale_idx"]
            and anchor["style_idx"] != other["style_idx"]
        )
    if mode == "layout_only":
        return (
            anchor["style_idx"] == other["style_idx"]
            and anchor["scale_idx"] == other["scale_idx"]
            and anchor["perm_idx"] != other["perm_idx"]
        )
    if mode == "magnitude_only":
        return (
            anchor["style_idx"] == other["style_idx"]
            and anchor["perm_idx"] == other["perm_idx"]
            and anchor["scale_idx"] != other["scale_idx"]
        )
    raise ValueError(f"unknown set-geometry mode: {mode}")


def _set_geometry_identity_metrics(
    examples: list[dict[str, Any]],
    *,
    anchor_scenario: str,
    mode: str,
) -> dict[str, Any]:
    anchors = [example for example in examples if example["scenario"] == anchor_scenario]
    if len(anchors) < 2 or len(examples) < 4:
        return {"error": "insufficient_examples"}

    hits: list[int] = []
    same_sims: list[float] = []
    other_sims: list[float] = []
    for entry in anchors:
        best_match: dict[str, Any] | None = None
        best_sim = None
        best_same = None
        best_other = None
        for other in examples:
            if not _set_geometry_mode_allowed(entry, other, mode=mode):
                continue
            sim = _cosine_similarity(entry["geometry_vec"], other["geometry_vec"])
            if sim is None:
                continue
            if best_sim is None or sim > best_sim:
                best_sim = sim
                best_match = other
            if other["scenario"] == entry["scenario"]:
                if best_same is None or sim > best_same:
                    best_same = sim
            else:
                if best_other is None or sim > best_other:
                    best_other = sim
        if best_match is not None:
            hits.append(int(best_match["scenario"] == entry["scenario"]))
        if best_same is not None:
            same_sims.append(best_same)
        if best_other is not None:
            other_sims.append(best_other)

    return {
        "n_examples": len(anchors),
        "nn_accuracy": _mean(hits),
        "same_geometry_cosine_mean": _mean(same_sims),
        "other_geometry_cosine_mean": _mean(other_sims),
        "geometry_identity_margin": None if not same_sims or not other_sims else float(np.mean(same_sims) - np.mean(other_sims)),
    }


def _split_example_ids(
    example_ids: list[str],
    *,
    seed: int,
    test_fraction: float,
) -> tuple[set[str], set[str]]:
    unique_ids = sorted({str(example_id) for example_id in example_ids})
    if not unique_ids:
        return set(), set()
    if len(unique_ids) == 1:
        return {unique_ids[0]}, {unique_ids[0]}
    rng = np.random.default_rng(seed)
    shuffled = list(unique_ids)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    n_test = min(len(shuffled) - 1, n_test)
    test_ids = set(shuffled[:n_test])
    train_ids = set(shuffled[n_test:])
    return train_ids, test_ids


def _score_distance_vector_for_profiles(
    score_coords: dict[str, tuple[float, float]],
    *,
    ordered_profiles: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    values: list[float] = []
    labels: list[str] = []
    for left_idx in range(len(ordered_profiles)):
        for right_idx in range(left_idx + 1, len(ordered_profiles)):
            left = ordered_profiles[left_idx]
            right = ordered_profiles[right_idx]
            left_coords = np.asarray(score_coords[left], dtype=np.float32)
            right_coords = np.asarray(score_coords[right], dtype=np.float32)
            values.append(float(np.linalg.norm(left_coords - right_coords)))
            labels.append(f"{left}__{right}")
    return np.asarray(values, dtype=np.float32), tuple(labels)


def _collect_set_geometry_context_examples(
    *,
    asset_rows: list[dict[str, Any]],
    activation_cache: dict[int, dict[str, np.ndarray]],
    row_key: str,
    layer: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in asset_rows:
        if str(row.get("family")) != SET_GEOMETRY_CONTROL_FAMILY:
            continue
        parsed = _parse_set_geometry_example_id(row.get("example_id"))
        if parsed is None:
            continue
        _, style_idx, perm_idx, scale_idx = parsed
        scenario = str(row.get("family_variant"))
        context_variant = str(row.get("context_variant"))
        profile_id = str(row.get("profile_id") or "")
        if profile_id not in SET_GEOMETRY_COORDS_BY_SCENARIO.get(scenario, {}):
            continue
        log_id = int(row["log_id"])
        acts = activation_cache.get(log_id)
        if not acts:
            continue
        key = f"{row_key}_{int(row['row_index'])}"
        if key not in acts:
            continue
        example_key = (str(row["example_id"]), context_variant)
        entry = grouped.setdefault(
            example_key,
            {
                "example_id": str(row["example_id"]),
                "context_variant": context_variant,
                "scenario": scenario,
                "style_idx": style_idx,
                "perm_idx": perm_idx,
                "scale_idx": scale_idx,
                "profiles": {},
                "score_coords": {},
            },
        )
        entry["profiles"][profile_id] = acts[key][layer].astype(np.float32)
        entry["score_coords"][profile_id] = (
            float(row.get("attractiveness_score", 0.0)),
            float(row.get("risk_adjusted_score", 0.0)),
        )

    finalized: list[dict[str, Any]] = []
    for (_, _), entry in sorted(grouped.items()):
        ordered_profiles = tuple(SET_GEOMETRY_COORDS_BY_SCENARIO[entry["scenario"]])
        if any(profile_id not in entry["profiles"] for profile_id in ordered_profiles):
            continue
        if any(profile_id not in entry["score_coords"] for profile_id in ordered_profiles):
            continue
        geometry_vec, pair_labels = _pairwise_distance_vector_for_profiles(
            entry["profiles"],
            ordered_profiles=ordered_profiles,
        )
        score_geometry_vec, score_labels = _score_distance_vector_for_profiles(
            entry["score_coords"],
            ordered_profiles=ordered_profiles,
        )
        finalized.append({
            **entry,
            "ordered_profiles": ordered_profiles,
            "geometry_vec": geometry_vec,
            "pair_labels": pair_labels,
            "score_geometry_vec": score_geometry_vec,
            "score_labels": score_labels,
        })
    return finalized


def _collect_set_geometry_coordinate_rows_for_context(
    *,
    example_ids: set[str],
    asset_rows: list[dict[str, Any]],
    activation_cache: dict[int, dict[str, np.ndarray]],
    row_key: str,
    layer: int,
    axis_index: int,
    context_variant: str,
) -> tuple[np.ndarray, np.ndarray]:
    X_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    for row in asset_rows:
        if str(row.get("family")) != SET_GEOMETRY_CONTROL_FAMILY:
            continue
        if str(row.get("context_variant")) != context_variant:
            continue
        if str(row.get("example_id")) not in example_ids:
            continue
        scenario = str(row.get("family_variant"))
        profile_id = str(row.get("profile_id") or "")
        coords = SET_GEOMETRY_COORDS_BY_SCENARIO.get(scenario, {}).get(profile_id)
        if coords is None:
            continue
        log_id = int(row["log_id"])
        acts = activation_cache.get(log_id)
        if not acts:
            continue
        key = f"{row_key}_{int(row['row_index'])}"
        if key not in acts:
            continue
        X_rows.append(acts[key][layer].astype(np.float32))
        y_rows.append(float(coords[axis_index]))
    if not X_rows:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.stack(X_rows), np.asarray(y_rows, dtype=np.float32)


def _set_geometry_context_realignment_metrics(examples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(examples) < 2:
        return {"error": "insufficient_examples"}

    base_spearmans: list[float] = []
    score_spearmans: list[float] = []
    for example in examples:
        latent_vec, latent_labels = _latent_distance_vector_for_scenario(example["scenario"])
        if tuple(latent_labels) != tuple(example["pair_labels"]):
            continue
        if tuple(example["score_labels"]) != tuple(example["pair_labels"]):
            continue
        act_vec = np.asarray(example["geometry_vec"], dtype=np.float32)
        score_vec = np.asarray(example["score_geometry_vec"], dtype=np.float32)

        base_corr = spearmanr(latent_vec, act_vec).correlation
        if base_corr is not None and not np.isnan(base_corr):
            base_spearmans.append(float(base_corr))

        score_corr = spearmanr(score_vec, act_vec).correlation
        if score_corr is not None and not np.isnan(score_corr):
            score_spearmans.append(float(score_corr))

    return {
        "n_examples": len(examples),
        "base_distance_spearman_mean": _mean(base_spearmans),
        "score_distance_spearman_mean": _mean(score_spearmans),
        "score_over_base_margin": None
        if not base_spearmans or not score_spearmans
        else float(np.mean(score_spearmans) - np.mean(base_spearmans)),
    }


def _set_geometry_context_deformation_metrics(
    examples: list[dict[str, Any]],
    *,
    source_context: str,
    target_context: str,
) -> dict[str, Any]:
    by_key = {
        (str(example["example_id"]), str(example["context_variant"])): example
        for example in examples
    }

    spearmans: list[float] = []
    cosines: list[float] = []
    activation_norms: list[float] = []
    score_norms: list[float] = []
    paired = 0

    for example_id in sorted({str(example["example_id"]) for example in examples}):
        source = by_key.get((example_id, source_context))
        target = by_key.get((example_id, target_context))
        if source is None or target is None:
            continue
        if tuple(source["pair_labels"]) != tuple(target["pair_labels"]):
            continue
        act_delta = np.asarray(target["geometry_vec"], dtype=np.float32) - np.asarray(source["geometry_vec"], dtype=np.float32)
        score_delta = np.asarray(target["score_geometry_vec"], dtype=np.float32) - np.asarray(source["score_geometry_vec"], dtype=np.float32)
        paired += 1
        activation_norms.append(float(np.linalg.norm(act_delta)))
        score_norms.append(float(np.linalg.norm(score_delta)))
        corr = spearmanr(score_delta, act_delta).correlation
        if corr is not None and not np.isnan(corr):
            spearmans.append(float(corr))
        cosine = _cosine_similarity(score_delta, act_delta)
        if cosine is not None:
            cosines.append(float(cosine))

    return {
        "n_examples": paired,
        "deformation_spearman_mean": _mean(spearmans),
        "deformation_cosine_mean": _mean(cosines),
        "activation_delta_norm_mean": _mean(activation_norms),
        "score_delta_norm_mean": _mean(score_norms),
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
        mode_keys: list[str] = []
        for per_layer in per_row_key.values():
            for key in per_layer:
                if key not in mode_keys:
                    mode_keys.append(key)
        for mode_key in mode_keys:
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


def _summarize_best_metric(
    results: dict[str, Any],
    *,
    margin_key: str,
    acc_key: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for scenario, per_row_key in results.items():
        best: tuple[str, float, int, float | None] | None = None
        for row_key, per_layer in per_row_key.items():
            for metrics in per_layer:
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
        summary[scenario] = None if best is None else {
            "representation": best[0],
            "margin": best[1],
            "layer": best[2],
            "nn_accuracy": best[3],
        }
    return summary


def _summarize_context_transfer(results: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for target_name, by_transfer in results.items():
        summary[target_name] = {}
        for transfer_key, per_row_key in by_transfer.items():
            best: tuple[str, float, int] | None = None
            for row_key, per_layer in per_row_key.items():
                for metrics in per_layer:
                    score = metrics.get("r2")
                    if score is None:
                        continue
                    if best is None or float(score) > best[1]:
                        best = (row_key, float(score), int(metrics["layer"]))
            summary[target_name][transfer_key] = None if best is None else {
                "representation": best[0],
                "r2": best[1],
                "layer": best[2],
            }
    return summary


def run_synthetic_market_representation_analysis(config: SyntheticMarketRepresentationConfig) -> dict[str, Any]:
    meta_rows, tick_rows, asset_rows = _load_structure_tables(config.structure_dir)
    if not meta_rows:
        return {"error": "no_synthetic_structure_metadata"}

    allowed = set(config.family_allowlist)
    all_tick_rows = [
        row for row in tick_rows
        if str(row.get("family")) in allowed
    ]
    all_log_ids = sorted({int(row["log_id"]) for row in all_tick_rows})
    tick_rows = [
        row for row in tick_rows
        if str(row.get("context_variant")) == config.context_variant
        and str(row.get("family")) in allowed
    ]
    log_ids = sorted({int(row["log_id"]) for row in tick_rows})
    if not log_ids:
        return {"error": "no_market_ticks_for_phase"}

    all_asset_rows = [
        row for row in asset_rows
        if int(row["log_id"]) in set(all_log_ids)
        and str(row.get("family")) in allowed
    ]
    asset_rows = [
        row for row in all_asset_rows
        if int(row["log_id"]) in set(log_ids)
        and str(row.get("context_variant")) == config.context_variant
    ]
    asset_by_log = _group_asset_rows(asset_rows)

    activation_cache = _preload_pooled_residuals(
        config.structure_dir,
        all_log_ids,
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
        "relation_invariance": {},
        "relation_rank_control": {},
        "relation_scale_control": {},
        "set_geometry_coordinate_regression": {},
        "set_geometry_alignment": {},
        "set_geometry_identity": {},
        "set_geometry_context_transfer": {},
        "set_geometry_context_realignment": {},
        "set_geometry_context_deformation": {},
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

    relation_rows = [row for row in asset_rows if str(row.get("family")) == RELATION_CONTROL_FAMILY]
    if relation_rows:
        relation_scenarios = sorted({str(row["family_variant"]) for row in relation_rows})
        for scenario in relation_scenarios:
            analysis["relation_invariance"].setdefault(scenario, {})
            analysis["relation_rank_control"].setdefault(scenario, {})
            analysis["relation_scale_control"].setdefault(scenario, {})

        for row_key in config.row_keys:
            relation_by_scenario_and_mode: dict[str, dict[str, list[dict[str, Any]]]] = {
                scenario: {
                    "full": [],
                    "style_only": [],
                    "layout_only": [],
                    "roster_only": [],
                    "magnitude_only": [],
                }
                for scenario in relation_scenarios
            }
            rank_by_scenario: dict[str, list[dict[str, Any]]] = {scenario: [] for scenario in relation_scenarios}
            scale_by_scenario: dict[str, list[dict[str, Any]]] = {scenario: [] for scenario in relation_scenarios}

            for layer in layers:
                examples = _collect_relation_invariance_examples(
                    asset_rows=relation_rows,
                    activation_cache=activation_cache,
                    row_key=row_key,
                    layer=layer,
                )
                for scenario in relation_scenarios:
                    for mode_key in ("full", "style_only", "layout_only", "roster_only", "magnitude_only"):
                        metrics = _focal_relation_invariance_metrics(
                            examples,
                            mode=mode_key,
                            anchor_scenario=scenario,
                        )
                        metrics["layer"] = layer
                        relation_by_scenario_and_mode[scenario][mode_key].append(metrics)

                    rank_metrics = _relation_over_rank_control_metrics(
                        examples,
                        anchor_scenario=scenario,
                    )
                    rank_metrics["layer"] = layer
                    rank_by_scenario[scenario].append(rank_metrics)

                    scale_metrics = _relation_over_magnitude_control_metrics(
                        examples,
                        anchor_scenario=scenario,
                    )
                    scale_metrics["layer"] = layer
                    scale_by_scenario[scenario].append(scale_metrics)

            for scenario in relation_scenarios:
                analysis["relation_invariance"][scenario][row_key] = relation_by_scenario_and_mode[scenario]
                analysis["relation_rank_control"][scenario][row_key] = rank_by_scenario[scenario]
                analysis["relation_scale_control"][scenario][row_key] = scale_by_scenario[scenario]

    set_geometry_rows = [row for row in asset_rows if str(row.get("family")) == SET_GEOMETRY_CONTROL_FAMILY]
    set_geometry_all_rows = [row for row in all_asset_rows if str(row.get("family")) == SET_GEOMETRY_CONTROL_FAMILY]
    if set_geometry_rows:
        for target_name, axis_index in (("latent_x", 0), ("latent_y", 1)):
            analysis["set_geometry_coordinate_regression"][target_name] = {}
            for row_key in config.row_keys:
                per_layer: list[dict[str, Any]] = []
                for layer in layers:
                    X_train, y_train = _collect_set_geometry_coordinate_rows(
                        log_ids=train_ids,
                        asset_rows=set_geometry_rows,
                        activation_cache=activation_cache,
                        row_key=row_key,
                        layer=layer,
                        axis_index=axis_index,
                    )
                    X_test, y_test = _collect_set_geometry_coordinate_rows(
                        log_ids=test_ids,
                        asset_rows=set_geometry_rows,
                        activation_cache=activation_cache,
                        row_key=row_key,
                        layer=layer,
                        axis_index=axis_index,
                    )
                    if X_train.size == 0 or X_test.size == 0:
                        per_layer.append({"layer": layer, "error": "insufficient_data"})
                        continue
                    probe = _train_regression_probe(X_train, y_train)
                    metrics = _evaluate_regression_probe(probe, X_test, y_test)
                    metrics["layer"] = layer
                    per_layer.append(metrics)
                analysis["set_geometry_coordinate_regression"][target_name][row_key] = per_layer

        geometry_scenarios = sorted({str(row["family_variant"]) for row in set_geometry_rows})
        for scenario in geometry_scenarios:
            analysis["set_geometry_alignment"].setdefault(scenario, {})
            analysis["set_geometry_identity"].setdefault(scenario, {})

        for row_key in config.row_keys:
            alignment_by_scenario: dict[str, list[dict[str, Any]]] = {
                scenario: [] for scenario in geometry_scenarios
            }
            identity_by_scenario_and_mode: dict[str, dict[str, list[dict[str, Any]]]] = {
                scenario: {
                    "full": [],
                    "style_only": [],
                    "layout_only": [],
                    "magnitude_only": [],
                }
                for scenario in geometry_scenarios
            }
            for layer in layers:
                examples = _collect_set_geometry_examples(
                    asset_rows=set_geometry_rows,
                    activation_cache=activation_cache,
                    row_key=row_key,
                    layer=layer,
                )
                for scenario in geometry_scenarios:
                    scenario_examples = [example for example in examples if example["scenario"] == scenario]
                    alignment_metrics = _set_geometry_alignment_metrics(scenario_examples)
                    alignment_metrics["layer"] = layer
                    alignment_by_scenario[scenario].append(alignment_metrics)

                    for mode_key in ("full", "style_only", "layout_only", "magnitude_only"):
                        identity_metrics = _set_geometry_identity_metrics(
                            examples,
                            anchor_scenario=scenario,
                            mode=mode_key,
                        )
                        identity_metrics["layer"] = layer
                        identity_by_scenario_and_mode[scenario][mode_key].append(identity_metrics)

            for scenario in geometry_scenarios:
                analysis["set_geometry_alignment"][scenario][row_key] = alignment_by_scenario[scenario]
                analysis["set_geometry_identity"][scenario][row_key] = identity_by_scenario_and_mode[scenario]

    set_geometry_context_variants = _ordered_set_geometry_context_variants(
        [str(row.get("context_variant")) for row in set_geometry_all_rows]
    )
    if len(set_geometry_context_variants) > 1:
        base_example_ids = sorted({str(row.get("example_id")) for row in set_geometry_all_rows})
        train_example_ids, test_example_ids = _split_example_ids(
            base_example_ids,
            seed=config.seed,
            test_fraction=config.test_fraction,
        )

        transfer_pairs = _set_geometry_context_transfer_pairs(set_geometry_context_variants)
        for target_name, axis_index in (("latent_x", 0), ("latent_y", 1)):
            analysis["set_geometry_context_transfer"][target_name] = {}
            for source_context, target_context in transfer_pairs:
                transfer_key = f"{source_context}_to_{target_context}"
                analysis["set_geometry_context_transfer"][target_name][transfer_key] = {}
                for row_key in config.row_keys:
                    per_layer: list[dict[str, Any]] = []
                    for layer in layers:
                        X_train, y_train = _collect_set_geometry_coordinate_rows_for_context(
                            example_ids=train_example_ids,
                            asset_rows=set_geometry_all_rows,
                            activation_cache=activation_cache,
                            row_key=row_key,
                            layer=layer,
                            axis_index=axis_index,
                            context_variant=source_context,
                        )
                        X_test, y_test = _collect_set_geometry_coordinate_rows_for_context(
                            example_ids=test_example_ids,
                            asset_rows=set_geometry_all_rows,
                            activation_cache=activation_cache,
                            row_key=row_key,
                            layer=layer,
                            axis_index=axis_index,
                            context_variant=target_context,
                        )
                        if X_train.size == 0 or X_test.size == 0:
                            per_layer.append({"layer": layer, "error": "insufficient_data"})
                            continue
                        probe = _train_regression_probe(X_train, y_train)
                        metrics = _evaluate_regression_probe(probe, X_test, y_test)
                        metrics["layer"] = layer
                        per_layer.append(metrics)
                    analysis["set_geometry_context_transfer"][target_name][transfer_key][row_key] = per_layer

        for context_variant in set_geometry_context_variants:
            analysis["set_geometry_context_realignment"][context_variant] = {}
            for row_key in config.row_keys:
                per_layer: list[dict[str, Any]] = []
                for layer in layers:
                    examples = _collect_set_geometry_context_examples(
                        asset_rows=set_geometry_all_rows,
                        activation_cache=activation_cache,
                        row_key=row_key,
                        layer=layer,
                    )
                    context_examples = [
                        example for example in examples
                        if example["context_variant"] == context_variant
                    ]
                    metrics = _set_geometry_context_realignment_metrics(context_examples)
                    metrics["layer"] = layer
                    per_layer.append(metrics)
                analysis["set_geometry_context_realignment"][context_variant][row_key] = per_layer

        deformation_pairs = _set_geometry_context_deformation_pairs(set_geometry_context_variants)
        for source_context, target_context in deformation_pairs:
            pair_key = f"{source_context}_to_{target_context}"
            analysis["set_geometry_context_deformation"][pair_key] = {}
            for row_key in config.row_keys:
                per_layer: list[dict[str, Any]] = []
                for layer in layers:
                    examples = _collect_set_geometry_context_examples(
                        asset_rows=set_geometry_all_rows,
                        activation_cache=activation_cache,
                        row_key=row_key,
                        layer=layer,
                    )
                    metrics = _set_geometry_context_deformation_metrics(
                        examples,
                        source_context=source_context,
                        target_context=target_context,
                    )
                    metrics["layer"] = layer
                    per_layer.append(metrics)
                analysis["set_geometry_context_deformation"][pair_key][row_key] = per_layer

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
        "relation_invariance": _summarize_mode_metric(
            analysis["relation_invariance"],
            margin_key="relation_margin",
            acc_key="nn_accuracy",
        ),
        "relation_rank_control": _summarize_best_metric(
            analysis["relation_rank_control"],
            margin_key="relation_over_rank_margin",
            acc_key="nn_accuracy",
        ),
        "relation_scale_control": _summarize_best_metric(
            analysis["relation_scale_control"],
            margin_key="relation_over_scale_margin",
            acc_key="nn_accuracy",
        ),
        "set_geometry_coordinate_regression": _summarize_regression(
            analysis["set_geometry_coordinate_regression"]
        ),
        "set_geometry_alignment": _summarize_best_metric(
            analysis["set_geometry_alignment"],
            margin_key="distance_spearman_mean",
            acc_key="closest_pair_accuracy",
        ),
        "set_geometry_identity": _summarize_mode_metric(
            analysis["set_geometry_identity"],
            margin_key="geometry_identity_margin",
            acc_key="nn_accuracy",
        ),
        "set_geometry_context_transfer": _summarize_context_transfer(
            analysis["set_geometry_context_transfer"]
        ),
        "set_geometry_context_realignment": _summarize_best_metric(
            analysis["set_geometry_context_realignment"],
            margin_key="score_over_base_margin",
            acc_key="score_distance_spearman_mean",
        ),
        "set_geometry_context_deformation": _summarize_best_metric(
            analysis["set_geometry_context_deformation"],
            margin_key="deformation_spearman_mean",
            acc_key="deformation_cosine_mean",
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
