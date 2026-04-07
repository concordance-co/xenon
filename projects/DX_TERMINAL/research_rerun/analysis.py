"""Analysis for blocked-valence and settings-twist real-prompt rerun captures."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pipelines.db import connect_neon
from projects.DX_TERMINAL.counterfactual.analysis import linear_cka, preload_all_activations, train_probe
from projects.DX_TERMINAL.decision_structure.analysis import (
    _evaluate_target_groups,
    _load_structure_labels,
    _preload_pooled_residuals,
    _split_log_ids,
    collect_pre_groups,
)


@dataclass(slots=True)
class ResearchRerunAnalysisConfig:
    decision_structure_dir: Path = Path("/data/activations/decision_structure")
    decision_results_path: Path = Path("/data/analysis_results/decision_structure")
    research_activations_dir: Path = Path("/projects/activations/research_rerun")
    output_dir: Path = Path("/projects/analysis_results/research_rerun")
    experiment_id: str = "blocked_valence_settings_twist_kickoff_v1"
    seed: int = 42
    test_fraction: float = 0.2
    num_workers: int = 8
    buy_probe_target: str = "is_buy_target"
    sell_probe_target: str = "is_sell_target"
    default_buy_row_key: str = "row_eos"
    default_buy_layer: int = 25
    default_sell_row_key: str = "row_mean"
    default_sell_layer: int = 2

    @property
    def research_run_dir(self) -> Path:
        return self.research_activations_dir / self.experiment_id

    @property
    def results_dir(self) -> Path:
        return self.output_dir / self.experiment_id


def _resolve_decision_results_path(path: Path) -> Path | None:
    if path.is_file():
        return path
    candidate = path / "decision_structure_results.json"
    if candidate.exists():
        return candidate
    return None


def _load_decision_summary(path: Path) -> dict[str, Any]:
    resolved = _resolve_decision_results_path(path)
    if resolved is None:
        return {}
    try:
        return json.loads(resolved.read_text()).get("summary", {})
    except Exception:
        return {}


def _fit_binary_row_probe(
    *,
    structure_dir: Path,
    target: str,
    row_key: str,
    layer: int,
    train_ids: set[int],
    test_ids: set[int],
    asset_by_log: dict[int, list[dict[str, Any]]],
    activation_cache: dict[int, dict[str, np.ndarray]],
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    train_groups = collect_pre_groups(
        log_ids=train_ids,
        asset_by_log=asset_by_log,
        structure_dir=structure_dir,
        target=target,
        layer=layer,
        row_key=row_key,
        activation_cache=activation_cache,
    )
    test_groups = collect_pre_groups(
        log_ids=test_ids,
        asset_by_log=asset_by_log,
        structure_dir=structure_dir,
        target=target,
        layer=layer,
        row_key=row_key,
        activation_cache=activation_cache,
    )
    metrics = _evaluate_target_groups(train_groups, test_groups, seed=seed)
    if "error" in metrics:
        raise RuntimeError(f"Could not fit row probe for {target}: {metrics['error']}")
    X_train = np.concatenate([group["X"] for group in train_groups])
    y_train = np.concatenate([group["y"] for group in train_groups])
    probe = train_probe(X_train, y_train, seed=seed)
    return probe, metrics


def _fit_last_token_binary_probe(
    *,
    tick_rows: list[dict[str, Any]],
    activation_cache: dict[int, dict[str, np.ndarray]],
    train_ids: set[int],
    test_ids: set[int],
    label_fn,
    filter_fn,
    layers: list[int],
    seed: int,
    num_workers: int,
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

    def _fit_one(layer: int) -> dict[str, Any]:
        X_train: list[np.ndarray] = []
        y_train: list[int] = []
        X_test: list[np.ndarray] = []
        y_test: list[int] = []
        for row in tick_rows:
            log_id = int(row["log_id"])
            if not filter_fn(row):
                continue
            acts = activation_cache.get(log_id, {})
            if "last_token" not in acts:
                continue
            vec = acts["last_token"][layer].astype(np.float32)
            label = int(label_fn(row))
            if log_id in train_ids:
                X_train.append(vec)
                y_train.append(label)
            elif log_id in test_ids:
                X_test.append(vec)
                y_test.append(label)
        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
            return {"layer": layer, "error": "insufficient_class_balance"}
        probe = train_probe(np.stack(X_train), np.array(y_train), seed=seed)
        probs = probe.predict_proba(np.stack(X_test))[:, 1]
        preds = probe.predict(np.stack(X_test))
        return {
            "layer": layer,
            "probe": probe,
            "accuracy": float(accuracy_score(y_test, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
            "auroc": float(roc_auc_score(y_test, probs)),
            "n_train": len(y_train),
            "n_test": len(y_test),
        }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        futures = {pool.submit(_fit_one, layer): layer for layer in layers}
        for fut in as_completed(futures):
            results.append(fut.result())
    valid = [row for row in results if "error" not in row]
    if not valid:
        raise RuntimeError("No valid last-token probe fits completed")
    best = max(valid, key=lambda row: (row["balanced_accuracy"], row["auroc"]))
    return {
        "best": {
            "layer": int(best["layer"]),
            "accuracy": float(best["accuracy"]),
            "balanced_accuracy": float(best["balanced_accuracy"]),
            "auroc": float(best["auroc"]),
            "n_train": int(best["n_train"]),
            "n_test": int(best["n_test"]),
        },
        "all_layers": sorted(
            [
                {
                    "layer": int(row["layer"]),
                    "accuracy": float(row["accuracy"]),
                    "balanced_accuracy": float(row["balanced_accuracy"]),
                    "auroc": float(row["auroc"]),
                }
                for row in valid
            ],
            key=lambda row: row["layer"],
        ),
        "probe": best["probe"],
    }


def _load_research_prompt_rows(experiment_id: str) -> list[dict[str, Any]]:
    conn = connect_neon()
    try:
        rows = conn.execute(
            """
            SELECT
                p.prompt_id AS capture_id,
                p.base_example_id,
                p.experiment_id,
                p.experiment_group,
                p.cohort_label,
                p.variant,
                p.n_rows,
                p.row_order,
                p.target_asset,
                p.block_reason,
                p.settings_signature,
                p.actionability_cell,
                p.metadata
            FROM research_rerun_prompts p
            WHERE p.experiment_id = %s
            ORDER BY p.experiment_group, p.base_example_id, p.variant
            """,
            [experiment_id],
        ).fetchall()
    finally:
        conn.close()
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        metadata = record.get("metadata")
        if isinstance(metadata, str):
            try:
                record["metadata"] = json.loads(metadata)
            except json.JSONDecodeError:
                record["metadata"] = {}
        elif metadata is None:
            record["metadata"] = {}
        parsed_rows.append(record)
    return parsed_rows


def _row_vectors(
    acts: dict[str, np.ndarray],
    *,
    row_key: str,
    layer: int,
    n_rows: int,
) -> np.ndarray | None:
    vectors: list[np.ndarray] = []
    for i in range(n_rows):
        key = f"{row_key}_{i}"
        if key not in acts:
            return None
        vectors.append(acts[key][layer].astype(np.float32))
    if not vectors:
        return None
    return np.stack(vectors, axis=0)


def _orthonormal_basis(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return np.empty((mat.shape[-1], 0), dtype=np.float32)
    _, s, vh = np.linalg.svd(mat.astype(np.float32), full_matrices=False)
    rank = int(np.sum(s > 1e-6))
    if rank == 0:
        return np.empty((mat.shape[-1], 0), dtype=np.float32)
    return vh[:rank].T.astype(np.float32)


def _parallel_fraction(delta: np.ndarray, basis: np.ndarray) -> float | None:
    norm_sq = float(np.dot(delta, delta))
    if norm_sq <= 0.0:
        return None
    if basis.size == 0:
        return 0.0
    coeffs = basis.T @ delta
    parallel = basis @ coeffs
    return float(np.dot(parallel, parallel) / norm_sq)


def _infer_valence_scores(
    *,
    acts: dict[str, np.ndarray],
    trade_probe: Any,
    trade_layer: int,
    side_probe: Any,
    side_layer: int,
) -> dict[str, float]:
    trade_vec = acts["last_token"][trade_layer].astype(np.float32).reshape(1, -1)
    side_vec = acts["last_token"][side_layer].astype(np.float32).reshape(1, -1)
    p_trade = float(trade_probe.predict_proba(trade_vec)[0, 1])
    p_buy = float(side_probe.predict_proba(side_vec)[0, 1])
    bullish = p_trade * p_buy
    bearish = p_trade * (1.0 - p_buy)
    neutral = 1.0 - p_trade
    scores = {
        "neutral": neutral,
        "bullish": bullish,
        "bearish": bearish,
        "trade_probability": p_trade,
        "buy_given_trade": p_buy,
    }
    scores["predicted_valence"] = max(
        ("neutral", "bullish", "bearish"),
        key=lambda key: scores[key],
    )
    return scores


def _score_prompt_rows(
    prompt_rows: list[dict[str, Any]],
    *,
    activation_cache: dict[str, dict[str, np.ndarray]],
    buy_probe: Any,
    buy_layer: int,
    buy_row_key: str,
    sell_probe: Any,
    sell_layer: int,
    sell_row_key: str,
    trade_probe: Any,
    trade_layer: int,
    side_probe: Any,
    side_layer: int,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in prompt_rows:
        capture_id = row["capture_id"]
        acts = activation_cache.get(capture_id, {})
        if not acts or "last_token" not in acts:
            continue
        n_rows = int(row["n_rows"])
        row_order = [str(symbol) for symbol in row.get("row_order") or []]
        buy_rows = _row_vectors(acts, row_key=buy_row_key, layer=buy_layer, n_rows=n_rows)
        sell_rows = _row_vectors(acts, row_key=sell_row_key, layer=sell_layer, n_rows=n_rows)
        if buy_rows is None or sell_rows is None:
            continue
        buy_scores = buy_probe.predict_proba(buy_rows)[:, 1]
        sell_scores = sell_probe.predict_proba(sell_rows)[:, 1]
        valence = _infer_valence_scores(
            acts=acts,
            trade_probe=trade_probe,
            trade_layer=trade_layer,
            side_probe=side_probe,
            side_layer=side_layer,
        )
        buy_top_idx = int(np.argmax(buy_scores))
        sell_top_idx = int(np.argmax(sell_scores))
        buy_symbol = row_order[buy_top_idx] if 0 <= buy_top_idx < len(row_order) else f"row_{buy_top_idx}"
        sell_symbol = row_order[sell_top_idx] if 0 <= sell_top_idx < len(row_order) else f"row_{sell_top_idx}"
        scored.append(
            {
                **row,
                "top_buy_row_index": buy_top_idx,
                "top_buy_score": float(buy_scores[buy_top_idx]),
                "top_buy_symbol": buy_symbol,
                "top_sell_row_index": sell_top_idx,
                "top_sell_score": float(sell_scores[sell_top_idx]),
                "top_sell_symbol": sell_symbol,
                **valence,
            }
        )
    return scored


def _blocked_pair_rows(scored_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_base: dict[str, dict[str, dict[str, Any]]] = {}
    for row in scored_rows:
        if row["experiment_group"] != "blocked_valence":
            continue
        by_base.setdefault(str(row["base_example_id"]), {})[str(row["variant"])] = row

    rows: list[dict[str, Any]] = []
    for base_example_id, variants in by_base.items():
        original = variants.get("original")
        clear = variants.get("clear_strategies")
        if not original or not clear:
            continue
        clear_label = str(clear["predicted_valence"])
        revealed_asset = None
        if clear_label == "bullish":
            revealed_asset = clear["top_buy_symbol"]
        elif clear_label == "bearish":
            revealed_asset = clear["top_sell_symbol"]
        rows.append(
            {
                "base_example_id": base_example_id,
                "original_capture_id": original["capture_id"],
                "clear_capture_id": clear["capture_id"],
                "block_reason": original.get("block_reason"),
                "settings_signature": original.get("settings_signature"),
                "actionability_cell": original.get("actionability_cell"),
                "original_predicted_valence": original["predicted_valence"],
                "clear_predicted_valence": clear_label,
                "original_top_buy_symbol": original.get("top_buy_symbol"),
                "original_top_sell_symbol": original.get("top_sell_symbol"),
                "clear_top_buy_symbol": clear.get("top_buy_symbol"),
                "clear_top_sell_symbol": clear.get("top_sell_symbol"),
                "original_trade_probability": float(original["trade_probability"]),
                "clear_trade_probability": float(clear["trade_probability"]),
                "delta_trade_probability": float(clear["trade_probability"] - original["trade_probability"]),
                "original_bullish": float(original["bullish"]),
                "original_bearish": float(original["bearish"]),
                "original_neutral": float(original["neutral"]),
                "clear_bullish": float(clear["bullish"]),
                "clear_bearish": float(clear["bearish"]),
                "clear_neutral": float(clear["neutral"]),
                "revealed_asset": revealed_asset,
                "top_buy_score": float(clear["top_buy_score"]),
                "top_sell_score": float(clear["top_sell_score"]),
            }
        )
    return rows


def _settings_triplet_rows(scored_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_base: dict[str, dict[str, dict[str, Any]]] = {}
    for row in scored_rows:
        if row["experiment_group"] != "settings_twist":
            continue
        by_base.setdefault(str(row["base_example_id"]), {})[str(row["variant"])] = row

    rows: list[dict[str, Any]] = []
    for base_example_id, variants in by_base.items():
        original = variants.get("original")
        low = variants.get("settings_all1")
        high = variants.get("settings_all5")
        if not original or not low or not high:
            continue
        rows.append(
            {
                "base_example_id": base_example_id,
                "original_capture_id": original["capture_id"],
                "all1_capture_id": low["capture_id"],
                "all5_capture_id": high["capture_id"],
                "cohort_label": original.get("cohort_label"),
                "target_asset": original.get("target_asset"),
                "settings_signature": original.get("settings_signature"),
                "actionability_cell": original.get("actionability_cell"),
                "original_predicted_valence": original["predicted_valence"],
                "all1_predicted_valence": low["predicted_valence"],
                "all5_predicted_valence": high["predicted_valence"],
                "original_trade_probability": float(original["trade_probability"]),
                "all1_trade_probability": float(low["trade_probability"]),
                "all5_trade_probability": float(high["trade_probability"]),
                "delta_trade_probability_all5_minus_all1": float(high["trade_probability"] - low["trade_probability"]),
                "all1_bullish": float(low["bullish"]),
                "all5_bullish": float(high["bullish"]),
                "all1_bearish": float(low["bearish"]),
                "all5_bearish": float(high["bearish"]),
                "bullish_shift_all5_minus_all1": float(high["bullish"] - low["bullish"]),
                "bearish_shift_all5_minus_all1": float(high["bearish"] - low["bearish"]),
                "neutral_shift_all5_minus_all1": float(high["neutral"] - low["neutral"]),
            }
        )
    return rows


def _settings_layer_metrics(
    triplet_rows: list[dict[str, Any]],
    activation_cache: dict[str, dict[str, np.ndarray]],
    *,
    layer: int,
) -> dict[str, Any]:
    orig_market: list[np.ndarray] = []
    low_market: list[np.ndarray] = []
    high_market: list[np.ndarray] = []
    orig_last: list[np.ndarray] = []
    low_last: list[np.ndarray] = []
    high_last: list[np.ndarray] = []
    low_settings: list[np.ndarray] = []
    high_settings: list[np.ndarray] = []
    last_parallel_fracs: list[float] = []
    settings_parallel_fracs: list[float] = []

    for row in triplet_rows:
        orig_id = str(row["original_capture_id"])
        low_id = str(row["all1_capture_id"])
        high_id = str(row["all5_capture_id"])

        orig = activation_cache.get(orig_id)
        low = activation_cache.get(low_id)
        high = activation_cache.get(high_id)
        if not orig or not low or not high:
            continue

        n_rows = 0
        while f"row_mean_{n_rows}" in orig:
            n_rows += 1
        if n_rows == 0 or "last_token" not in orig or "last_token" not in low or "last_token" not in high:
            continue

        orig_row_mat = _row_vectors(orig, row_key="row_mean", layer=layer, n_rows=n_rows)
        low_row_mat = _row_vectors(low, row_key="row_mean", layer=layer, n_rows=n_rows)
        high_row_mat = _row_vectors(high, row_key="row_mean", layer=layer, n_rows=n_rows)
        if orig_row_mat is None or low_row_mat is None or high_row_mat is None:
            continue

        orig_market.append(orig_row_mat.mean(axis=0))
        low_market.append(low_row_mat.mean(axis=0))
        high_market.append(high_row_mat.mean(axis=0))
        orig_last.append(orig["last_token"][layer].astype(np.float32))
        low_last.append(low["last_token"][layer].astype(np.float32))
        high_last.append(high["last_token"][layer].astype(np.float32))

        if "active_settings_eos" in low and "active_settings_eos" in high:
            low_settings.append(low["active_settings_eos"][layer].astype(np.float32))
            high_settings.append(high["active_settings_eos"][layer].astype(np.float32))

        basis = _orthonormal_basis(orig_row_mat)
        last_frac = _parallel_fraction(high["last_token"][layer].astype(np.float32) - low["last_token"][layer].astype(np.float32), basis)
        if last_frac is not None:
            last_parallel_fracs.append(last_frac)
        if "active_settings_eos" in low and "active_settings_eos" in high:
            settings_frac = _parallel_fraction(
                high["active_settings_eos"][layer].astype(np.float32) - low["active_settings_eos"][layer].astype(np.float32),
                basis,
            )
            if settings_frac is not None:
                settings_parallel_fracs.append(settings_frac)

    if not orig_market:
        return {"layer": layer, "error": "no_aligned_triplets"}

    orig_market_mat = np.stack(orig_market)
    low_market_mat = np.stack(low_market)
    high_market_mat = np.stack(high_market)
    orig_last_mat = np.stack(orig_last)
    low_last_mat = np.stack(low_last)
    high_last_mat = np.stack(high_last)

    result = {
        "layer": layer,
        "n_triplets": len(orig_market),
        "row_mean_cka_original_all1": float(linear_cka(orig_market_mat, low_market_mat)),
        "row_mean_cka_original_all5": float(linear_cka(orig_market_mat, high_market_mat)),
        "last_token_cka_original_all1": float(linear_cka(orig_last_mat, low_last_mat)),
        "last_token_cka_original_all5": float(linear_cka(orig_last_mat, high_last_mat)),
        "last_token_cka_all1_all5": float(linear_cka(low_last_mat, high_last_mat)),
        "last_token_parallel_fraction_mean": float(np.mean(last_parallel_fracs)) if last_parallel_fracs else None,
    }
    if low_settings and high_settings:
        low_settings_mat = np.stack(low_settings)
        high_settings_mat = np.stack(high_settings)
        result["active_settings_cka_all1_all5"] = float(linear_cka(low_settings_mat, high_settings_mat))
        result["active_settings_parallel_fraction_mean"] = (
            float(np.mean(settings_parallel_fracs)) if settings_parallel_fracs else None
        )
    return result


def _count_dict(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return {name: int(count) for name, count in Counter(str(row.get(key) or "NONE") for row in rows).most_common()}


def run_research_rerun_analysis(config: ResearchRerunAnalysisConfig) -> dict[str, Any]:
    meta_rows, tick_rows, asset_by_log = _load_structure_labels(config.decision_structure_dir)
    decision_log_ids = sorted({int(row["log_id"]) for row in meta_rows if int(row["log_id"]) in asset_by_log})
    print(
        "Loaded decision-structure training set: "
        f"{len(meta_rows)} metadata rows, {len(tick_rows)} tick rows, {len(asset_by_log)} asset-grouped logs",
        flush=True,
    )
    train_ids, test_ids = _split_log_ids(decision_log_ids, seed=config.seed, test_fraction=config.test_fraction)
    print(
        f"Split decision-structure training set: train={len(train_ids)} test={len(test_ids)}",
        flush=True,
    )
    decision_cache = _preload_pooled_residuals(
        config.decision_structure_dir,
        decision_log_ids,
        max_workers=config.num_workers,
    )

    decision_summary = _load_decision_summary(config.decision_results_path)
    buy_summary = decision_summary.get(config.buy_probe_target, {}).get("best_pre", {})
    sell_summary = decision_summary.get(config.sell_probe_target, {}).get("best_pre", {})
    buy_row_key = str(buy_summary.get("representation") or config.default_buy_row_key)
    buy_layer = int(buy_summary.get("layer") if buy_summary.get("layer") is not None else config.default_buy_layer)
    sell_row_key = str(sell_summary.get("representation") or config.default_sell_row_key)
    sell_layer = int(sell_summary.get("layer") if sell_summary.get("layer") is not None else config.default_sell_layer)

    buy_probe, buy_metrics = _fit_binary_row_probe(
        structure_dir=config.decision_structure_dir,
        target=config.buy_probe_target,
        row_key=buy_row_key,
        layer=buy_layer,
        train_ids=train_ids,
        test_ids=test_ids,
        asset_by_log=asset_by_log,
        activation_cache=decision_cache,
        seed=config.seed,
    )
    sell_probe, sell_metrics = _fit_binary_row_probe(
        structure_dir=config.decision_structure_dir,
        target=config.sell_probe_target,
        row_key=sell_row_key,
        layer=sell_layer,
        train_ids=train_ids,
        test_ids=test_ids,
        asset_by_log=asset_by_log,
        activation_cache=decision_cache,
        seed=config.seed,
    )

    sample_acts = next(iter(decision_cache.values()))
    num_layers = int(sample_acts["last_token"].shape[0])
    layers = list(range(num_layers))
    print(f"Loaded decision-structure activation cache across {num_layers} layers", flush=True)

    decision_probe = _fit_last_token_binary_probe(
        tick_rows=tick_rows,
        activation_cache=decision_cache,
        train_ids=train_ids,
        test_ids=test_ids,
        label_fn=lambda row: row.get("decision_type") == "trade",
        filter_fn=lambda row: row.get("decision_type") in {"trade", "record_observation"},
        layers=layers,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    side_probe = _fit_last_token_binary_probe(
        tick_rows=tick_rows,
        activation_cache=decision_cache,
        train_ids=train_ids,
        test_ids=test_ids,
        label_fn=lambda row: row.get("trade_side") == "buy",
        filter_fn=lambda row: row.get("decision_type") == "trade" and row.get("trade_side") in {"buy", "sell"},
        layers=layers,
        seed=config.seed,
        num_workers=config.num_workers,
    )

    prompt_rows = _load_research_prompt_rows(config.experiment_id)
    print(
        f"Loaded research rerun prompts for {config.experiment_id}: {len(prompt_rows)} prompt variants",
        flush=True,
    )
    capture_ids = [str(row["capture_id"]) for row in prompt_rows]
    research_cache = preload_all_activations(
        config.research_run_dir,
        capture_ids,
        max_workers=config.num_workers,
    )
    print(
        f"Preloaded research rerun activations: {len(research_cache)}/{len(capture_ids)} captures",
        flush=True,
    )

    scored_rows = _score_prompt_rows(
        prompt_rows,
        activation_cache=research_cache,
        buy_probe=buy_probe,
        buy_layer=buy_layer,
        buy_row_key=buy_row_key,
        sell_probe=sell_probe,
        sell_layer=sell_layer,
        sell_row_key=sell_row_key,
        trade_probe=decision_probe["probe"],
        trade_layer=int(decision_probe["best"]["layer"]),
        side_probe=side_probe["probe"],
        side_layer=int(side_probe["best"]["layer"]),
    )
    print(f"Scored {len(scored_rows)} rerun prompts with transferred probes", flush=True)
    blocked_rows = _blocked_pair_rows(scored_rows)
    settings_rows = _settings_triplet_rows(scored_rows)
    print(
        f"Constructed paired analysis tables: blocked_pairs={len(blocked_rows)} settings_triplets={len(settings_rows)}",
        flush=True,
    )

    settings_layer_metrics: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, config.num_workers)) as pool:
        futures = {
            pool.submit(_settings_layer_metrics, settings_rows, research_cache, layer=layer): layer
            for layer in layers
        }
        for fut in as_completed(futures):
            settings_layer_metrics.append(fut.result())
    settings_layer_metrics.sort(key=lambda row: row["layer"])
    print(f"Computed settings layer metrics for {len(settings_layer_metrics)} layers", flush=True)

    blocked_asset_counts = {
        label: {
            asset: int(count)
            for asset, count in Counter(
                row["revealed_asset"] for row in blocked_rows
                if row["clear_predicted_valence"] == label and row.get("revealed_asset")
            ).most_common(10)
        }
        for label in ("bullish", "bearish")
    }
    blocked_alignment = {
        "bullish_revealed_matches_original_top_buy": int(sum(
            row["clear_predicted_valence"] == "bullish"
            and row.get("revealed_asset")
            and row.get("revealed_asset") == row.get("original_top_buy_symbol")
            for row in blocked_rows
        )),
        "bearish_revealed_matches_original_top_sell": int(sum(
            row["clear_predicted_valence"] == "bearish"
            and row.get("revealed_asset")
            and row.get("revealed_asset") == row.get("original_top_sell_symbol")
            for row in blocked_rows
        )),
    }
    valid_settings_layers = [row for row in settings_layer_metrics if "error" not in row]
    best_parallel = max(
        valid_settings_layers,
        key=lambda row: row.get("last_token_parallel_fraction_mean") if row.get("last_token_parallel_fraction_mean") is not None else -1.0,
    ) if valid_settings_layers else None
    settings_more_bullish_high = int(sum(
        row["bullish_shift_all5_minus_all1"] > 0 for row in settings_rows
    ))
    settings_more_bearish_high = int(sum(
        row["bearish_shift_all5_minus_all1"] > 0 for row in settings_rows
    ))
    settings_more_neutral_high = int(sum(
        row["neutral_shift_all5_minus_all1"] > 0 for row in settings_rows
    ))

    summary = {
        "probe_training": {
            "buy_row_probe": {
                "row_key": buy_row_key,
                "layer": buy_layer,
                **buy_metrics,
            },
            "sell_row_probe": {
                "row_key": sell_row_key,
                "layer": sell_layer,
                **sell_metrics,
            },
            "decision_type_last_token_probe": decision_probe["best"],
            "trade_side_last_token_probe": side_probe["best"],
        },
        "blocked_valence": {
            "n_pairs": len(blocked_rows),
            "original_valence_counts": _count_dict(blocked_rows, "original_predicted_valence"),
            "clear_valence_counts": _count_dict(blocked_rows, "clear_predicted_valence"),
            "mean_original_trade_probability": float(np.mean([row["original_trade_probability"] for row in blocked_rows])) if blocked_rows else None,
            "mean_clear_trade_probability": float(np.mean([row["clear_trade_probability"] for row in blocked_rows])) if blocked_rows else None,
            "mean_delta_trade_probability": float(np.mean([row["delta_trade_probability"] for row in blocked_rows])) if blocked_rows else None,
            "clear_revealed_asset_counts": blocked_asset_counts,
            "revealed_asset_alignment": blocked_alignment,
            "block_reason_counts": _count_dict(blocked_rows, "block_reason"),
        },
        "settings_twist": {
            "n_triplets": len(settings_rows),
            "original_valence_counts": _count_dict(settings_rows, "original_predicted_valence"),
            "all1_valence_counts": _count_dict(settings_rows, "all1_predicted_valence"),
            "all5_valence_counts": _count_dict(settings_rows, "all5_predicted_valence"),
            "mean_all1_trade_probability": float(np.mean([row["all1_trade_probability"] for row in settings_rows])) if settings_rows else None,
            "mean_all5_trade_probability": float(np.mean([row["all5_trade_probability"] for row in settings_rows])) if settings_rows else None,
            "mean_delta_trade_probability_all5_minus_all1": float(np.mean([row["delta_trade_probability_all5_minus_all1"] for row in settings_rows])) if settings_rows else None,
            "n_more_bullish_high": settings_more_bullish_high,
            "n_more_bearish_high": settings_more_bearish_high,
            "n_more_neutral_high": settings_more_neutral_high,
            "cohort_counts": _count_dict(settings_rows, "cohort_label"),
            "best_last_token_parallel_layer": best_parallel,
        },
    }

    config.results_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(scored_rows), config.results_dir / "prompt_scores.parquet", compression="snappy")
    pq.write_table(pa.Table.from_pylist(blocked_rows), config.results_dir / "blocked_pairs.parquet", compression="snappy")
    pq.write_table(pa.Table.from_pylist(settings_rows), config.results_dir / "settings_triplets.parquet", compression="snappy")
    pq.write_table(pa.Table.from_pylist(settings_layer_metrics), config.results_dir / "settings_layer_metrics.parquet", compression="snappy")

    results = {
        "experiment_id": config.experiment_id,
        "summary": summary,
        "settings_layer_metrics": settings_layer_metrics,
        "decision_type_probe_layers": decision_probe["all_layers"],
        "trade_side_probe_layers": side_probe["all_layers"],
    }
    (config.results_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    return results


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze blocked-valence and settings-twist rerun captures")
    p.add_argument("--experiment-id", default="blocked_valence_settings_twist_kickoff_v1")
    p.add_argument("--decision-structure-dir", type=Path, default=Path("data/activations/decision_structure"))
    p.add_argument("--decision-results-path", type=Path, default=Path("data/analysis_results/decision_structure"))
    p.add_argument("--research-activations-dir", type=Path, default=Path("data/activations/research_rerun"))
    p.add_argument("--output-dir", type=Path, default=Path("data/analysis_results/research_rerun"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=8)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config = ResearchRerunAnalysisConfig(
        decision_structure_dir=args.decision_structure_dir,
        decision_results_path=args.decision_results_path,
        research_activations_dir=args.research_activations_dir,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
        seed=args.seed,
        test_fraction=args.test_fraction,
        num_workers=args.num_workers,
    )
    results = run_research_rerun_analysis(config)
    print(json.dumps(results["summary"], indent=2, default=str))


if __name__ == "__main__":
    main()
