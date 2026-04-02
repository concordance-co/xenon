"""Sanity checks for decision-structure probes against raw market metrics.

This module reconstructs a held-out probe for a chosen target, then compares
its within-snapshot asset ranking against raw metric rankings to estimate
whether the probe is mostly following obvious numeric salience.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from research.counterfactual.analysis import train_probe


@dataclass
class DecisionStructureSanityConfig:
    structure_dir: Path = Path("data/activations/decision_structure")
    results_path: Path = Path("data/analysis_results/decision_structure/decision_structure_results.json")
    output_path: Path = Path("data/analysis_results/decision_structure/buy_probe_metric_sanity.json")
    target: str = "is_buy_target"
    summary_bucket: str = "best_pre"
    seed: int = 42
    test_fraction: float = 0.2
    metrics: tuple[str, ...] = field(default_factory=lambda: (
        "pct_5m",
        "pct_1h",
        "net_flow_5m",
        "vol_5m",
        "vol_1h",
        "unique_traders_5m",
        "is_top_5m_gainer",
        "is_top_net_flow",
        "is_momentum_divergence_leader",
        "is_flow_surprise",
        "is_participation_momentum_leader",
    ))


def _split_log_ids(log_ids: list[int], *, seed: int, test_fraction: float) -> tuple[set[int], set[int]]:
    ordered = list(log_ids)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_test = max(1, int(len(ordered) * test_fraction))
    return set(ordered[n_test:]), set(ordered[:n_test])


def _target_present(target: str, rows: list[dict[str, Any]]) -> bool:
    return any(bool(r.get(target)) for r in rows)


def _load_asset_rows(structure_dir: Path) -> dict[int, list[dict[str, Any]]]:
    asset_rows = pq.read_table(structure_dir / "asset_labels.parquet").to_pylist()
    asset_by_log: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in asset_rows:
        asset_by_log[int(row["log_id"])].append(dict(row))
    for rows in asset_by_log.values():
        rows.sort(key=lambda r: int(r["row_index"]))
    return asset_by_log


def _safe_load_residual(path: Path) -> dict[str, np.ndarray] | None:
    from safetensors.numpy import load_file

    if not path.exists():
        return None
    try:
        return load_file(str(path))
    except Exception:
        return None


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearman(values_a: np.ndarray, values_b: np.ndarray) -> float | None:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return None
    rank_a = _average_ranks(values_a.astype(np.float64))
    rank_b = _average_ranks(values_b.astype(np.float64))
    std_a = float(rank_a.std())
    std_b = float(rank_b.std())
    if std_a == 0.0 or std_b == 0.0:
        return None
    corr = np.corrcoef(rank_a, rank_b)[0, 1]
    return float(corr)


def _roc_auc_binary(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # one-based average rank
        ranks[order[i:j]] = avg_rank
        i = j
    sum_pos = float(ranks[y_true == 1].sum())
    auc = (sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def _metric_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def run_decision_structure_sanity(config: DecisionStructureSanityConfig) -> dict[str, Any]:
    structure_dir = config.structure_dir
    residual_dir = structure_dir / "residual"

    results = json.loads(config.results_path.read_text())
    choice = results["summary"][config.target][config.summary_bucket]
    representation = str(choice["representation"])
    layer = int(choice["layer"])

    asset_by_log = _load_asset_rows(structure_dir)
    log_ids = sorted(asset_by_log)
    train_ids, test_ids = _split_log_ids(log_ids, seed=config.seed, test_fraction=config.test_fraction)

    bad = Counter()

    X_train: list[np.ndarray] = []
    y_train: list[np.ndarray] = []
    train_groups_used = 0
    for log_id in sorted(train_ids):
        rows = asset_by_log[log_id]
        if not _target_present(config.target, rows):
            continue
        acts = _safe_load_residual(residual_dir / f"{log_id}.safetensors")
        if acts is None:
            bad["train_missing_or_corrupt"] += 1
            continue
        X_rows: list[np.ndarray] = []
        y_rows: list[int] = []
        for row in rows:
            key = f"{representation}_{int(row['row_index'])}"
            if key not in acts:
                X_rows = []
                bad["train_missing_key"] += 1
                break
            X_rows.append(acts[key][layer].astype(np.float32))
            y_rows.append(int(bool(row.get(config.target))))
        if not X_rows or len(set(y_rows)) < 2:
            continue
        X_train.append(np.stack(X_rows))
        y_train.append(np.array(y_rows, dtype=np.int64))
        train_groups_used += 1

    if not X_train:
        return {"error": "no_train_groups", "bad_counts": dict(bad)}

    probe = train_probe(np.concatenate(X_train), np.concatenate(y_train), seed=config.seed)

    snapshot_rows: list[dict[str, Any]] = []
    probe_aurocs: list[float] = []
    probe_hit1: list[float] = []
    metric_hit1: dict[str, list[float]] = {metric: [] for metric in config.metrics}
    metric_rhos: dict[str, list[float]] = {metric: [] for metric in config.metrics}

    for log_id in sorted(test_ids):
        rows = asset_by_log[log_id]
        if not _target_present(config.target, rows):
            continue
        y = np.asarray([int(bool(r.get(config.target))) for r in rows], dtype=np.int64)
        if len(set(y.tolist())) < 2:
            continue
        acts = _safe_load_residual(residual_dir / f"{log_id}.safetensors")
        if acts is None:
            bad["test_missing_or_corrupt"] += 1
            continue
        X_rows = []
        for row in rows:
            key = f"{representation}_{int(row['row_index'])}"
            if key not in acts:
                X_rows = []
                bad["test_missing_key"] += 1
                break
            X_rows.append(acts[key][layer].astype(np.float32))
        if not X_rows:
            continue

        X = np.stack(X_rows)
        scores = probe.predict_proba(X)[:, 1]
        auc = _roc_auc_binary(y, scores)
        probe_top = int(np.argmax(scores))
        true_top = int(np.argmax(y))

        row: dict[str, Any] = {
            "log_id": log_id,
            "n_assets": len(rows),
            "probe_top_index": probe_top,
            "true_top_index": true_top,
        }
        if auc is not None:
            row["probe_auroc"] = auc
            probe_aurocs.append(auc)
        probe_hit1.append(1.0 if probe_top == true_top else 0.0)

        for metric in config.metrics:
            vals = np.asarray([float(r.get(metric) or 0.0) for r in rows], dtype=np.float64)
            rho = _spearman(scores, vals)
            row[f"rho_{metric}"] = rho
            if rho is not None:
                metric_rhos[metric].append(rho)
            metric_hit1[metric].append(1.0 if int(np.argmax(vals)) == true_top else 0.0)

        snapshot_rows.append(row)

    output = {
        "target": config.target,
        "summary_bucket": config.summary_bucket,
        "representation": representation,
        "layer": layer,
        "n_total_log_ids": len(log_ids),
        "n_train_log_ids": len(train_ids),
        "n_test_log_ids": len(test_ids),
        "n_train_groups_used": train_groups_used,
        "n_test_groups_used": len(snapshot_rows),
        "bad_counts": dict(bad),
        "probe_metrics": {
            "mean_auroc": float(np.mean(probe_aurocs)) if probe_aurocs else None,
            "mean_hit_at_1": float(np.mean(probe_hit1)) if probe_hit1 else None,
        },
        "metric_rank_correlation": {
            metric: _metric_summary(values) for metric, values in metric_rhos.items()
        },
        "metric_top1_hit_rate": {
            metric: float(np.mean(values)) if values else None
            for metric, values in metric_hit1.items()
        },
        "per_snapshot": snapshot_rows,
    }

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(output, indent=2))
    return output


def main(argv: list[str] | None = None) -> None:
    _ = argv
    result = run_decision_structure_sanity(DecisionStructureSanityConfig())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
