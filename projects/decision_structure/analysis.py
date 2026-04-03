"""Probe analysis over pooled real-decision structure activations.

Consumes outputs from :mod:`projects.decision_structure` and measures
when asset-binding targets become decodable:

- ``is_target_asset``: any traded asset
- ``is_buy_target``: bullish / bought asset
- ``is_sell_target``: bearish / sold asset

Each target is evaluated on row-level market states and on row+downstream
representations (settings, portfolio, constraints, previous decisions,
last token) to localize where the model binds an eventual action to an asset.
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from projects.counterfactual.analysis import (
    evaluate_probe_per_snapshot,
    train_probe,
)


@dataclass
class DecisionStructureAnalysisConfig:
    structure_dir: Path = Path("data/activations/decision_structure")
    output_dir: Path = Path("data/analysis_results/decision_structure")
    row_key: str = "row_mean"
    layers: list[int] | None = None
    seed: int = 42
    test_fraction: float = 0.2
    num_workers: int = 8
    targets: tuple[str, ...] = field(default_factory=lambda: (
        "is_target_asset",
        "is_buy_target",
        "is_sell_target",
    ))
    downstream_positions: tuple[str, ...] = field(default_factory=lambda: (
        "active_settings_eos",
        "portfolio_eos",
        "constraints_eos",
        "prev_decisions_eos",
        "last_token",
    ))


def _target_row_filter(target: str, rows: list[dict[str, Any]]) -> bool:
    if target == "is_buy_target":
        return any(bool(r.get("is_buy_target")) for r in rows)
    if target == "is_sell_target":
        return any(bool(r.get("is_sell_target")) for r in rows)
    if target == "is_target_asset":
        return any(bool(r.get("is_target_asset")) for r in rows)
    return True


def _split_log_ids(log_ids: list[int], *, seed: int, test_fraction: float) -> tuple[set[int], set[int]]:
    ordered = list(log_ids)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_test = max(1, int(len(ordered) * test_fraction))
    return set(ordered[n_test:]), set(ordered[:n_test])


def _load_structure_labels(structure_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    meta = pq.read_table(structure_dir / "metadata.parquet").to_pylist()
    tick_rows = pq.read_table(structure_dir / "tick_labels.parquet").to_pylist()
    asset_rows = pq.read_table(structure_dir / "asset_labels.parquet").to_pylist()
    asset_by_log: dict[int, list[dict[str, Any]]] = {}
    for row in asset_rows:
        log_id = int(row["log_id"])
        asset_by_log.setdefault(log_id, []).append(dict(row))
    for log_id in asset_by_log:
        asset_by_log[log_id].sort(key=lambda r: int(r["row_index"]))
    return meta, tick_rows, asset_by_log


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
                print(f"Preloaded pooled residuals: {loaded}/{total}", flush=True)
    return cache


def collect_pre_groups(
    *,
    log_ids: set[int],
    asset_by_log: dict[int, list[dict[str, Any]]],
    structure_dir: Path,
    target: str,
    layer: int,
    row_key: str,
    activation_cache: dict[int, dict[str, np.ndarray]] | None = None,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for log_id in sorted(log_ids):
        rows = asset_by_log.get(log_id)
        if not rows or not _target_row_filter(target, rows):
            continue
        acts = activation_cache.get(log_id, {}) if activation_cache is not None else _load_pooled_residual(structure_dir, log_id)
        if not acts:
            continue
        X_rows: list[np.ndarray] = []
        y: list[int] = []
        for row in rows:
            key = f"{row_key}_{int(row['row_index'])}"
            if key not in acts:
                X_rows = []
                break
            X_rows.append(acts[key][layer].astype(np.float32))
            y.append(int(bool(row.get(target))))
        if not X_rows or len(set(y)) < 2:
            continue
        groups.append({
            "X": np.stack(X_rows),
            "y": np.array(y, dtype=np.int64),
            "snapshot_id": str(log_id),
        })
    return groups


def collect_concat_groups(
    *,
    log_ids: set[int],
    asset_by_log: dict[int, list[dict[str, Any]]],
    structure_dir: Path,
    target: str,
    layer: int,
    row_key: str,
    position_key: str,
    activation_cache: dict[int, dict[str, np.ndarray]] | None = None,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for log_id in sorted(log_ids):
        rows = asset_by_log.get(log_id)
        if not rows or not _target_row_filter(target, rows):
            continue
        acts = activation_cache.get(log_id, {}) if activation_cache is not None else _load_pooled_residual(structure_dir, log_id)
        if not acts or position_key not in acts:
            continue
        pos = acts[position_key][layer].astype(np.float32)
        X_rows: list[np.ndarray] = []
        y: list[int] = []
        for row in rows:
            key = f"{row_key}_{int(row['row_index'])}"
            if key not in acts:
                X_rows = []
                break
            row_vec = acts[key][layer].astype(np.float32)
            X_rows.append(np.concatenate([row_vec, pos]))
            y.append(int(bool(row.get(target))))
        if not X_rows or len(set(y)) < 2:
            continue
        groups.append({
            "X": np.stack(X_rows),
            "y": np.array(y, dtype=np.int64),
            "snapshot_id": str(log_id),
        })
    return groups


def _mean_metric(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _evaluate_target_groups(
    train_groups: list[dict[str, Any]],
    test_groups: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    if not train_groups or not test_groups:
        return {"error": "insufficient_data"}

    X_train = np.concatenate([g["X"] for g in train_groups])
    y_train = np.concatenate([g["y"] for g in train_groups])
    if len(np.unique(y_train)) < 2:
        return {"error": "single_class_train"}

    probe = train_probe(X_train, y_train, seed=seed)
    metrics = evaluate_probe_per_snapshot(probe, test_groups)
    return {
        "auroc": _mean_metric(metrics["auroc"]),
        "hit_at_1": _mean_metric(metrics["hit_at_1"]),
        "mrr": _mean_metric(metrics["mrr"]),
        "balanced_accuracy": _mean_metric(metrics["balanced_accuracy"]),
        "n_groups": len(test_groups),
    }


def summarize_probe_results(results: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for target, reps in results.get("targets", {}).items():
        best_pre: tuple[str, float, int] | None = None
        best_post: tuple[str, float, int] | None = None
        for rep_key, per_layer in reps.items():
            for layer_result in per_layer:
                score = layer_result.get("auroc")
                if score is None:
                    continue
                candidate = (rep_key, float(score), int(layer_result["layer"]))
                if rep_key in {"row_mean", "row_eos"}:
                    if best_pre is None or candidate[1] > best_pre[1]:
                        best_pre = candidate
                else:
                    if best_post is None or candidate[1] > best_post[1]:
                        best_post = candidate
        summary[target] = {
            "best_pre": {
                "representation": best_pre[0],
                "auroc": best_pre[1],
                "layer": best_pre[2],
            } if best_pre is not None else None,
            "best_post": {
                "representation": best_post[0],
                "auroc": best_post[1],
                "layer": best_post[2],
            } if best_post is not None else None,
            "best_post_minus_best_pre": (
                best_post[1] - best_pre[1]
                if best_pre is not None and best_post is not None
                else None
            ),
        }
    return summary


def run_decision_structure_analysis(config: DecisionStructureAnalysisConfig) -> dict[str, Any]:
    meta_rows, _, asset_by_log = _load_structure_labels(config.structure_dir)
    log_ids = sorted({int(r["log_id"]) for r in meta_rows if int(r["log_id"]) in asset_by_log})
    if not log_ids:
        return {"error": "no_pooled_decision_structure"}
    print(
        f"Loaded decision structure labels: {len(meta_rows)} metadata rows, "
        f"{len(asset_by_log)} asset-grouped ticks",
        flush=True,
    )

    train_ids, test_ids = _split_log_ids(
        log_ids,
        seed=config.seed,
        test_fraction=config.test_fraction,
    )
    print(
        f"Split log_ids into train/test: train={len(train_ids)} test={len(test_ids)}",
        flush=True,
    )

    sample_acts = _load_pooled_residual(config.structure_dir, log_ids[0])
    if not sample_acts or "last_token" not in sample_acts:
        return {"error": "missing_pooled_residuals"}
    print(
        f"Sample pooled residual keys: {len(sample_acts)}; last_token layers={sample_acts['last_token'].shape[0]}",
        flush=True,
    )

    activation_cache = _preload_pooled_residuals(
        config.structure_dir,
        log_ids,
        max_workers=config.num_workers,
    )
    print(
        f"Activation preload complete: cached {len(activation_cache)}/{len(log_ids)} pooled residual files",
        flush=True,
    )

    num_layers = int(sample_acts["last_token"].shape[0])
    layers = config.layers or list(range(num_layers))
    print(f"Evaluating layers: {layers[0]}..{layers[-1]} ({len(layers)} total)", flush=True)

    results: dict[str, Any] = {
        "layers": layers,
        "row_key": config.row_key,
        "targets": {},
    }

    for target in config.targets:
        print(f"=== Target: {target} ===", flush=True)
        target_results: dict[str, list[dict[str, Any]]] = {}

        for row_key in ("row_mean", "row_eos"):
            print(f"Running pre representation: {row_key}", flush=True)
            per_layer: list[dict[str, Any]] = []
            for layer in layers:
                train_groups = collect_pre_groups(
                    log_ids=train_ids,
                    asset_by_log=asset_by_log,
                    structure_dir=config.structure_dir,
                    target=target,
                    layer=layer,
                    row_key=row_key,
                    activation_cache=activation_cache,
                )
                test_groups = collect_pre_groups(
                    log_ids=test_ids,
                    asset_by_log=asset_by_log,
                    structure_dir=config.structure_dir,
                    target=target,
                    layer=layer,
                    row_key=row_key,
                    activation_cache=activation_cache,
                )
                layer_result = _evaluate_target_groups(train_groups, test_groups, seed=config.seed)
                layer_result["layer"] = layer
                per_layer.append(layer_result)
                if (layer + 1) % 8 == 0 or layer == layers[-1]:
                    print(
                        f"  {target} {row_key} layer {layer}: auroc={layer_result.get('auroc')}",
                        flush=True,
                    )
            target_results[row_key] = per_layer

        for position_key in config.downstream_positions:
            rep_key = f"{config.row_key}+{position_key}"
            print(f"Running post representation: {rep_key}", flush=True)
            per_layer = []
            for layer in layers:
                train_groups = collect_concat_groups(
                    log_ids=train_ids,
                    asset_by_log=asset_by_log,
                    structure_dir=config.structure_dir,
                    target=target,
                    layer=layer,
                    row_key=config.row_key,
                    position_key=position_key,
                    activation_cache=activation_cache,
                )
                test_groups = collect_concat_groups(
                    log_ids=test_ids,
                    asset_by_log=asset_by_log,
                    structure_dir=config.structure_dir,
                    target=target,
                    layer=layer,
                    row_key=config.row_key,
                    position_key=position_key,
                    activation_cache=activation_cache,
                )
                layer_result = _evaluate_target_groups(train_groups, test_groups, seed=config.seed)
                layer_result["layer"] = layer
                per_layer.append(layer_result)
                if (layer + 1) % 8 == 0 or layer == layers[-1]:
                    print(
                        f"  {target} {rep_key} layer {layer}: auroc={layer_result.get('auroc')}",
                        flush=True,
                    )
            target_results[rep_key] = per_layer

        results["targets"][target] = target_results
        print(f"Completed target: {target}", flush=True)

    results["summary"] = summarize_probe_results(results)
    print("Summary computed for decision structure analysis", flush=True)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.output_dir / "decision_structure_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote decision structure results to {out_path}", flush=True)
    return results


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze pooled real-decision structure activations")
    p.add_argument("--structure-dir", type=Path, default=Path("data/activations/decision_structure"))
    p.add_argument("--output-dir", type=Path, default=Path("data/analysis_results/decision_structure"))
    p.add_argument("--row-key", choices=["row_mean", "row_eos"], default="row_mean")
    p.add_argument("--layers", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=8)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    parsed_layers: list[int] | None = None
    if args.layers:
        parsed_layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    config = DecisionStructureAnalysisConfig(
        structure_dir=args.structure_dir,
        output_dir=args.output_dir,
        row_key=args.row_key,
        layers=parsed_layers,
        seed=args.seed,
        test_fraction=args.test_fraction,
        num_workers=args.num_workers,
    )
    results = run_decision_structure_analysis(config)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
