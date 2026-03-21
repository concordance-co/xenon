"""Analysis for the synthetic policy-algebra dataset."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file

from pipelines.db import connect_neon
from pipelines.interp.counterfactual.analysis import evaluate_probe_per_snapshot, train_probe


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _split_groups(groups: list[str], *, seed: int, test_fraction: float) -> tuple[set[str], set[str]]:
    ordered = list(groups)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_test = max(1, int(len(ordered) * test_fraction))
    return set(ordered[n_test:]), set(ordered[:n_test])


def _load_examples(phase_name: str) -> list[dict[str, Any]]:
    conn = connect_neon()
    try:
        rows = conn.execute(
            """
            SELECT log_id, example_id, family, family_variant, labels_json
            FROM synthetic_market_examples_v0
            WHERE phase_name = %s
            ORDER BY log_id
            """,
            (phase_name,),
        ).fetchall()
    finally:
        conn.close()

    examples: list[dict[str, Any]] = []
    for row in rows:
        labels = row["labels_json"]
        if isinstance(labels, str):
            labels = json.loads(labels)
        examples.append(
            {
                "log_id": int(row["log_id"]),
                "example_id": str(row["example_id"]),
                "family": str(row["family"]),
                "family_variant": str(row["family_variant"]),
                "labels": dict(labels),
            }
        )
    return examples


def _load_asset_rows(phase_name: str) -> dict[int, list[dict[str, Any]]]:
    conn = connect_neon()
    try:
        rows = conn.execute(
            """
            SELECT log_id, row_index, symbol
            FROM synthetic_market_assets_v0
            WHERE phase_name = %s
            ORDER BY log_id, row_index
            """,
            (phase_name,),
        ).fetchall()
    finally:
        conn.close()

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["log_id"]), []).append(
            {
                "row_index": int(row["row_index"]),
                "symbol": str(row["symbol"]),
            }
        )
    return grouped


def _preload_residuals(
    structure_dir: Path,
    log_ids: list[int],
    *,
    max_workers: int,
) -> dict[int, dict[str, np.ndarray]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

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
            if loaded % 25 == 0 or loaded == total:
                print(f"Preloaded synthetic policy residuals: {loaded}/{total}", flush=True)
    return cache


def _train_multiclass_probe(X_train: np.ndarray, y_train: list[str], *, seed: int) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            random_state=seed,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def _collect_asset_probe_data(
    examples: list[dict[str, Any]],
    asset_by_log: dict[int, list[dict[str, Any]]],
    activation_cache: dict[int, dict[str, np.ndarray]],
    *,
    group_filter: set[str],
    row_key: str,
    layer: int,
    label_key: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    snapshot_groups: list[dict[str, Any]] = []

    for example in examples:
        scenario_group = str(example["labels"]["scenario_group"])
        if scenario_group not in group_filter:
            continue
        log_id = int(example["log_id"])
        acts = activation_cache.get(log_id)
        rows = asset_by_log.get(log_id)
        if not acts or not rows:
            continue
        target_symbol = example["labels"].get(label_key)
        if not target_symbol:
            continue

        group_X: list[np.ndarray] = []
        group_y: list[int] = []
        for row in rows:
            key = f"{row_key}_{row['row_index']}"
            if key not in acts:
                continue
            group_X.append(acts[key][layer].astype(np.float32))
            group_y.append(int(row["symbol"] == target_symbol))
        if not group_X or len(set(group_y)) < 2:
            continue

        X_rows.extend(group_X)
        y_rows.extend(group_y)
        snapshot_groups.append(
            {
                "X": np.stack(group_X),
                "y": np.asarray(group_y, dtype=np.int64),
                "snapshot_id": f"{log_id}",
            }
        )

    if not X_rows:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64), []
    return np.stack(X_rows), np.asarray(y_rows, dtype=np.int64), snapshot_groups


def _collect_tick_classification_data(
    examples: list[dict[str, Any]],
    activation_cache: dict[int, dict[str, np.ndarray]],
    *,
    group_filter: set[str],
    section_key: str,
    layer: int,
    label_key: str,
) -> tuple[np.ndarray, list[str], list[int]]:
    X_rows: list[np.ndarray] = []
    y_rows: list[str] = []
    log_ids: list[int] = []
    for example in examples:
        scenario_group = str(example["labels"]["scenario_group"])
        if scenario_group not in group_filter:
            continue
        log_id = int(example["log_id"])
        acts = activation_cache.get(log_id)
        if not acts or section_key not in acts:
            continue
        label = example["labels"].get(label_key)
        if label is None:
            label = "NONE"
        elif label == "":
            label = "NONE"
        X_rows.append(acts[section_key][layer].astype(np.float32))
        y_rows.append(str(label))
        log_ids.append(log_id)
    if not X_rows:
        return np.zeros((0, 0), dtype=np.float32), [], []
    return np.stack(X_rows), y_rows, log_ids


def _evaluate_multiclass(
    probe: Any,
    X_test: np.ndarray,
    y_test: list[str],
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    pred = probe.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "n_rows": int(len(y_test)),
    }


def _policy_invariance_summary(
    *,
    examples: list[dict[str, Any]],
    asset_by_log: dict[int, list[dict[str, Any]]],
    activation_cache: dict[int, dict[str, np.ndarray]],
    train_groups: set[str],
    test_groups: set[str],
    best_market: dict[str, Any],
    best_action: dict[str, Any],
    best_policy_asset: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    # Refit probes on the best discovered layers/states.
    market_row_key = best_market["row_key"]
    market_layer = int(best_market["layer"])
    X_train, y_train, _ = _collect_asset_probe_data(
        examples,
        asset_by_log,
        activation_cache,
        group_filter=train_groups,
        row_key=market_row_key,
        layer=market_layer,
        label_key="market_best_asset",
    )
    market_probe = train_probe(X_train, y_train, seed=seed)

    def _top_symbol_for_log(log_id: int) -> str | None:
        acts = activation_cache.get(log_id)
        rows = asset_by_log.get(log_id)
        if not acts or not rows:
            return None
        scored: list[tuple[float, str]] = []
        for row in rows:
            key = f"{market_row_key}_{row['row_index']}"
            if key not in acts:
                continue
            score = float(market_probe.predict_proba(acts[key][market_layer][None, :])[:, 1][0])
            scored.append((score, row["symbol"]))
        if not scored:
            return None
        scored.sort(reverse=True)
        return scored[0][1]

    by_group: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        group = str(example["labels"]["scenario_group"])
        if group not in test_groups:
            continue
        by_group.setdefault(group, []).append(example)

    permission_ok = 0
    permission_total = 0
    strategy_ok = 0
    strategy_total = 0
    for group, rows in by_group.items():
        family = rows[0]["family"]
        top_symbols = {
            _top_symbol_for_log(int(row["log_id"]))
            for row in rows
        }
        top_symbols.discard(None)
        if family == "permission_grid":
            permission_total += 1
            if len(top_symbols) == 1:
                permission_ok += 1
        elif family == "strategy_override_grid":
            strategy_total += 1
            if len(top_symbols) == 1:
                strategy_ok += 1

    # Refit action/policy-best probes for held-out scenario summary.
    action_section = best_action["section_key"]
    action_layer = int(best_action["layer"])
    X_train_action, y_train_action, _ = _collect_tick_classification_data(
        examples,
        activation_cache,
        group_filter=train_groups,
        section_key=action_section,
        layer=action_layer,
        label_key="expected_action_type",
    )
    action_probe = _train_multiclass_probe(X_train_action, y_train_action, seed=seed)

    policy_section = best_policy_asset["section_key"]
    policy_layer = int(best_policy_asset["layer"])
    X_train_policy, y_train_policy, _ = _collect_tick_classification_data(
        examples,
        activation_cache,
        group_filter=train_groups,
        section_key=policy_section,
        layer=policy_layer,
        label_key="policy_best_asset",
    )
    policy_probe = _train_multiclass_probe(X_train_policy, y_train_policy, seed=seed)

    risk_pair_total = 0
    risk_pair_correct = 0
    for group, rows in by_group.items():
        if rows[0]["family"] != "risk_gate_grid":
            continue
        by_variant = {row["family_variant"]: row for row in rows}
        if not {"low_risk", "high_risk"} <= set(by_variant):
            continue
        low = by_variant["low_risk"]
        high = by_variant["high_risk"]
        low_pred = action_probe.predict(
            activation_cache[int(low["log_id"])][action_section][action_layer][None, :]
        )[0]
        high_pred = action_probe.predict(
            activation_cache[int(high["log_id"])][action_section][action_layer][None, :]
        )[0]
        low_policy = policy_probe.predict(
            activation_cache[int(low["log_id"])][policy_section][policy_layer][None, :]
        )[0]
        high_policy = policy_probe.predict(
            activation_cache[int(high["log_id"])][policy_section][policy_layer][None, :]
        )[0]
        risk_pair_total += 1
        if (
            low_policy == str(low["labels"].get("policy_best_asset") or "NONE")
            and high_policy == str(high["labels"].get("policy_best_asset") or "NONE")
            and low_pred == str(low["labels"].get("expected_action_type") or "NONE")
            and high_pred == str(high["labels"].get("expected_action_type") or "NONE")
        ):
            risk_pair_correct += 1

    return {
        "permission_top_symbol_invariance": (
            float(permission_ok / permission_total) if permission_total else None
        ),
        "strategy_top_symbol_invariance": (
            float(strategy_ok / strategy_total) if strategy_total else None
        ),
        "risk_pair_policy_accuracy": (
            float(risk_pair_correct / risk_pair_total) if risk_pair_total else None
        ),
        "n_permission_groups": permission_total,
        "n_strategy_groups": strategy_total,
        "n_risk_groups": risk_pair_total,
    }


def _aggregate_repeated_invariance(
    *,
    examples: list[dict[str, Any]],
    asset_by_log: dict[int, list[dict[str, Any]]],
    activation_cache: dict[int, dict[str, np.ndarray]],
    scenario_groups: list[str],
    best_market: dict[str, Any],
    best_action: dict[str, Any],
    best_policy_asset: dict[str, Any],
    base_seed: int,
    test_fraction: float,
    repeats: int,
) -> dict[str, Any]:
    metrics = {
        "permission_top_symbol_invariance": [],
        "strategy_top_symbol_invariance": [],
        "risk_pair_policy_accuracy": [],
    }
    group_sizes = {
        "n_permission_groups": [],
        "n_strategy_groups": [],
        "n_risk_groups": [],
    }

    for offset in range(max(1, repeats)):
        seed = base_seed + offset
        train_groups, test_groups = _split_groups(
            scenario_groups,
            seed=seed,
            test_fraction=test_fraction,
        )
        summary = _policy_invariance_summary(
            examples=examples,
            asset_by_log=asset_by_log,
            activation_cache=activation_cache,
            train_groups=train_groups,
            test_groups=test_groups,
            best_market=best_market,
            best_action=best_action,
            best_policy_asset=best_policy_asset,
            seed=seed,
        )
        for key in metrics:
            value = summary.get(key)
            if value is not None:
                metrics[key].append(float(value))
        for key in group_sizes:
            value = summary.get(key)
            if value is not None:
                group_sizes[key].append(int(value))

    aggregated = {
        "repeats": int(max(1, repeats)),
    }
    for key, values in metrics.items():
        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        else:
            aggregated[key] = None
    for key, values in group_sizes.items():
        aggregated[key] = {
            "mean": float(np.mean(values)) if values else None,
            "min": int(np.min(values)) if values else None,
            "max": int(np.max(values)) if values else None,
        }
    return aggregated


@dataclass
class SyntheticPolicyAnalysisConfig:
    phase_name: str = "policy_algebra_v1"
    structure_dir: Path = Path("data/activations/synthetic_structure/policy_algebra_v1")
    output_dir: Path = Path("data/analysis_results/synthetic_policy/policy_algebra_v1")
    row_keys: tuple[str, ...] = ("row_mean", "row_eos")
    section_keys: tuple[str, ...] = (
        "active_settings_eos",
        "portfolio_eos",
        "active_strategies_eos",
        "constraints_eos",
        "last_token",
    )
    seed: int = 42
    test_fraction: float = 0.25
    max_workers: int = 8
    invariance_repeats: int = 16


def run_synthetic_policy_analysis(config: SyntheticPolicyAnalysisConfig) -> dict[str, Any]:
    examples = _load_examples(config.phase_name)
    asset_by_log = _load_asset_rows(config.phase_name)
    log_ids = [int(row["log_id"]) for row in examples]
    activation_cache = _preload_residuals(config.structure_dir, log_ids, max_workers=config.max_workers)
    if not activation_cache:
        raise RuntimeError(f"No pooled residuals found under {config.structure_dir}")

    sample = next(iter(activation_cache.values()))
    layers = list(range(int(sample["last_token"].shape[0])))

    scenario_groups = sorted({str(row["labels"]["scenario_group"]) for row in examples})
    train_groups, test_groups = _split_groups(scenario_groups, seed=config.seed, test_fraction=config.test_fraction)
    print(
        f"Synthetic policy split: train_groups={len(train_groups)} test_groups={len(test_groups)}",
        flush=True,
    )

    market_probe_results: list[dict[str, Any]] = []
    for row_key in config.row_keys:
        for layer in layers:
            X_train, y_train, _ = _collect_asset_probe_data(
                examples,
                asset_by_log,
                activation_cache,
                group_filter=train_groups,
                row_key=row_key,
                layer=layer,
                label_key="market_best_asset",
            )
            _, _, snapshot_groups = _collect_asset_probe_data(
                examples,
                asset_by_log,
                activation_cache,
                group_filter=test_groups,
                row_key=row_key,
                layer=layer,
                label_key="market_best_asset",
            )
            if X_train.size == 0 or not snapshot_groups:
                continue
            probe = train_probe(X_train, y_train, seed=config.seed)
            metrics = evaluate_probe_per_snapshot(probe, snapshot_groups)
            market_probe_results.append(
                {
                    "row_key": row_key,
                    "layer": layer,
                    "auroc": _mean(metrics["auroc"]),
                    "hit_at_1": _mean(metrics["hit_at_1"]),
                    "mrr": _mean(metrics["mrr"]),
                    "balanced_accuracy": _mean(metrics["balanced_accuracy"]),
                    "n_groups": len(metrics["hit_at_1"]),
                }
            )

    tick_results: dict[str, list[dict[str, Any]]] = {
        "expected_action_type": [],
        "policy_best_asset": [],
        "permission_mode": [],
    }
    for target in tick_results:
        for section_key in config.section_keys:
            for layer in layers:
                X_train, y_train, _ = _collect_tick_classification_data(
                    examples,
                    activation_cache,
                    group_filter=train_groups,
                    section_key=section_key,
                    layer=layer,
                    label_key=target,
                )
                X_test, y_test, _ = _collect_tick_classification_data(
                    examples,
                    activation_cache,
                    group_filter=test_groups,
                    section_key=section_key,
                    layer=layer,
                    label_key=target,
                )
                if X_train.size == 0 or X_test.size == 0:
                    continue
                probe = _train_multiclass_probe(X_train, y_train, seed=config.seed)
                metrics = _evaluate_multiclass(probe, X_test, y_test)
                tick_results[target].append(
                    {
                        "section_key": section_key,
                        "layer": layer,
                        **metrics,
                    }
                )

    best_market = max(market_probe_results, key=lambda row: (row.get("hit_at_1") or -1.0, row.get("auroc") or -1.0))
    best_action = max(tick_results["expected_action_type"], key=lambda row: row.get("accuracy") or -1.0)
    best_policy_asset = max(tick_results["policy_best_asset"], key=lambda row: row.get("accuracy") or -1.0)
    best_permission = max(tick_results["permission_mode"], key=lambda row: row.get("accuracy") or -1.0)

    invariance = _policy_invariance_summary(
        examples=examples,
        asset_by_log=asset_by_log,
        activation_cache=activation_cache,
        train_groups=train_groups,
        test_groups=test_groups,
        best_market=best_market,
        best_action=best_action,
        best_policy_asset=best_policy_asset,
        seed=config.seed,
    )
    repeated_invariance = _aggregate_repeated_invariance(
        examples=examples,
        asset_by_log=asset_by_log,
        activation_cache=activation_cache,
        scenario_groups=scenario_groups,
        best_market=best_market,
        best_action=best_action,
        best_policy_asset=best_policy_asset,
        base_seed=config.seed,
        test_fraction=config.test_fraction,
        repeats=config.invariance_repeats,
    )

    summary = {
        "market_best_asset_probe": best_market,
        "expected_action_type_classifier": best_action,
        "policy_best_asset_classifier": best_policy_asset,
        "permission_mode_classifier": best_permission,
        "invariance": invariance,
        "repeated_invariance": repeated_invariance,
    }

    result = {
        "phase_name": config.phase_name,
        "n_examples": len(examples),
        "layers": layers,
        "summary": summary,
        "market_best_asset_probe": market_probe_results,
        "tick_classification": tick_results,
    }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run synthetic policy-algebra analysis")
    parser.add_argument("--phase-name", default="policy_algebra_v1")
    parser.add_argument("--structure-dir", type=Path, default=Path("data/activations/synthetic_structure/policy_algebra_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis_results/synthetic_policy/policy_algebra_v1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args(argv)

    run_synthetic_policy_analysis(
        SyntheticPolicyAnalysisConfig(
            phase_name=args.phase_name,
            structure_dir=args.structure_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            test_fraction=args.test_fraction,
            max_workers=args.max_workers,
        )
    )


if __name__ == "__main__":
    main()
