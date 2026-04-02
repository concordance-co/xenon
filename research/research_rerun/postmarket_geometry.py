from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from pipelines.db import connect_neon
from research.counterfactual.analysis import preload_all_activations


STATE_KEYS = (
    "market_mean",
    "market_eos",
    "active_settings_eos",
    "portfolio_eos",
    "constraints_eos",
    "last_token",
)

RISK_GROUP = "risk_postmarket_geometry"
AFFORDANCE_GROUP = "affordance_postmarket_geometry"

BASE_CONTEXT_BY_GROUP = {
    RISK_GROUP: "risk_3",
    AFFORDANCE_GROUP: "market_only",
}


@dataclass
class PostMarketGeometryConfig:
    research_activations_dir: Path = Path("data/activations/research_rerun")
    output_dir: Path = Path("data/analysis_results/research_postmarket_geometry")
    experiment_id: str = "real_postmarket_geometry_bridge_v1"
    seed: int = 42
    test_fraction: float = 0.2
    num_workers: int = 8

    @property
    def run_dir(self) -> Path:
        return self.research_activations_dir / self.experiment_id

    @property
    def results_dir(self) -> Path:
        return self.output_dir / self.experiment_id


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    if av.size == 0 or bv.size == 0 or av.size != bv.size:
        return None
    if np.allclose(av, av[0]) or np.allclose(bv, bv[0]):
        return None
    corr = spearmanr(av, bv).correlation
    if corr is None or np.isnan(corr):
        return None
    return float(corr)


def _pairwise_distance_vector(coords: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for left_idx in range(coords.shape[0]):
        for right_idx in range(left_idx + 1, coords.shape[0]):
            values.append(float(np.linalg.norm(coords[left_idx] - coords[right_idx])))
    return np.asarray(values, dtype=np.float32)


def _center_rows(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def _matrix_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float | None:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-12:
        return None
    return float(np.dot(av, bv) / denom)


def _load_prompt_rows(experiment_id: str, experiment_group: str) -> list[dict[str, Any]]:
    with connect_neon() as conn:
        rows = conn.execute(
            """
            SELECT
                p.prompt_id AS capture_id,
                p.base_example_id,
                p.variant,
                p.row_order,
                p.metadata
            FROM research_rerun_prompts p
            WHERE p.experiment_id = %s
              AND p.experiment_group = %s
            ORDER BY p.base_example_id, p.variant
            """,
            [experiment_id, experiment_group],
        ).fetchall()
    parsed = []
    for row in rows:
        record = dict(row)
        metadata = record.get("metadata")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        record["metadata"] = metadata or {}
        parsed.append(record)
    return parsed


def _ordered_contexts(experiment_group: str, prompt_rows: list[dict[str, Any]]) -> list[str]:
    unique = sorted({str(row["variant"]) for row in prompt_rows})
    if experiment_group == RISK_GROUP:
        return sorted(unique, key=lambda text: int(text.split("_", 1)[1]))
    def _affordance_key(text: str) -> tuple[int, int]:
        if text == "market_only":
            return (0, 0)
        return (1, int(text.split("_", 1)[1]))
    return sorted(unique, key=_affordance_key)


def _base_coords_array(metadata: dict[str, Any]) -> np.ndarray:
    selected_symbols = list(metadata.get("selected_symbols") or [])
    coords_by_symbol = metadata.get("base_coords") or {}
    coords = np.asarray([coords_by_symbol[symbol] for symbol in selected_symbols], dtype=np.float32)
    return _center_rows(coords)


def _score_coords_array(metadata: dict[str, Any]) -> np.ndarray:
    selected_symbols = list(metadata.get("selected_symbols") or [])
    coords_by_symbol = metadata.get("score_coords") or {}
    coords = np.asarray([coords_by_symbol[symbol] for symbol in selected_symbols], dtype=np.float32)
    return _center_rows(coords)


def _collect_state_rows(
    *,
    prompt_rows: list[dict[str, Any]],
    activation_cache: dict[str, dict[str, np.ndarray]],
    example_ids: set[str],
    state_key: str,
    layer: int,
    context_variant: str,
    target_kind: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    X_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    row_meta: list[dict[str, Any]] = []
    for row in prompt_rows:
        if str(row["variant"]) != context_variant:
            continue
        if str(row["base_example_id"]) not in example_ids:
            continue
        acts = activation_cache.get(str(row["capture_id"]))
        if not acts or state_key not in acts:
            continue
        metadata = row.get("metadata") or {}
        target = _base_coords_array(metadata) if target_kind == "base" else _score_coords_array(metadata)
        if target.ndim != 2 or target.shape[1] != 2 or target.shape[0] < 2:
            continue
        X_rows.append(acts[state_key][layer].astype(np.float32))
        y_rows.append(target.reshape(-1))
        row_meta.append(
            {
                "base_example_id": str(row["base_example_id"]),
                "context_variant": str(row["variant"]),
                "base_coords": _base_coords_array(metadata),
                "score_coords": _score_coords_array(metadata),
            }
        )
    if not X_rows:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32), []
    return np.asarray(X_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32), row_meta


def _split_example_ids(example_ids: list[str], *, seed: int, test_fraction: float) -> tuple[set[str], set[str]]:
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


def _train_probe(X: np.ndarray, Y: np.ndarray) -> Ridge:
    probe = Ridge(alpha=1.0, fit_intercept=True, solver="svd")
    probe.fit(X, Y)
    return probe


def _per_dim_r2(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    scores: list[float] = []
    for dim in range(y_true.shape[1]):
        true = y_true[:, dim]
        pred = y_pred[:, dim]
        ss_tot = float(np.sum((true - true.mean()) ** 2))
        if ss_tot <= 1e-12:
            continue
        ss_res = float(np.sum((true - pred) ** 2))
        scores.append(float(1.0 - ss_res / ss_tot))
    return scores


def _evaluate_geometry_predictions(
    y_pred: np.ndarray,
    row_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    base_spearmans: list[float] = []
    score_spearmans: list[float] = []
    base_cosines: list[float] = []
    score_cosines: list[float] = []
    step_norms: list[float] = []
    for pred_flat, meta in zip(y_pred, row_meta, strict=False):
        base_coords = np.asarray(meta["base_coords"], dtype=np.float32)
        score_coords = np.asarray(meta["score_coords"], dtype=np.float32)
        if base_coords.ndim != 2 or base_coords.shape[1] != 2 or base_coords.shape[0] < 2:
            continue
        pred = _center_rows(pred_flat.reshape(base_coords.shape[0], base_coords.shape[1]))
        pred_vec = _pairwise_distance_vector(pred)
        base_vec = _pairwise_distance_vector(base_coords)
        score_vec = _pairwise_distance_vector(score_coords)
        base_corr = _safe_spearman(pred_vec, base_vec)
        score_corr = _safe_spearman(pred_vec, score_vec)
        if base_corr is not None:
            base_spearmans.append(base_corr)
        if score_corr is not None:
            score_spearmans.append(score_corr)
        base_cos = _matrix_cosine_similarity(pred_vec, base_vec)
        score_cos = _matrix_cosine_similarity(pred_vec, score_vec)
        if base_cos is not None:
            base_cosines.append(base_cos)
        if score_cos is not None:
            score_cosines.append(score_cos)
        step_norms.append(float(np.linalg.norm(score_vec - base_vec)))
    return {
        "base_distance_spearman_mean": _mean(base_spearmans),
        "score_distance_spearman_mean": _mean(score_spearmans),
        "base_distance_cosine_mean": _mean(base_cosines),
        "score_distance_cosine_mean": _mean(score_cosines),
        "score_over_base_margin": None
        if not score_spearmans or not base_spearmans
        else float(np.mean(np.asarray(score_spearmans, dtype=np.float64)) - np.mean(np.asarray(base_spearmans, dtype=np.float64))),
        "decoded_geometry_step_norm_mean": _mean(step_norms),
    }


def _summarize_best(results: dict[str, Any], metric_key: str) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for container_key, container in results.items():
        best_entry: dict[str, Any] | None = None
        best_value = None
        for state_key, rows in container.items():
            for row in rows:
                value = row.get(metric_key)
                if value is None:
                    continue
                if best_value is None or float(value) > best_value:
                    best_value = float(value)
                    best_entry = {
                        "state_key": state_key,
                        "layer": int(row["layer"]),
                        metric_key: best_value,
                    }
        best[container_key] = best_entry
    return best


def run_postmarket_geometry_analysis(config: PostMarketGeometryConfig) -> dict[str, Any]:
    results: dict[str, Any] = {
        "experiment_id": config.experiment_id,
        "groups": {},
    }

    for experiment_group in (RISK_GROUP, AFFORDANCE_GROUP):
        prompt_rows = _load_prompt_rows(config.experiment_id, experiment_group)
        if not prompt_rows:
            results["groups"][experiment_group] = {"error": "no_prompts"}
            continue

        contexts = _ordered_contexts(experiment_group, prompt_rows)
        base_context = BASE_CONTEXT_BY_GROUP[experiment_group]
        capture_ids = [str(row["capture_id"]) for row in prompt_rows]
        activation_cache = preload_all_activations(config.run_dir, capture_ids, max_workers=config.num_workers)
        if not activation_cache:
            results["groups"][experiment_group] = {"error": "no_activations"}
            continue
        sample_acts = next(iter(activation_cache.values()))
        available_states = [state_key for state_key in STATE_KEYS if state_key in sample_acts]
        layers = list(range(int(sample_acts["last_token"].shape[0])))
        example_ids = sorted({str(row["base_example_id"]) for row in prompt_rows})
        train_ids, test_ids = _split_example_ids(example_ids, seed=config.seed, test_fraction=config.test_fraction)

        group_results: dict[str, Any] = {
            "contexts": contexts,
            "base_context": base_context,
            "n_examples": len(example_ids),
            "layers": layers,
            "state_keys": available_states,
            "coordinate_transfer": {},
            "realignment": {},
            "selected_states": {},
        }

        for context in contexts:
            group_results["coordinate_transfer"][context] = {}
            group_results["realignment"][context] = {}
            for state_key in available_states:
                transfer_rows: list[dict[str, Any]] = []
                realignment_rows: list[dict[str, Any]] = []
                for layer in layers:
                    X_train, Y_train, _ = _collect_state_rows(
                        prompt_rows=prompt_rows,
                        activation_cache=activation_cache,
                        example_ids=train_ids,
                        state_key=state_key,
                        layer=layer,
                        context_variant=base_context,
                        target_kind="base",
                    )
                    X_test, Y_test, row_meta = _collect_state_rows(
                        prompt_rows=prompt_rows,
                        activation_cache=activation_cache,
                        example_ids=test_ids,
                        state_key=state_key,
                        layer=layer,
                        context_variant=context,
                        target_kind="base",
                    )
                    if X_train.size == 0 or X_test.size == 0:
                        transfer_rows.append({"layer": layer, "error": "insufficient_rows"})
                        realignment_rows.append({"layer": layer, "error": "insufficient_rows"})
                        continue
                    probe = _train_probe(X_train, Y_train)
                    y_pred = probe.predict(X_test)
                    dim_scores = _per_dim_r2(Y_test, y_pred)
                    transfer_rows.append(
                        {
                            "layer": layer,
                            "coord_r2_mean": _mean(dim_scores),
                            "coord_r2_dims": dim_scores,
                        }
                    )
                    realignment_rows.append(
                        {
                            "layer": layer,
                            **_evaluate_geometry_predictions(y_pred, row_meta),
                        }
                    )
                group_results["coordinate_transfer"][context][state_key] = transfer_rows
                group_results["realignment"][context][state_key] = realignment_rows

        group_results["summary"] = {
            "best_transfer": _summarize_best(group_results["coordinate_transfer"], "coord_r2_mean"),
            "best_realignment": _summarize_best(group_results["realignment"], "score_over_base_margin"),
        }
        results["groups"][experiment_group] = group_results

    config.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.results_dir / "results.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote post-market geometry analysis to {output_path}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze real DX post-market risk and affordance ladders")
    parser.add_argument("--experiment-id", default="real_postmarket_geometry_bridge_v1")
    parser.add_argument("--research-activations-dir", type=Path, default=Path("data/activations/research_rerun"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis_results/research_postmarket_geometry"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    config = PostMarketGeometryConfig(
        research_activations_dir=args.research_activations_dir,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
        seed=args.seed,
        test_fraction=args.test_fraction,
        num_workers=args.num_workers,
    )
    results = run_postmarket_geometry_analysis(config)
    print(json.dumps(results.get("groups", {}), indent=2))


if __name__ == "__main__":
    main()
