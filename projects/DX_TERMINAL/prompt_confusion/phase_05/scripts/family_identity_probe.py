"""Phase 05 Direction 2b: Layer-wise family-identity probe.

Fits a 4-class linear probe (strategy_family as label) at each captured
layer, on both residual stream and router logits. Accuracy over depth
identifies where family identity becomes linearly readable in each
data source -- the "branching depth."

If family saturates at different layers in router vs residual, that
distinguishes routing-driven branching from content-driven branching.

Usage:
    uv run --extra interp --extra modal modal run \\
        projects/DX_TERMINAL/prompt_confusion/phase_05/scripts/family_identity_probe.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "xenon-prompt-confusion-phase5-family-identity-probe"

DEFAULT_CAPTURE_RUN_ID = "16474bceae4e"
DEFAULT_ACTIVATIONS_SUBDIR = f"workflows/conflict_probe_v3/{DEFAULT_CAPTURE_RUN_ID}"
DEFAULT_OUTPUT_SUBDIR = "prompt_confusion/phase_05/family_identity_probe"
DEFAULT_BASE_RELATION = "workflow_dataset_conflict_probe_v3_v1"

CAPTURED_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]
FAMILY_LABELS = (
    "activity_force_observe",
    "activity_force_trade",
    "trade_size_force_large",
    "trade_size_force_small",
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name("xenon-data", create_if_missing=True)
neon_secret = modal.Secret.from_name("xenon-neon")

base_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("numpy", "pyarrow", "psycopg[binary]", "safetensors", "scikit-learn")
    .add_local_python_source("pipelines")
)


def _load_rows(base_relation: str) -> list[dict[str, Any]]:
    from pipelines.db import connect_neon, ensure_schema

    sql = f"""
    SELECT
        log_id,
        strategy_family,
        matched_pair_id
    FROM {base_relation}
    ORDER BY log_id
    """
    with connect_neon(autocommit=True) as conn:
        ensure_schema(conn)
        rows = conn.execute(sql).fetchall()
    return [dict(row) for row in rows]


def _load_compact_features(
    activations_dir: Path,
    data_source: str,
    layer: int,
) -> dict[int, Any]:
    from safetensors import safe_open

    path = activations_dir / "compact" / f"{data_source}_prompt_eos_layer{layer}.safetensors"
    with safe_open(str(path), framework="numpy") as f:
        features = f.get_tensor("features")
        log_ids = f.get_tensor("log_ids")
    return {int(lid): features[i] for i, lid in enumerate(log_ids)}


def _probe_layer(
    X: Any,
    y: Any,
    groups: Any,
    *,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)

    unique_groups = len(set(groups.tolist()))
    if unique_groups < 2 or len(np.unique(y)) < 2:
        return {"balanced_accuracy": None, "n_folds": 0, "n_examples": int(len(y))}

    actual_folds = max(2, min(n_folds, unique_groups))
    splitter = StratifiedGroupKFold(n_splits=actual_folds, shuffle=True, random_state=seed)

    accs: list[float] = []
    bals: list[float] = []
    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
            ),
        )
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[test_idx])
        accs.append(float(accuracy_score(y[test_idx], pred)))
        bals.append(float(balanced_accuracy_score(y[test_idx], pred)))

    return {
        "accuracy_mean": round(float(np.mean(accs)), 4),
        "accuracy_std": round(float(np.std(accs)), 4),
        "balanced_accuracy": round(float(np.mean(bals)), 4),
        "balanced_accuracy_std": round(float(np.std(bals)), 4),
        "chance_baseline": round(1.0 / len(FAMILY_LABELS), 4),
        "n_folds": int(len(bals)),
        "n_examples": int(len(y)),
        "n_groups": int(unique_groups),
    }


@app.function(
    volumes={"/data": data_volume},
    image=base_image,
    timeout=1800,
    cpu=6,
    secrets=[neon_secret],
)
def run_family_identity_probe(
    *,
    activations_subdir: str = DEFAULT_ACTIVATIONS_SUBDIR,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    base_relation: str = DEFAULT_BASE_RELATION,
    layers: list[int] | None = None,
    data_sources: list[str] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    import numpy as np

    activations_dir = Path("/data/activations") / activations_subdir
    output_dir = Path("/data/analysis_results") / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = layers or CAPTURED_LAYERS
    data_sources = tuple(data_sources or ("residual", "router"))

    rows = _load_rows(base_relation)
    family_to_idx = {fam: i for i, fam in enumerate(FAMILY_LABELS)}

    results: list[dict[str, Any]] = []

    for data_source in data_sources:
        for layer in layers:
            features = _load_compact_features(activations_dir, data_source, layer)
            X, y, g = [], [], []
            for row in rows:
                log_id = int(row["log_id"])
                if log_id not in features:
                    continue
                fam = str(row["strategy_family"])
                if fam not in family_to_idx:
                    continue
                X.append(features[log_id])
                y.append(family_to_idx[fam])
                g.append(str(row["matched_pair_id"]))
            if not X:
                continue

            metrics = _probe_layer(np.stack(X), y, g, seed=seed)
            results.append({
                "data_source": data_source,
                "layer": int(layer),
                **metrics,
            })

    # Find saturation point per data source (first layer within 95% of the max)
    saturation: dict[str, dict[str, Any]] = {}
    for data_source in data_sources:
        layer_results = [r for r in results if r["data_source"] == data_source and r.get("balanced_accuracy") is not None]
        if not layer_results:
            continue
        max_acc = max(r["balanced_accuracy"] for r in layer_results)
        threshold = max_acc * 0.95
        saturation_layer = next(
            (r["layer"] for r in sorted(layer_results, key=lambda r: r["layer"])
             if r["balanced_accuracy"] >= threshold),
            None,
        )
        saturation[data_source] = {
            "max_balanced_accuracy": max_acc,
            "saturation_layer_95pct": saturation_layer,
        }

    summary = {
        "capture_run_id": activations_subdir.split("/")[-1],
        "base_relation": base_relation,
        "layers": list(layers),
        "data_sources": list(data_sources),
        "family_labels": list(FAMILY_LABELS),
        "n_examples": len(rows),
        "per_layer_results": results,
        "saturation_by_source": saturation,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


@app.local_entrypoint()
def main() -> None:
    result = run_family_identity_probe.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
