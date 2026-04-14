"""Phase 05 Direction 1: Cross-family probe transfer.

Tests whether (1a) the aligned-vs-conflict detection signal and (1b) the
strategy-vs-setting arbitration signal transfer across family groupings.

For each (data_source, layer, direction, sub_question):
  - within-family baseline: train+test within the family grouping (grouped k-fold)
  - cross-family transfer:  train on one grouping, test on the other
  - transfer delta:         cross minus within
  - cosine similarity of probe weight vectors between the two family groupings

Usage (local wrapper):
    uv run --extra interp --extra modal modal run \\
        projects/DX_TERMINAL/prompt_confusion/phase_05/scripts/cross_family_transfer.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "xenon-prompt-confusion-phase5-cross-family-transfer"

DEFAULT_CAPTURE_RUN_ID = "16474bceae4e"
DEFAULT_ACTIVATIONS_SUBDIR = f"workflows/conflict_probe_v3/{DEFAULT_CAPTURE_RUN_ID}"
DEFAULT_OUTPUT_SUBDIR = "prompt_confusion/phase_05/cross_family_transfer"
DEFAULT_BASE_RELATION = "workflow_dataset_conflict_probe_v3_v1"
DEFAULT_CONFLICT_READOUT_RELATION = "workflow_dataset_conflict_probe_v3_conflict_readout_side_v1"

CAPTURED_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]

SIZE_FAMILIES = ("trade_size_force_large", "trade_size_force_small")
ACTIVITY_FAMILIES = ("activity_force_trade", "activity_force_observe")
FAMILY_GROUPINGS = {
    "size": SIZE_FAMILIES,
    "activity": ACTIVITY_FAMILIES,
}

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name("xenon-data", create_if_missing=True)
neon_secret = modal.Secret.from_name("xenon-neon")

base_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("numpy", "pyarrow", "psycopg[binary]", "safetensors", "scikit-learn")
    .add_local_python_source("pipelines")
)


def _load_detection_rows(base_relation: str) -> list[dict[str, Any]]:
    from pipelines.db import connect_neon, ensure_schema

    sql = f"""
    SELECT
        log_id,
        strategy_family,
        conflict_present,
        matched_pair_id
    FROM {base_relation}
    ORDER BY log_id
    """
    with connect_neon(autocommit=True) as conn:
        ensure_schema(conn)
        rows = conn.execute(sql).fetchall()
    return [dict(row) for row in rows]


def _load_arbitration_rows(relation: str, base_relation: str) -> list[dict[str, Any]]:
    from pipelines.db import connect_neon, ensure_schema

    sql = f"""
    SELECT
        v.log_id,
        v.workflow_label,
        v.arbitration_group_id,
        d.strategy_family,
        d.matched_pair_id
    FROM {relation} v
    JOIN {base_relation} d USING (log_id)
    ORDER BY v.log_id
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
    """Return dict {log_id -> feature_vector} for one layer/source."""
    from safetensors import safe_open

    path = activations_dir / "compact" / f"{data_source}_prompt_eos_layer{layer}.safetensors"
    with safe_open(str(path), framework="numpy") as f:
        features = f.get_tensor("features")
        log_ids = f.get_tensor("log_ids")
    return {int(lid): features[i] for i, lid in enumerate(log_ids)}


def _train_probe(X: Any, y: Any, *, seed: int = 42) -> Any:
    """Train a balanced LogisticRegression probe and return the fitted pipeline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

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
    pipe.fit(X, y)
    return pipe


def _within_family_cv(X: Any, y: Any, groups: Any, *, n_folds: int = 5, seed: int = 42) -> dict[str, Any]:
    """Stratified group k-fold balanced accuracy within a single family grouping."""
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedGroupKFold

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)

    unique_groups = len(set(groups.tolist()))
    if unique_groups < 2 or len(np.unique(y)) < 2:
        return {"balanced_accuracy": None, "n_folds": 0, "n_examples": int(len(y))}

    actual_folds = max(2, min(n_folds, unique_groups))
    splitter = StratifiedGroupKFold(n_splits=actual_folds, shuffle=True, random_state=seed)

    scores: list[float] = []
    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        pipe = _train_probe(X[train_idx], y[train_idx], seed=seed)
        pred = pipe.predict(X[test_idx])
        scores.append(float(balanced_accuracy_score(y[test_idx], pred)))

    return {
        "balanced_accuracy": round(float(np.mean(scores)), 4),
        "balanced_accuracy_std": round(float(np.std(scores)), 4),
        "n_folds": int(len(scores)),
        "n_examples": int(len(y)),
    }


def _cross_family_transfer(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    *,
    seed: int = 42,
) -> tuple[dict[str, Any], Any]:
    """Train on X_train, evaluate on X_test. Return (metrics, fitted_pipeline)."""
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int64)

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return (
            {"balanced_accuracy": None, "n_train": int(len(y_train)), "n_test": int(len(y_test))},
            None,
        )

    pipe = _train_probe(X_train, y_train, seed=seed)
    pred = pipe.predict(X_test)
    return (
        {
            "balanced_accuracy": round(float(balanced_accuracy_score(y_test, pred)), 4),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        },
        pipe,
    )


def _extract_probe_direction(pipe: Any) -> Any:
    """Extract the learned weight vector in standardized-feature space."""
    import numpy as np

    if pipe is None:
        return None
    clf = pipe.named_steps["logisticregression"]
    coef = np.asarray(clf.coef_, dtype=np.float32)
    return coef[0] if coef.shape[0] == 1 else coef.mean(axis=0)


def _cosine(u: Any, v: Any) -> float | None:
    import numpy as np

    if u is None or v is None:
        return None
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return None
    return round(float(np.dot(u, v) / (nu * nv)), 4)


def _run_one_direction(
    *,
    rows: list[dict[str, Any]],
    label_fn: Any,
    group_col: str,
    activations_dir: Path,
    data_sources: tuple[str, ...],
    layers: list[int],
    seed: int,
) -> list[dict[str, Any]]:
    """Run the full within/cross/cosine battery for one sub-question (1a or 1b)."""
    import numpy as np

    # Partition rows by family grouping
    rows_by_grouping: dict[str, list[dict[str, Any]]] = {name: [] for name in FAMILY_GROUPINGS}
    for row in rows:
        fam = str(row["strategy_family"])
        for grouping_name, members in FAMILY_GROUPINGS.items():
            if fam in members:
                rows_by_grouping[grouping_name].append(row)
                break

    results: list[dict[str, Any]] = []

    for data_source in data_sources:
        for layer in layers:
            features = _load_compact_features(activations_dir, data_source, layer)

            # Build per-grouping X, y, groups
            packed: dict[str, dict[str, Any]] = {}
            for grouping_name, grouping_rows in rows_by_grouping.items():
                X, y, g = [], [], []
                for row in grouping_rows:
                    log_id = int(row["log_id"])
                    if log_id not in features:
                        continue
                    X.append(features[log_id])
                    y.append(int(label_fn(row)))
                    g.append(str(row[group_col]))
                if not X:
                    packed[grouping_name] = {"X": None, "y": None, "g": None}
                    continue
                packed[grouping_name] = {
                    "X": np.stack(X),
                    "y": np.asarray(y, dtype=np.int64),
                    "g": np.asarray(g),
                }

            # Within-family baselines + full-family probe for cosine comparison
            baselines: dict[str, dict[str, Any]] = {}
            full_directions: dict[str, Any] = {}
            for grouping_name, bundle in packed.items():
                if bundle["X"] is None:
                    baselines[grouping_name] = {"balanced_accuracy": None}
                    full_directions[grouping_name] = None
                    continue
                baselines[grouping_name] = _within_family_cv(
                    bundle["X"], bundle["y"], bundle["g"], seed=seed
                )
                # Fit once on the full grouping for direction extraction
                if len(np.unique(bundle["y"])) >= 2:
                    full_pipe = _train_probe(bundle["X"], bundle["y"], seed=seed)
                    full_directions[grouping_name] = _extract_probe_direction(full_pipe)
                else:
                    full_directions[grouping_name] = None

            # Cross-family transfers
            transfers: dict[str, dict[str, Any]] = {}
            for train_name in FAMILY_GROUPINGS:
                for test_name in FAMILY_GROUPINGS:
                    if train_name == test_name:
                        continue
                    train_bundle = packed[train_name]
                    test_bundle = packed[test_name]
                    if train_bundle["X"] is None or test_bundle["X"] is None:
                        transfers[f"{train_name}_to_{test_name}"] = {"balanced_accuracy": None}
                        continue
                    metrics, _ = _cross_family_transfer(
                        train_bundle["X"], train_bundle["y"],
                        test_bundle["X"], test_bundle["y"],
                        seed=seed,
                    )
                    # Transfer delta: cross-family minus the test-side within-family baseline
                    within_on_test_side = baselines[test_name].get("balanced_accuracy")
                    delta: float | None = None
                    if metrics["balanced_accuracy"] is not None and within_on_test_side is not None:
                        delta = round(float(metrics["balanced_accuracy"]) - float(within_on_test_side), 4)
                    metrics["transfer_delta_vs_test_within"] = delta
                    transfers[f"{train_name}_to_{test_name}"] = metrics

            cosine_size_activity = _cosine(
                full_directions.get("size"), full_directions.get("activity")
            )

            results.append({
                "data_source": data_source,
                "layer": int(layer),
                "within_family_baseline": baselines,
                "cross_family_transfer": transfers,
                "cosine_similarity_size_vs_activity": cosine_size_activity,
            })

    return results


@app.function(
    volumes={"/data": data_volume},
    image=base_image,
    timeout=3600,
    cpu=6,
    secrets=[neon_secret],
)
def run_cross_family_transfer(
    *,
    activations_subdir: str = DEFAULT_ACTIVATIONS_SUBDIR,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    base_relation: str = DEFAULT_BASE_RELATION,
    arbitration_relation: str = DEFAULT_CONFLICT_READOUT_RELATION,
    layers: list[int] | None = None,
    data_sources: list[str] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    activations_dir = Path("/data/activations") / activations_subdir
    output_dir = Path("/data/analysis_results") / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = layers or CAPTURED_LAYERS
    data_sources = tuple(data_sources or ("residual", "router"))

    # Direction 1a: detection transfer (label = conflict_present, group = matched_pair_id)
    detection_rows = _load_detection_rows(base_relation)
    detection_results = _run_one_direction(
        rows=detection_rows,
        label_fn=lambda r: 1 if bool(r["conflict_present"]) else 0,
        group_col="matched_pair_id",
        activations_dir=activations_dir,
        data_sources=data_sources,
        layers=layers,
        seed=seed,
    )

    # Direction 1b: arbitration transfer (label = workflow_label, group = arbitration_group_id)
    arbitration_rows = _load_arbitration_rows(arbitration_relation, base_relation)
    arbitration_results = _run_one_direction(
        rows=arbitration_rows,
        label_fn=lambda r: 1 if str(r["workflow_label"]) == "setting" else 0,
        group_col="arbitration_group_id",
        activations_dir=activations_dir,
        data_sources=data_sources,
        layers=layers,
        seed=seed,
    )

    summary = {
        "capture_run_id": activations_subdir.split("/")[-1],
        "base_relation": base_relation,
        "arbitration_relation": arbitration_relation,
        "layers": list(layers),
        "data_sources": list(data_sources),
        "n_detection_rows": len(detection_rows),
        "n_arbitration_rows": len(arbitration_rows),
        "direction_1a_detection": detection_results,
        "direction_1b_arbitration": arbitration_results,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


@app.local_entrypoint()
def main() -> None:
    result = run_cross_family_transfer.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
