"""Phase 05 confound battery.

Five tests to separate mechanistic signal from surface-text confounds
before committing to further structural claims on the Phase 05 data.

Tests:
  1. Lexical family identity -- can CountVectorizer + LR decode
     strategy_family from raw user_text? (Addresses: "are we just
     reading the prompt?")
  2. Lexical cross-family detection transfer -- baseline for what
     "no surface generalization" looks like across family groupings.
  3. Within-family lexical-holdout detection -- train on v0 strategy
     wording, test on v1 (and the same for settings). Runs both a
     lexical baseline and an activation probe so we can compare.
     (Addresses: is the within-family signal surface or deeper?)
  4. Family-residualized conflict probe -- per-family mean-center the
     activations, then probe conflict_present. Survives or not?
     (Addresses: "0 transferability is suspect -- is anything shared
     once family is removed?")
  5. Regularization sweep on cross-family detection transfer -- vary
     C to check whether the collapse is a regularization artifact
     rather than a real absence of shared structure.

All tests use the Phase 04 capture (16474bceae4e) and the 288-row
workflow_dataset_conflict_probe_v3_v1 view.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "xenon-prompt-confusion-phase5-confound-battery"
DEFAULT_CAPTURE_RUN_ID = "16474bceae4e"
DEFAULT_ACTIVATIONS_SUBDIR = f"workflows/conflict_probe_v3/{DEFAULT_CAPTURE_RUN_ID}"
DEFAULT_OUTPUT_SUBDIR = "prompt_confusion/phase_05/confound_battery"
DEFAULT_BASE_RELATION = "workflow_dataset_conflict_probe_v3_v1"

# Layer selection. Test 3 (lexical holdout) and test 4 (family residualization)
# sweep all captured layers to let us see whether any surviving conflict signal
# builds with depth (consistent with a constructed semantic feature) or is flat
# (consistent with a shallow embedding artifact that survives holdout by
# coincidence). Test 5 (regularization sweep) stays on focus layers to cap
# runtime -- the structure is already well-characterized there.
CAPTURED_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]
FOCUS_LAYERS = [0, 24, 36]

SIZE_FAMILIES = ("trade_size_force_large", "trade_size_force_small")
ACTIVITY_FAMILIES = ("activity_force_trade", "activity_force_observe")
FAMILY_GROUPINGS = {"size": SIZE_FAMILIES, "activity": ACTIVITY_FAMILIES}
FAMILY_LABELS = ACTIVITY_FAMILIES + SIZE_FAMILIES

REG_SWEEP_C = [0.01, 0.1, 1.0, 10.0, 100.0]

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name("xenon-data", create_if_missing=True)
neon_secret = modal.Secret.from_name("xenon-neon")

base_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("numpy", "pyarrow", "psycopg[binary]", "safetensors", "scikit-learn")
    .add_local_python_source("pipelines")
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_rows(base_relation: str) -> list[dict[str, Any]]:
    from pipelines.db import connect_neon, ensure_schema

    sql = f"""
    SELECT
        log_id,
        user_text,
        strategy_family,
        conflict_present,
        matched_pair_id,
        strategy_lexical_split,
        setting_lexical_split
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


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

def _activation_probe(
    X_train: Any, y_train: Any, X_test: Any, y_test: Any, *, C: float = 1.0, seed: int = 42
) -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int64)

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {
            "balanced_accuracy": None,
            "auroc": None,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        }

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed),
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, pred)), 4),
        "auroc": round(float(roc_auc_score(y_test, proba)), 4),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def _lexical_probe(
    texts_train: list[str], y_train: Any, texts_test: list[str], y_test: Any, *, seed: int = 42
) -> dict[str, Any]:
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline

    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {
            "balanced_accuracy": None,
            "auroc": None,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        }

    pipe = make_pipeline(
        CountVectorizer(ngram_range=(1, 2), min_df=1),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed),
    )
    pipe.fit(texts_train, y_train)
    pred = pipe.predict(texts_test)
    proba = pipe.predict_proba(texts_test)[:, 1]
    return {
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, pred)), 4),
        "auroc": round(float(roc_auc_score(y_test, proba)), 4),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def _activation_grouped_cv(
    X: Any, y: Any, groups: Any, *, C: float = 1.0, n_folds: int = 5, seed: int = 42
) -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)

    unique_groups = len(set(groups.tolist()))
    if unique_groups < 2 or len(np.unique(y)) < 2:
        return {
            "balanced_accuracy": None,
            "auroc": None,
            "n_folds": 0,
            "n_examples": int(len(y)),
        }

    actual_folds = max(2, min(n_folds, unique_groups))
    splitter = StratifiedGroupKFold(n_splits=actual_folds, shuffle=True, random_state=seed)

    bal_scores: list[float] = []
    auroc_scores: list[float] = []
    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed),
        )
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[test_idx])
        bal_scores.append(float(balanced_accuracy_score(y[test_idx], pred)))
        if len(np.unique(y[test_idx])) >= 2:
            proba = pipe.predict_proba(X[test_idx])[:, 1]
            auroc_scores.append(float(roc_auc_score(y[test_idx], proba)))

    return {
        "balanced_accuracy": round(float(np.mean(bal_scores)), 4),
        "balanced_accuracy_std": round(float(np.std(bal_scores)), 4),
        "auroc": round(float(np.mean(auroc_scores)), 4) if auroc_scores else None,
        "auroc_std": round(float(np.std(auroc_scores)), 4) if auroc_scores else None,
        "n_folds": int(len(bal_scores)),
        "n_examples": int(len(y)),
        "n_groups": int(unique_groups),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test1_lexical_family_identity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Stratified group k-fold lexical classifier for strategy_family."""
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline

    family_to_idx = {fam: i for i, fam in enumerate(FAMILY_LABELS)}
    texts = [str(r["user_text"]) for r in rows]
    y = np.asarray([family_to_idx[str(r["strategy_family"])] for r in rows], dtype=np.int64)
    groups = np.asarray([str(r["matched_pair_id"]) for r in rows])

    unique_groups = len(set(groups.tolist()))
    n_folds = max(2, min(5, unique_groups))
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)

    bal_scores: list[float] = []
    auroc_scores: list[float] = []
    labels = list(range(len(FAMILY_LABELS)))
    for train_idx, test_idx in splitter.split(texts, y, groups=groups):
        pipe = make_pipeline(
            CountVectorizer(ngram_range=(1, 2), min_df=1),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42),
        )
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        pipe.fit(train_texts, y[train_idx])
        pred = pipe.predict(test_texts)
        bal_scores.append(float(balanced_accuracy_score(y[test_idx], pred)))
        if len(np.unique(y[test_idx])) >= 2:
            proba = pipe.predict_proba(test_texts)
            auroc_scores.append(
                float(roc_auc_score(y[test_idx], proba, multi_class="ovr", labels=labels))
            )

    return {
        "balanced_accuracy": round(float(np.mean(bal_scores)), 4),
        "balanced_accuracy_std": round(float(np.std(bal_scores)), 4),
        "auroc_ovr_macro": round(float(np.mean(auroc_scores)), 4) if auroc_scores else None,
        "auroc_ovr_macro_std": round(float(np.std(auroc_scores)), 4) if auroc_scores else None,
        "chance_baseline": round(1.0 / len(FAMILY_LABELS), 4),
        "auroc_chance": 0.5,
        "n_examples": int(len(y)),
        "n_folds": int(len(bal_scores)),
        "comparison_activation_residual_L36": 0.9893,
    }


def _test2_lexical_cross_family_detection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Lexical baseline for cross-family detection transfer."""
    by_grouping: dict[str, list[dict[str, Any]]] = {name: [] for name in FAMILY_GROUPINGS}
    for row in rows:
        fam = str(row["strategy_family"])
        for name, members in FAMILY_GROUPINGS.items():
            if fam in members:
                by_grouping[name].append(row)
                break

    results: dict[str, dict[str, Any]] = {}
    for train_name, train_rows in by_grouping.items():
        for test_name, test_rows in by_grouping.items():
            if train_name == test_name:
                continue
            texts_train = [str(r["user_text"]) for r in train_rows]
            y_train = [1 if bool(r["conflict_present"]) else 0 for r in train_rows]
            texts_test = [str(r["user_text"]) for r in test_rows]
            y_test = [1 if bool(r["conflict_present"]) else 0 for r in test_rows]
            results[f"{train_name}_to_{test_name}"] = _lexical_probe(
                texts_train, y_train, texts_test, y_test
            )
    return results


def _test3_within_family_lexical_holdout(
    rows: list[dict[str, Any]],
    activations_dir: Path,
    layers: list[int],
) -> dict[str, Any]:
    """Within-family train/test split by strategy_lexical_split and setting_lexical_split.

    Compares a lexical baseline (CountVectorizer+LR on user_text) and an
    activation probe (LogisticRegression on compact residual) to see if
    the within-family conflict signal survives holding out half the
    lexical variants.
    """
    import numpy as np

    by_grouping: dict[str, list[dict[str, Any]]] = {name: [] for name in FAMILY_GROUPINGS}
    for row in rows:
        fam = str(row["strategy_family"])
        for name, members in FAMILY_GROUPINGS.items():
            if fam in members:
                by_grouping[name].append(row)
                break

    out: dict[str, dict[str, Any]] = {}
    for grouping_name, grouping_rows in by_grouping.items():
        out[grouping_name] = {}
        for split_col in ("strategy_lexical_split", "setting_lexical_split"):
            train_rows = [r for r in grouping_rows if str(r[split_col]) == "train"]
            test_rows = [r for r in grouping_rows if str(r[split_col]) == "test"]
            if not train_rows or not test_rows:
                out[grouping_name][split_col] = {"error": "empty split"}
                continue

            # Lexical baseline
            lex_metrics = _lexical_probe(
                [str(r["user_text"]) for r in train_rows],
                [1 if bool(r["conflict_present"]) else 0 for r in train_rows],
                [str(r["user_text"]) for r in test_rows],
                [1 if bool(r["conflict_present"]) else 0 for r in test_rows],
            )

            # Activation probe per layer (residual only; router is noisier and
            # we already have strong signal on residual to test)
            layer_results: list[dict[str, Any]] = []
            for layer in layers:
                features = _load_compact_features(activations_dir, "residual", layer)
                X_train, y_train, X_test, y_test = [], [], [], []
                for r in train_rows:
                    lid = int(r["log_id"])
                    if lid not in features:
                        continue
                    X_train.append(features[lid])
                    y_train.append(1 if bool(r["conflict_present"]) else 0)
                for r in test_rows:
                    lid = int(r["log_id"])
                    if lid not in features:
                        continue
                    X_test.append(features[lid])
                    y_test.append(1 if bool(r["conflict_present"]) else 0)
                if not X_train or not X_test:
                    continue
                metrics = _activation_probe(
                    np.stack(X_train), y_train, np.stack(X_test), y_test
                )
                metrics["layer"] = int(layer)
                layer_results.append(metrics)

            out[grouping_name][split_col] = {
                "lexical_baseline": lex_metrics,
                "activation_probe_residual": layer_results,
            }
    return out


def _test4_family_residualized_conflict(
    rows: list[dict[str, Any]],
    activations_dir: Path,
    layers: list[int],
) -> dict[str, Any]:
    """Project activations onto the family-null subspace, then probe conflict.

    Procedure per (data_source, layer):
      1. Fit a 4-class LogisticRegression on strategy_family (raw features,
         no scaling, so weight directions live directly in activation space).
      2. Treat the classifier's coef_ rows as the family subspace. Build the
         orthogonal projector P = W.T (W W.T)^+ W (pinv for rank safety --
         multinomial coef_ is rank 3, not 4).
      3. Compute X_null = X - X @ P. This is the component of each activation
         that has zero projection onto any family-classifier direction.
      4. Sanity check: a fresh family probe on X_null should collapse to
         near-chance. We report this as family_acc_on_null.
      5. Probe conflict_present on X_null with the same grouped k-fold as the
         raw baseline and compare.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    results: list[dict[str, Any]] = []
    family_to_idx = {fam: i for i, fam in enumerate(FAMILY_LABELS)}

    for data_source in ("residual", "router"):
        for layer in layers:
            features = _load_compact_features(activations_dir, data_source, layer)
            X, y, g, fam_idx = [], [], [], []
            for r in rows:
                lid = int(r["log_id"])
                if lid not in features:
                    continue
                fam = str(r["strategy_family"])
                if fam not in family_to_idx:
                    continue
                X.append(features[lid])
                y.append(1 if bool(r["conflict_present"]) else 0)
                g.append(str(r["matched_pair_id"]))
                fam_idx.append(family_to_idx[fam])
            if not X:
                continue
            X = np.stack(X).astype(np.float32)
            y_arr = np.asarray(y, dtype=np.int64)
            fam_idx_arr = np.asarray(fam_idx, dtype=np.int64)

            # Step 1: fit family classifier on raw features (no scaler).
            family_lr = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=4000, random_state=42,
            )
            family_lr.fit(X, fam_idx_arr)
            W = np.asarray(family_lr.coef_, dtype=np.float64)  # [n_classes, d]
            family_acc_raw = float(
                balanced_accuracy_score(fam_idx_arr, family_lr.predict(X))
            )

            # Steps 2-3: project into orthogonal complement of W's row space.
            WWT = W @ W.T
            WWT_pinv = np.linalg.pinv(WWT)
            P = W.T @ WWT_pinv @ W  # [d, d] projector onto family subspace
            X_null = (X.astype(np.float64) - X.astype(np.float64) @ P).astype(np.float32)

            # Step 4: sanity check -- family should no longer be linearly
            # readable in the null space. Refit rather than reusing W so we are
            # asking "can any linear probe recover family from X_null."
            family_lr_null = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=4000, random_state=42,
            )
            family_lr_null.fit(X_null, fam_idx_arr)
            family_acc_on_null = float(
                balanced_accuracy_score(fam_idx_arr, family_lr_null.predict(X_null))
            )

            # Step 5: probe conflict on raw and null-space activations.
            raw = _activation_grouped_cv(X, y_arr, g)
            null = _activation_grouped_cv(X_null, y_arr, g)

            results.append({
                "data_source": data_source,
                "layer": int(layer),
                "family_subspace_rank": int(np.linalg.matrix_rank(W)),
                "family_accuracy_raw_training_fit": round(family_acc_raw, 4),
                "family_accuracy_on_null_training_fit": round(family_acc_on_null, 4),
                "raw_conflict_probe": raw,
                "family_residualized_conflict_probe": null,
                "conflict_delta_raw_minus_null_balanced_accuracy": (
                    round(raw["balanced_accuracy"] - null["balanced_accuracy"], 4)
                    if raw.get("balanced_accuracy") is not None and null.get("balanced_accuracy") is not None
                    else None
                ),
                "conflict_delta_raw_minus_null_auroc": (
                    round(raw["auroc"] - null["auroc"], 4)
                    if raw.get("auroc") is not None and null.get("auroc") is not None
                    else None
                ),
            })
    return results


def _test5_regularization_sweep(
    rows: list[dict[str, Any]],
    activations_dir: Path,
    layers: list[int],
    c_values: list[float],
) -> dict[str, Any]:
    """Vary C on cross-family detection transfer. Does the collapse change?"""
    import numpy as np

    by_grouping: dict[str, list[dict[str, Any]]] = {name: [] for name in FAMILY_GROUPINGS}
    for row in rows:
        fam = str(row["strategy_family"])
        for name, members in FAMILY_GROUPINGS.items():
            if fam in members:
                by_grouping[name].append(row)
                break

    results: list[dict[str, Any]] = []
    for layer in layers:
        features = _load_compact_features(activations_dir, "residual", layer)
        packed: dict[str, dict[str, Any]] = {}
        for grouping_name, grouping_rows in by_grouping.items():
            X, y = [], []
            for r in grouping_rows:
                lid = int(r["log_id"])
                if lid not in features:
                    continue
                X.append(features[lid])
                y.append(1 if bool(r["conflict_present"]) else 0)
            if not X:
                packed[grouping_name] = {"X": None, "y": None}
                continue
            packed[grouping_name] = {
                "X": np.stack(X).astype(np.float32),
                "y": np.asarray(y, dtype=np.int64),
            }

        for c in c_values:
            for train_name in FAMILY_GROUPINGS:
                for test_name in FAMILY_GROUPINGS:
                    if train_name == test_name:
                        continue
                    tb, te = packed[train_name], packed[test_name]
                    if tb["X"] is None or te["X"] is None:
                        continue
                    metrics = _activation_probe(tb["X"], tb["y"], te["X"], te["y"], C=c)
                    results.append({
                        "layer": int(layer),
                        "C": c,
                        "direction": f"{train_name}_to_{test_name}",
                        **metrics,
                    })
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@app.function(
    volumes={"/data": data_volume},
    image=base_image,
    timeout=3600,
    cpu=6,
    secrets=[neon_secret],
)
def run_confound_battery(
    *,
    activations_subdir: str = DEFAULT_ACTIVATIONS_SUBDIR,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    base_relation: str = DEFAULT_BASE_RELATION,
    layers: list[int] | None = None,
) -> dict[str, Any]:
    activations_dir = Path("/data/activations") / activations_subdir
    output_dir = Path("/data/analysis_results") / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = layers or FOCUS_LAYERS

    rows = _load_rows(base_relation)

    test1 = _test1_lexical_family_identity(rows)
    test2 = _test2_lexical_cross_family_detection(rows)
    # Tests 3 and 4 sweep all captured layers so we can read the layer
    # progression of any surviving conflict signal.
    test3 = _test3_within_family_lexical_holdout(rows, activations_dir, CAPTURED_LAYERS)
    test4 = _test4_family_residualized_conflict(rows, activations_dir, CAPTURED_LAYERS)
    # Test 5 stays on focus layers -- its structure is already clear there.
    test5 = _test5_regularization_sweep(rows, activations_dir, layers, REG_SWEEP_C)

    summary = {
        "capture_run_id": activations_subdir.split("/")[-1],
        "base_relation": base_relation,
        "layers": list(layers),
        "n_examples": len(rows),
        "test1_lexical_family_identity": test1,
        "test2_lexical_cross_family_detection": test2,
        "test3_within_family_lexical_holdout": test3,
        "test4_family_residualized_conflict": test4,
        "test5_regularization_sweep": test5,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


@app.local_entrypoint()
def main() -> None:
    result = run_confound_battery.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
