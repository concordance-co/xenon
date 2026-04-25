#!/usr/bin/env python3
"""Probe response-label candidates on full-public response replay captures."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from pipelines_v2.api import ModalVolumeStore, PostgresCatalog, PostgresSource, TransferPolicy
from pipelines_v2.storage.artifacts import CaptureArtifact, artifact_from_manifest


DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
ARTIFACT_STORE_NAME = "xenon-data"
CAPTURE_ROOT = "/data/artifacts/morebench_phase_03_experiment03_response_labels"
DEFAULT_CAPTURE_ARTIFACT_ID = "capture_1_f2a9e4531dec"
GENERATION_RESULT = Path("artifacts/_modal_cache/generation_run_1_d6e12a467208/result.json")
RUBRIC_SCORES = Path(
    "projects/MOREBENCH/phase_03/reports/experiment_03_full_public_rubric_judge/manual_scores_merged.jsonl"
)
REPORT_DIR = Path("projects/MOREBENCH/phase_03/reports/experiment_03_response_label_probe")

LAYERS = (0, 4, 8, 16, 28, 36, 40, 44)
LABELS = ("helpful_harmless_off_diagonal", "strong_helpful", "strong_harmless")
GENERATED_VIEWS = ("generated_first_third", "generated_middle_third", "generated_last_third", "generated_total")
ALL_VIEWS = ("prompt_end", *GENERATED_VIEWS)
MIN_HOLDOUT_TEST_N = 25
RANDOM_STATE = 13


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _catalog() -> PostgresCatalog:
    return PostgresCatalog(source=PostgresSource.from_env(DB_ENV_VAR))


def _store() -> ModalVolumeStore:
    return ModalVolumeStore(
        name=ARTIFACT_STORE_NAME,
        root=CAPTURE_ROOT,
        transfer_policy=TransferPolicy(allow_large_transfer=True),
    )


def _load_capture_artifact(capture_artifact_id: str) -> CaptureArtifact:
    manifest = _catalog().load_artifact(capture_artifact_id)
    if manifest is None:
        raise RuntimeError(f"Could not load artifact manifest {capture_artifact_id!r}")
    artifact = artifact_from_manifest(manifest, store=_store())
    if not isinstance(artifact, CaptureArtifact):
        raise TypeError(f"Artifact {capture_artifact_id!r} is not a capture artifact")
    return artifact


def _load_rows() -> list[dict[str, Any]]:
    generation = json.loads(GENERATION_RESULT.read_text(encoding="utf-8"))
    rubric_by_id = {row["row_id"]: row for row in _read_jsonl(RUBRIC_SCORES)}
    rows: list[dict[str, Any]] = []
    for item in generation["rows"]:
        example = dict(item.get("example") or {})
        labels = dict(example.get("labels") or {})
        row_id = str(item.get("example_key") or example.get("key") or "")
        rubric = rubric_by_id.get(row_id)
        if not row_id or rubric is None:
            continue
        helpful_score = int(rubric["helpful_score"])
        harmless_score = int(rubric["harmless_score"])
        off_diagonal = ""
        if helpful_score == 3 and harmless_score < 3:
            off_diagonal = "helpful_over_harmless"
        elif helpful_score < 3 and harmless_score == 3:
            off_diagonal = "harmless_over_helpful"
        rows.append(
            {
                "key": row_id,
                "source_family": str(labels.get("source_family") or ""),
                "context": str(labels.get("context") or ""),
                "strong_helpful": "true" if helpful_score == 3 else "false",
                "strong_harmless": "true" if harmless_score == 3 else "false",
                "helpful_harmless_off_diagonal": off_diagonal,
                "helpful_score": helpful_score,
                "harmless_score": harmless_score,
            }
        )
    return rows


def _pool_generated(values: np.ndarray, view: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        return np.asarray(array, dtype=np.float32).reshape(-1)
    n_tokens = int(array.shape[0])
    if n_tokens <= 0:
        raise ValueError("empty generated sequence")
    if view == "generated_total":
        return array.mean(axis=0)
    first = 0
    second = max(1, n_tokens // 3)
    third = max(second + 1, (2 * n_tokens) // 3) if n_tokens >= 3 else n_tokens
    if view == "generated_first_third":
        return array[first:second].mean(axis=0)
    if view == "generated_middle_third":
        return array[second:third].mean(axis=0) if second < third else array.mean(axis=0)
    if view == "generated_last_third":
        return array[third:n_tokens].mean(axis=0) if third < n_tokens else array[-1]
    raise ValueError(view)


def _pool_prompt(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 2:
        return array[-1]
    return array.reshape(-1)


def _load_prompt_vectors(capture: CaptureArtifact) -> dict[int, dict[str, np.ndarray]]:
    payload = capture.feature("prompt_end_residual").load()
    vectors: dict[int, dict[str, np.ndarray]] = {}
    for layer in LAYERS:
        layer_payload = payload["layers"][str(layer)]
        layer_vectors: dict[str, np.ndarray] = {}
        for key, record in layer_payload.items():
            values = np.asarray(record["values"], dtype=np.float32)
            layer_vectors[str(key)] = _pool_prompt(values)
        vectors[layer] = layer_vectors
    return vectors


def _load_generated_vectors(capture: CaptureArtifact) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    payload = capture.feature("generated_sequence_residual").load()
    vectors_by_view: dict[str, dict[int, dict[str, np.ndarray]]] = {
        view: {} for view in GENERATED_VIEWS
    }
    for layer in LAYERS:
        layer_payload = payload["layers"][str(layer)]
        for view in GENERATED_VIEWS:
            vectors_by_view[view][layer] = {}
        for key, record in layer_payload.items():
            values = np.asarray(record["values"], dtype=np.float32)
            for view in GENERATED_VIEWS:
                vectors_by_view[view][layer][str(key)] = _pool_generated(values, view)
    return vectors_by_view


def _supported_rows(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get(label)]


def _make_xy(rows: list[dict[str, Any]], vectors: dict[str, np.ndarray], label: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    selected = [row for row in _supported_rows(rows, label) if row["key"] in vectors]
    encoder = LabelEncoder()
    y = encoder.fit_transform([str(row[label]) for row in selected])
    x = np.stack([vectors[row["key"]] for row in selected], axis=0).astype(np.float32)
    groups = [str(row["source_family"]) for row in selected]
    return x, y, groups


def _fit_scores(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE),
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    scores = model.predict_proba(x_test)[:, 1]
    return pred, scores


def _score(y_true: np.ndarray, pred: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    out: dict[str, float | None] = {"balanced_accuracy": float(balanced_accuracy_score(y_true, pred))}
    try:
        out["auroc"] = float(roc_auc_score(y_true, scores))
    except ValueError:
        out["auroc"] = None
    return out


def _cv_eval(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    counts = Counter(y.tolist())
    n_splits = min(5, min(counts.values()))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    pred = np.zeros_like(y)
    scores = np.zeros(len(y), dtype=np.float32)
    for train_idx, test_idx in splitter.split(x, y):
        fold_pred, fold_scores = _fit_scores(x[train_idx], y[train_idx], x[test_idx])
        pred[test_idx] = fold_pred
        scores[test_idx] = fold_scores
    return {"n_splits": int(n_splits), **_score(y, pred, scores)}


def _source_holdout_eval(x: np.ndarray, y: np.ndarray, groups: list[str]) -> dict[str, Any] | None:
    folds = []
    group_array = np.asarray(groups)
    for group in sorted(set(groups)):
        test_idx = np.where(group_array == group)[0]
        train_idx = np.where(group_array != group)[0]
        if len(test_idx) < MIN_HOLDOUT_TEST_N:
            continue
        if len(set(y[test_idx].tolist())) < 2 or len(set(y[train_idx].tolist())) < 2:
            continue
        pred, scores = _fit_scores(x[train_idx], y[train_idx], x[test_idx])
        folds.append({"heldout": group, "n_test": int(len(test_idx)), **_score(y[test_idx], pred, scores)})
    if not folds:
        return None
    valid_aurocs = [fold["auroc"] for fold in folds if fold["auroc"] is not None]
    return {
        "n_splits": len(folds),
        "balanced_accuracy": float(np.mean([fold["balanced_accuracy"] for fold in folds])),
        "auroc": float(np.mean(valid_aurocs)) if valid_aurocs else None,
        "folds": folds,
    }


def run_analysis(capture_artifact_id: str = DEFAULT_CAPTURE_ARTIFACT_ID) -> dict[str, Any]:
    capture = _load_capture_artifact(capture_artifact_id)
    rows = _load_rows()
    label_counts = {
        label: dict(Counter(str(row[label]) for row in _supported_rows(rows, label)))
        for label in LABELS
    }
    results: dict[str, Any] = {
        "capture_artifact_id": capture_artifact_id,
        "layers": list(LAYERS),
        "views": list(ALL_VIEWS),
        "labels": list(LABELS),
        "label_counts": label_counts,
        "metrics": {},
    }

    prompt_vectors = _load_prompt_vectors(capture)
    _evaluate_view(results, rows, "prompt_end", prompt_vectors)
    del prompt_vectors

    generated_vectors_by_view = _load_generated_vectors(capture)
    for view in GENERATED_VIEWS:
        _evaluate_view(results, rows, view, generated_vectors_by_view[view])
    del generated_vectors_by_view

    results["best"] = _best_results(results)
    return results


def _evaluate_view(
    results: dict[str, Any],
    rows: list[dict[str, Any]],
    view: str,
    vectors_by_layer: dict[int, dict[str, np.ndarray]],
) -> None:
    results["metrics"][view] = {}
    for label in LABELS:
        results["metrics"][view][label] = {}
        for layer in LAYERS:
            x, y, groups = _make_xy(rows, vectors_by_layer[layer], label)
            cv = _cv_eval(x, y)
            holdout = _source_holdout_eval(x, y, groups)
            results["metrics"][view][label][str(layer)] = {
                "n": int(len(y)),
                "class_counts": dict(Counter(y.tolist())),
                "cv": cv,
                "source_family_holdout": holdout,
            }


def _primary_score(record: dict[str, Any]) -> float:
    holdout = record.get("source_family_holdout")
    if holdout and holdout.get("balanced_accuracy") is not None:
        return float(holdout["balanced_accuracy"])
    return float(record["cv"]["balanced_accuracy"])


def _best_results(results: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for label in LABELS:
        candidates = []
        for view in ALL_VIEWS:
            for layer, record in results["metrics"][view][label].items():
                candidates.append((view, layer, _primary_score(record), record))
        view, layer, score, record = max(candidates, key=lambda item: item[2])
        best[label] = {
            "view": view,
            "layer": int(layer),
            "primary_balanced_accuracy": score,
            "cv_balanced_accuracy": record["cv"]["balanced_accuracy"],
            "cv_auroc": record["cv"]["auroc"],
            "source_family_holdout": record["source_family_holdout"],
        }
    return best


def write_report(summary: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Experiment 03 Response Label Probe",
        "",
        f"- capture artifact: `{summary['capture_artifact_id']}`",
        f"- labels: `{', '.join(summary['labels'])}`",
        f"- views: `{', '.join(summary['views'])}`",
        f"- layers: `{', '.join(str(layer) for layer in summary['layers'])}`",
        "",
        "## Best Results",
        "",
        "| label | best view | layer | holdout BA | holdout AUROC | CV BA | CV AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, record in summary["best"].items():
        holdout = record.get("source_family_holdout") or {}
        lines.append(
            "| {label} | {view} | {layer} | {hba:.3f} | {hauc} | {cvba:.3f} | {cvauc:.3f} |".format(
                label=label,
                view=record["view"],
                layer=record["layer"],
                hba=float(holdout.get("balanced_accuracy") or 0.0),
                hauc="" if holdout.get("auroc") is None else f"{float(holdout['auroc']):.3f}",
                cvba=float(record["cv_balanced_accuracy"]),
                cvauc=float(record["cv_auroc"]),
            )
        )
    lines.extend(["", "## Label Counts", ""])
    for label, counts in summary["label_counts"].items():
        lines.append(f"- `{label}`: `{counts}`")
    lines.extend(["", "## Notes", ""])
    lines.append("- Primary score is source-family holdout balanced accuracy when available.")
    lines.append("- Generated views are mean pooled over token thirds or the full generated sequence.")
    lines.append("- Prompt-end is the final non-whitespace prompt token before the replayed assistant response.")
    (REPORT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary = run_analysis()
    write_report(summary)
    print(REPORT_DIR)


if __name__ == "__main__":
    main()
