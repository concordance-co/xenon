#!/usr/bin/env python3
"""Run scaled F.1 process probes on collapsed criterion-family labels.

This is an exploratory follow-up to the process-supervision annotation pass:
collapse covered rubric criteria into a small set of process-shaped row labels,
then test whether prompt-final residuals predict those labels better than cheap
text/length baselines under source-family holdout.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipelines_v2.api import ModalVolumeStore, PostgresCatalog, PostgresSource, TransferPolicy
from pipelines_v2.storage.artifacts import CaptureArtifact, artifact_from_manifest


DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
ARTIFACT_STORE_NAME = "xenon-data"
CAPTURE_ROOT = "/data/artifacts/morebench_phase_03_experiment03_response_labels"
CAPTURE_ARTIFACT_ID = "capture_1_f2a9e4531dec"

ROOT = Path("projects/MOREBENCH/phase_03")
BASE_DIR = ROOT / "reports/experiment_03_process_supervision"
PACKET_DIR = BASE_DIR / "annotation_packet"
ANNOTATION_PATH = BASE_DIR / "process_supervision_annotations.jsonl"
REPORT_DIR = ROOT / "reports/experiment_03_scaled_f1_process_probe"
ASSET_DIR = REPORT_DIR / "assets"
TABLE_DIR = REPORT_DIR / "tables"

LAYERS = (0, 4, 8, 16, 28, 36, 40, 44)
PRIMARY_FAMILIES = (
    "decision_procedure",
    "identify_options",
    "tailored_next_steps",
    "epistemic_uncertainty",
    "risk_mitigation",
    "identify_stakeholders",
)
SECONDARY_FAMILIES = ("uncertainty_incomplete_info",)
TARGET_FAMILIES = (*PRIMARY_FAMILIES, *SECONDARY_FAMILIES)

MIN_CLASS_N = 30
MIN_HOLDOUT_TEST_N = 25
RANDOM_STATE = 23
SHUFFLED_PERMUTATIONS = int(os.environ.get("MOREBENCH_F1_NULL_PERMUTATIONS", "50"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _catalog() -> PostgresCatalog:
    return PostgresCatalog(source=PostgresSource.from_env(DB_ENV_VAR))


def _store() -> ModalVolumeStore:
    return ModalVolumeStore(
        name=ARTIFACT_STORE_NAME,
        root=CAPTURE_ROOT,
        transfer_policy=TransferPolicy(allow_large_transfer=True),
    )


def _load_capture() -> CaptureArtifact:
    manifest = _catalog().load_artifact(CAPTURE_ARTIFACT_ID)
    if manifest is None:
        raise RuntimeError(f"Could not load capture artifact {CAPTURE_ARTIFACT_ID!r}")
    artifact = artifact_from_manifest(manifest, store=_store())
    if not isinstance(artifact, CaptureArtifact):
        raise TypeError(f"Artifact {CAPTURE_ARTIFACT_ID!r} is not a capture artifact")
    return artifact


def _pool_prompt(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 2:
        return array[-1]
    return array.reshape(-1)


def _load_prompt_vectors(capture: CaptureArtifact) -> dict[int, dict[str, np.ndarray]]:
    payload = capture.feature("prompt_end_residual").load()
    out: dict[int, dict[str, np.ndarray]] = {}
    for layer in LAYERS:
        layer_payload = payload["layers"][str(layer)]
        out[layer] = {
            str(key): _pool_prompt(record["values"])
            for key, record in layer_payload.items()
        }
    return out


def _family_set(annotation: Mapping[str, Any]) -> set[str]:
    covered: set[str] = set()
    items = annotation.get("criterion_coverage")
    if not isinstance(items, list):
        return covered
    for item in items:
        if not isinstance(item, Mapping) or not bool(item.get("covered")):
            continue
        family_id = str(item.get("family_id") or "").strip()
        if family_id:
            covered.add(family_id)
    return covered


def _fit_scores(x_train: Any, y_train: np.ndarray, x_test: Any) -> tuple[np.ndarray, np.ndarray]:
    model = make_pipeline(
        StandardScaler(with_mean=not hasattr(x_train, "tocsr")),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
    )
    model.fit(x_train, y_train)
    return model.predict(x_test), model.predict_proba(x_test)[:, 1]


def _fit_scores_sparse(x_train: Any, y_train: np.ndarray, x_test: Any) -> tuple[np.ndarray, np.ndarray]:
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)
    return model.predict(x_test), model.predict_proba(x_test)[:, 1]


def _score(y_true: np.ndarray, pred: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred))
    }
    try:
        out["auroc"] = float(roc_auc_score(y_true, scores))
    except ValueError:
        out["auroc"] = None
    return out


def _cv_eval(x: Any, y: np.ndarray, *, sparse: bool = False) -> dict[str, Any] | None:
    counts = Counter(y.tolist())
    if len(counts) < 2 or min(counts.values()) < 2:
        return None
    n_splits = min(5, min(counts.values()))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    pred = np.zeros_like(y)
    scores = np.zeros(len(y), dtype=np.float32)
    for train_idx, test_idx in splitter.split(np.zeros(len(y)), y):
        if sparse:
            fold_pred, fold_scores = _fit_scores_sparse(x[train_idx], y[train_idx], x[test_idx])
        else:
            fold_pred, fold_scores = _fit_scores(x[train_idx], y[train_idx], x[test_idx])
        pred[test_idx] = fold_pred
        scores[test_idx] = fold_scores
    return {"n_splits": int(n_splits), **_score(y, pred, scores)}


def _source_holdout_eval(x: Any, y: np.ndarray, groups: list[str], *, sparse: bool = False) -> dict[str, Any] | None:
    folds = []
    group_array = np.asarray(groups)
    for group in sorted(set(groups)):
        test_idx = np.where(group_array == group)[0]
        train_idx = np.where(group_array != group)[0]
        if len(test_idx) < MIN_HOLDOUT_TEST_N:
            continue
        if len(set(y[test_idx].tolist())) < 2 or len(set(y[train_idx].tolist())) < 2:
            continue
        if sparse:
            pred, scores = _fit_scores_sparse(x[train_idx], y[train_idx], x[test_idx])
        else:
            pred, scores = _fit_scores(x[train_idx], y[train_idx], x[test_idx])
        folds.append({"heldout": str(group), "n_test": int(len(test_idx)), **_score(y[test_idx], pred, scores)})
    if not folds:
        return None
    valid_aurocs = [fold["auroc"] for fold in folds if fold.get("auroc") is not None]
    return {
        "n_splits": int(len(folds)),
        "balanced_accuracy": float(np.mean([fold["balanced_accuracy"] for fold in folds])),
        "auroc": float(np.mean(valid_aurocs)) if valid_aurocs else None,
        "folds": folds,
    }


def _text_eval(texts: list[str], y: np.ndarray, groups: list[str]) -> dict[str, Any] | None:
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=40_000)
    x = vectorizer.fit_transform(texts)
    return _source_holdout_eval(x, y, groups, sparse=True)


def _length_eval(features: np.ndarray, y: np.ndarray, groups: list[str]) -> dict[str, Any] | None:
    return _source_holdout_eval(features.astype(np.float32), y, groups)


def _source_data() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = {str(row["row_id"]): row for row in _read_jsonl(PACKET_DIR / "rows.jsonl")}
    annotations = {str(row["row_id"]): row for row in _read_jsonl(ANNOTATION_PATH)}
    return rows, annotations


def _make_matrix(
    rows: Mapping[str, Mapping[str, Any]],
    annotations: Mapping[str, Mapping[str, Any]],
    vectors: Mapping[str, np.ndarray],
    family_id: str,
) -> tuple[list[str], np.ndarray, np.ndarray, list[str], list[str], np.ndarray]:
    keys = sorted(set(rows) & set(annotations) & set(vectors))
    y = np.asarray(
        [1 if family_id in _family_set(annotations[key]) else 0 for key in keys],
        dtype=np.int64,
    )
    x = np.stack([vectors[key] for key in keys], axis=0).astype(np.float32)
    groups = [str(rows[key].get("source_family") or "") for key in keys]
    texts = [str(rows[key].get("dilemma") or "") for key in keys]
    length_features = np.asarray(
        [
            [
                float(rows[key].get("response_char_length") or 0),
                float(rows[key].get("response_word_count") or 0),
                float(len(str(rows[key].get("dilemma") or ""))),
            ]
            for key in keys
        ],
        dtype=np.float32,
    )
    return keys, x, y, groups, texts, length_features


def _best_layer(layer_records: Mapping[str, Any]) -> dict[str, Any] | None:
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for layer, record in layer_records.items():
        holdout = record.get("source_family_holdout")
        score = holdout.get("auroc") if isinstance(holdout, Mapping) else None
        if score is not None:
            candidates.append((float(score), int(layer), record))
    if not candidates:
        return None
    score, layer, record = max(candidates, key=lambda item: item[0])
    holdout = record.get("source_family_holdout") or {}
    cv = record.get("cv") or {}
    return {
        "layer": int(layer),
        "source_family_holdout_auroc": score,
        "source_family_holdout_balanced_accuracy": holdout.get("balanced_accuracy"),
        "cv_auroc": cv.get("auroc"),
        "cv_balanced_accuracy": cv.get("balanced_accuracy"),
    }


def _shuffle_null(
    layer_matrices: Mapping[int, np.ndarray],
    y: np.ndarray,
    groups: list[str],
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    best_aurocs: list[float] = []
    for _ in range(permutations):
        y_perm = rng.permutation(y)
        layer_aurocs: list[float] = []
        for layer in LAYERS:
            holdout = _source_holdout_eval(layer_matrices[layer], y_perm, groups)
            if holdout and holdout.get("auroc") is not None:
                layer_aurocs.append(float(holdout["auroc"]))
        if layer_aurocs:
            best_aurocs.append(max(layer_aurocs))
    if not best_aurocs:
        return {"permutations": permutations, "valid_permutations": 0, "best_holdout_auroc_p95": None}
    return {
        "permutations": int(permutations),
        "valid_permutations": int(len(best_aurocs)),
        "best_holdout_auroc_mean": float(np.mean(best_aurocs)),
        "best_holdout_auroc_p95": float(np.percentile(best_aurocs, 95)),
        "best_holdout_aurocs": [float(value) for value in best_aurocs],
    }


def _target_group(family_id: str) -> str:
    if family_id in PRIMARY_FAMILIES:
        return "primary_process"
    if family_id in SECONDARY_FAMILIES:
        return "secondary_uncertainty_duplicate_check"
    return "other"


def run_analysis() -> dict[str, Any]:
    rows, annotations = _source_data()
    capture = _load_capture()
    prompt_vectors = _load_prompt_vectors(capture)

    summary: dict[str, Any] = {
        "capture_artifact_id": CAPTURE_ARTIFACT_ID,
        "label_source": str(ANNOTATION_PATH),
        "report_dir": str(REPORT_DIR),
        "layers": list(LAYERS),
        "target_families": list(TARGET_FAMILIES),
        "primary_families": list(PRIMARY_FAMILIES),
        "secondary_families": list(SECONDARY_FAMILIES),
        "notes": [
            "Exploratory scaled F.1 readout; process-label gates were intentionally bypassed at user request.",
            "Row label is true if any covered criterion in the row belongs to the target family.",
            "Primary split is source-family leave-one-out; best layer is selected by holdout AUROC.",
            "Shuffled-label null reports best-over-layers source-family-holdout AUROC across permutations.",
        ],
        "families": {},
    }

    for family_idx, family_id in enumerate(TARGET_FAMILIES):
        print(f"[{family_idx + 1}/{len(TARGET_FAMILIES)}] {family_id}", flush=True)
        layer_matrices: dict[int, np.ndarray] = {}
        groups: list[str] | None = None
        texts: list[str] | None = None
        length_features: np.ndarray | None = None
        y: np.ndarray | None = None
        layer_records: dict[str, Any] = {}

        for layer in LAYERS:
            _, x, y_layer, groups_layer, texts_layer, length_layer = _make_matrix(
                rows,
                annotations,
                prompt_vectors[layer],
                family_id,
            )
            layer_matrices[layer] = x
            if y is None:
                y = y_layer
                groups = groups_layer
                texts = texts_layer
                length_features = length_layer
            layer_records[str(layer)] = {
                "cv": _cv_eval(x, y_layer),
                "source_family_holdout": _source_holdout_eval(x, y_layer, groups_layer),
            }

        assert y is not None and groups is not None and texts is not None and length_features is not None
        counts = Counter(y.tolist())
        family_record: dict[str, Any] = {
            "target_group": _target_group(family_id),
            "n": int(len(y)),
            "class_counts": {str(key): int(value) for key, value in sorted(counts.items())},
            "eligible": bool(counts.get(0, 0) >= MIN_CLASS_N and counts.get(1, 0) >= MIN_CLASS_N),
            "prompt_text_baseline": _text_eval(texts, y, groups),
            "length_baseline": _length_eval(length_features, y, groups),
            "frequency_baseline_auroc": 0.5,
            "layers": layer_records,
        }
        family_record["best"] = _best_layer(layer_records)
        family_record["shuffle_null"] = _shuffle_null(
            layer_matrices,
            y,
            groups,
            permutations=SHUFFLED_PERMUTATIONS,
            seed=RANDOM_STATE + family_idx,
        )
        best = family_record.get("best") or {}
        null = family_record.get("shuffle_null") or {}
        real = best.get("source_family_holdout_auroc")
        null_values = null.get("best_holdout_aurocs") or []
        family_record["shuffle_null"]["empirical_p_best_ge_real"] = (
            float((1 + sum(float(value) >= float(real) for value in null_values)) / (1 + len(null_values)))
            if real is not None and null_values
            else None
        )
        summary["families"][family_id] = family_record

    summary["best_table"] = _best_table(summary)
    return summary


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _best_table(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family_id, record in summary.get("families", {}).items():
        best = record.get("best") or {}
        text = record.get("prompt_text_baseline") or {}
        length = record.get("length_baseline") or {}
        null = record.get("shuffle_null") or {}
        rows.append(
            {
                "family_id": family_id,
                "target_group": record.get("target_group"),
                "n_pos": int(record.get("class_counts", {}).get("1", 0)),
                "n_neg": int(record.get("class_counts", {}).get("0", 0)),
                "best_layer": best.get("layer"),
                "holdout_auroc": best.get("source_family_holdout_auroc"),
                "holdout_balanced_accuracy": best.get("source_family_holdout_balanced_accuracy"),
                "cv_auroc": best.get("cv_auroc"),
                "cv_balanced_accuracy": best.get("cv_balanced_accuracy"),
                "prompt_text_holdout_auroc": text.get("auroc"),
                "prompt_text_holdout_balanced_accuracy": text.get("balanced_accuracy"),
                "length_holdout_auroc": length.get("auroc"),
                "length_holdout_balanced_accuracy": length.get("balanced_accuracy"),
                "shuffle_best_auroc_p95": null.get("best_holdout_auroc_p95"),
                "shuffle_empirical_p": null.get("empirical_p_best_ge_real"),
            }
        )
    rows.sort(key=lambda row: (row["target_group"] != "primary_process", -(row["holdout_auroc"] or 0.0)))
    return rows


def _write_report(summary: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (TABLE_DIR / "best_results.json").write_text(json.dumps(summary["best_table"], indent=2), encoding="utf-8")
    _write_plots(summary)

    lines = [
        "# Experiment 03 Scaled F.1 Process Probe",
        "",
        f"- capture artifact: `{summary['capture_artifact_id']}`",
        f"- label source: `{summary['label_source']}`",
        f"- layers: `{', '.join(str(layer) for layer in summary['layers'])}`",
        f"- shuffled-label permutations: `{SHUFFLED_PERMUTATIONS}`",
        "- status: exploratory, gates bypassed intentionally; use as triage, not a Level 2 claim by itself.",
        "",
        "## Collapsed Labels",
        "",
        "| family | group | pos/neg | best layer | holdout AUROC | holdout BA | CV AUROC | text AUROC | length AUROC | null p95 | emp p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["best_table"]:
        lines.append(
            "| {family_id} | {target_group} | {n_pos}/{n_neg} | {best_layer} | {holdout_auroc} | {holdout_ba} | {cv_auroc} | {text_auroc} | {length_auroc} | {null_p95} | {emp_p} |".format(
                family_id=row["family_id"],
                target_group=row["target_group"],
                n_pos=row["n_pos"],
                n_neg=row["n_neg"],
                best_layer=row["best_layer"],
                holdout_auroc=_fmt(row["holdout_auroc"]),
                holdout_ba=_fmt(row["holdout_balanced_accuracy"]),
                cv_auroc=_fmt(row["cv_auroc"]),
                text_auroc=_fmt(row["prompt_text_holdout_auroc"]),
                length_auroc=_fmt(row["length_holdout_auroc"]),
                null_p95=_fmt(row["shuffle_best_auroc_p95"]),
                emp_p=_fmt(row["shuffle_empirical_p"]),
            )
        )
    lines.extend(
        [
            "",
            "## Reading Rules",
            "",
            "- A useful process label should clear prompt-text and length baselines, not just random CV.",
            "- The shuffled null is best-over-layers, so it is intentionally stricter than a single-layer null.",
            "- `uncertainty_incomplete_info` is included as a secondary duplicate-check against `epistemic_uncertainty`.",
            "",
            "## Charts",
            "",
            "![Best holdout AUROC](assets/best_holdout_auroc.png)",
            "",
            "![Layer curves](assets/layer_curves.png)",
        ]
    )
    (REPORT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_html(summary)


def _write_plots(summary: Mapping[str, Any]) -> None:
    table = summary["best_table"]
    labels = [row["family_id"] for row in table]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - 0.25, [row["holdout_auroc"] or 0.0 for row in table], width=0.25, label="Activation holdout AUROC")
    ax.bar(x, [row["prompt_text_holdout_auroc"] or 0.0 for row in table], width=0.25, label="Prompt text AUROC")
    ax.bar(x + 0.25, [row["shuffle_best_auroc_p95"] or 0.0 for row in table], width=0.25, label="Shuffled best-layer p95")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(0.7, color="#944", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.4, 0.85)
    ax.set_ylabel("AUROC")
    ax.set_title("Scaled F.1: Best Source-Family Holdout AUROC")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "best_holdout_auroc.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for family_id, record in summary["families"].items():
        ys = []
        for layer in LAYERS:
            holdout = record["layers"][str(layer)].get("source_family_holdout") or {}
            ys.append(holdout.get("auroc"))
        ax.plot(LAYERS, ys, marker="o", label=family_id)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(0.7, color="#944", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Source-family holdout AUROC")
    ax.set_title("Scaled F.1 Layer Curves")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "layer_curves.png", dpi=180)
    plt.close(fig)


def _write_html(summary: Mapping[str, Any]) -> None:
    rows = []
    for row in summary["best_table"]:
        rows.append(
            "<tr>"
            f"<td>{row['family_id']}</td>"
            f"<td>{row['target_group']}</td>"
            f"<td>{row['n_pos']}/{row['n_neg']}</td>"
            f"<td>{row['best_layer']}</td>"
            f"<td>{_fmt(row['holdout_auroc'])}</td>"
            f"<td>{_fmt(row['holdout_balanced_accuracy'])}</td>"
            f"<td>{_fmt(row['prompt_text_holdout_auroc'])}</td>"
            f"<td>{_fmt(row['length_holdout_auroc'])}</td>"
            f"<td>{_fmt(row['shuffle_best_auroc_p95'])}</td>"
            f"<td>{_fmt(row['shuffle_empirical_p'])}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MoReBench Scaled F.1 Process Probe</title>
  <style>
    body {{ font: 16px/1.5 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 40px; max-width: 1180px; }}
    h1, h2 {{ line-height: 1.15; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    img {{ max-width: 100%; border: 1px solid #e5e5e5; border-radius: 8px; margin: 12px 0 28px; }}
    code {{ background: #f7f7f7; border-radius: 6px; padding: 1px 4px; }}
  </style>
</head>
<body>
  <h1>MoReBench Scaled F.1 Process Probe</h1>
  <p>Exploratory collapsed criterion-family readout from prompt-final residuals. Gates were bypassed intentionally; this report is triage, not a confirmatory claim.</p>
  <table>
    <thead><tr><th>Family</th><th>Group</th><th>Pos/Neg</th><th>Best Layer</th><th>Holdout AUROC</th><th>Holdout BA</th><th>Text AUROC</th><th>Length AUROC</th><th>Null p95</th><th>Emp p</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Best Holdout AUROC</h2>
  <img src="assets/best_holdout_auroc.png" alt="Best holdout AUROC">
  <h2>Layer Curves</h2>
  <img src="assets/layer_curves.png" alt="Layer curves">
</body>
</html>
"""
    (REPORT_DIR / "report.html").write_text(html, encoding="utf-8")


def main() -> None:
    summary = run_analysis()
    _write_report(summary)
    print(json.dumps({"report_dir": str(REPORT_DIR), "families": list(summary["families"])}, indent=2))


if __name__ == "__main__":
    main()
