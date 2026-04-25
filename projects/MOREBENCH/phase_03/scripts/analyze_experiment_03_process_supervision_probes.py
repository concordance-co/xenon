#!/usr/bin/env python3
"""Run MoReBench process-supervision probes after annotation gates pass."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
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
LABELABILITY_PATH = BASE_DIR / "labelability_summary.json"
SUMMARY_PATH = BASE_DIR / "process_probe_summary.json"
REPORT_PATH = BASE_DIR / "process_probe_report.md"

LAYERS = (0, 4, 8, 16, 28, 36, 40, 44)
B_PRIMARY_LAYER = 16
B_EXPLORATORY_LAYERS = (8, 28, 36, 40, 44)
MIN_POS = 30
MIN_NEG = 30
RANDOM_STATE = 19
SHUFFLED_PERMUTATIONS = 50


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


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


def _span(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, Mapping):
        return None
    start = value.get("char_start")
    end = value.get("char_end")
    if start is None or end is None:
        return None
    try:
        start_i = int(start)
        end_i = int(end)
    except (TypeError, ValueError):
        return None
    if start_i < 0 or end_i <= start_i:
        return None
    return start_i, end_i


def _family_set(annotation: Mapping[str, Any]) -> set[str]:
    out: set[str] = set()
    for item in annotation.get("criterion_coverage", []) if isinstance(annotation.get("criterion_coverage"), list) else []:
        if isinstance(item, Mapping) and bool(item.get("covered")):
            family_id = str(item.get("family_id") or "").strip()
            if family_id:
                out.add(family_id)
    return out


def _claim_family_set(claim: Mapping[str, Any]) -> set[str]:
    values = claim.get("covered_family_ids")
    return {str(value) for value in values} if isinstance(values, list) else set()


def _source_data() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    rows = {row["row_id"]: row for row in _read_jsonl(PACKET_DIR / "rows.jsonl")}
    annotations = {row["row_id"]: row for row in _read_jsonl(ANNOTATION_PATH)}
    families = _read_json(BASE_DIR / "criterion_families.json")
    family_ids = [str(item["family_id"]) for item in families.get("families", [])]
    return rows, annotations, family_ids


def _fit_scores(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE),
    )
    model.fit(x_train, y_train)
    return model.predict(x_test), model.predict_proba(x_test)[:, 1]


def _score(y_true: np.ndarray, pred: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    out: dict[str, float | None] = {"balanced_accuracy": float(balanced_accuracy_score(y_true, pred))}
    try:
        out["auroc"] = float(roc_auc_score(y_true, scores))
    except ValueError:
        out["auroc"] = None
    return out


def _cv_eval(x: np.ndarray, y: np.ndarray) -> dict[str, Any] | None:
    counts = Counter(y.tolist())
    if len(counts) < 2 or min(counts.values()) < 2:
        return None
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
        if len(test_idx) < 25:
            continue
        if len(set(y[test_idx].tolist())) < 2 or len(set(y[train_idx].tolist())) < 2:
            continue
        pred, scores = _fit_scores(x[train_idx], y[train_idx], x[test_idx])
        folds.append({"heldout": group, "n_test": int(len(test_idx)), **_score(y[test_idx], pred, scores)})
    if not folds:
        return None
    return {
        "n_splits": len(folds),
        "balanced_accuracy": float(np.mean([fold["balanced_accuracy"] for fold in folds])),
        "auroc": float(np.nanmean([fold["auroc"] for fold in folds if fold["auroc"] is not None])),
        "folds": folds,
    }


def _text_eval(texts: list[str], y: np.ndarray, groups: list[str]) -> dict[str, Any] | None:
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=40_000)
    x = vectorizer.fit_transform(texts)
    return _source_holdout_eval_sparse(x, y, groups)


def _source_holdout_eval_sparse(x: Any, y: np.ndarray, groups: list[str]) -> dict[str, Any] | None:
    folds = []
    group_array = np.asarray(groups)
    for group in sorted(set(groups)):
        test_idx = np.where(group_array == group)[0]
        train_idx = np.where(group_array != group)[0]
        if len(test_idx) < 25:
            continue
        if len(set(y[test_idx].tolist())) < 2 or len(set(y[train_idx].tolist())) < 2:
            continue
        model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE)
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[test_idx])
        scores = model.predict_proba(x[test_idx])[:, 1]
        folds.append({"heldout": group, "n_test": int(len(test_idx)), **_score(y[test_idx], pred, scores)})
    if not folds:
        return None
    return {
        "n_splits": len(folds),
        "balanced_accuracy": float(np.mean([fold["balanced_accuracy"] for fold in folds])),
        "auroc": float(np.nanmean([fold["auroc"] for fold in folds if fold["auroc"] is not None])),
        "folds": folds,
    }


def _length_eval(features: np.ndarray, y: np.ndarray, groups: list[str]) -> dict[str, Any] | None:
    return _source_holdout_eval(features.astype(np.float32), y, groups)


def _load_prompt_vectors(capture: CaptureArtifact) -> dict[int, dict[str, np.ndarray]]:
    payload = capture.feature("prompt_end_residual").load()
    out: dict[int, dict[str, np.ndarray]] = {}
    for layer in LAYERS:
        layer_payload = payload["layers"][str(layer)]
        out[layer] = {
            str(key): np.asarray(record["values"], dtype=np.float32).reshape(-1)
            for key, record in layer_payload.items()
        }
    return out


def _load_generated_payload(capture: CaptureArtifact) -> dict[str, Any]:
    return capture.feature("generated_sequence_residual").load()


def _pool_span(values: np.ndarray, response_len: int, span: tuple[int, int] | None) -> np.ndarray | None:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] <= 0 or response_len <= 0 or span is None:
        return None
    start, end = span
    token_start = max(0, min(arr.shape[0] - 1, math.floor((start / response_len) * arr.shape[0])))
    token_end = max(token_start + 1, min(arr.shape[0], math.ceil((end / response_len) * arr.shape[0])))
    return arr[token_start:token_end].mean(axis=0)


def _eligible_families(annotations: Mapping[str, Mapping[str, Any]], family_ids: list[str]) -> list[str]:
    eligible = []
    rows = list(annotations.values())
    for family_id in family_ids:
        positives = sum(1 for row in rows if family_id in _family_set(row))
        negatives = len(rows) - positives
        if positives >= MIN_POS and negatives >= MIN_NEG:
            eligible.append(family_id)
    return eligible


def _f1_prompt_probe(capture: CaptureArtifact, rows: Mapping[str, Mapping[str, Any]], annotations: Mapping[str, Mapping[str, Any]], family_ids: list[str]) -> dict[str, Any]:
    prompt_vectors = _load_prompt_vectors(capture)
    eligible = _eligible_families(annotations, family_ids)
    results: dict[str, Any] = {"eligible_families": eligible, "families": {}}
    keys = sorted(set(rows) & set(annotations))
    groups = [str(rows[key].get("source_family") or "") for key in keys]
    prompt_texts = [str(rows[key].get("dilemma") or "") for key in keys]
    length_features = np.asarray([[rows[key].get("response_char_length") or 0, rows[key].get("response_word_count") or 0] for key in keys], dtype=np.float32)

    for family_id in eligible:
        y = np.asarray([1 if family_id in _family_set(annotations[key]) else 0 for key in keys], dtype=np.int64)
        family_record: dict[str, Any] = {
            "n": int(len(y)),
            "class_counts": dict(Counter(y.tolist())),
            "prompt_text_baseline": _text_eval(prompt_texts, y, groups),
            "length_baseline": _length_eval(length_features, y, groups),
            "frequency_baseline_auroc": 0.5,
            "layers": {},
        }
        for layer in LAYERS:
            x = np.stack([prompt_vectors[layer][key] for key in keys], axis=0).astype(np.float32)
            cv = _cv_eval(x, y)
            holdout = _source_holdout_eval(x, y, groups)
            family_record["layers"][str(layer)] = {"cv": cv, "source_family_holdout": holdout}
        results["families"][family_id] = family_record
    return results


def _claims_for_family(rows: Mapping[str, Mapping[str, Any]], annotations: Mapping[str, Mapping[str, Any]], family_id: str) -> tuple[list[dict[str, Any]], np.ndarray, list[str], list[str]]:
    examples: list[dict[str, Any]] = []
    labels: list[int] = []
    groups: list[str] = []
    texts: list[str] = []
    for row_id, annotation in annotations.items():
        row = rows.get(row_id)
        if row is None:
            continue
        response = str(row.get("response") or "")
        claims = annotation.get("claims")
        if not isinstance(claims, list):
            continue
        for claim in claims:
            if not isinstance(claim, Mapping):
                continue
            span = _span(claim)
            if span is None:
                continue
            examples.append({"row_id": row_id, "span": span})
            labels.append(1 if family_id in _claim_family_set(claim) else 0)
            groups.append(str(row.get("source_family") or ""))
            texts.append(response[span[0] : span[1]])
    return examples, np.asarray(labels, dtype=np.int64), groups, texts


def _f2_claim_probe(capture: CaptureArtifact, rows: Mapping[str, Mapping[str, Any]], annotations: Mapping[str, Mapping[str, Any]], family_ids: list[str]) -> dict[str, Any]:
    payload = _load_generated_payload(capture)
    eligible = []
    for family_id in family_ids:
        _, y, _, _ = _claims_for_family(rows, annotations, family_id)
        if len(y) and int(y.sum()) >= MIN_POS and int((1 - y).sum()) >= MIN_NEG:
            eligible.append(family_id)
    results: dict[str, Any] = {"eligible_families": eligible, "alignment": "proportional_char_to_token", "families": {}}
    for family_id in eligible:
        examples, y, groups, texts = _claims_for_family(rows, annotations, family_id)
        family_record: dict[str, Any] = {
            "n": int(len(y)),
            "class_counts": dict(Counter(y.tolist())),
            "claim_text_baseline": _text_eval(texts, y, groups),
            "frequency_baseline_auroc": 0.5,
            "layers": {},
        }
        for layer in LAYERS:
            layer_payload = payload["layers"][str(layer)]
            pooled = []
            keep_y = []
            keep_groups = []
            for example, label, group in zip(examples, y.tolist(), groups, strict=True):
                rec = layer_payload.get(example["row_id"])
                if rec is None:
                    continue
                response_len = len(str(rows[example["row_id"]].get("response") or ""))
                vec = _pool_span(np.asarray(rec["values"], dtype=np.float32), response_len, example["span"])
                if vec is None:
                    continue
                pooled.append(vec)
                keep_y.append(label)
                keep_groups.append(group)
            x = np.stack(pooled, axis=0).astype(np.float32)
            y_arr = np.asarray(keep_y, dtype=np.int64)
            family_record["layers"][str(layer)] = {
                "cv": _cv_eval(x, y_arr),
                "source_family_holdout": _source_holdout_eval(x, y_arr, keep_groups),
            }
        results["families"][family_id] = family_record
    return results


def _b_examples(rows: Mapping[str, Mapping[str, Any]], annotations: Mapping[str, Mapping[str, Any]], negative_kind: str) -> tuple[list[dict[str, Any]], np.ndarray, list[str], list[str]]:
    examples: list[dict[str, Any]] = []
    labels: list[int] = []
    groups: list[str] = []
    texts: list[str] = []
    for row_id, annotation in annotations.items():
        row = rows.get(row_id)
        if row is None:
            continue
        response = str(row.get("response") or "")
        commitment = _mapping(annotation.get("commitment"))
        if not bool(commitment.get("has_commitment")):
            continue
        pos_span = _span(commitment)
        controls = _mapping(annotation.get("control_spans"))
        neg_span = _span(controls.get(negative_kind))
        if pos_span is None or neg_span is None:
            continue
        for span, label in ((pos_span, 1), (neg_span, 0)):
            examples.append({"row_id": row_id, "span": span})
            labels.append(label)
            groups.append(str(row.get("source_family") or ""))
            texts.append(response[span[0] : span[1]])
    return examples, np.asarray(labels, dtype=np.int64), groups, texts


def _span_binary_probe(capture: CaptureArtifact, rows: Mapping[str, Mapping[str, Any]], examples: list[dict[str, Any]], y: np.ndarray, groups: list[str], layers: tuple[int, ...]) -> dict[str, Any]:
    payload = _load_generated_payload(capture)
    result: dict[str, Any] = {"layers": {}}
    for layer in layers:
        pooled = []
        keep_y = []
        keep_groups = []
        for example, label, group in zip(examples, y.tolist(), groups, strict=True):
            rec = payload["layers"][str(layer)].get(example["row_id"])
            if rec is None:
                continue
            response_len = len(str(rows[example["row_id"]].get("response") or ""))
            vec = _pool_span(np.asarray(rec["values"], dtype=np.float32), response_len, example["span"])
            if vec is None:
                continue
            pooled.append(vec)
            keep_y.append(label)
            keep_groups.append(group)
        x = np.stack(pooled, axis=0).astype(np.float32)
        y_arr = np.asarray(keep_y, dtype=np.int64)
        result["layers"][str(layer)] = {
            "n": int(len(y_arr)),
            "class_counts": dict(Counter(y_arr.tolist())),
            "cv": _cv_eval(x, y_arr),
            "source_family_holdout": _source_holdout_eval(x, y_arr, keep_groups),
        }
    return result


def _b_commitment_probe(capture: CaptureArtifact, rows: Mapping[str, Mapping[str, Any]], annotations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    examples, y, groups, texts = _b_examples(rows, annotations, "matched_mid_reasoning")
    null_examples, null_y, null_groups, null_texts = _b_examples(rows, annotations, "same_position_noncommitment")
    layers = (B_PRIMARY_LAYER, *B_EXPLORATORY_LAYERS)
    return {
        "primary_layer": B_PRIMARY_LAYER,
        "commitment_vs_mid": {
            **_span_binary_probe(capture, rows, examples, y, groups, layers),
            "viewport_text_baseline": _text_eval(texts, y, groups),
        },
        "same_position_null": {
            **_span_binary_probe(capture, rows, null_examples, null_y, null_groups, layers),
            "viewport_text_baseline": _text_eval(null_texts, null_y, null_groups),
        },
    }


def _residualize_train_test(x_train: np.ndarray, x_test: np.ndarray, z_train: np.ndarray, z_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = LinearRegression()
    model.fit(z_train, x_train)
    return x_train - model.predict(z_train), x_test - model.predict(z_test)


def _c_residualized_eval(x: np.ndarray, z: np.ndarray, y: np.ndarray, groups: list[str]) -> dict[str, Any] | None:
    folds = []
    group_array = np.asarray(groups)
    for group in sorted(set(groups)):
        test_idx = np.where(group_array == group)[0]
        train_idx = np.where(group_array != group)[0]
        if len(test_idx) < 25:
            continue
        if len(set(y[test_idx].tolist())) < 2 or len(set(y[train_idx].tolist())) < 2:
            continue
        xr_train, xr_test = _residualize_train_test(x[train_idx], x[test_idx], z[train_idx], z[test_idx])
        pred, scores = _fit_scores(xr_train, y[train_idx], xr_test)
        folds.append({"heldout": group, "n_test": int(len(test_idx)), **_score(y[test_idx], pred, scores)})
    if not folds:
        return None
    return {
        "n_splits": len(folds),
        "balanced_accuracy": float(np.mean([fold["balanced_accuracy"] for fold in folds])),
        "auroc": float(np.nanmean([fold["auroc"] for fold in folds if fold["auroc"] is not None])),
        "folds": folds,
    }


def _c_probe(capture: CaptureArtifact, rows: Mapping[str, Mapping[str, Any]], annotations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = _load_generated_payload(capture)
    keys = []
    labels = []
    groups = []
    z_rows = []
    for row_id, annotation in annotations.items():
        c = _mapping(annotation.get("consideration"))
        label = str(c.get("early_collapse_vs_sustained") or "")
        if label not in {"early_collapse", "sustained_multi_consideration"}:
            continue
        row = rows.get(row_id)
        if row is None:
            continue
        commitment_span = _span(_mapping(annotation.get("commitment")))
        precommit_chars = commitment_span[0] if commitment_span else len(str(row.get("response") or ""))
        claims = annotation.get("claims") if isinstance(annotation.get("claims"), list) else []
        keys.append(row_id)
        labels.append(1 if label == "sustained_multi_consideration" else 0)
        groups.append(str(row.get("source_family") or ""))
        z_rows.append([
            float(row.get("response_char_length") or 0),
            float(row.get("response_word_count") or 0),
            float(precommit_chars),
            float(len(claims)),
        ])
    y = np.asarray(labels, dtype=np.int64)
    z = np.asarray(z_rows, dtype=np.float32)
    result: dict[str, Any] = {
        "n": int(len(y)),
        "class_counts": dict(Counter(y.tolist())),
        "length_only_baseline": _length_eval(z, y, groups),
        "layers": {},
    }
    for layer in LAYERS:
        pooled = []
        keep_y = []
        keep_z = []
        keep_groups = []
        for key, label, z_row, group in zip(keys, y.tolist(), z.tolist(), groups, strict=True):
            rec = payload["layers"][str(layer)].get(key)
            if rec is None:
                continue
            values = np.asarray(rec["values"], dtype=np.float32)
            pooled.append(values.mean(axis=0))
            keep_y.append(label)
            keep_z.append(z_row)
            keep_groups.append(group)
        x = np.stack(pooled, axis=0).astype(np.float32)
        y_arr = np.asarray(keep_y, dtype=np.int64)
        z_arr = np.asarray(keep_z, dtype=np.float32)
        result["layers"][str(layer)] = {
            "residualized_source_family_holdout": _c_residualized_eval(x, z_arr, y_arr, keep_groups),
            "raw_source_family_holdout": _source_holdout_eval(x, y_arr, keep_groups),
        }
    return result


def _best_family_metrics(families: Mapping[str, Any]) -> dict[str, Any]:
    best = {}
    for family_id, record in families.items():
        candidates = []
        for layer, payload in record.get("layers", {}).items():
            holdout = payload.get("source_family_holdout")
            auroc = holdout.get("auroc") if isinstance(holdout, Mapping) else None
            if auroc is not None:
                candidates.append((float(auroc), int(layer), payload))
        if candidates:
            auroc, layer, payload = max(candidates, key=lambda item: item[0])
            best[family_id] = {"best_layer": layer, "best_auroc": auroc, "record": payload}
    return best


def main() -> None:
    labelability = _read_json(LABELABILITY_PATH)
    gates = _mapping(_mapping(labelability.get("agreement")).get("gates"))
    rows, annotations, family_ids = _source_data()
    capture = _load_capture()
    summary: dict[str, Any] = {
        "capture_artifact_id": CAPTURE_ARTIFACT_ID,
        "labelability_gates": dict(gates),
        "layers": list(LAYERS),
        "tracks": {},
    }

    if gates.get("F"):
        f1 = _f1_prompt_probe(capture, rows, annotations, family_ids)
        f2 = _f2_claim_probe(capture, rows, annotations, family_ids)
        summary["tracks"]["F.1"] = f1
        summary["tracks"]["F.2"] = f2
        summary["tracks"]["F.1_best"] = _best_family_metrics(f1["families"])
        summary["tracks"]["F.2_best"] = _best_family_metrics(f2["families"])
    if gates.get("B"):
        summary["tracks"]["B"] = _b_commitment_probe(capture, rows, annotations)
    if gates.get("C"):
        summary["tracks"]["C"] = _c_probe(capture, rows, annotations)

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Experiment 03 Process-Supervision Probes",
        "",
        f"- capture artifact: `{CAPTURE_ARTIFACT_ID}`",
        f"- labelability gates: `{dict(gates)}`",
        f"- tracks run: `{', '.join(summary['tracks'].keys())}`",
        "",
    ]
    for track in ("F.1", "F.2"):
        best = summary["tracks"].get(f"{track}_best")
        if not isinstance(best, Mapping):
            continue
        aurocs = [float(item["best_auroc"]) for item in best.values()]
        lines.extend([
            f"## {track}",
            "",
            f"- eligible families: `{len(best)}`",
            f"- mean best holdout AUROC: `{float(np.mean(aurocs)):.3f}`" if aurocs else "- mean best holdout AUROC: `NA`",
            f"- share above 0.75: `{float(np.mean([value >= 0.75 for value in aurocs])):.3f}`" if aurocs else "- share above 0.75: `NA`",
            "",
        ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"tracks": list(summary["tracks"].keys()), "summary": str(SUMMARY_PATH)}, indent=2))


if __name__ == "__main__":
    main()
