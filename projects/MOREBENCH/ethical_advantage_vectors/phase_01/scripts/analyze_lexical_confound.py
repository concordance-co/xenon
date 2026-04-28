"""Lexical-confound checks for ethical-vs-self-advantage probes."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from projects.MOREBENCH.ethical_advantage_vectors.phase_01.scripts import analyze_activation_probes as activation


PHASE_ROOT = Path("projects/MOREBENCH/ethical_advantage_vectors/phase_01")
DEFAULT_GENERATION_ROWS = (
    PHASE_ROOT
    / "reports"
    / "v2_full_capture"
    / "report_12abda8dec51_9d7390f7"
    / "results"
    / "generate_v2_responses_results.json"
)
DEFAULT_ACTION_ROWS = PHASE_ROOT / "reports" / "behavior_smoke_analysis" / "v2_full40" / "scored_rows.jsonl"
DEFAULT_REPORT_DIR = PHASE_ROOT / "reports" / "lexical_confound" / "v2_full40"
DEFAULT_CAPTURE_ID = "capture_1_2461d8ccdc41"

ETHICAL_CONDITIONS = {"P_ethical_01", "P_ethical_02"}
NEGATIVE_CONDITIONS = {"P_self_serving_01", "P_self_serving_02", "P_exploit_01"}
PRIMARY_CONDITIONS = ETHICAL_CONDITIONS | NEGATIVE_CONDITIONS
LAYERS = (16, 24, 32, 40)
SLICES = ("prompt_end", "generated_first_16", "generated_full")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _prompt_text(prompt: Any) -> str:
    if not isinstance(prompt, list):
        return ""
    parts: list[str] = []
    for item in prompt:
        if isinstance(item, Mapping):
            role = str(item.get("role") or "")
            content = str(item.get("content") or "")
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _prefix_words(text: str, n_words: int) -> str:
    words = re.findall(r"\S+", text)
    return " ".join(words[:n_words])


def _rows_by_key(generation_rows_path: Path, action_rows_path: Path) -> dict[str, dict[str, Any]]:
    generation = _read_json(generation_rows_path)
    action_by_key = {str(row["example_key"]): str(row["action_label"]) for row in _read_jsonl(action_rows_path)}
    rows = generation.get("rows")
    if not isinstance(rows, list):
        raise TypeError(f"{generation_rows_path} must contain a rows list")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        example = row.get("example") if isinstance(row.get("example"), Mapping) else {}
        labels = example.get("labels") if isinstance(example.get("labels"), Mapping) else {}
        metadata = example.get("metadata") if isinstance(example.get("metadata"), Mapping) else {}
        key = str(row.get("example_key") or example.get("key") or "")
        generated_text = str(row.get("generated_text") or "")
        if not key:
            continue
        prompt_text = _prompt_text(example.get("prompt"))
        out[key] = {
            "key": key,
            "dilemma_id": str(labels.get("dilemma_id") or ""),
            "condition_id": str(labels.get("condition_id") or ""),
            "pole": str(labels.get("pole") or ""),
            "sample_index": int(labels.get("sample_index") or 0),
            "action_label": action_by_key.get(key, "missing"),
            "generated_text": generated_text,
            "generated_prefix16_text": _prefix_words(generated_text, 16),
            "generated_prefix32_text": _prefix_words(generated_text, 32),
            "prompt_text": prompt_text,
            "instruction_text": str(metadata.get("instruction") or ""),
            "dilemma_text": str(metadata.get("dilemma_text") or ""),
        }
    return out


def _target_rows(rows_by_key: dict[str, dict[str, Any]], target: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in rows_by_key.values():
        cid = row["condition_id"]
        label: int | None = None
        if target == "prompt_pole":
            if cid in NEGATIVE_CONDITIONS:
                label = 1
            elif cid in ETHICAL_CONDITIONS:
                label = 0
        elif target == "observed_action":
            if row["action_label"] == "self_advantage":
                label = 1
            elif row["action_label"] == "ethical":
                label = 0
        elif target == "observed_action_within_negative":
            if cid not in NEGATIVE_CONDITIONS:
                continue
            if row["action_label"] == "self_advantage":
                label = 1
            elif row["action_label"] == "ethical":
                label = 0
        else:
            raise ValueError(f"unknown target {target!r}")
        if label is None:
            continue
        out = dict(row)
        out["label"] = label
        rows.append(out)
    return rows


def _splits(y: np.ndarray, groups: np.ndarray, *, seed: int):
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 5 and min(np.bincount(y.astype(int))) >= 5:
        splitter = GroupKFold(n_splits=5)
        return "group_kfold_dilemma", list(splitter.split(np.zeros((len(y), 1)), y, groups))
    n_splits = min(5, int(min(np.bincount(y.astype(int)))))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return "stratified_kfold_fallback", list(splitter.split(np.zeros((len(y), 1)), y))


def _text_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True),
        LogisticRegression(
            C=0.5,
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
        ),
    )


def _dense_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
        ),
    )


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(set(y_true.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _fit_text_scores(train_text: list[str], train_y: np.ndarray, test_text: list[str]) -> tuple[np.ndarray, np.ndarray]:
    model = _text_model()
    model.fit(train_text, train_y)
    train_scores = model.decision_function(train_text)
    test_scores = model.decision_function(test_text)
    return np.asarray(train_scores, dtype=np.float64), np.asarray(test_scores, dtype=np.float64)


def _fit_dense_scores(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = _dense_model()
    model.fit(train_x, train_y)
    train_scores = model.decision_function(train_x)
    test_scores = model.decision_function(test_x)
    return np.asarray(train_scores, dtype=np.float64), np.asarray(test_scores, dtype=np.float64)


def _text_cv(rows: list[dict[str, Any]], text_field: str, *, seed: int) -> dict[str, Any]:
    y = np.asarray([row["label"] for row in rows], dtype=np.int64)
    groups = np.asarray([row["dilemma_id"] for row in rows], dtype=object)
    text = [str(row[text_field]) for row in rows]
    split_kind, splits = _splits(y, groups, seed=seed)
    aucs: list[float] = []
    for train_idx, test_idx in splits:
        if len(set(y[train_idx].tolist())) < 2 or len(set(y[test_idx].tolist())) < 2:
            continue
        _, scores = _fit_text_scores([text[i] for i in train_idx], y[train_idx], [text[i] for i in test_idx])
        aucs.append(_safe_auc(y[test_idx], scores))
    return {
        "n": int(len(rows)),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "split_kind": split_kind,
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "fold_aucs": aucs,
    }


def _condition_auc(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y = np.asarray([row["label"] for row in rows], dtype=np.int64)
    score = np.asarray([1.0 if row["condition_id"] in NEGATIVE_CONDITIONS else 0.0 for row in rows])
    return {
        "n": int(len(rows)),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "auc": _safe_auc(y, score),
    }


def _residualize(test_scores: np.ndarray, train_scores: np.ndarray, train_text_scores: np.ndarray, test_text_scores: np.ndarray) -> np.ndarray:
    reg = LinearRegression()
    reg.fit(train_text_scores.reshape(-1, 1), train_scores)
    pred = reg.predict(test_text_scores.reshape(-1, 1))
    return test_scores - pred


def _combined_auc(
    train_text_scores: np.ndarray,
    test_text_scores: np.ndarray,
    train_act_scores: np.ndarray,
    test_act_scores: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
) -> float:
    train_pair = np.column_stack([train_text_scores, train_act_scores])
    test_pair = np.column_stack([test_text_scores, test_act_scores])
    model = make_pipeline(
        FunctionTransformer(lambda x: np.asarray(x, dtype=np.float64), validate=False),
        StandardScaler(),
        LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced", max_iter=2000),
    )
    model.fit(train_pair, train_y)
    return _safe_auc(test_y, model.decision_function(test_pair))


def _activation_incremental_cv(
    rows: list[dict[str, Any]],
    feats: dict[str, np.ndarray],
    *,
    text_field: str,
    seed: int,
) -> dict[str, Any]:
    filtered = [row for row in rows if row["key"] in feats]
    y = np.asarray([row["label"] for row in filtered], dtype=np.int64)
    groups = np.asarray([row["dilemma_id"] for row in filtered], dtype=object)
    text = [str(row[text_field]) for row in filtered]
    x = np.stack([feats[row["key"]] for row in filtered], axis=0).astype(np.float32)
    split_kind, splits = _splits(y, groups, seed=seed)

    text_aucs: list[float] = []
    act_aucs: list[float] = []
    combined_aucs: list[float] = []
    residual_act_aucs: list[float] = []
    for train_idx, test_idx in splits:
        if len(set(y[train_idx].tolist())) < 2 or len(set(y[test_idx].tolist())) < 2:
            continue
        train_text_scores, test_text_scores = _fit_text_scores(
            [text[i] for i in train_idx],
            y[train_idx],
            [text[i] for i in test_idx],
        )
        train_act_scores, test_act_scores = _fit_dense_scores(x[train_idx], y[train_idx], x[test_idx])
        residual_scores = _residualize(test_act_scores, train_act_scores, train_text_scores, test_text_scores)

        text_aucs.append(_safe_auc(y[test_idx], test_text_scores))
        act_aucs.append(_safe_auc(y[test_idx], test_act_scores))
        combined_aucs.append(
            _combined_auc(train_text_scores, test_text_scores, train_act_scores, test_act_scores, y[train_idx], y[test_idx])
        )
        residual_auc = _safe_auc(y[test_idx], residual_scores)
        residual_act_aucs.append(max(residual_auc, 1.0 - residual_auc))

    text_mean = float(np.mean(text_aucs)) if text_aucs else float("nan")
    act_mean = float(np.mean(act_aucs)) if act_aucs else float("nan")
    combined_mean = float(np.mean(combined_aucs)) if combined_aucs else float("nan")
    residual_mean = float(np.mean(residual_act_aucs)) if residual_act_aucs else float("nan")
    return {
        "n": int(len(filtered)),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "text_field": text_field,
        "split_kind": split_kind,
        "text_auc_mean": text_mean,
        "activation_auc_mean": act_mean,
        "combined_auc_mean": combined_mean,
        "combined_delta_over_text": float(combined_mean - text_mean) if not math.isnan(combined_mean) and not math.isnan(text_mean) else float("nan"),
        "residual_activation_abs_auc_mean": residual_mean,
        "fold_text_aucs": text_aucs,
        "fold_activation_aucs": act_aucs,
        "fold_combined_aucs": combined_aucs,
        "fold_residual_activation_abs_aucs": residual_act_aucs,
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.3f}"
    return str(value)


def _write_report(summary: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Ethical Advantage Lexical-Confound Analysis",
        "",
        f"- generation rows: `{summary['generation_rows_path']}`",
        f"- action rows: `{summary['action_rows_path']}`",
        f"- capture: `{summary['capture_artifact_id']}`",
        "",
        "## Text-Only Baselines",
        "",
        "AUROC from TF-IDF text only, grouped by dilemma. For first-16 activation claims, the fair lexical comparator is `generated_prefix16_text`; for full-response claims it is `generated_text`; for prompt-end it is prompt/instruction text.",
        "",
        "| target | text field | n | positives | AUROC |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary["text_baselines"]:
        lines.append(
            f"| {row['target']} | {row['text_field']} | {row['n']} | {row['positive']} | {_fmt(row['auc_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Condition-Only Baselines",
            "",
            "| target | n | positives | AUROC |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summary["condition_baselines"]:
        lines.append(f"| {row['target']} | {row['n']} | {row['positive']} | {_fmt(row['auc'])} |")

    lines.extend(
        [
            "",
            "## Activation Beyond Text",
            "",
            "Each row uses the matching text field for the activation slice. `combined delta` is AUROC(text+activation) - AUROC(text). `resid act abs` is activation-score AUROC after linearly removing the text score, sign-normalized away from 0.5.",
            "",
            "| target | slice | layer | text field | n | text AUROC | act AUROC | text+act AUROC | combined delta | resid act abs |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["activation_incremental"]:
        lines.append(
            f"| {row['target']} | {row['slice']} | {row['layer']} | {row['text_field']} | {row['n']} | "
            f"{_fmt(row['text_auc_mean'])} | {_fmt(row['activation_auc_mean'])} | {_fmt(row['combined_auc_mean'])} | "
            f"{_fmt(row['combined_delta_over_text'])} | {_fmt(row['residual_activation_abs_auc_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- If text-only AUROC is already near the activation AUROC, the probe is lexically vulnerable.",
            "- If `combined delta` is near zero, activation adds little beyond text for that target/slice.",
            "- Full-response generated activations are the highest-risk slice because the behavior labeler also reads the full response text.",
        ]
    )
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    *,
    capture_id: str,
    generation_rows_path: Path,
    action_rows_path: Path,
    report_dir: Path,
    seed: int,
) -> None:
    rows_by_key = _rows_by_key(generation_rows_path, action_rows_path)
    targets = {
        "prompt_pole": _target_rows(rows_by_key, "prompt_pole"),
        "observed_action": _target_rows(rows_by_key, "observed_action"),
        "observed_action_within_negative": _target_rows(rows_by_key, "observed_action_within_negative"),
    }

    text_fields = ("instruction_text", "prompt_text", "generated_prefix16_text", "generated_prefix32_text", "generated_text")
    text_baselines: list[dict[str, Any]] = []
    condition_baselines: list[dict[str, Any]] = []
    for target, rows in targets.items():
        condition_baselines.append({"target": target, **_condition_auc(rows)})
        for text_field in text_fields:
            text_baselines.append({"target": target, "text_field": text_field, **_text_cv(rows, text_field, seed=seed)})

    capture = activation._load_capture(capture_id)
    activation_incremental: list[dict[str, Any]] = []
    text_field_by_slice = {
        "prompt_end": "prompt_text",
        "generated_first_16": "generated_prefix16_text",
        "generated_full": "generated_text",
    }
    for target, rows in targets.items():
        for slice_name in SLICES:
            text_field = text_field_by_slice[slice_name]
            for layer in LAYERS:
                feats = activation._feature_map(capture, layer=layer, slice_name=slice_name)
                activation_incremental.append(
                    {
                        "target": target,
                        "slice": slice_name,
                        "layer": layer,
                        **_activation_incremental_cv(rows, feats, text_field=text_field, seed=seed),
                    }
                )

    summary = {
        "capture_artifact_id": capture_id,
        "generation_rows_path": str(generation_rows_path),
        "action_rows_path": str(action_rows_path),
        "text_baselines": text_baselines,
        "condition_baselines": condition_baselines,
        "activation_incremental": activation_incremental,
    }
    _write_report(summary, report_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-id", default=DEFAULT_CAPTURE_ID)
    parser.add_argument("--generation-rows", type=Path, default=DEFAULT_GENERATION_ROWS)
    parser.add_argument("--action-rows", type=Path, default=DEFAULT_ACTION_ROWS)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    analyze(
        capture_id=args.capture_id,
        generation_rows_path=args.generation_rows,
        action_rows_path=args.action_rows,
        report_dir=args.report_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
