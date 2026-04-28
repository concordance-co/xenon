"""Lexical controls for generated-slice PCA/behavior correlations.

This analysis asks whether behavioral-composite correlations with full-response
activation PCs survive after controlling for response text with a simple TF-IDF
ridge model. It intentionally avoids sklearn so it can run in the project env.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import analyze_generated_slices as slices
from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import analyze_within_dilemma_pca as pca


PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_03")
LABEL_ROOT = PHASE_ROOT / "reports" / "model_judged_labels"
DEFAULT_REPORT_DIR = LABEL_ROOT / "generated_slice_lexical_control"

TOKEN_RE = re.compile(r"[a-z][a-z0-9_'-]{1,}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _labels(row: Mapping[str, Any]) -> Mapping[str, Any]:
    example = row.get("example")
    if not isinstance(example, Mapping):
        return {}
    labels = example.get("labels")
    return labels if isinstance(labels, Mapping) else {}


def _filtered_matrix(
    *,
    rows_by_key: Mapping[str, Mapping[str, Any]],
    feats: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    grouped: dict[str, list[tuple[str, str, np.ndarray, Mapping[str, Any]]]] = defaultdict(list)
    wanted = set(pca.CONDITION_ORDER)
    for key, row in rows_by_key.items():
        if key not in feats:
            continue
        labels = _labels(row)
        dilemma_id = str(labels.get("dilemma_id") or "")
        condition_id = str(labels.get("condition_id") or "")
        if not dilemma_id or condition_id not in wanted:
            continue
        grouped[dilemma_id].append((key, condition_id, np.asarray(feats[key], dtype=np.float32), labels))
    vectors: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for dilemma_id in sorted(grouped):
        items = grouped[dilemma_id]
        present = {condition_id for _, condition_id, _, _ in items}
        if wanted - present:
            continue
        stack = np.stack([vec for _, _, vec, _ in items], axis=0)
        center = stack.mean(axis=0)
        for key, condition_id, vec, labels in items:
            vectors.append(vec - center)
            meta.append(
                {
                    "key": key,
                    "dilemma_id": dilemma_id,
                    "condition_id": condition_id,
                    "condition_role": labels.get("condition_role"),
                    "condition_theory": labels.get("condition_theory"),
                }
            )
    if not vectors:
        raise RuntimeError("no complete within-dilemma vectors")
    return np.stack(vectors, axis=0), meta


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 3 or float(a.std()) < 1e-12 or float(b.std()) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom < 1e-12:
        return float("nan")
    return float(1 - np.sum((y - pred) ** 2) / denom)


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _build_vocab(texts: list[str], *, max_features: int, min_df: int) -> list[str]:
    df: Counter[str] = Counter()
    tf: Counter[str] = Counter()
    for text in texts:
        terms = _tokenize(text)
        tf.update(terms)
        df.update(set(terms))
    candidates = [
        term for term, count in df.items()
        if count >= min_df and len(term) > 1
    ]
    candidates.sort(key=lambda term: (tf[term], df[term], term), reverse=True)
    return candidates[:max_features]


def _tfidf_matrix(texts: list[str], vocab: list[str]) -> np.ndarray:
    index = {term: i for i, term in enumerate(vocab)}
    df = np.zeros(len(vocab), dtype=np.float32)
    counts_by_doc: list[Counter[str]] = []
    for text in texts:
        counts = Counter(term for term in _tokenize(text) if term in index)
        counts_by_doc.append(counts)
        for term in counts:
            df[index[term]] += 1
    n = len(texts)
    idf = np.log((1 + n) / (1 + df)) + 1
    x = np.zeros((n, len(vocab)), dtype=np.float32)
    for row, counts in enumerate(counts_by_doc):
        total = sum(counts.values()) or 1
        for term, count in counts.items():
            x[row, index[term]] = (count / total) * idf[index[term]]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return x / norms


def _standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (x_train - mu) / sd, (x_test - mu) / sd


def _ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, *, alpha: float) -> np.ndarray:
    x_train = np.column_stack([np.ones(x_train.shape[0], dtype=np.float32), x_train])
    x_test = np.column_stack([np.ones(x_test.shape[0], dtype=np.float32), x_test])
    reg = np.eye(x_train.shape[1], dtype=np.float32) * alpha
    reg[0, 0] = 0.0
    beta = np.linalg.pinv(x_train.T @ x_train + reg) @ x_train.T @ y_train
    return x_test @ beta


def _cv_predict_by_dilemma(
    *,
    texts: list[str],
    y: np.ndarray,
    dilemmas: list[str],
    max_features: int,
    min_df: int,
    alpha: float,
) -> np.ndarray:
    pred = np.zeros_like(y, dtype=np.float32)
    for dilemma in sorted(set(dilemmas)):
        test = np.asarray([d == dilemma for d in dilemmas], dtype=bool)
        train = ~test
        vocab = _build_vocab([texts[i] for i in np.where(train)[0]], max_features=max_features, min_df=min_df)
        x_train = _tfidf_matrix([texts[i] for i in np.where(train)[0]], vocab)
        x_test = _tfidf_matrix([texts[i] for i in np.where(test)[0]], vocab)
        x_train, x_test = _standardize_train_test(x_train, x_test)
        pred[test] = _ridge_predict(x_train, y[train], x_test, alpha=alpha)
    return pred


def _load_label_vectors(meta: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    content = {row["example_key"]: row for row in _load_jsonl(LABEL_ROOT / "content_scores.jsonl")}
    process = {row["example_key"]: row for row in _load_jsonl(LABEL_ROOT / "process_scores.jsonl")}
    rows = []
    for item in meta:
        allv = {**content[str(item["key"])], **process[str(item["key"])]}
        outcome = np.mean([allv["harm_welfare"], allv["public_interest_social_impact"]])
        procedural = np.mean(
            [
                allv["legality_compliance"],
                allv["procedural_escalation"],
                allv["risk_mitigation"],
                allv["conditional_recommendation"],
                allv["moral_uncertainty"],
            ]
        )
        decisive = float(allv["priority_resolution"])
        principle = np.mean(
            [
                allv["rights_autonomy"],
                allv["fairness_justice"],
                allv["honesty_truthfulness"],
                allv["responsibility_accountability"],
                allv["loyalty_trust"],
                allv["virtue_character"],
            ]
        )
        rows.append(
            {
                "procedural_minus_decisive": procedural - decisive,
                "procedural_risk_management": procedural,
                "decisive_resolution": decisive,
                "principle_minus_outcome": principle - outcome,
                "virtue_character": float(allv["virtue_character"]),
            }
        )
    return {key: np.asarray([row[key] for row in rows], dtype=np.float32) for key in rows[0]}


def _prefix_words(text: str, n: int) -> str:
    return " ".join(str(text).strip().split()[:n])


def analyze(
    *,
    slice_name: str,
    text_scope: str,
    max_features: int,
    min_df: int,
    alpha: float,
    components: int,
) -> dict[str, Any]:
    rows_by_key, _ = pca._load_combined_rows()
    base_capture = pca.paired._load_capture(pca.BASE_CAPTURE_ID)
    contract_capture = pca.paired._load_capture(pca.CONTRACTARIAN_CAPTURE_ID)
    feats: dict[str, np.ndarray] = {}
    for capture in (base_capture, contract_capture):
        feats.update(
            slices._feature_slice_map(
                capture,
                site="generated_sequence_residual",
                layer=32,
                slice_name=slice_name,
            )
        )
    matrix, meta = _filtered_matrix(rows_by_key=rows_by_key, feats=feats)
    fit = pca._pca(matrix, n_components=components)
    manifest = {row["example_key"]: row for row in _load_jsonl(LABEL_ROOT / "manifest.jsonl")}
    full_texts = [str(manifest[str(item["key"])]["response_text"]) for item in meta]
    if text_scope == "full_response":
        texts = full_texts
    elif text_scope == "first16_words":
        texts = [_prefix_words(text, 16) for text in full_texts]
    else:
        raise ValueError(f"unknown text_scope {text_scope!r}")
    dilemmas = [str(item["dilemma_id"]) for item in meta]
    labels = _load_label_vectors(meta)

    # CV text predictions for both behavior labels and PC scores.
    pc_preds = {
        f"PC{i + 1}": _cv_predict_by_dilemma(
            texts=texts,
            y=fit["scores"][:, i],
            dilemmas=dilemmas,
            max_features=max_features,
            min_df=min_df,
            alpha=alpha,
        )
        for i in range(components)
    }
    label_preds = {
        name: _cv_predict_by_dilemma(
            texts=texts,
            y=values,
            dilemmas=dilemmas,
            max_features=max_features,
            min_df=min_df,
            alpha=alpha,
        )
        for name, values in labels.items()
    }

    rows = []
    for i in range(components):
        pc_name = f"PC{i + 1}"
        pc_scores = fit["scores"][:, i]
        pc_resid = pc_scores - pc_preds[pc_name]
        for label_name, values in labels.items():
            label_resid = values - label_preds[label_name]
            rows.append(
                {
                    "pc": i + 1,
                    "label": label_name,
                    "raw_corr": _corr(pc_scores, values),
                    "pc_text_resid_corr": _corr(pc_resid, values),
                    "label_text_resid_corr": _corr(pc_scores, label_resid),
                    "both_text_resid_corr": _corr(pc_resid, label_resid),
                    "text_predicts_pc_r2": _r2(pc_scores, pc_preds[pc_name]),
                    "text_predicts_label_r2": _r2(values, label_preds[label_name]),
                }
            )

    return {
        "n_rows": int(matrix.shape[0]),
        "n_dilemmas": len(set(dilemmas)),
        "n_conditions": len({str(item["condition_id"]) for item in meta}),
        "layer": 32,
        "slice": slice_name,
        "text_scope": text_scope,
        "components": components,
        "max_features": max_features,
        "min_df": min_df,
        "ridge_alpha": alpha,
        "explained_variance_ratio": fit["explained_variance_ratio"].astype(float).tolist(),
        "rows": rows,
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.3f}"
    return str(value)


def write_report(summary: Mapping[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Generated Slice Lexical Control",
        "",
        f"- rows: `{summary['n_rows']}`",
        f"- layer/slice: `L{summary['layer']} {summary['slice']}`",
        f"- text scope: `{summary['text_scope']}`",
        f"- TF-IDF max features: `{summary['max_features']}`",
        f"- TF-IDF min df: `{summary['min_df']}`",
        f"- ridge alpha: `{summary['ridge_alpha']}`",
        "",
        "| PC | label | raw r | PC text-resid r | label text-resid r | both text-resid r | text->PC R2 | text->label R2 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    focus = {"procedural_minus_decisive", "principle_minus_outcome", "virtue_character"}
    for row in summary["rows"]:
        if int(row["pc"]) <= 3 and row["label"] in focus:
            lines.append(
                f"| {row['pc']} | `{row['label']}` | {_fmt(row['raw_corr'])} | "
                f"{_fmt(row['pc_text_resid_corr'])} | {_fmt(row['label_text_resid_corr'])} | "
                f"{_fmt(row['both_text_resid_corr'])} | {_fmt(row['text_predicts_pc_r2'])} | "
                f"{_fmt(row['text_predicts_label_r2'])} |"
            )
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", choices=("full", "first_16", "first_third", "middle_third", "last_third"), default="full")
    parser.add_argument("--text-scope", choices=("full_response", "first16_words"), default="full_response")
    parser.add_argument("--max-features", type=int, default=2000)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    args = parser.parse_args()
    summary = analyze(
        slice_name=args.slice,
        text_scope=args.text_scope,
        max_features=args.max_features,
        min_df=args.min_df,
        alpha=args.alpha,
        components=args.components,
    )
    write_report(summary, Path(args.report_dir))
    print(json.dumps({"report": str(Path(args.report_dir) / "report.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
