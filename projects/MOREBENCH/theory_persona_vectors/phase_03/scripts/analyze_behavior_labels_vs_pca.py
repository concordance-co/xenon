"""Relate behavioral labels to within-dilemma PCA scores."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import analyze_within_dilemma_pca as pca


PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_03")
DEFAULT_REPORT_DIR = PHASE_ROOT / "reports" / "behavior_labels_vs_pca"
ETHICAL_LABEL_DIR = PHASE_ROOT / "reports" / "ethical_content_labels"
PROCESS_LABEL_DIR = PHASE_ROOT / "reports" / "process_feature_labels"

CONTENT_FEATURES = (
    "harm_welfare",
    "rights_autonomy",
    "fairness_justice",
    "honesty_truthfulness",
    "responsibility_accountability",
    "loyalty_trust",
    "legality_compliance",
    "public_interest_social_impact",
    "virtue_character",
    "care_compassion",
)

PROCESS_FEATURES = (
    "stakeholder_identification",
    "consequence_forecasting",
    "tradeoff_acknowledged",
    "priority_resolution",
    "moral_uncertainty",
    "risk_mitigation",
    "conditional_recommendation",
    "procedural_escalation",
)


def _latest_jsonl(directory: Path) -> Path:
    candidates = sorted(directory.glob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no .jsonl label output found under {directory}")
    return candidates[0]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _score_from_row(row: dict[str, Any], feature: str) -> float:
    for container_key in ("scores", "labels", "dimensions", "process_features"):
        value = row.get(container_key)
        if isinstance(value, dict) and feature in value:
            return float(value[feature])
    if feature in row:
        return float(row[feature])
    return float("nan")


def _example_key(row: dict[str, Any]) -> str:
    return str(row.get("example_key") or row.get("key") or "").strip()


def _load_labels(path: Path, features: tuple[str, ...]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in _read_jsonl(path):
        key = _example_key(row)
        if not key:
            continue
        scores = {feature: _score_from_row(row, feature) for feature in features}
        out[key] = scores
    return out


def _fit_pca_scores(*, layer: int, components: int) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
    rows_by_key, _ = pca._load_combined_rows()
    feats = pca._load_feature_map(site="generated_sequence_residual", layer=layer, slice_name="first_16")
    matrix, meta, _ = pca._build_matrix(rows_by_key=rows_by_key, feats=feats)
    fit = pca._pca(matrix, n_components=components)
    return fit["scores"], meta, fit["explained_variance_ratio"]


def _standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=0, keepdims=True)
    sigma = x_train.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    return (x_train - mu) / sigma, (x_test - mu) / sigma


def _one_hot(values: list[str]) -> tuple[np.ndarray, list[str]]:
    categories = sorted(set(values))
    index = {category: i for i, category in enumerate(categories)}
    x = np.zeros((len(values), len(categories)), dtype=np.float32)
    for row, value in enumerate(values):
        x[row, index[value]] = 1.0
    return x, categories


def _ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, *, alpha: float = 1.0) -> np.ndarray:
    x_train = np.column_stack([np.ones(x_train.shape[0], dtype=np.float32), x_train])
    x_test = np.column_stack([np.ones(x_test.shape[0], dtype=np.float32), x_test])
    reg = np.eye(x_train.shape[1], dtype=np.float32) * alpha
    reg[0, 0] = 0.0
    beta = np.linalg.pinv(x_train.T @ x_train + reg) @ x_train.T @ y_train
    return x_test @ beta


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom < 1e-12:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def _loo_dilemma_r2(x: np.ndarray, y: np.ndarray, dilemmas: list[str]) -> float:
    preds = np.zeros_like(y, dtype=np.float32)
    dilemma_values = sorted(set(dilemmas))
    for dilemma in dilemma_values:
        test = np.asarray([d == dilemma for d in dilemmas], dtype=bool)
        train = ~test
        x_train, x_test = _standardize_train_test(x[train], x[test])
        preds[test] = _ridge_predict(x_train, y[train], x_test)
    return _r2(y, preds)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return float("nan")
    a = a[mask]
    b = b[mask]
    if float(a.std()) < 1e-12 or float(b.std()) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _feature_distribution(values: np.ndarray) -> dict[str, Any]:
    clean = values[np.isfinite(values)]
    counts = Counter(int(v) for v in clean)
    n = int(clean.size)
    majority = max(counts.values()) / n if n else float("nan")
    return {
        "n": n,
        "mean": float(clean.mean()) if n else float("nan"),
        "std": float(clean.std()) if n else float("nan"),
        "counts": {str(key): int(value) for key, value in sorted(counts.items())},
        "majority_rate": float(majority),
        "low_variance_drop_candidate": bool(n and majority >= 0.9),
    }


def _condition_means(meta: list[dict[str, Any]], matrix: np.ndarray, columns: list[str]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for row, item in zip(matrix, meta, strict=True):
        grouped[str(item["condition_id"])].append(np.asarray(row, dtype=np.float32))
    out = {}
    for condition_id in pca.CONDITION_ORDER:
        rows = grouped.get(condition_id, [])
        if rows:
            mean = np.stack(rows, axis=0).mean(axis=0)
            out[condition_id] = {col: float(mean[i]) for i, col in enumerate(columns)}
    return out


def analyze(*, ethical_path: Path, process_path: Path, layer: int, components: int) -> dict[str, Any]:
    ethical = _load_labels(ethical_path, CONTENT_FEATURES)
    process = _load_labels(process_path, PROCESS_FEATURES)
    scores, meta, evr = _fit_pca_scores(layer=layer, components=components)

    joined_rows = []
    x_content = []
    x_process = []
    x_all = []
    pc_rows = []
    joined_meta = []
    for pc_score, item in zip(scores, meta, strict=True):
        key = str(item["key"])
        if key not in ethical or key not in process:
            continue
        content_vec = np.asarray([ethical[key][feature] for feature in CONTENT_FEATURES], dtype=np.float32)
        process_vec = np.asarray([process[key][feature] for feature in PROCESS_FEATURES], dtype=np.float32)
        if not (np.isfinite(content_vec).all() and np.isfinite(process_vec).all()):
            continue
        x_content.append(content_vec)
        x_process.append(process_vec)
        x_all.append(np.concatenate([content_vec, process_vec]))
        pc_rows.append(np.asarray(pc_score, dtype=np.float32))
        joined_meta.append(item)
        joined_rows.append(
            {
                "key": key,
                "dilemma_id": item["dilemma_id"],
                "condition_id": item["condition_id"],
                "condition_role": item["condition_role"],
                "condition_theory": item["condition_theory"],
            }
        )

    if not joined_rows:
        raise RuntimeError("no rows joined between labels and PCA scores")

    x_content_arr = np.stack(x_content, axis=0)
    x_process_arr = np.stack(x_process, axis=0)
    x_all_arr = np.stack(x_all, axis=0)
    pc_arr = np.stack(pc_rows, axis=0)
    dilemmas = [str(item["dilemma_id"]) for item in joined_meta]
    theories = [str(item["condition_theory"]) for item in joined_meta]
    condition_ids = [str(item["condition_id"]) for item in joined_meta]
    x_theory, theory_categories = _one_hot(theories)
    x_condition, condition_categories = _one_hot(condition_ids)

    feature_names = list(CONTENT_FEATURES) + list(PROCESS_FEATURES)
    all_labels = x_all_arr

    feature_summaries = {
        name: _feature_distribution(all_labels[:, i])
        for i, name in enumerate(feature_names)
    }

    correlations = []
    regressions = []
    for pc_idx in range(components):
        y = pc_arr[:, pc_idx]
        correlations.extend(
            {
                "pc": pc_idx + 1,
                "feature": name,
                "correlation": _corr(all_labels[:, i], y),
            }
            for i, name in enumerate(feature_names)
        )
        for name, x in (
            ("content", x_content_arr),
            ("process", x_process_arr),
            ("content_plus_process", x_all_arr),
            ("theory_onehot", x_theory),
            ("condition_onehot", x_condition),
        ):
            regressions.append(
                {
                    "pc": pc_idx + 1,
                    "predictor_set": name,
                    "loo_dilemma_r2": _loo_dilemma_r2(x, y, dilemmas),
                }
            )

    label_condition_means = _condition_means(joined_meta, all_labels, feature_names)
    pc_condition_means = _condition_means(joined_meta, pc_arr, [f"PC{i}" for i in range(1, components + 1)])

    return {
        "layer": layer,
        "components": components,
        "ethical_path": str(ethical_path),
        "process_path": str(process_path),
        "joined_n": len(joined_rows),
        "dilemma_n": len(set(dilemmas)),
        "condition_n": len(set(condition_ids)),
        "explained_variance_ratio": evr.astype(float).tolist(),
        "theory_categories": theory_categories,
        "condition_categories": condition_categories,
        "feature_summaries": feature_summaries,
        "correlations": correlations,
        "regressions": regressions,
        "label_condition_means": label_condition_means,
        "pc_condition_means": pc_condition_means,
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.3f}"
    return str(value)


def write_report(summary: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Behavioral Labels vs PCA",
        "",
        f"- ethical labels: `{summary['ethical_path']}`",
        f"- process labels: `{summary['process_path']}`",
        f"- layer: `L{summary['layer']} generated first16`",
        f"- joined rows: `{summary['joined_n']}`",
        f"- dilemmas: `{summary['dilemma_n']}`",
        f"- conditions: `{summary['condition_n']}`",
        "",
        "## Feature Distributions",
        "",
        "| feature | mean | std | counts | majority rate | drop? |",
        "|---|---:|---:|---|---:|---|",
    ]
    for name, stats in summary["feature_summaries"].items():
        lines.append(
            f"| `{name}` | {_fmt(stats['mean'])} | {_fmt(stats['std'])} | "
            f"`{json.dumps(stats['counts'], sort_keys=True)}` | {_fmt(stats['majority_rate'])} | "
            f"`{stats['low_variance_drop_candidate']}` |"
        )
    lines.extend(["", "## LOO Dilemma R2 Predicting Activation PCs", ""])
    lines.append("| PC | content | process | content+process | theory one-hot | condition one-hot |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    by_pc_pred = defaultdict(dict)
    for row in summary["regressions"]:
        by_pc_pred[int(row["pc"])][row["predictor_set"]] = row["loo_dilemma_r2"]
    for pc_idx in range(1, summary["components"] + 1):
        row = by_pc_pred[pc_idx]
        lines.append(
            f"| {pc_idx} | {_fmt(row.get('content', float('nan')))} | {_fmt(row.get('process', float('nan')))} | "
            f"{_fmt(row.get('content_plus_process', float('nan')))} | {_fmt(row.get('theory_onehot', float('nan')))} | "
            f"{_fmt(row.get('condition_onehot', float('nan')))} |"
        )
    lines.extend(["", "## Strongest Feature Correlations By PC", ""])
    correlations = summary["correlations"]
    for pc_idx in range(1, summary["components"] + 1):
        rows = [row for row in correlations if int(row["pc"]) == pc_idx and math.isfinite(float(row["correlation"]))]
        rows.sort(key=lambda row: abs(float(row["correlation"])), reverse=True)
        lines.append(f"### PC{pc_idx}")
        lines.append("")
        lines.append("| feature | corr |")
        lines.append("|---|---:|")
        for row in rows[:10]:
            lines.append(f"| `{row['feature']}` | {_fmt(float(row['correlation']))} |")
        lines.append("")
    (report_dir / "report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ethical-labels", default=None)
    parser.add_argument("--process-labels", default=None)
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    args = parser.parse_args()

    ethical_path = Path(args.ethical_labels) if args.ethical_labels else _latest_jsonl(ETHICAL_LABEL_DIR)
    process_path = Path(args.process_labels) if args.process_labels else _latest_jsonl(PROCESS_LABEL_DIR)
    summary = analyze(ethical_path=ethical_path, process_path=process_path, layer=args.layer, components=args.components)
    write_report(summary, Path(args.report_dir))
    print(json.dumps({"report": str(Path(args.report_dir) / "report.md"), "joined_n": summary["joined_n"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
