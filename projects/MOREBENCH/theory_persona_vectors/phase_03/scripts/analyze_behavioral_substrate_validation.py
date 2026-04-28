"""Behavioral substrate validation for Phase 03 model-judged labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_03")
LABEL_ROOT = PHASE_ROOT / "reports" / "model_judged_labels"
DEFAULT_REPORT_DIR = LABEL_ROOT / "behavioral_substrate_validation"

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
FEATURES = CONTENT_FEATURES + PROCESS_FEATURES


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _standardize(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (x - mu) / sd


def _condition_separation_stat(x: np.ndarray, conditions: list[str]) -> float:
    overall = x.mean(axis=0)
    stat = 0.0
    for condition in sorted(set(conditions)):
        idx = np.asarray([c == condition for c in conditions], dtype=bool)
        center = x[idx].mean(axis=0)
        stat += int(idx.sum()) * float(np.sum((center - overall) ** 2))
    return stat


def _within_dilemma_condition_null(
    *,
    x: np.ndarray,
    conditions: list[str],
    dilemmas: list[str],
    trials: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    indices_by_dilemma: dict[str, list[int]] = defaultdict(list)
    for i, dilemma in enumerate(dilemmas):
        indices_by_dilemma[dilemma].append(i)
    values = []
    for _ in range(trials):
        fake = list(conditions)
        for idxs in indices_by_dilemma.values():
            shuffled = [fake[i] for i in idxs]
            rng.shuffle(shuffled)
            for i, value in zip(idxs, shuffled, strict=True):
                fake[i] = value
        values.append(_condition_separation_stat(x, fake))
    return values


def _ridge_mahalanobis(
    *,
    x: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    ridge: float = 0.2,
) -> float:
    cov = np.cov(x, rowvar=False)
    scale = float(np.trace(cov) / max(1, cov.shape[0]))
    cov = cov + np.eye(cov.shape[0]) * ridge * scale
    diff = a - b
    return float(diff @ np.linalg.pinv(cov) @ diff)


def analyze(*, trials: int, seed: int) -> dict[str, Any]:
    content = {row["example_key"]: row for row in _load_jsonl(LABEL_ROOT / "content_scores.jsonl")}
    process = {row["example_key"]: row for row in _load_jsonl(LABEL_ROOT / "process_scores.jsonl")}
    manifest = _load_jsonl(LABEL_ROOT / "manifest.jsonl")
    rows = []
    for row in manifest:
        key = row["example_key"]
        values = {**content[key], **process[key]}
        rec = {
            "example_key": key,
            "dilemma_id": row["dilemma_id"],
            "condition_id": row["condition_id"],
            "source": row["source"],
        }
        for feature in FEATURES:
            rec[feature] = int(values[feature])
        rec["procedural_risk_management"] = float(
            np.mean(
                [
                    values["legality_compliance"],
                    values["procedural_escalation"],
                    values["risk_mitigation"],
                    values["conditional_recommendation"],
                    values["moral_uncertainty"],
                ]
            )
        )
        rec["decisive_resolution"] = float(values["priority_resolution"])
        rec["procedural_decisive_axis"] = rec["procedural_risk_management"] - rec["decisive_resolution"]
        rec["virtue_character_singleton"] = float(values["virtue_character"])
        rows.append(rec)

    feature_names = list(FEATURES) + [
        "procedural_risk_management",
        "decisive_resolution",
        "procedural_decisive_axis",
        "virtue_character_singleton",
    ]
    x_raw = np.asarray([[row[f] for f in feature_names] for row in rows], dtype=np.float32)
    x = _standardize(x_raw)
    conditions = [row["condition_id"] for row in rows]
    dilemmas = [row["dilemma_id"] for row in rows]

    stat = _condition_separation_stat(x, conditions)
    null = _within_dilemma_condition_null(x=x, conditions=conditions, dilemmas=dilemmas, trials=trials, seed=seed)
    p_value = (1 + sum(value >= stat for value in null)) / (1 + len(null))

    by_condition: dict[str, list[int]] = defaultdict(list)
    for i, condition in enumerate(conditions):
        by_condition[condition].append(i)
    condition_means = {}
    for condition, idxs in sorted(by_condition.items()):
        arr = x_raw[idxs]
        condition_means[condition] = {
            "n": len(idxs),
            **{feature: float(arr[:, j].mean()) for j, feature in enumerate(feature_names)},
        }

    pairwise = []
    for i, a in enumerate(sorted(by_condition)):
        for b in sorted(by_condition)[i + 1 :]:
            a_mean = x[by_condition[a]].mean(axis=0)
            b_mean = x[by_condition[b]].mean(axis=0)
            pairwise.append(
                {
                    "condition_a": a,
                    "condition_b": b,
                    "euclidean_z": float(np.linalg.norm(a_mean - b_mean)),
                    "mahalanobis_ridge": _ridge_mahalanobis(x=x, a=a_mean, b=b_mean),
                    "procedural_decisive_delta": condition_means[a]["procedural_decisive_axis"]
                    - condition_means[b]["procedural_decisive_axis"],
                    "virtue_delta": condition_means[a]["virtue_character_singleton"]
                    - condition_means[b]["virtue_character_singleton"],
                }
            )
    pairwise.sort(key=lambda row: row["euclidean_z"], reverse=True)

    feature_eta = []
    for j, feature in enumerate(feature_names):
        y = x_raw[:, j]
        overall = float(y.mean())
        ss_total = float(np.sum((y - overall) ** 2))
        ss_between = 0.0
        for idxs in by_condition.values():
            mean = float(y[idxs].mean())
            ss_between += len(idxs) * (mean - overall) ** 2
        feature_eta.append(
            {
                "feature": feature,
                "eta_squared_by_condition": ss_between / ss_total if ss_total > 1e-12 else float("nan"),
            }
        )
    feature_eta.sort(key=lambda row: row["eta_squared_by_condition"], reverse=True)

    return {
        "n_rows": len(rows),
        "n_dilemmas": len(set(dilemmas)),
        "n_conditions": len(set(conditions)),
        "features": feature_names,
        "condition_separation_stat": stat,
        "within_dilemma_permutation_null_p95": float(np.percentile(null, 95)),
        "within_dilemma_permutation_p_value": p_value,
        "condition_counts": dict(sorted(Counter(conditions).items())),
        "condition_means": condition_means,
        "top_pairwise_distances": pairwise[:40],
        "feature_eta_squared": feature_eta,
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_report(summary: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Behavioral Substrate Validation",
        "",
        f"- rows: `{summary['n_rows']}`",
        f"- dilemmas: `{summary['n_dilemmas']}`",
        f"- conditions: `{summary['n_conditions']}`",
        f"- global condition-separation stat: `{_fmt(summary['condition_separation_stat'])}`",
        f"- within-dilemma permutation null p95: `{_fmt(summary['within_dilemma_permutation_null_p95'])}`",
        f"- permutation p-value: `{_fmt(summary['within_dilemma_permutation_p_value'])}`",
        "",
        "## Most Condition-Sensitive Labels",
        "",
        "| feature | eta^2 by condition |",
        "|---|---:|",
    ]
    for row in summary["feature_eta_squared"][:20]:
        lines.append(f"| `{row['feature']}` | {_fmt(row['eta_squared_by_condition'])} |")

    lines.extend(["", "## Condition Means: Primary Axes", ""])
    lines.append("| condition | n | procedural_risk | decisive | procedural_decisive | virtue |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for condition, row in summary["condition_means"].items():
        lines.append(
            f"| `{condition}` | {row['n']} | {_fmt(row['procedural_risk_management'])} | "
            f"{_fmt(row['decisive_resolution'])} | {_fmt(row['procedural_decisive_axis'])} | "
            f"{_fmt(row['virtue_character_singleton'])} |"
        )

    lines.extend(["", "## Largest Pairwise Behavioral Distances", ""])
    lines.append("| condition A | condition B | euclidean z | ridge mahalanobis | proc-decisive delta | virtue delta |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in summary["top_pairwise_distances"][:25]:
        lines.append(
            f"| `{row['condition_a']}` | `{row['condition_b']}` | {_fmt(row['euclidean_z'])} | "
            f"{_fmt(row['mahalanobis_ridge'])} | {_fmt(row['procedural_decisive_delta'])} | "
            f"{_fmt(row['virtue_delta'])} |"
        )
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    args = parser.parse_args()
    summary = analyze(trials=args.trials, seed=args.seed)
    write_report(summary, Path(args.report_dir))
    print(json.dumps({"report": str(Path(args.report_dir) / "report.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
