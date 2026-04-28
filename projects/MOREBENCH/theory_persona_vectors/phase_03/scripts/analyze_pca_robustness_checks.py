"""Robustness checks for Phase 03 within-dilemma PCA geometry."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import analyze_behavior_labels_vs_pca as labels_pca
from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import analyze_generated_slices as slices
from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import analyze_within_dilemma_pca as pca


PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_03")
DEFAULT_REPORT_DIR = PHASE_ROOT / "reports" / "model_judged_labels" / "pca_robustness_checks"
LABEL_ROOT = PHASE_ROOT / "reports" / "model_judged_labels"

BASE_CONDITIONS = tuple(
    condition for condition in pca.CONDITION_ORDER if "contractarian" not in condition
)
POSITIVE_THEORY_CONDITIONS = tuple(
    condition for condition in pca.CONDITION_ORDER if condition.startswith("P_")
)
NON_MORAL_OR_DIAGNOSTIC_CONDITIONS = (
    "N_neutral_01",
    "N_neutral_02",
    "N_anti_deont_01",
    "N_anti_util_01",
    "N_anti_virtue_01",
    "N_anti_contract_01",
    "N_anti_contractarian_01",
)
NEUTRAL_ONLY_CONDITIONS = ("N_neutral_01", "N_neutral_02")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 3 or float(a.std()) < 1e-12 or float(b.std()) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    design = np.column_stack([np.ones(y.shape[0]), x])
    beta = np.linalg.pinv(design.T @ design) @ design.T @ y
    return y - design @ beta


def _token_count(text: str) -> int:
    return len(str(text).strip().split())


def _labels(row: Mapping[str, Any]) -> Mapping[str, Any]:
    example = row.get("example")
    if not isinstance(example, Mapping):
        return {}
    labels = example.get("labels")
    return labels if isinstance(labels, Mapping) else {}


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    example = row.get("example")
    if not isinstance(example, Mapping):
        return {}
    metadata = example.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _filtered_matrix(
    *,
    rows_by_key: Mapping[str, Mapping[str, Any]],
    feats: Mapping[str, np.ndarray],
    conditions: tuple[str, ...],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    wanted = set(conditions)
    grouped: dict[str, list[tuple[str, str, np.ndarray, Mapping[str, Any]]]] = defaultdict(list)
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
        if set(conditions) - present:
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
        raise RuntimeError(f"no complete dilemmas for conditions: {conditions}")
    return np.stack(vectors, axis=0), meta


def _moral_active_direction(
    *,
    rows_by_key: Mapping[str, Mapping[str, Any]],
    feats: Mapping[str, np.ndarray],
    positive_conditions: tuple[str, ...],
    baseline_conditions: tuple[str, ...],
) -> np.ndarray:
    deltas: list[np.ndarray] = []
    pos_set = set(positive_conditions)
    base_set = set(baseline_conditions)
    grouped: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for key, row in rows_by_key.items():
        if key not in feats:
            continue
        labels = _labels(row)
        dilemma_id = str(labels.get("dilemma_id") or "")
        condition_id = str(labels.get("condition_id") or "")
        if dilemma_id and condition_id in pos_set | base_set:
            grouped[dilemma_id][condition_id] = np.asarray(feats[key], dtype=np.float32)
    for condition_vectors in grouped.values():
        if not pos_set.issubset(condition_vectors) or not base_set.issubset(condition_vectors):
            continue
        pos = np.stack([condition_vectors[c] for c in positive_conditions], axis=0).mean(axis=0)
        neg = np.stack([condition_vectors[c] for c in baseline_conditions], axis=0).mean(axis=0)
        deltas.append(pos - neg)
    if not deltas:
        raise RuntimeError("no complete moral-active centroid deltas")
    direction = np.stack(deltas, axis=0).mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise RuntimeError("moral-active direction has near-zero norm")
    return direction / norm


def _project_out(matrix: np.ndarray, direction: np.ndarray) -> np.ndarray:
    unit = direction / max(float(np.linalg.norm(direction)), 1e-12)
    return matrix - (matrix @ unit)[:, None] * unit[None, :]


def _condition_centroids(scores: np.ndarray, meta: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for score, item in zip(scores, meta, strict=True):
        grouped[str(item["condition_id"])].append(np.asarray(score, dtype=np.float32))
    return {condition: np.stack(rows, axis=0).mean(axis=0) for condition, rows in grouped.items()}


def _pc_summary(
    *,
    name: str,
    matrix: np.ndarray,
    meta: list[dict[str, Any]],
    components: int,
    label_vectors: Mapping[str, np.ndarray],
    token_counts: np.ndarray,
) -> dict[str, Any]:
    fit = pca._pca(matrix, n_components=components)
    centroids = _condition_centroids(fit["scores"], meta)
    pcs: list[dict[str, Any]] = []
    for pc_idx in range(components):
        scores = fit["scores"][:, pc_idx]
        residual_scores = _residualize(scores, token_counts)
        label_corrs = {}
        for label_name, values in label_vectors.items():
            label_corrs[label_name] = {
                "raw": _corr(scores, values),
                "pc_length_resid": _corr(residual_scores, values),
                "both_length_resid": _corr(residual_scores, _residualize(values, token_counts)),
            }
        values = sorted(
            ((condition, float(scores[pc_idx])) for condition, scores in centroids.items()),
            key=lambda row: row[1],
        )
        pcs.append(
            {
                "pc": pc_idx + 1,
                "explained_variance_ratio": float(fit["explained_variance_ratio"][pc_idx]),
                "label_correlations": label_corrs,
                "condition_extremes": {
                    "negative": values[:5],
                    "positive": values[-5:][::-1],
                },
            }
        )
    return {
        "name": name,
        "n_rows": int(matrix.shape[0]),
        "n_dilemmas": len({str(item["dilemma_id"]) for item in meta}),
        "n_conditions": len({str(item["condition_id"]) for item in meta}),
        "n_features": int(matrix.shape[1]),
        "explained_variance_ratio": fit["explained_variance_ratio"].astype(float).tolist(),
        "pcs": pcs,
        "components": fit["components"].astype(float).tolist(),
    }


def _load_label_vectors(meta: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    content = {row["example_key"]: row for row in _load_jsonl(LABEL_ROOT / "content_scores.jsonl")}
    process = {row["example_key"]: row for row in _load_jsonl(LABEL_ROOT / "process_scores.jsonl")}
    manifest = {row["example_key"]: row for row in _load_jsonl(LABEL_ROOT / "manifest.jsonl")}
    rows = []
    token_counts = []
    for item in meta:
        key = str(item["key"])
        all_values = {**content[key], **process[key]}
        outcome = np.mean([all_values["harm_welfare"], all_values["public_interest_social_impact"]])
        procedural = np.mean(
            [
                all_values["legality_compliance"],
                all_values["procedural_escalation"],
                all_values["risk_mitigation"],
                all_values["conditional_recommendation"],
                all_values["moral_uncertainty"],
            ]
        )
        decisive = float(all_values["priority_resolution"])
        principle = np.mean(
            [
                all_values["rights_autonomy"],
                all_values["fairness_justice"],
                all_values["honesty_truthfulness"],
                all_values["responsibility_accountability"],
                all_values["loyalty_trust"],
                all_values["virtue_character"],
            ]
        )
        rows.append(
            {
                "outcome_content_only": outcome,
                "procedural_risk_management": procedural,
                "decisive_resolution": decisive,
                "procedural_minus_decisive": procedural - decisive,
                "principle_integrity": principle,
                "principle_minus_outcome": principle - outcome,
            }
        )
        token_counts.append(_token_count(manifest[key]["response_text"]))
    return (
        {name: np.asarray([row[name] for row in rows], dtype=np.float32) for name in rows[0]},
        np.asarray(token_counts, dtype=np.float32),
    )


def _run_one(
    *,
    name: str,
    rows_by_key: Mapping[str, Mapping[str, Any]],
    feats: Mapping[str, np.ndarray],
    conditions: tuple[str, ...],
    components: int,
    residualize_moral_active: bool,
) -> dict[str, Any]:
    matrix, meta = _filtered_matrix(rows_by_key=rows_by_key, feats=feats, conditions=conditions)
    if residualize_moral_active:
        moral_active = _moral_active_direction(
            rows_by_key=rows_by_key,
            feats=feats,
            positive_conditions=tuple(c for c in POSITIVE_THEORY_CONDITIONS if c in conditions),
            baseline_conditions=tuple(c for c in NON_MORAL_OR_DIAGNOSTIC_CONDITIONS if c in conditions),
        )
        matrix = _project_out(matrix, moral_active)
    label_vectors, token_counts = _load_label_vectors(meta)
    return _pc_summary(
        name=name,
        matrix=matrix,
        meta=meta,
        components=components,
        label_vectors=label_vectors,
        token_counts=token_counts,
    )


def _run_project_out(
    *,
    name: str,
    rows_by_key: Mapping[str, Mapping[str, Any]],
    feats: Mapping[str, np.ndarray],
    conditions: tuple[str, ...],
    components: int,
    baseline_conditions: tuple[str, ...],
) -> dict[str, Any]:
    matrix, meta = _filtered_matrix(rows_by_key=rows_by_key, feats=feats, conditions=conditions)
    moral_active = _moral_active_direction(
        rows_by_key=rows_by_key,
        feats=feats,
        positive_conditions=tuple(c for c in POSITIVE_THEORY_CONDITIONS if c in conditions),
        baseline_conditions=tuple(c for c in baseline_conditions if c in conditions),
    )
    matrix = _project_out(matrix, moral_active)
    label_vectors, token_counts = _load_label_vectors(meta)
    out = _pc_summary(
        name=name,
        matrix=matrix,
        meta=meta,
        components=components,
        label_vectors=label_vectors,
        token_counts=token_counts,
    )
    out["projected_out_baseline_conditions"] = list(baseline_conditions)
    out["projected_out_positive_conditions"] = [
        c for c in POSITIVE_THEORY_CONDITIONS if c in conditions
    ]
    return out


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.3f}"
    return str(value)


def _write_report(summary: Mapping[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# PCA Robustness Checks",
        "",
        f"- layer: `L{summary['layer']}`",
        f"- components: `{summary['components']}`",
        "",
    ]
    for result in summary["results"]:
        lines.extend(
            [
                f"## {result['name']}",
                "",
                f"- rows: `{result['n_rows']}`",
                f"- dilemmas: `{result['n_dilemmas']}`",
                f"- conditions: `{result['n_conditions']}`",
                f"- EVR top 5: `{[_fmt(x) for x in result['explained_variance_ratio'][:5]]}`",
                "",
                "| PC | EVR | proc-minus-decisive r | principle-minus-outcome r | procedural r | decisive r |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for pc in result["pcs"][:5]:
            corrs = pc["label_correlations"]
            lines.append(
                f"| {pc['pc']} | {_fmt(pc['explained_variance_ratio'])} | "
                f"{_fmt(corrs['procedural_minus_decisive']['both_length_resid'])} | "
                f"{_fmt(corrs['principle_minus_outcome']['both_length_resid'])} | "
                f"{_fmt(corrs['procedural_risk_management']['both_length_resid'])} | "
                f"{_fmt(corrs['decisive_resolution']['both_length_resid'])} |"
            )
        lines.extend(["", "### Condition Extremes", ""])
        for pc in result["pcs"][:3]:
            neg = ", ".join(f"{condition} ({_fmt(score)})" for condition, score in pc["condition_extremes"]["negative"])
            pos = ", ".join(f"{condition} ({_fmt(score)})" for condition, score in pc["condition_extremes"]["positive"])
            lines.append(f"- PC{pc['pc']} negative: {neg}")
            lines.append(f"- PC{pc['pc']} positive: {pos}")
        lines.append("")
    (report_dir / "report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    args = parser.parse_args()

    rows_by_key, generation_rows = pca._load_combined_rows()
    base_capture = pca.paired._load_capture(pca.BASE_CAPTURE_ID)
    contract_capture = pca.paired._load_capture(pca.CONTRACTARIAN_CAPTURE_ID)

    def load_feats(slice_name: str) -> dict[str, np.ndarray]:
        feats: dict[str, np.ndarray] = {}
        feats.update(
            slices._feature_slice_map(
                base_capture,
                site="generated_sequence_residual",
                layer=args.layer,
                slice_name=slice_name,
            )
        )
        feats.update(
            slices._feature_slice_map(
                contract_capture,
                site="generated_sequence_residual",
                layer=args.layer,
                slice_name=slice_name,
            )
        )
        return feats

    first16_feats = load_feats("first_16")
    full_feats = load_feats("full")
    results = [
        _run_one(
            name="all18_first16",
            rows_by_key=rows_by_key,
            feats=first16_feats,
            conditions=pca.CONDITION_ORDER,
            components=args.components,
            residualize_moral_active=False,
        ),
        _run_one(
            name="batchA_15_first16_no_contractarian",
            rows_by_key=rows_by_key,
            feats=first16_feats,
            conditions=BASE_CONDITIONS,
            components=args.components,
            residualize_moral_active=False,
        ),
        _run_one(
            name="all18_first16_project_out_moral_active",
            rows_by_key=rows_by_key,
            feats=first16_feats,
            conditions=pca.CONDITION_ORDER,
            components=args.components,
            residualize_moral_active=True,
        ),
        _run_project_out(
            name="all18_first16_project_out_positive_vs_neutral_only",
            rows_by_key=rows_by_key,
            feats=first16_feats,
            conditions=pca.CONDITION_ORDER,
            components=args.components,
            baseline_conditions=NEUTRAL_ONLY_CONDITIONS,
        ),
        _run_one(
            name="all18_full_response",
            rows_by_key=rows_by_key,
            feats=full_feats,
            conditions=pca.CONDITION_ORDER,
            components=args.components,
            residualize_moral_active=False,
        ),
    ]

    summary = {
        "layer": args.layer,
        "components": args.components,
        "generation_rows": {key: str(value) for key, value in generation_rows.items()},
        "results": results,
    }

    # Components are large; keep them in summary for downstream cosine checks,
    # but omit from the compact report.
    _write_report(summary, Path(args.report_dir))
    print(json.dumps({"report": str(Path(args.report_dir) / "report.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
