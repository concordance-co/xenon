"""Plot generated first-16 theory-vs-neutral AUROC curves across layers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import analyze_generated_slices as slices
from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import analyze_natural_prompt_paired as paired


DEFAULT_REPORT_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_03/reports/all_theories_brief_recommendation_report")
DEFAULT_REPORT_DIR = Path("projects/MOREBENCH/theory_persona_vectors/phase_03/reports/all_theories_brief_recommendation_first16_auroc_curve")
LAYERS = (0, 4, 16, 24, 32, 40)
THEORY_CONSTRUCTIONS = {
    "deont": ("P_deont_01", "N_neutral_01"),
    "util": ("P_util_01", "N_neutral_01"),
    "generic": ("N_generic_moral_01", "N_neutral_01"),
}


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Mann-Whitney AUROC with average tie handling.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(scores, dtype=np.float64)
    i = 0
    while i < scores.size:
        j = i + 1
        while j < scores.size and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    pos_rank_sum = float(ranks[labels == 1].sum())
    return (pos_rank_sum - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)


def _condition_features(
    *,
    row_index: dict[tuple[str, str], dict[str, Any]],
    feats: dict[str, np.ndarray],
    pos_condition: str,
    neg_condition: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    dilemma_ids: list[str] = []
    for dilemma_id in sorted({d for d, _ in row_index}):
        pos = row_index.get((dilemma_id, pos_condition))
        neg = row_index.get((dilemma_id, neg_condition))
        if not pos or not neg:
            continue
        if pos["key"] not in feats or neg["key"] not in feats:
            continue
        xs.extend([feats[pos["key"]], feats[neg["key"]]])
        ys.extend([1, 0])
        dilemma_ids.extend([dilemma_id, dilemma_id])
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int32), dilemma_ids


def _loo_auroc(
    *,
    row_index: dict[tuple[str, str], dict[str, Any]],
    feats: dict[str, np.ndarray],
    pos_condition: str,
    neg_condition: str,
) -> float:
    scores: list[float] = []
    labels: list[int] = []
    dilemma_ids = sorted({d for d, _ in row_index})
    for heldout in dilemma_ids:
        train_deltas: list[np.ndarray] = []
        held_x: list[np.ndarray] = []
        held_y: list[int] = []
        for dilemma_id in dilemma_ids:
            pos = row_index.get((dilemma_id, pos_condition))
            neg = row_index.get((dilemma_id, neg_condition))
            if not pos or not neg:
                continue
            if pos["key"] not in feats or neg["key"] not in feats:
                continue
            if dilemma_id == heldout:
                held_x.extend([feats[pos["key"]], feats[neg["key"]]])
                held_y.extend([1, 0])
            else:
                train_deltas.append(feats[pos["key"]] - feats[neg["key"]])
        if not train_deltas or len(held_x) != 2:
            continue
        direction = np.stack(train_deltas, axis=0).mean(axis=0)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            continue
        unit = direction / norm
        scores.extend([float(np.dot(x, unit)) for x in held_x])
        labels.extend(held_y)
    return _auroc(np.asarray(scores), np.asarray(labels))


def _fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def _write_outputs(summary: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    for theory in THEORY_CONSTRUCTIONS:
        rows = [row for row in summary["rows"] if row["theory"] == theory]
        ax.plot([row["layer"] for row in rows], [row["loo_auroc"] for row in rows], marker="o", label=theory)
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Leave-one-dilemma-out AUROC")
    ax.set_title("Generated First-16 Theory-vs-Neutral AUROC")
    ax.set_ylim(0.45, 1.01)
    ax.set_xticks(list(LAYERS))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "auroc_curve.png")
    plt.close(fig)

    lines = [
        "# Generated First-16 AUROC Curve",
        "",
        f"- capture artifact: `{summary['capture_artifact_id']}`",
        f"- generation rows: `{summary['generation_rows_path']}`",
        f"- slice: `first_16` generated tokens",
        f"- metric: leave-one-dilemma-out AUROC against neutral",
        "",
        "![AUROC curve](/Users/trentelmore/Projects/concordance/xenon/projects/MOREBENCH/theory_persona_vectors/phase_03/reports/all_theories_brief_recommendation_first16_auroc_curve/auroc_curve.png)",
        "",
        "| theory | layer | LOO AUROC | in-sample AUROC |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(f"| {row['theory']} | {row['layer']} | {_fmt(row['loo_auroc'])} | {_fmt(row['insample_auroc'])} |")
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-id", default="capture_1_1d7271d73617")
    parser.add_argument("--generation-rows", default=None)
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    args = parser.parse_args()

    generation_rows = Path(args.generation_rows) if args.generation_rows else paired._latest_generation_rows_path(Path(args.report_root))
    row_index = paired._row_index(paired._rows(generation_rows))
    capture = paired._load_capture(args.capture_id)

    rows: list[dict[str, Any]] = []
    for layer in LAYERS:
        feats = slices._feature_slice_map(capture, site="generated_sequence_residual", layer=layer, slice_name="first_16")
        for theory, (pos, neg) in THEORY_CONSTRUCTIONS.items():
            x, y, _ = _condition_features(row_index=row_index, feats=feats, pos_condition=pos, neg_condition=neg)
            direction = x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0)
            norm = float(np.linalg.norm(direction))
            unit = direction / norm if norm > 1e-12 else direction
            scores = x @ unit
            rows.append(
                {
                    "theory": theory,
                    "layer": layer,
                    "loo_auroc": _loo_auroc(row_index=row_index, feats=feats, pos_condition=pos, neg_condition=neg),
                    "insample_auroc": _auroc(scores, y),
                }
            )

    summary = {
        "capture_artifact_id": args.capture_id,
        "generation_rows_path": str(generation_rows),
        "rows": rows,
        "layers": list(LAYERS),
        "theories": list(THEORY_CONSTRUCTIONS),
    }
    _write_outputs(summary, Path(args.report_dir))
    print(f"wrote {Path(args.report_dir) / 'report.md'}")


if __name__ == "__main__":
    main()
