"""Prompt-vs-generated analysis for the controlled deontology isolation capture."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import analyze_generated_slices as slices
from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import analyze_natural_prompt_paired as paired
from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import plot_generated_first16_auroc_curve as auroc_curve


PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_03")
DEFAULT_REPORT_ROOT = PHASE_ROOT / "reports" / "deont_prompt_isolation_report"
DEFAULT_OUTPUT_DIR = PHASE_ROOT / "reports" / "deont_prompt_isolation_analysis"
DEFAULT_CAPTURE_ID = None
LAYERS = (0, 4, 16, 24, 32, 40)
SITE_SPECS = (
    ("prompt_end", "prompt_end_residual", "full"),
    ("generated_full", "generated_sequence_residual", "full"),
    ("generated_first_16", "generated_sequence_residual", "first_16"),
)
DIRECT_TASKS = (
    ("deont01_vs_generic", "P_deont_iso_01", "N_generic_moral_iso_01"),
    ("deont02_vs_generic", "P_deont_iso_02", "N_generic_moral_iso_01"),
    ("deont01_vs_neutral", "P_deont_iso_01", "N_neutral_iso_01"),
    ("deont02_vs_neutral", "P_deont_iso_02", "N_neutral_iso_01"),
    ("deont01_vs_deont02", "P_deont_iso_01", "P_deont_iso_02"),
    ("deont01_vs_anti", "P_deont_iso_01", "N_anti_deont_iso_01"),
    ("deont02_vs_anti", "P_deont_iso_02", "N_anti_deont_iso_01"),
)
TRANSFER_TASKS = (
    ("deont01_vs_generic", "deont02_vs_generic", "P_deont_iso_01", "N_generic_moral_iso_01", "P_deont_iso_02", "N_generic_moral_iso_01"),
    ("deont01_vs_neutral", "deont02_vs_neutral", "P_deont_iso_01", "N_neutral_iso_01", "P_deont_iso_02", "N_neutral_iso_01"),
)


def _latest_generation_rows_path(report_root: Path) -> Path:
    candidates = sorted(
        report_root.glob("report_*/results/generate_natural_responses_results.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no generate_natural_responses result found under {report_root}")
    return candidates[0]


def _latest_capture_id(report_root: Path) -> str:
    candidates = sorted(report_root.glob("report_*/report.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no report.json found under {report_root}")
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    inputs = payload.get("inputs")
    if not isinstance(inputs, list):
        raise TypeError("report.json must contain inputs")
    for item in inputs:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("step_name") or "") == "capture_residuals":
            artifact_id = str(item.get("artifact_id") or "")
            if artifact_id:
                return artifact_id
    raise KeyError("could not find capture_residuals artifact_id in report.json")


def _feature_map(capture: Any, *, site: str, layer: int, slice_name: str) -> dict[str, np.ndarray]:
    if site == "prompt_end_residual":
        return paired._capture_layer_features(capture, site=site, layer=layer)
    return slices._feature_slice_map(capture, site=site, layer=layer, slice_name=slice_name)


def _paired_examples(
    *,
    dilemma_ids: Sequence[str],
    row_index: dict[tuple[str, str], dict[str, Any]],
    feats: dict[str, np.ndarray],
    pos_condition: str,
    neg_condition: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dilemma_id in dilemma_ids:
        pos = row_index.get((dilemma_id, pos_condition))
        neg = row_index.get((dilemma_id, neg_condition))
        if not pos or not neg:
            continue
        if pos["key"] not in feats or neg["key"] not in feats:
            continue
        rows.append(
            {
                "dilemma_id": dilemma_id,
                "pos_key": pos["key"],
                "neg_key": neg["key"],
                "pos_vec": feats[pos["key"]],
                "neg_vec": feats[neg["key"]],
            }
        )
    return rows


def _direction_from_pairs(rows: list[dict[str, Any]]) -> np.ndarray:
    if not rows:
        return np.empty((0,), dtype=np.float32)
    deltas = np.stack([row["pos_vec"] - row["neg_vec"] for row in rows], axis=0)
    return deltas.mean(axis=0)


def _auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    return float(auroc_curve._auroc(np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int32)))


def _balanced_accuracy(scores: Sequence[float], labels: Sequence[int]) -> float:
    score_arr = np.asarray(scores, dtype=np.float64)
    label_arr = np.asarray(labels, dtype=np.int32)
    preds = (score_arr >= 0.0).astype(np.int32)
    pos = label_arr == 1
    neg = label_arr == 0
    pos_acc = float(np.mean(preds[pos] == 1)) if np.any(pos) else float("nan")
    neg_acc = float(np.mean(preds[neg] == 0)) if np.any(neg) else float("nan")
    if math.isnan(pos_acc) or math.isnan(neg_acc):
        return float("nan")
    return 0.5 * (pos_acc + neg_acc)


def _loo_same_task(
    *,
    dilemma_ids: Sequence[str],
    row_index: dict[tuple[str, str], dict[str, Any]],
    feats: dict[str, np.ndarray],
    pos_condition: str,
    neg_condition: str,
) -> dict[str, Any]:
    scores: list[float] = []
    labels: list[int] = []
    margins: list[float] = []
    for heldout in dilemma_ids:
        train_rows = _paired_examples(
            dilemma_ids=[d for d in dilemma_ids if d != heldout],
            row_index=row_index,
            feats=feats,
            pos_condition=pos_condition,
            neg_condition=neg_condition,
        )
        test_rows = _paired_examples(
            dilemma_ids=[heldout],
            row_index=row_index,
            feats=feats,
            pos_condition=pos_condition,
            neg_condition=neg_condition,
        )
        if not train_rows or len(test_rows) != 1:
            continue
        direction = _direction_from_pairs(train_rows)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            continue
        unit = direction / norm
        pos_score = float(np.dot(test_rows[0]["pos_vec"], unit))
        neg_score = float(np.dot(test_rows[0]["neg_vec"], unit))
        scores.extend([pos_score, neg_score])
        labels.extend([1, 0])
        margins.append(pos_score - neg_score)
    if not scores:
        return {
            "n_pairs": 0,
            "auroc": float("nan"),
            "balanced_accuracy": float("nan"),
            "median_margin": float("nan"),
            "pair_margin_positive_rate": float("nan"),
        }
    return {
        "n_pairs": len(margins),
        "auroc": _auroc(scores, labels),
        "balanced_accuracy": _balanced_accuracy(scores, labels),
        "median_margin": float(np.median(margins)),
        "pair_margin_positive_rate": float(np.mean(np.asarray(margins) > 0)),
    }


def _loo_transfer(
    *,
    dilemma_ids: Sequence[str],
    row_index: dict[tuple[str, str], dict[str, Any]],
    feats: dict[str, np.ndarray],
    train_pos: str,
    train_neg: str,
    eval_pos: str,
    eval_neg: str,
) -> dict[str, Any]:
    scores: list[float] = []
    labels: list[int] = []
    margins: list[float] = []
    for heldout in dilemma_ids:
        train_rows = _paired_examples(
            dilemma_ids=[d for d in dilemma_ids if d != heldout],
            row_index=row_index,
            feats=feats,
            pos_condition=train_pos,
            neg_condition=train_neg,
        )
        test_rows = _paired_examples(
            dilemma_ids=[heldout],
            row_index=row_index,
            feats=feats,
            pos_condition=eval_pos,
            neg_condition=eval_neg,
        )
        if not train_rows or len(test_rows) != 1:
            continue
        direction = _direction_from_pairs(train_rows)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            continue
        unit = direction / norm
        pos_score = float(np.dot(test_rows[0]["pos_vec"], unit))
        neg_score = float(np.dot(test_rows[0]["neg_vec"], unit))
        scores.extend([pos_score, neg_score])
        labels.extend([1, 0])
        margins.append(pos_score - neg_score)
    if not scores:
        return {
            "n_pairs": 0,
            "auroc": float("nan"),
            "balanced_accuracy": float("nan"),
            "median_margin": float("nan"),
            "pair_margin_positive_rate": float("nan"),
        }
    return {
        "n_pairs": len(margins),
        "auroc": _auroc(scores, labels),
        "balanced_accuracy": _balanced_accuracy(scores, labels),
        "median_margin": float(np.median(margins)),
        "pair_margin_positive_rate": float(np.mean(np.asarray(margins) > 0)),
    }


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.3f}"
    return str(value)


def _write_report(summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Deont Prompt Isolation Capture Analysis",
        "",
        f"- capture artifact: `{summary['capture_artifact_id']}`",
        f"- generation rows: `{summary['generation_rows_path']}`",
        f"- layers: `{', '.join(str(x) for x in summary['layers'])}`",
        "",
        "## Direct Readouts",
        "",
        "| site | layer | task | pairs | pair acc | BA | AUROC | median margin |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["direct_rows"]:
        lines.append(
            f"| {row['site_name']} | {row['layer']} | {row['task']} | {row['n_pairs']} | "
            f"{_fmt(row['pair_margin_positive_rate'])} | {_fmt(row['balanced_accuracy'])} | "
            f"{_fmt(row['auroc'])} | {_fmt(row['median_margin'])} |"
        )
    lines.extend(
        [
            "",
            "## Transfer Readouts",
            "",
            "| site | layer | train task | eval task | pairs | pair acc | BA | AUROC | median margin |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["transfer_rows"]:
        lines.append(
            f"| {row['site_name']} | {row['layer']} | {row['train_task']} | {row['eval_task']} | "
            f"{row['n_pairs']} | {_fmt(row['pair_margin_positive_rate'])} | {_fmt(row['balanced_accuracy'])} | "
            f"{_fmt(row['auroc'])} | {_fmt(row['median_margin'])} |"
        )
    lines.extend(
        [
            "",
            "## Prompt-vs-Generated Direction Cosines",
            "",
            "| layer | task | cos(prompt_end, generated_full) | cos(prompt_end, generated_first_16) | cos(generated_full, generated_first_16) |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in summary["cosine_rows"]:
        lines.append(
            f"| {row['layer']} | {row['task']} | {_fmt(row['prompt_to_generated_full'])} | "
            f"{_fmt(row['prompt_to_generated_first16'])} | {_fmt(row['generated_full_to_first16'])} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-id", default=DEFAULT_CAPTURE_ID)
    parser.add_argument("--generation-rows", default=None)
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    report_root = Path(args.report_root)
    generation_rows = Path(args.generation_rows) if args.generation_rows else _latest_generation_rows_path(report_root)
    capture_id = args.capture_id or _latest_capture_id(report_root)

    rows = paired._rows(generation_rows)
    row_index = paired._row_index(rows)
    dilemma_ids = sorted({d for d, _ in row_index})
    capture = paired._load_capture(capture_id)

    direct_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    direction_bank: dict[tuple[str, int, str], np.ndarray] = {}

    for site_name, site, slice_name in SITE_SPECS:
        for layer in LAYERS:
            feats = _feature_map(capture, site=site, layer=layer, slice_name=slice_name)
            for task, pos_condition, neg_condition in DIRECT_TASKS:
                rows_same = _paired_examples(
                    dilemma_ids=dilemma_ids,
                    row_index=row_index,
                    feats=feats,
                    pos_condition=pos_condition,
                    neg_condition=neg_condition,
                )
                direction_bank[(site_name, layer, task)] = _direction_from_pairs(rows_same)
                direct_rows.append(
                    {
                        "site_name": site_name,
                        "layer": layer,
                        "task": task,
                        **_loo_same_task(
                            dilemma_ids=dilemma_ids,
                            row_index=row_index,
                            feats=feats,
                            pos_condition=pos_condition,
                            neg_condition=neg_condition,
                        ),
                    }
                )
            for train_task, eval_task, train_pos, train_neg, eval_pos, eval_neg in TRANSFER_TASKS:
                transfer_rows.append(
                    {
                        "site_name": site_name,
                        "layer": layer,
                        "train_task": train_task,
                        "eval_task": eval_task,
                        **_loo_transfer(
                            dilemma_ids=dilemma_ids,
                            row_index=row_index,
                            feats=feats,
                            train_pos=train_pos,
                            train_neg=train_neg,
                            eval_pos=eval_pos,
                            eval_neg=eval_neg,
                        ),
                    }
                )

    cosine_rows: list[dict[str, Any]] = []
    for layer in LAYERS:
        for task, _, _ in DIRECT_TASKS:
            cosine_rows.append(
                {
                    "layer": layer,
                    "task": task,
                    "prompt_to_generated_full": _cos(
                        direction_bank[("prompt_end", layer, task)],
                        direction_bank[("generated_full", layer, task)],
                    ),
                    "prompt_to_generated_first16": _cos(
                        direction_bank[("prompt_end", layer, task)],
                        direction_bank[("generated_first_16", layer, task)],
                    ),
                    "generated_full_to_first16": _cos(
                        direction_bank[("generated_full", layer, task)],
                        direction_bank[("generated_first_16", layer, task)],
                    ),
                }
            )

    summary = {
        "capture_artifact_id": capture_id,
        "generation_rows_path": str(generation_rows),
        "layers": list(LAYERS),
        "direct_rows": direct_rows,
        "transfer_rows": transfer_rows,
        "cosine_rows": cosine_rows,
    }
    _write_report(summary, Path(args.output_dir))
    print(f"wrote {Path(args.output_dir) / 'report.md'}")


if __name__ == "__main__":
    main()
