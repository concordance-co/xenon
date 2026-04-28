"""Generated-token slice geometry for phase 03 brief-recommendation capture."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import analyze_natural_prompt_paired as paired


PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_03")
DEFAULT_REPORT_ROOT = PHASE_ROOT / "reports" / "all_theories_brief_recommendation_report"
DEFAULT_REPORT_DIR = PHASE_ROOT / "reports" / "all_theories_brief_recommendation_generated_slices"

THEORY_CONSTRUCTIONS = {
    "deont": ("P_deont_01", "N_neutral_01"),
    "util": ("P_util_01", "N_neutral_01"),
    "virtue": ("P_virtue_01", "N_neutral_01"),
    "contract": ("P_contract_01", "N_neutral_01"),
}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _slice_array(arr: np.ndarray, slice_name: str) -> np.ndarray:
    if arr.ndim == 1:
        return arr
    n = arr.shape[0]
    if n <= 0:
        return np.empty((0,), dtype=np.float32)
    if slice_name == "full":
        return arr.mean(axis=0)
    if slice_name == "first_third":
        return arr[: max(1, n // 3)].mean(axis=0)
    if slice_name == "middle_third":
        start = n // 3
        end = max(start + 1, (2 * n) // 3)
        return arr[start:end].mean(axis=0)
    if slice_name == "last_third":
        return arr[(2 * n) // 3 :].mean(axis=0)
    if slice_name == "first_16":
        return arr[: min(16, n)].mean(axis=0)
    raise ValueError(f"unknown slice {slice_name!r}")


def _feature_slice_map(capture: Any, *, site: str, layer: int, slice_name: str) -> dict[str, np.ndarray]:
    payload = paired._feature_payload(capture, site)
    layer_payload = payload.get("layers", {}).get(str(layer))
    if not isinstance(layer_payload, Mapping):
        raise RuntimeError(f"missing {site} L{layer}")
    out: dict[str, np.ndarray] = {}
    for key, rec in layer_payload.items():
        values = rec.get("values") if isinstance(rec, Mapping) else None
        if values is None:
            continue
        arr = np.asarray(values, dtype=np.float32)
        vec = _slice_array(arr, slice_name)
        if vec.size:
            out[str(key)] = vec
    return out


def _paired_direction(
    *,
    row_index: dict[tuple[str, str], dict[str, Any]],
    feats: dict[str, np.ndarray],
    pos_condition: str,
    neg_condition: str,
) -> tuple[np.ndarray, np.ndarray]:
    deltas: list[np.ndarray] = []
    for dilemma_id in sorted({d for d, _ in row_index}):
        pos = row_index.get((dilemma_id, pos_condition))
        neg = row_index.get((dilemma_id, neg_condition))
        if not pos or not neg:
            continue
        if pos["key"] not in feats or neg["key"] not in feats:
            continue
        deltas.append(feats[pos["key"]] - feats[neg["key"]])
    if not deltas:
        return np.empty((0,), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
    stacked = np.stack(deltas, axis=0)
    return stacked.mean(axis=0), stacked


def _split_gap(deltas: np.ndarray, *, trials: int, layer: int) -> dict[str, float]:
    if deltas.shape[0] < 4:
        return {"real_median": float("nan"), "null_p95": float("nan"), "gap": float("nan")}
    real = paired._split_half_distribution(deltas, n_trials=trials, seed=1000 + layer)
    null = paired._sign_flip_null_distribution(deltas, n_trials=trials, seed=2000 + layer)
    return {
        "real_median": float(np.median(real)),
        "null_p95": float(np.percentile(null, 95)),
        "gap": float(np.median(real) - np.percentile(null, 95)),
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.3f}"
    return str(value)


def _write_report(summary: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Generated Slice Geometry",
        "",
        f"- capture artifact: `{summary['capture_artifact_id']}`",
        f"- generation rows: `{summary['generation_rows_path']}`",
        f"- layer: L{summary['layer']}",
        "",
        "| theory | slice | split gap | cos(slice, full generated) | cos(slice, prompt-end) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['theory']} | {row['slice']} | {_fmt(row['gap'])} | "
            f"{_fmt(row['cos_to_full_generated'])} | {_fmt(row['cos_to_prompt_end'])} |"
        )
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--generation-rows", default=None)
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--trials", type=int, default=128)
    args = parser.parse_args()

    generation_rows = Path(args.generation_rows) if args.generation_rows else paired._latest_generation_rows_path(Path(args.report_root))
    row_index = paired._row_index(paired._rows(generation_rows))
    capture = paired._load_capture(args.capture_id)

    prompt_feats = _feature_slice_map(capture, site="prompt_end_residual", layer=args.layer, slice_name="full")
    slice_names = ("full", "first_16", "first_third", "middle_third", "last_third")
    generated_feats_by_slice = {
        name: _feature_slice_map(capture, site="generated_sequence_residual", layer=args.layer, slice_name=name)
        for name in slice_names
    }

    full_dirs: dict[str, np.ndarray] = {}
    prompt_dirs: dict[str, np.ndarray] = {}
    for theory, (pos, neg) in THEORY_CONSTRUCTIONS.items():
        full_dirs[theory], _ = _paired_direction(
            row_index=row_index,
            feats=generated_feats_by_slice["full"],
            pos_condition=pos,
            neg_condition=neg,
        )
        prompt_dirs[theory], _ = _paired_direction(
            row_index=row_index,
            feats=prompt_feats,
            pos_condition=pos,
            neg_condition=neg,
        )

    rows: list[dict[str, Any]] = []
    for theory, (pos, neg) in THEORY_CONSTRUCTIONS.items():
        for slice_name in slice_names:
            direction, deltas = _paired_direction(
                row_index=row_index,
                feats=generated_feats_by_slice[slice_name],
                pos_condition=pos,
                neg_condition=neg,
            )
            gap = _split_gap(deltas, trials=args.trials, layer=args.layer)
            rows.append(
                {
                    "theory": theory,
                    "slice": slice_name,
                    **gap,
                    "cos_to_full_generated": _cos(direction, full_dirs[theory]),
                    "cos_to_prompt_end": _cos(direction, prompt_dirs[theory]),
                }
            )

    summary = {
        "capture_artifact_id": args.capture_id,
        "generation_rows_path": str(generation_rows),
        "layer": args.layer,
        "trials": args.trials,
        "rows": rows,
    }
    _write_report(summary, Path(args.report_dir))
    print(f"wrote {Path(args.report_dir) / 'report.md'}")


if __name__ == "__main__":
    main()
