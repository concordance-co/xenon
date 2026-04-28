"""Layer sweep for Phase 03 generated first-16 theory directions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import analyze_generated_slices as slices
from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import analyze_natural_prompt_paired as paired


DEFAULT_REPORT_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_03/reports/all_theories_brief_recommendation_report")
DEFAULT_REPORT_DIR = Path("projects/MOREBENCH/theory_persona_vectors/phase_03/reports/all_theories_brief_recommendation_first16_layer_sweep")
LAYERS = (0, 4, 16, 24, 32, 40)
THEORY_CONSTRUCTIONS = {
    "deont": ("P_deont_01", "N_neutral_01"),
    "util": ("P_util_01", "N_neutral_01"),
    "generic": ("N_generic_moral_01", "N_neutral_01"),
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
        "# Generated First-16 Layer Sweep",
        "",
        f"- capture artifact: `{summary['capture_artifact_id']}`",
        f"- generation rows: `{summary['generation_rows_path']}`",
        f"- slice: `first_16` generated tokens",
        "",
        "| theory | layer | real median | null p95 | gap | cosine to L32 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['theory']} | {row['layer']} | {_fmt(row['real_median'])} | {_fmt(row['null_p95'])} | "
            f"{_fmt(row['gap'])} | {_fmt(row['cos_to_l32'])} |"
        )
    lines.extend(
        [
            "",
            "## Suggested Write Layers",
            "",
            "Use this as write-site evidence, not as a new readout claim. Earlier layers near the first sustained gap uptick are better first causal write candidates than the latest/highest readout layer.",
            "",
        ]
    )
    for theory, suggestion in summary["suggested_write_layers"].items():
        lines.append(f"- `{theory}`: `{suggestion}`")
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _suggest(rows: list[dict[str, Any]], theory: str) -> str:
    theory_rows = [row for row in rows if row["theory"] == theory and not math.isnan(float(row["gap"]))]
    if not theory_rows:
        return "none"
    # Pick the earliest layer whose gap clears 0.30 and whose direction is still
    # moderately aligned with L32. This is a write-site heuristic, not a gate.
    for row in sorted(theory_rows, key=lambda r: int(r["layer"])):
        if float(row["gap"]) >= 0.30 and (math.isnan(float(row["cos_to_l32"])) or float(row["cos_to_l32"]) >= 0.35):
            return f"L{row['layer']} first layer with gap>=0.30 and cos_to_L32>=0.35"
    best = max(theory_rows, key=lambda r: float(r["gap"]))
    return f"L{best['layer']} best observed gap fallback"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-id", default="capture_1_1d7271d73617")
    parser.add_argument("--generation-rows", default=None)
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--trials", type=int, default=128)
    args = parser.parse_args()

    generation_rows = Path(args.generation_rows) if args.generation_rows else paired._latest_generation_rows_path(Path(args.report_root))
    row_index = paired._row_index(paired._rows(generation_rows))
    capture = paired._load_capture(args.capture_id)

    l32_dirs: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    per_layer_dirs: dict[tuple[str, int], np.ndarray] = {}
    per_layer_deltas: dict[tuple[str, int], np.ndarray] = {}
    for layer in LAYERS:
        feats = slices._feature_slice_map(capture, site="generated_sequence_residual", layer=layer, slice_name="first_16")
        for theory, (pos, neg) in THEORY_CONSTRUCTIONS.items():
            direction, deltas = slices._paired_direction(
                row_index=row_index,
                feats=feats,
                pos_condition=pos,
                neg_condition=neg,
            )
            per_layer_dirs[(theory, layer)] = direction
            per_layer_deltas[(theory, layer)] = deltas
            if layer == 32:
                l32_dirs[theory] = direction

    for layer in LAYERS:
        for theory in THEORY_CONSTRUCTIONS:
            direction = per_layer_dirs[(theory, layer)]
            deltas = per_layer_deltas[(theory, layer)]
            gap = slices._split_gap(deltas, trials=args.trials, layer=layer)
            rows.append(
                {
                    "theory": theory,
                    "layer": layer,
                    **gap,
                    "cos_to_l32": slices._cos(direction, l32_dirs.get(theory, np.empty((0,), dtype=np.float32))),
                }
            )

    summary = {
        "capture_artifact_id": args.capture_id,
        "generation_rows_path": str(generation_rows),
        "trials": args.trials,
        "layers": list(LAYERS),
        "rows": rows,
        "suggested_write_layers": {theory: _suggest(rows, theory) for theory in THEORY_CONSTRUCTIONS},
    }
    _write_report(summary, Path(args.report_dir))
    print(f"wrote {Path(args.report_dir) / 'report.md'}")


if __name__ == "__main__":
    main()
