"""Prompt-end vs generated-state geometry for phase 03 brief-recommendation capture."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import analyze_natural_prompt_paired as paired


PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_03")
DEFAULT_REPORT_ROOT = PHASE_ROOT / "reports" / "all_theories_brief_recommendation_report"
DEFAULT_REPORT_DIR = PHASE_ROOT / "reports" / "all_theories_brief_recommendation_geometry"

SITES = ("prompt_end_residual", "generated_sequence_residual")
LAYERS = (0, 4, 16, 24, 32, 40)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _stats(values: list[float]) -> dict[str, float]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return {"median": float("nan"), "p95": float("nan")}
    return {"median": float(np.median(clean)), "p95": float(np.percentile(clean, 95))}


def _direction_table(
    *,
    capture: Any,
    row_index: dict[tuple[str, str], dict[str, Any]],
    site: str,
    layer: int,
    trials: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    feats = paired._capture_layer_features(capture, site=site, layer=layer)
    directions: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for theory in paired.THEORIES:
        for construction, (pos, neg) in paired._contrast_pairs(theory).items():
            deltas, meta = paired._paired_deltas(
                row_index=row_index,
                feats=feats,
                pos_condition=pos,
                neg_condition=neg,
            )
            name = f"{paired.THEORY_SHORT[theory]}::{construction}"
            if deltas.shape[0] < 4:
                rows.append({"name": name, "theory": theory, "construction": construction, "n": int(deltas.shape[0])})
                continue
            direction = deltas.mean(axis=0)
            directions[name] = direction
            real = _stats(paired._split_half_distribution(deltas, n_trials=trials, seed=1000 + layer))
            null = _stats(paired._sign_flip_null_distribution(deltas, n_trials=trials, seed=2000 + layer))
            rows.append(
                {
                    "name": name,
                    "theory": theory,
                    "construction": construction,
                    "n": int(deltas.shape[0]),
                    "norm": float(np.linalg.norm(direction)),
                    "split_half_median": real["median"],
                    "null_p95": null["p95"],
                    "gap": real["median"] - null["p95"],
                    "mean_pos_tokens": float(np.mean([m["pos_tokens"] for m in meta])),
                    "mean_neg_tokens": float(np.mean([m["neg_tokens"] for m in meta])),
                }
            )
    return directions, rows


def _cross_theory(directions: dict[str, np.ndarray], construction_suffix: str) -> dict[str, dict[str, float]]:
    selected: dict[str, np.ndarray] = {}
    for theory in paired.THEORIES:
        short = paired.THEORY_SHORT[theory]
        name = f"{short}::{short}_{construction_suffix}"
        if name in directions:
            selected[short] = directions[name]
    return {a: {b: _cos(va, vb) for b, vb in selected.items()} for a, va in selected.items()}


def _generic_alignment(directions: dict[str, np.ndarray]) -> dict[str, float]:
    generic = directions.get("deont::deont_neutral_short")
    # The generic-moral direction is shared, so compute it directly by name below
    # in main where the feature table can see N_generic_moral_01 - N_neutral_01.
    if generic is None:
        return {}
    return {}


def _shared_generic_direction(
    *,
    capture: Any,
    row_index: dict[tuple[str, str], dict[str, Any]],
    site: str,
    layer: int,
) -> np.ndarray:
    feats = paired._capture_layer_features(capture, site=site, layer=layer)
    deltas, _ = paired._paired_deltas(
        row_index=row_index,
        feats=feats,
        pos_condition=paired.GENERIC_MORAL,
        neg_condition=paired.NEUTRAL_SHORT,
    )
    if deltas.shape[0] < 4:
        return np.empty((0,), dtype=np.float32)
    return deltas.mean(axis=0)


def _format_float(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.3f}"
    return str(value)


def _write_report(summary: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Prompt-End vs Generated Geometry")
    lines.append("")
    lines.append(f"- capture artifact: `{summary['capture_artifact_id']}`")
    lines.append(f"- generation rows: `{summary['generation_rows_path']}`")
    lines.append("")
    lines.append("## Cross-Locus Cosines")
    lines.append("")
    lines.append("| layer | theory | construction | cos(prompt_end, generated) | prompt gap | generated gap |")
    lines.append("|---:|---|---|---:|---:|---:|")
    for row in summary["cross_locus"]:
        lines.append(
            f"| {row['layer']} | {row['theory']} | {row['construction']} | "
            f"{_format_float(row['cross_locus_cosine'])} | {_format_float(row['prompt_gap'])} | "
            f"{_format_float(row['generated_gap'])} |"
        )
    lines.append("")
    lines.append("## Generic-Moral Alignment")
    lines.append("")
    lines.append("| site | layer | theory | cos(theory-neutral, generic-neutral) |")
    lines.append("|---|---:|---|---:|")
    for row in summary["generic_alignment"]:
        lines.append(
            f"| {row['site']} | {row['layer']} | {row['theory']} | "
            f"{_format_float(row['cosine_to_generic'])} |"
        )
    lines.append("")
    lines.append("## Cross-Theory Cosines: Neutral-Short Construction")
    for block in summary["cross_theory"]:
        lines.append("")
        lines.append(f"### {block['site']} L{block['layer']}")
        matrix = block["matrix"]
        names = sorted(matrix)
        lines.append("")
        lines.append("| | " + " | ".join(names) + " |")
        lines.append("|---" + "|---:" * len(names) + "|")
        for a in names:
            lines.append(f"| {a} | " + " | ".join(_format_float(matrix[a].get(b)) for b in names) + " |")
    lines.append("")
    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--generation-rows", default=None)
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--trials", type=int, default=128)
    args = parser.parse_args()

    generation_rows = Path(args.generation_rows) if args.generation_rows else paired._latest_generation_rows_path(Path(args.report_root))
    rows = paired._rows(generation_rows)
    row_idx = paired._row_index(rows)
    capture = paired._load_capture(args.capture_id)

    direction_by_site_layer: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    stats_by_site_layer: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    generic_by_site_layer: dict[tuple[str, int], np.ndarray] = {}
    for site in SITES:
        for layer in LAYERS:
            dirs, rows_stats = _direction_table(capture=capture, row_index=row_idx, site=site, layer=layer, trials=args.trials)
            direction_by_site_layer[(site, layer)] = dirs
            stats_by_site_layer[(site, layer)] = {r["name"]: r for r in rows_stats}
            generic_by_site_layer[(site, layer)] = _shared_generic_direction(
                capture=capture,
                row_index=row_idx,
                site=site,
                layer=layer,
            )

    cross_locus: list[dict[str, Any]] = []
    for layer in LAYERS:
        prompt_dirs = direction_by_site_layer[("prompt_end_residual", layer)]
        generated_dirs = direction_by_site_layer[("generated_sequence_residual", layer)]
        for name, prompt_vec in sorted(prompt_dirs.items()):
            if name not in generated_dirs:
                continue
            p_stat = stats_by_site_layer[("prompt_end_residual", layer)].get(name, {})
            g_stat = stats_by_site_layer[("generated_sequence_residual", layer)].get(name, {})
            theory_short, construction = name.split("::", 1)
            cross_locus.append(
                {
                    "layer": layer,
                    "theory": theory_short,
                    "construction": construction,
                    "cross_locus_cosine": _cos(prompt_vec, generated_dirs[name]),
                    "prompt_gap": p_stat.get("gap", float("nan")),
                    "generated_gap": g_stat.get("gap", float("nan")),
                }
            )

    generic_alignment: list[dict[str, Any]] = []
    for (site, layer), dirs in direction_by_site_layer.items():
        generic_vec = generic_by_site_layer[(site, layer)]
        if generic_vec.size == 0:
            continue
        for theory in paired.THEORIES:
            short = paired.THEORY_SHORT[theory]
            name = f"{short}::{short}_neutral_short"
            if name in dirs:
                generic_alignment.append(
                    {
                        "site": site,
                        "layer": layer,
                        "theory": short,
                        "cosine_to_generic": _cos(dirs[name], generic_vec),
                    }
                )

    cross_theory: list[dict[str, Any]] = []
    for site in SITES:
        for layer in LAYERS:
            cross_theory.append(
                {
                    "site": site,
                    "layer": layer,
                    "matrix": _cross_theory(direction_by_site_layer[(site, layer)], "neutral_short"),
                }
            )

    summary = {
        "capture_artifact_id": args.capture_id,
        "generation_rows_path": str(generation_rows),
        "trials": args.trials,
        "sites": list(SITES),
        "layers": list(LAYERS),
        "cross_locus": cross_locus,
        "generic_alignment": generic_alignment,
        "cross_theory": cross_theory,
    }
    _write_report(summary, Path(args.report_dir))
    print(f"wrote {Path(args.report_dir) / 'report.md'}")


if __name__ == "__main__":
    main()
