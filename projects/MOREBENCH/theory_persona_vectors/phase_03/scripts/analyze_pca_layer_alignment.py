"""Cross-layer alignment for within-dilemma PCA components."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import analyze_within_dilemma_pca as pca


DEFAULT_REPORT_DIR = pca.PHASE_ROOT / "reports" / "within_dilemma_pca_layer_alignment"


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.3f}"
    return str(value)


def _fit_components(*, rows_by_key: dict[str, dict[str, Any]], site: str, slice_name: str, layers: list[int], components: int) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for layer in layers:
        feats = pca._load_feature_map(site=site, layer=layer, slice_name=slice_name)
        matrix, _, _ = pca._build_matrix(rows_by_key=rows_by_key, feats=feats)
        out[layer] = pca._pca(matrix, n_components=components)["components"]
    return out


def _alignment_matrix(a: np.ndarray, b: np.ndarray) -> list[list[float]]:
    return [[abs(_cos(a[i], b[j])) for j in range(b.shape[0])] for i in range(a.shape[0])]


def _best_matches(a: np.ndarray, b: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    matrix = _alignment_matrix(a, b)
    for i, values in enumerate(matrix):
        best_j = int(np.nanargmax(np.asarray(values, dtype=np.float32)))
        rows.append({"source_pc": i + 1, "target_pc": best_j + 1, "abs_cosine": float(values[best_j])})
    return rows


def analyze(*, layers: list[int], components: int) -> dict[str, Any]:
    rows_by_key, generation_rows = pca._load_combined_rows()
    generated = _fit_components(
        rows_by_key=rows_by_key,
        site="generated_sequence_residual",
        slice_name="first_16",
        layers=layers,
        components=components,
    )
    prompt = _fit_components(
        rows_by_key=rows_by_key,
        site="prompt_end_residual",
        slice_name="full",
        layers=layers,
        components=components,
    )

    def site_summary(components_by_layer: dict[int, np.ndarray]) -> dict[str, Any]:
        adjacent = []
        for left, right in zip(layers[:-1], layers[1:], strict=True):
            adjacent.append(
                {
                    "left_layer": left,
                    "right_layer": right,
                    "best_matches": _best_matches(components_by_layer[left], components_by_layer[right]),
                    "matrix": _alignment_matrix(components_by_layer[left], components_by_layer[right]),
                }
            )
        reference_layer = 32 if 32 in components_by_layer else layers[-1]
        reference = []
        for layer in layers:
            if layer == reference_layer:
                continue
            reference.append(
                {
                    "reference_layer": reference_layer,
                    "target_layer": layer,
                    "best_matches": _best_matches(components_by_layer[reference_layer], components_by_layer[layer]),
                    "matrix": _alignment_matrix(components_by_layer[reference_layer], components_by_layer[layer]),
                }
            )
        return {"adjacent": adjacent, "reference": reference}

    return {
        "layers": layers,
        "components": components,
        "generation_rows": {key: str(value) for key, value in generation_rows.items()},
        "generated_first16": site_summary(generated),
        "prompt_end": site_summary(prompt),
    }


def write_report(summary: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Within-Dilemma PCA Cross-Layer Alignment",
        "",
        "Cosines are absolute because PCA component signs are arbitrary.",
        "Rows report the best target-layer PC match for each source-layer PC.",
        "",
    ]
    for site_key, title in (("generated_first16", "Generated First-16"), ("prompt_end", "Prompt-End")):
        lines.extend([f"## {title}", "", "### Adjacent Layers", ""])
        for block in summary[site_key]["adjacent"]:
            lines.append(f"#### L{block['left_layer']} -> L{block['right_layer']}")
            lines.append("")
            lines.append("| source PC | best target PC | abs cosine |")
            lines.append("|---:|---:|---:|")
            for row in block["best_matches"]:
                lines.append(f"| {row['source_pc']} | {row['target_pc']} | {_fmt(row['abs_cosine'])} |")
            lines.append("")
        lines.extend(["### L32 Reference", ""])
        for block in summary[site_key]["reference"]:
            lines.append(f"#### L{block['reference_layer']} -> L{block['target_layer']}")
            lines.append("")
            lines.append("| source PC | best target PC | abs cosine |")
            lines.append("|---:|---:|---:|")
            for row in block["best_matches"]:
                lines.append(f"| {row['source_pc']} | {row['target_pc']} | {_fmt(row['abs_cosine'])} |")
            lines.append("")
    (report_dir / "report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 4, 16, 24, 32, 40])
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    args = parser.parse_args()

    summary = analyze(layers=args.layers, components=args.components)
    write_report(summary, Path(args.report_dir))
    print(json.dumps({"report": str(Path(args.report_dir) / "report.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
