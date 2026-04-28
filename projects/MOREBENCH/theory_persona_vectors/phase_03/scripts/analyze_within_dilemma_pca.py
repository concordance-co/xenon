"""Within-dilemma PCA discovery over Phase 03 theory-persona captures."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import analyze_generated_slices as slices
from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import analyze_natural_prompt_paired as paired


PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_03")
DEFAULT_REPORT_DIR = PHASE_ROOT / "reports" / "within_dilemma_pca_discovery"
BASE_REPORT_ROOT = PHASE_ROOT / "reports" / "all_theories_brief_recommendation_report"
BASE_REPORT_DIR = BASE_REPORT_ROOT / "report_6aa730c32d87_8c1df9a2"
CONTRACTARIAN_REPORT_ROOT = (
    BASE_REPORT_ROOT
    / "morebench_theory_persona_vectors_phase03_brief_recommendation_smoke_anti_contractarian_contractarian_contractarian"
)

BASE_CAPTURE_ID = "capture_1_1d7271d73617"
CONTRACTARIAN_CAPTURE_ID = "capture_1_c24f680774a7"
LAYERS = (16, 32)
DEFAULT_COMPONENTS = 8

CONDITION_ORDER = (
    "N_neutral_01",
    "N_neutral_02",
    "N_generic_moral_01",
    "P_deont_01",
    "P_deont_02",
    "P_util_01",
    "P_util_02",
    "P_virtue_01",
    "P_virtue_02",
    "P_contract_01",
    "P_contract_02",
    "P_contractarian_01",
    "P_contractarian_02",
    "N_anti_deont_01",
    "N_anti_util_01",
    "N_anti_virtue_01",
    "N_anti_contract_01",
    "N_anti_contractarian_01",
)


def _latest_generation_rows(root: Path) -> Path:
    candidates = sorted(root.glob("report_*/results/generate_natural_responses_results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no generation result under {root}")
    return candidates[0]


def _rows_by_key(path: Path) -> dict[str, dict[str, Any]]:
    rows = paired._rows(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        example = row.get("example") if isinstance(row.get("example"), Mapping) else {}
        key = str(row.get("example_key") or example.get("key") or "")
        if key:
            out[key] = dict(row)
    return out


def _labels(row: Mapping[str, Any]) -> dict[str, Any]:
    example = row.get("example") if isinstance(row.get("example"), Mapping) else {}
    labels = example.get("labels") if isinstance(example.get("labels"), Mapping) else {}
    return dict(labels)


def _condition_role(condition_id: str, labels_by_condition: Mapping[str, Mapping[str, Any]]) -> str:
    labels = labels_by_condition.get(condition_id, {})
    return str(labels.get("condition_role") or "")


def _condition_theory(condition_id: str, labels_by_condition: Mapping[str, Mapping[str, Any]]) -> str:
    labels = labels_by_condition.get(condition_id, {})
    return str(labels.get("condition_theory") or "")


def _load_combined_rows() -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    base_generation = BASE_REPORT_DIR / "results" / "generate_natural_responses_results.json"
    contractarian_generation = _latest_generation_rows(CONTRACTARIAN_REPORT_ROOT)
    rows = {}
    rows.update(_rows_by_key(base_generation))
    rows.update(_rows_by_key(contractarian_generation))
    return rows, {
        "base_generation_rows": base_generation,
        "contractarian_generation_rows": contractarian_generation,
    }


def _load_feature_map(*, site: str, layer: int, slice_name: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for capture_id in (BASE_CAPTURE_ID, CONTRACTARIAN_CAPTURE_ID):
        capture = paired._load_capture(capture_id)
        if site == "generated_sequence_residual":
            out.update(slices._feature_slice_map(capture, site=site, layer=layer, slice_name=slice_name))
        else:
            raw = paired._capture_layer_features(capture, site=site, layer=layer)
            out.update(raw)
    return out


def _build_matrix(
    *,
    rows_by_key: Mapping[str, Mapping[str, Any]],
    feats: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    grouped: dict[str, list[tuple[str, str, np.ndarray, dict[str, Any]]]] = defaultdict(list)
    labels_by_condition: dict[str, dict[str, Any]] = {}
    for key, row in rows_by_key.items():
        if key not in feats:
            continue
        labels = _labels(row)
        dilemma_id = str(labels.get("dilemma_id") or "")
        condition_id = str(labels.get("condition_id") or "")
        if not dilemma_id or condition_id not in CONDITION_ORDER:
            continue
        labels_by_condition.setdefault(condition_id, labels)
        grouped[dilemma_id].append((key, condition_id, np.asarray(feats[key], dtype=np.float32), labels))

    vectors: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for dilemma_id in sorted(grouped):
        items = grouped[dilemma_id]
        condition_ids = {condition_id for _, condition_id, _, _ in items}
        if set(CONDITION_ORDER) - condition_ids:
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
        raise RuntimeError("no complete within-dilemma centered vectors found")
    return np.stack(vectors, axis=0), meta, labels_by_condition


def _pca(matrix: np.ndarray, n_components: int) -> dict[str, np.ndarray]:
    x = matrix.astype(np.float32)
    # Within-dilemma centering is already done. Remove any remaining global offset.
    x = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    n = max(1, x.shape[0] - 1)
    eigenvalues = (s**2) / n
    total = float(np.sum(eigenvalues))
    components = vt[:n_components].astype(np.float32)
    scores = x @ components.T
    return {
        "components": components,
        "scores": scores.astype(np.float32),
        "singular_values": s[:n_components].astype(np.float32),
        "explained_variance": eigenvalues[:n_components].astype(np.float32),
        "explained_variance_ratio": (eigenvalues[:n_components] / total).astype(np.float32) if total > 0 else np.zeros(n_components, dtype=np.float32),
    }


def _condition_centroid_scores(scores: np.ndarray, meta: list[dict[str, Any]]) -> dict[str, list[float]]:
    by_condition: dict[str, list[np.ndarray]] = defaultdict(list)
    for score, item in zip(scores, meta, strict=True):
        by_condition[str(item["condition_id"])].append(np.asarray(score, dtype=np.float32))
    out: dict[str, list[float]] = {}
    for condition_id in CONDITION_ORDER:
        arrs = by_condition.get(condition_id, [])
        if arrs:
            out[condition_id] = np.stack(arrs, axis=0).mean(axis=0).astype(float).tolist()
    return out


def _condition_top_bottom(
    centroid_scores: Mapping[str, list[float]],
    labels_by_condition: Mapping[str, Mapping[str, Any]],
    *,
    pc_index: int,
    k: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    values = []
    for condition_id, scores in centroid_scores.items():
        if len(scores) <= pc_index:
            continue
        values.append((condition_id, float(scores[pc_index])))
    values.sort(key=lambda item: item[1])

    def pack(items: list[tuple[str, float]]) -> list[dict[str, Any]]:
        return [
            {
                "condition_id": condition_id,
                "score": score,
                "role": _condition_role(condition_id, labels_by_condition),
                "theory": _condition_theory(condition_id, labels_by_condition),
            }
            for condition_id, score in items
        ]

    return {"negative": pack(values[:k]), "positive": pack(values[-k:][::-1])}


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return float("nan")
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _cv_loading_stability(matrix: np.ndarray, meta: list[dict[str, Any]], *, n_components: int, seed: int) -> list[dict[str, float]]:
    dilemmas = sorted({str(item["dilemma_id"]) for item in meta})
    rng = np.random.default_rng(seed)
    rng.shuffle(dilemmas)
    split = set(dilemmas[: len(dilemmas) // 2])

    train_idx = [i for i, item in enumerate(meta) if str(item["dilemma_id"]) in split]
    test_idx = [i for i, item in enumerate(meta) if str(item["dilemma_id"]) not in split]
    train = matrix[train_idx]
    test = matrix[test_idx]
    train_meta = [meta[i] for i in train_idx]
    test_meta = [meta[i] for i in test_idx]

    fit = _pca(train, n_components=n_components)
    components = fit["components"]
    train_scores = (train - train.mean(axis=0, keepdims=True)) @ components.T
    test_scores = (test - train.mean(axis=0, keepdims=True)) @ components.T
    train_centroids = _condition_centroid_scores(train_scores, train_meta)
    test_centroids = _condition_centroid_scores(test_scores, test_meta)

    rows: list[dict[str, float]] = []
    common_conditions = [condition for condition in CONDITION_ORDER if condition in train_centroids and condition in test_centroids]
    for pc in range(n_components):
        train_vec = np.asarray([train_centroids[c][pc] for c in common_conditions], dtype=np.float32)
        test_vec = np.asarray([test_centroids[c][pc] for c in common_conditions], dtype=np.float32)
        value = _corr(train_vec, test_vec)
        if not math.isnan(value) and value < 0:
            value = -value
        rows.append({"pc": pc + 1, "abs_condition_loading_corr": value})
    return rows


def _analyze_site_layer(
    *,
    site: str,
    layer: int,
    slice_name: str,
    rows_by_key: Mapping[str, Mapping[str, Any]],
    n_components: int,
) -> dict[str, Any]:
    feats = _load_feature_map(site=site, layer=layer, slice_name=slice_name)
    matrix, meta, labels_by_condition = _build_matrix(rows_by_key=rows_by_key, feats=feats)
    fit = _pca(matrix, n_components=n_components)
    centroid_scores = _condition_centroid_scores(fit["scores"], meta)
    pcs = []
    for pc in range(n_components):
        pcs.append(
            {
                "pc": pc + 1,
                "explained_variance_ratio": float(fit["explained_variance_ratio"][pc]),
                "singular_value": float(fit["singular_values"][pc]),
                "condition_extremes": _condition_top_bottom(centroid_scores, labels_by_condition, pc_index=pc),
            }
        )
    return {
        "site": site,
        "slice": slice_name,
        "layer": layer,
        "n_rows": int(matrix.shape[0]),
        "n_dilemmas": len({str(item["dilemma_id"]) for item in meta}),
        "n_conditions": len({str(item["condition_id"]) for item in meta}),
        "n_features": int(matrix.shape[1]),
        "explained_variance_ratio": fit["explained_variance_ratio"].astype(float).tolist(),
        "condition_centroid_scores": centroid_scores,
        "cv_loading_stability": _cv_loading_stability(matrix, meta, n_components=n_components, seed=1701 + layer),
        "pcs": pcs,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def _write_report(summary: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Within-Dilemma PCA Discovery",
        "",
        "PCA is run on residual activations after centering each dilemma across the full condition manifold.",
        "This removes the large scenario-content axis and asks which prompt-induced response-state modes remain.",
        "",
        "## Inputs",
        "",
        f"- base capture: `{summary['base_capture_id']}`",
        f"- contractarian capture: `{summary['contractarian_capture_id']}`",
        f"- base generation rows: `{summary['generation_rows']['base_generation_rows']}`",
        f"- contractarian generation rows: `{summary['generation_rows']['contractarian_generation_rows']}`",
        f"- conditions: `{len(CONDITION_ORDER)}`",
        "",
    ]
    for result in summary["results"]:
        label = f"{result['site']} L{result['layer']} {result['slice']}"
        lines.extend(
            [
                f"## {label}",
                "",
                f"- rows: `{result['n_rows']}`",
                f"- dilemmas: `{result['n_dilemmas']}`",
                f"- conditions: `{result['n_conditions']}`",
                f"- features: `{result['n_features']}`",
                "",
                "### Scree",
                "",
                "| PC | variance ratio | cumulative | CV condition-loading corr |",
                "|---:|---:|---:|---:|",
            ]
        )
        cumulative = 0.0
        cv_by_pc = {int(row["pc"]): row["abs_condition_loading_corr"] for row in result["cv_loading_stability"]}
        for idx, value in enumerate(result["explained_variance_ratio"], start=1):
            cumulative += float(value)
            lines.append(f"| {idx} | {_fmt(float(value))} | {_fmt(cumulative)} | {_fmt(float(cv_by_pc.get(idx, float('nan'))))} |")
        lines.extend(["", "### Condition Extremes", ""])
        for pc in result["pcs"][:5]:
            lines.append(f"#### PC{pc['pc']} ({_fmt(pc['explained_variance_ratio'])} variance)")
            lines.append("")
            lines.append("| side | condition | score | role | theory |")
            lines.append("|---|---|---:|---|---|")
            for side in ("positive", "negative"):
                for item in pc["condition_extremes"][side]:
                    lines.append(
                        f"| {side} | `{item['condition_id']}` | {_fmt(float(item['score']))} | "
                        f"`{item['role']}` | `{item['theory']}` |"
                    )
            lines.append("")

    (report_dir / "report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--components", type=int, default=DEFAULT_COMPONENTS)
    parser.add_argument("--layers", nargs="+", type=int, default=list(LAYERS))
    parser.add_argument("--include-prompt-end", action="store_true")
    args = parser.parse_args()

    rows_by_key, generation_rows = _load_combined_rows()
    results = []
    for layer in args.layers:
        results.append(
            _analyze_site_layer(
                site="generated_sequence_residual",
                layer=layer,
                slice_name="first_16",
                rows_by_key=rows_by_key,
                n_components=args.components,
            )
        )
        if args.include_prompt_end:
            results.append(
                _analyze_site_layer(
                    site="prompt_end_residual",
                    layer=layer,
                    slice_name="full",
                    rows_by_key=rows_by_key,
                    n_components=args.components,
                )
            )

    summary = {
        "base_capture_id": BASE_CAPTURE_ID,
        "contractarian_capture_id": CONTRACTARIAN_CAPTURE_ID,
        "generation_rows": {key: str(value) for key, value in generation_rows.items()},
        "conditions": list(CONDITION_ORDER),
        "results": results,
    }
    _write_report(summary, Path(args.report_dir))
    print(json.dumps({"report": str(Path(args.report_dir) / "report.md"), "results": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
