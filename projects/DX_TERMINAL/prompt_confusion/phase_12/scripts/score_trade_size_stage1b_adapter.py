from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from projects.DX_TERMINAL.prompt_confusion.neon import connect_neon
from projects.DX_TERMINAL.prompt_confusion.paths import phase_outputs_dir, phase_root
from projects.DX_TERMINAL.prompt_confusion.phase_12.scripts.transfer_bridge_neon import fetch_table_rows

ROOT = phase_root("phase_12", __file__)
FEATURE_CACHE = ROOT / "outputs" / "real_complaint_transfer_feature_cache"
TRANSFER_BRIDGE_DIR = ROOT / "outputs" / "transfer_bridge"
PHASE_09_DATASET = phase_outputs_dir("phase_09", __file__) / "phase_09_dataset" / "phase_09_dataset.jsonl"
CAPTURED_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def load_vector(tensors: safe_open, tensor_key: str) -> np.ndarray:
    return np.asarray(tensors.get_tensor(tensor_key), dtype=np.float32).reshape(-1)


def build_trade_size_direction() -> dict[str, Any]:
    records = load_jsonl(PHASE_09_DATASET)
    label_map = {
        row["example_id"]: bool(row["conflict_present"])
        for row in records
        if row.get("target_dimension") == "trade_size" and not bool(row.get("edge_conflict"))
    }
    metadata = load_json(FEATURE_CACHE / "trade_size_residual_prompt_eos.metadata.json")
    tensor_path = FEATURE_CACHE / "trade_size_feature_tensors.safetensors"

    directions: dict[str, list[float]] = {}
    thresholds: dict[str, float] = {}
    aligned_means: dict[str, float] = {}
    conflict_means: dict[str, float] = {}

    with safe_open(str(tensor_path), framework="np", device="cpu") as tensors:
        for layer in CAPTURED_LAYERS:
            entries = metadata["layers"][str(layer)]
            keys = [key for key in entries.keys() if key in label_map]
            aligned_keys = [key for key in keys if not label_map[key]]
            conflict_keys = [key for key in keys if label_map[key]]
            aligned = np.stack([load_vector(tensors, entries[key]["values"]["__tensor_key__"]) for key in aligned_keys], axis=0)
            conflict = np.stack([load_vector(tensors, entries[key]["values"]["__tensor_key__"]) for key in conflict_keys], axis=0)
            direction = conflict.mean(axis=0) - aligned.mean(axis=0)
            norm = float(np.linalg.norm(direction))
            unit = direction if norm == 0.0 else direction / norm
            aligned_scores = aligned @ unit
            conflict_scores = conflict @ unit
            if float(conflict_scores.mean()) < float(aligned_scores.mean()):
                unit = -unit
                aligned_scores = -aligned_scores
                conflict_scores = -conflict_scores
            directions[str(layer)] = unit.tolist()
            aligned_means[str(layer)] = float(aligned_scores.mean())
            conflict_means[str(layer)] = float(conflict_scores.mean())
            thresholds[str(layer)] = 0.5 * (aligned_means[str(layer)] + conflict_means[str(layer)])

    return {
        "direction_layers": directions,
        "thresholds": thresholds,
        "aligned_means": aligned_means,
        "conflict_means": conflict_means,
    }


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def load_dataset_rows(dataset_path: Path | None, dataset_table: str | None) -> dict[str, dict[str, Any]]:
    if dataset_table:
        with connect_neon() as conn:
            return {row["example_id"]: row for row in fetch_table_rows(conn, dataset_table)}
    if dataset_path is None:
        raise ValueError("Either dataset_path or dataset_table is required")
    return {row["example_id"]: row for row in load_jsonl(dataset_path)}


def score_artifact(
    artifact_root: Path,
    dataset_path: Path | None,
    output_prefix: str,
    *,
    dataset_table: str | None = None,
) -> dict[str, Any]:
    metadata = load_json(artifact_root / "features" / "residual_prompt_last.metadata.json")
    tensor_path = artifact_root / "features" / "feature_tensors.safetensors"
    dataset_rows = load_dataset_rows(dataset_path, dataset_table)
    direction = build_trade_size_direction()

    per_example_rows: list[dict[str, Any]] = []
    layer_summary: dict[str, Any] = {}

    with safe_open(str(tensor_path), framework="np", device="cpu") as tensors:
        for layer in CAPTURED_LAYERS:
            layer_key = str(layer)
            unit = np.asarray(direction["direction_layers"][layer_key], dtype=np.float32)
            threshold = float(direction["thresholds"][layer_key])
            scores: list[float] = []
            by_label: dict[str, list[float]] = defaultdict(list)

            for example_id, payload in metadata["layers"][layer_key].items():
                row = dataset_rows[example_id]
                tensor_key = payload["values"]["__tensor_key__"]
                score = float(np.dot(load_vector(tensors, tensor_key), unit))
                label = str(row["adapter_alignment_label"])
                scores.append(score)
                by_label[label].append(score)
                per_example_rows.append(
                    {
                        "example_id": example_id,
                        "layer": layer,
                        "score": score,
                        "threshold": threshold,
                        "predicted_conflict": bool(score >= threshold),
                        "adapter_alignment_label": label,
                        "strategy_size_preference": row["strategy_size_preference"],
                        "slider_size_bucket": row["slider_size_bucket"],
                        "complaint_type": row.get("complaint_type"),
                        "root_cause": row.get("root_cause"),
                    }
                )

            label_summaries = {label: summarize(vals) for label, vals in by_label.items()}
            conflict_mean = label_summaries.get("conflict", {}).get("mean")
            aligned_mean = label_summaries.get("aligned", {}).get("mean")
            layer_summary[layer_key] = {
                "overall": summarize(scores),
                "by_label": label_summaries,
                "threshold": threshold,
                "mean_delta_conflict_minus_aligned": None
                if conflict_mean is None or aligned_mean is None
                else float(conflict_mean - aligned_mean),
                "predicted_conflict_rate_by_label": {
                    label: float(sum(score >= threshold for score in vals) / len(vals))
                    for label, vals in by_label.items()
                },
            }

    output_dir = TRANSFER_BRIDGE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / f"{output_prefix}_scores.jsonl"
    summary_path = output_dir / f"{output_prefix}_summary.json"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in per_example_rows:
            handle.write(json.dumps(row))
            handle.write("\n")
    summary = {
        "artifact_root": str(artifact_root),
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "dataset_table": dataset_table,
        "rows": len(per_example_rows),
        "layers": layer_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dataset-table", default="dx_terminal_trade_size_stage1b_adapter_strict_v1")
    parser.add_argument("--output-prefix", default="trade_size_stage1b_adapter_strict")
    args = parser.parse_args()

    summary = score_artifact(
        artifact_root=Path(args.artifact_root),
        dataset_path=Path(args.dataset) if args.dataset else None,
        output_prefix=str(args.output_prefix),
        dataset_table=str(args.dataset_table) if args.dataset_table else None,
    )
    best = max(
        ((layer, payload["mean_delta_conflict_minus_aligned"]) for layer, payload in summary["layers"].items() if payload["mean_delta_conflict_minus_aligned"] is not None),
        key=lambda item: item[1],
    )
    print(f"Best layer by conflict-aligned mean delta: L{best[0]} delta={best[1]:.4f}")


if __name__ == "__main__":
    main()
