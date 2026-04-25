from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from projects.DX_TERMINAL.prompt_confusion.neon import connect_neon
from projects.DX_TERMINAL.prompt_confusion.paths import phase_outputs_dir, phase_root
from projects.DX_TERMINAL.prompt_confusion.phase_12.scripts.transfer_bridge_neon import fetch_table_rows

ROOT = phase_root("phase_12", __file__)
FEATURE_CACHE = ROOT / "outputs" / "real_complaint_transfer_feature_cache"
REAL_DATASET = ROOT / "outputs" / "real_complaint_transfer" / "real_complaint_transfer_dataset.jsonl"
REAL_TABLE = "dx_terminal_real_complaint_transfer_ticks_v1"
OUTPUT_DIR = ROOT / "outputs" / "real_complaint_transfer_scores"

PHASE_09_DATASET = phase_outputs_dir("phase_09", __file__) / "phase_09_dataset" / "phase_09_dataset.jsonl"
PHASE_10_DATASET = phase_outputs_dir("phase_10", __file__) / "phase_10_dataset" / "phase_10_dataset.jsonl"

CAPTURED_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44)
REAL_SITES = (
    ("prompt_last", "real_shard_a_residual_prompt_last.metadata.json", "real_shard_b_residual_prompt_last.metadata.json"),
    ("user_last", "real_shard_a_residual_user_last.metadata.json", "real_shard_b_residual_user_last.metadata.json"),
    ("system_last", "real_shard_a_residual_system_last.metadata.json", "real_shard_b_residual_system_last.metadata.json"),
)
SYNTHETIC_SPECS = {
    "trade_size": {
        "dataset_path": PHASE_09_DATASET,
        "metadata_name": "trade_size_residual_prompt_eos.metadata.json",
        "tensor_name": "trade_size_feature_tensors.safetensors",
        "target_dimension": "trade_size",
    },
    "risk_preference": {
        "dataset_path": PHASE_10_DATASET,
        "metadata_name": "risk_residual_prompt_eos.metadata.json",
        "tensor_name": "risk_feature_tensors.safetensors",
        "target_dimension": "risk_preference",
    },
}
SUMMARY_GROUPS = ("complaint_type", "root_cause", "label", "fault", "system_fault", "tool")


@dataclass
class FeatureBundle:
    metadata: dict[str, Any]
    tensor_path: Path


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


def load_feature_bundle(metadata_name: str, tensor_name: str) -> FeatureBundle:
    return FeatureBundle(
        metadata=load_json(FEATURE_CACHE / metadata_name),
        tensor_path=FEATURE_CACHE / tensor_name,
    )


def load_vector(tensors: safe_open, tensor_key: str) -> np.ndarray:
    return np.asarray(tensors.get_tensor(tensor_key), dtype=np.float32).reshape(-1)


def build_synthetic_direction(
    *,
    dataset_path: Path,
    metadata_name: str,
    tensor_name: str,
    target_dimension: str,
) -> dict[str, Any]:
    records = load_jsonl(dataset_path)
    label_map = {
        row["example_id"]: bool(row["conflict_present"])
        for row in records
        if row.get("target_dimension") == target_dimension and not bool(row.get("edge_conflict"))
    }
    bundle = load_feature_bundle(metadata_name, tensor_name)
    directions: dict[str, list[float]] = {}
    thresholds: dict[str, float] = {}
    aligned_means: dict[str, float] = {}
    conflict_means: dict[str, float] = {}

    with safe_open(str(bundle.tensor_path), framework="np", device="cpu") as tensors:
        for layer in CAPTURED_LAYERS:
            layer_entries = bundle.metadata["layers"][str(layer)]
            keys = [key for key in layer_entries.keys() if key in label_map]
            aligned_keys = [key for key in keys if not label_map[key]]
            conflict_keys = [key for key in keys if label_map[key]]
            aligned = np.stack(
                [load_vector(tensors, layer_entries[key]["values"]["__tensor_key__"]) for key in aligned_keys],
                axis=0,
            )
            conflict = np.stack(
                [load_vector(tensors, layer_entries[key]["values"]["__tensor_key__"]) for key in conflict_keys],
                axis=0,
            )
            vector = conflict.mean(axis=0) - aligned.mean(axis=0)
            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                unit = vector
            else:
                unit = vector / norm
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
        "target_dimension": target_dimension,
        "direction_layers": directions,
        "aligned_means": aligned_means,
        "conflict_means": conflict_means,
        "thresholds": thresholds,
        "n_examples": len(label_map),
    }


def shard_rows_from_metadata(metadata_name: str) -> list[str]:
    metadata = load_json(FEATURE_CACHE / metadata_name)
    return list(metadata["layers"]["32"].keys())


def build_real_record_map() -> dict[str, dict[str, Any]]:
    if REAL_DATASET.exists():
        rows = load_jsonl(REAL_DATASET)
    else:
        with connect_neon() as conn:
            rows = fetch_table_rows(conn, REAL_TABLE)
    return {row["example_id"]: row for row in rows}


def summarize_numeric(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def project_real_examples(
    *,
    site_name: str,
    metadata_name_a: str,
    metadata_name_b: str,
    direction_name: str,
    direction_payload: dict[str, Any],
    real_record_map: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bundle_a = load_feature_bundle(metadata_name_a, "real_shard_a_feature_tensors.safetensors")
    bundle_b = load_feature_bundle(metadata_name_b, "real_shard_b_feature_tensors.safetensors")
    per_example_rows: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {"layers": {}, "site": site_name, "direction": direction_name}

    for layer in CAPTURED_LAYERS:
        layer_key = str(layer)
        vector = np.asarray(direction_payload["direction_layers"][layer_key], dtype=np.float32)
        threshold = float(direction_payload["thresholds"][layer_key])
        score_rows: list[dict[str, Any]] = []

        for bundle, shard_name in ((bundle_a, "a"), (bundle_b, "b")):
            layer_entries = bundle.metadata["layers"][layer_key]
            with safe_open(str(bundle.tensor_path), framework="np", device="cpu") as tensors:
                for example_id, payload in layer_entries.items():
                    record = real_record_map.get(example_id)
                    if record is None:
                        continue
                    tensor_key = payload["values"]["__tensor_key__"]
                    score = float(np.dot(load_vector(tensors, tensor_key), vector))
                    predicted_conflict = score >= threshold
                    row = {
                        "example_id": example_id,
                        "layer": layer,
                        "site": site_name,
                        "direction": direction_name,
                        "score": score,
                        "threshold": threshold,
                        "predicted_conflict": bool(predicted_conflict),
                        "tensor_key": tensor_key,
                        "shard": shard_name,
                        "complaint_type": record.get("complaint_type"),
                        "root_cause": record.get("root_cause"),
                        "label": record.get("label"),
                        "fault": record.get("fault"),
                        "system_fault": record.get("system_fault"),
                        "tool": record.get("tool"),
                        "trace_id": record.get("trace_id"),
                        "tick_index": record.get("tick_index"),
                    }
                    per_example_rows.append(row)
                    score_rows.append(row)

        values = [row["score"] for row in score_rows]
        group_summaries: dict[str, dict[str, Any]] = {}
        for group_name in SUMMARY_GROUPS:
            grouped: dict[str, list[float]] = defaultdict(list)
            for row in score_rows:
                value = row.get(group_name)
                grouped[str(value) if value is not None else "__null__"].append(row["score"])
            ranked = sorted(
                (
                    {
                        "value": value,
                        **summarize_numeric(scores),
                    }
                    for value, scores in grouped.items()
                    if scores
                ),
                key=lambda item: (-item["mean"], item["value"]),
            )
            group_summaries[group_name] = {"top": ranked[:10], "bottom": sorted(ranked, key=lambda item: (item["mean"], item["value"]))[:10]}

        top_examples = sorted(score_rows, key=lambda row: row["score"], reverse=True)[:15]
        bottom_examples = sorted(score_rows, key=lambda row: row["score"])[:15]
        aggregate["layers"][layer_key] = {
            "overall": summarize_numeric(values),
            "threshold": threshold,
            "group_summaries": group_summaries,
            "top_examples": top_examples,
            "bottom_examples": bottom_examples,
        }

    return per_example_rows, aggregate


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    synthetic_directions = {
        name: build_synthetic_direction(**spec)
        for name, spec in SYNTHETIC_SPECS.items()
    }
    (OUTPUT_DIR / "synthetic_direction_summary.json").write_text(
        json.dumps(synthetic_directions, indent=2),
        encoding="utf-8",
    )

    real_record_map = build_real_record_map()
    all_rows: list[dict[str, Any]] = []
    all_aggregates: dict[str, Any] = {}

    for site_name, metadata_name_a, metadata_name_b in REAL_SITES:
        for direction_name, direction_payload in synthetic_directions.items():
            rows, aggregate = project_real_examples(
                site_name=site_name,
                metadata_name_a=metadata_name_a,
                metadata_name_b=metadata_name_b,
                direction_name=direction_name,
                direction_payload=direction_payload,
                real_record_map=real_record_map,
            )
            all_rows.extend(rows)
            all_aggregates[f"{direction_name}__{site_name}"] = aggregate

    with (OUTPUT_DIR / "real_transfer_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row) + "\n")

    (OUTPUT_DIR / "real_transfer_summary.json").write_text(
        json.dumps(
            {
                "rows": len(all_rows),
                "directions": list(synthetic_directions.keys()),
                "sites": [site for site, *_ in REAL_SITES],
                "aggregates": all_aggregates,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {len(all_rows)} projected rows to {OUTPUT_DIR / 'real_transfer_scores.jsonl'}")
    print(f"Wrote summary to {OUTPUT_DIR / 'real_transfer_summary.json'}")


if __name__ == "__main__":
    main()
