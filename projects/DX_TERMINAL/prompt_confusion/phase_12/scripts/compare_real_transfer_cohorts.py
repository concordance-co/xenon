from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open


ROOT = Path("/Users/trentelmore/Projects/concordance/xenon-dashboard/projects/DX_TERMINAL/prompt_confusion/phase_12")
FEATURE_CACHE = ROOT / "outputs" / "real_complaint_transfer_feature_cache"
REAL_DATASET = ROOT / "outputs" / "real_complaint_transfer" / "real_complaint_transfer_dataset.jsonl"
SCORE_DIR = ROOT / "outputs" / "real_complaint_transfer_scores"

BASELINE_ROOT_CAUSES = {"USER_EXPECTATION_MISMATCH", "CORRECT_BEHAVIOR", "MARKET_LEGITIMATE"}
CAPTURED_LAYERS = ("28", "32", "36", "40", "44")


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


def baseline_record_map() -> dict[str, dict[str, Any]]:
    rows = load_jsonl(REAL_DATASET)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("root_cause") in BASELINE_ROOT_CAUSES or row.get("label") == "market":
            out[row["example_id"]] = row
    return out


def project_baseline() -> list[dict[str, Any]]:
    directions = load_json(SCORE_DIR / "synthetic_direction_summary.json")
    record_map = baseline_record_map()
    site_specs = [
        (
            "prompt_last",
            "baseline_shard_a_residual_prompt_last.metadata.json",
            "baseline_shard_b_residual_prompt_last.metadata.json",
        ),
        (
            "user_last",
            "baseline_shard_a_residual_user_last.metadata.json",
            "baseline_shard_b_residual_user_last.metadata.json",
        ),
        (
            "system_last",
            "baseline_shard_a_residual_system_last.metadata.json",
            "baseline_shard_b_residual_system_last.metadata.json",
        ),
    ]
    tensor_specs = [
        ("baseline_shard_a_feature_tensors.safetensors", "a"),
        ("baseline_shard_b_feature_tensors.safetensors", "b"),
    ]
    rows: list[dict[str, Any]] = []
    for direction_name, payload in directions.items():
        for site_name, meta_a, meta_b in site_specs:
            metas = [load_json(FEATURE_CACHE / meta_a), load_json(FEATURE_CACHE / meta_b)]
            tensors_paths = [FEATURE_CACHE / tensor_specs[0][0], FEATURE_CACHE / tensor_specs[1][0]]
            for layer in CAPTURED_LAYERS:
                vector = np.asarray(payload["direction_layers"][layer], dtype=np.float32)
                threshold = float(payload["thresholds"][layer])
                for meta, tensor_path, (_, shard_tag) in zip(metas, tensors_paths, tensor_specs):
                    layer_entries = meta["layers"][layer]
                    with safe_open(str(tensor_path), framework="np", device="cpu") as tensors:
                        for example_id, item in layer_entries.items():
                            record = record_map.get(example_id)
                            if record is None:
                                continue
                            tensor_key = item["values"]["__tensor_key__"]
                            score = float(np.dot(load_vector(tensors, tensor_key), vector))
                            rows.append(
                                {
                                    "cohort": "baseline_control",
                                    "direction": direction_name,
                                    "site": site_name,
                                    "layer": int(layer),
                                    "score": score,
                                    "threshold": threshold,
                                    "predicted_conflict": bool(score >= threshold),
                                    "example_id": example_id,
                                    "root_cause": record.get("root_cause"),
                                    "complaint_type": record.get("complaint_type"),
                                    "label": record.get("label"),
                                    "fault": record.get("fault"),
                                    "shard": shard_tag,
                                }
                            )
    return rows


def load_high_signal_scores() -> list[dict[str, Any]]:
    rows = load_jsonl(SCORE_DIR / "real_transfer_scores.jsonl")
    for row in rows:
        row["cohort"] = "high_signal"
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    pred_grouped: dict[tuple[str, str, str, int], list[bool]] = defaultdict(list)
    for row in rows:
        key = (row["cohort"], row["direction"], row["site"], int(row["layer"]))
        grouped[key].append(float(row["score"]))
        pred_grouped[key].append(bool(row["predicted_conflict"]))
    for key, values in grouped.items():
        arr = np.asarray(values, dtype=np.float32)
        pred = np.asarray(pred_grouped[key], dtype=np.bool_)
        summary["|".join(map(str, key))] = {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p10": float(np.quantile(arr, 0.10)),
            "p50": float(np.quantile(arr, 0.50)),
            "p90": float(np.quantile(arr, 0.90)),
            "predicted_conflict_rate": float(pred.mean()),
        }
    return summary


def main() -> None:
    baseline_rows = project_baseline()
    high_rows = load_high_signal_scores()
    all_rows = high_rows + baseline_rows

    out_jsonl = SCORE_DIR / "baseline_control_scores.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for row in baseline_rows:
            handle.write(json.dumps(row) + "\n")

    comparison = {
        "summary": summarize(all_rows),
        "high_signal_count": len(high_rows),
        "baseline_control_count": len(baseline_rows),
    }
    out_json = SCORE_DIR / "high_signal_vs_baseline_summary.json"
    out_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"Wrote {len(baseline_rows)} baseline rows to {out_jsonl}")
    print(f"Wrote comparison summary to {out_json}")


if __name__ == "__main__":
    main()
