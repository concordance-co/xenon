from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open


ROOT = Path("/Users/trentelmore/Projects/concordance/xenon-dashboard/projects/DX_TERMINAL/prompt_confusion/phase_12")
FEATURE_CACHE = ROOT / "outputs" / "real_complaint_transfer_feature_cache"
TRANSFER_BRIDGE_DIR = ROOT / "outputs" / "transfer_bridge"
PHASE_09_DATASET = ROOT.parent / "phase_09" / "outputs" / "phase_09_dataset" / "phase_09_dataset.jsonl"
CAPTURED_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44)

STAGE_SPECS = {
    "stage1a": {
        "dataset": TRANSFER_BRIDGE_DIR / "trade_size_stage1a_template_control.jsonl",
        "artifact_root": Path("/Users/trentelmore/.xenon/pipelines_v2/cache/capture_1_1def4f9a"),
    },
    "stage1b": {
        "dataset": TRANSFER_BRIDGE_DIR / "trade_size_stage1b_adapter_strict.jsonl",
        "artifact_root": Path("/Users/trentelmore/.xenon/pipelines_v2/cache/capture_1_45d4a738"),
    },
}

OUTPUT_SUMMARY = TRANSFER_BRIDGE_DIR / "trade_size_bridge_probe_summary.json"


@dataclass
class ProbeFit:
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    std: np.ndarray


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


def build_synthetic_direction() -> dict[str, np.ndarray]:
    rows = load_jsonl(PHASE_09_DATASET)
    label_map = {
        row["example_id"]: bool(row["conflict_present"])
        for row in rows
        if row.get("target_dimension") == "trade_size" and not bool(row.get("edge_conflict"))
    }
    metadata = load_json(FEATURE_CACHE / "trade_size_residual_prompt_eos.metadata.json")
    tensor_path = FEATURE_CACHE / "trade_size_feature_tensors.safetensors"
    directions: dict[str, np.ndarray] = {}
    with safe_open(str(tensor_path), framework="np", device="cpu") as tensors:
        for layer in CAPTURED_LAYERS:
            entries = metadata["layers"][str(layer)]
            aligned_keys = [key for key in entries if key in label_map and not label_map[key]]
            conflict_keys = [key for key in entries if key in label_map and label_map[key]]
            aligned = np.stack([load_vector(tensors, entries[key]["values"]["__tensor_key__"]) for key in aligned_keys], axis=0)
            conflict = np.stack([load_vector(tensors, entries[key]["values"]["__tensor_key__"]) for key in conflict_keys], axis=0)
            direction = conflict.mean(axis=0) - aligned.mean(axis=0)
            norm = float(np.linalg.norm(direction))
            unit = direction if norm == 0.0 else direction / norm
            if float((conflict @ unit).mean()) < float((aligned @ unit).mean()):
                unit = -unit
            directions[str(layer)] = unit.astype(np.float32)
    return directions


def load_stage_matrix(dataset_path: Path, artifact_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    rows = {row["example_id"]: row for row in load_jsonl(dataset_path)}
    metadata = load_json(artifact_root / "features" / "residual_prompt_last.metadata.json")
    tensor_path = artifact_root / "features" / "feature_tensors.safetensors"
    matrices: dict[str, np.ndarray] = {}
    with safe_open(str(tensor_path), framework="np", device="cpu") as tensors:
        for layer in CAPTURED_LAYERS:
            layer_entries = metadata["layers"][str(layer)]
            ordered_keys = [example_id for example_id in layer_entries if example_id in rows]
            matrix = np.stack(
                [load_vector(tensors, layer_entries[example_id]["values"]["__tensor_key__"]) for example_id in ordered_keys],
                axis=0,
            )
            matrices[str(layer)] = matrix
            if layer == 32:
                rows = {example_id: rows[example_id] for example_id in ordered_keys}
    return rows, matrices


def labels_from_rows(rows: dict[str, dict[str, Any]]) -> np.ndarray:
    ordered = list(rows.values())
    return np.asarray([1.0 if row["adapter_alignment_label"] == "conflict" else 0.0 for row in ordered], dtype=np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def fit_logistic_probe(x: np.ndarray, y: np.ndarray, *, l2: float = 1e-2, steps: int = 800, lr: float = 0.1) -> ProbeFit:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    xz = (x - mean) / std
    weights = np.zeros(x.shape[1], dtype=np.float32)
    bias = 0.0
    n = float(x.shape[0])
    for _ in range(steps):
        logits = xz @ weights + bias
        probs = sigmoid(logits)
        error = probs - y
        grad_w = (xz.T @ error) / n + l2 * weights
        grad_b = float(error.mean())
        weights -= lr * grad_w
        bias -= lr * grad_b
    return ProbeFit(weights=weights, bias=float(bias), mean=mean.astype(np.float32), std=std.astype(np.float32))


def probe_scores(fit: ProbeFit, x: np.ndarray) -> np.ndarray:
    xz = (x - fit.mean) / fit.std
    return xz @ fit.weights + fit.bias


def auc_from_scores(y: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    comparisons = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an == 0.0 or bn == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (an * bn))


def main() -> None:
    synthetic_directions = build_synthetic_direction()
    stage_rows: dict[str, dict[str, dict[str, Any]]] = {}
    stage_matrices: dict[str, dict[str, np.ndarray]] = {}
    stage_labels: dict[str, np.ndarray] = {}

    for stage_name, spec in STAGE_SPECS.items():
        rows, matrices = load_stage_matrix(spec["dataset"], spec["artifact_root"])
        stage_rows[stage_name] = rows
        stage_matrices[stage_name] = matrices
        stage_labels[stage_name] = labels_from_rows(rows)

    summary: dict[str, Any] = {"layers": {}, "datasets": {}}
    for stage_name, labels in stage_labels.items():
        summary["datasets"][stage_name] = {
            "count": int(labels.size),
            "conflict": int(labels.sum()),
            "aligned": int(labels.size - labels.sum()),
        }

    for layer in CAPTURED_LAYERS:
        key = str(layer)
        synthetic = synthetic_directions[key]
        fits: dict[str, ProbeFit] = {}
        row: dict[str, Any] = {
            "synthetic_direction_auroc": {},
            "probe_within_auroc": {},
            "probe_cross_auroc": {},
            "probe_synthetic_cosine": {},
            "probe_probe_cosine": {},
        }

        for stage_name in STAGE_SPECS:
            x = stage_matrices[stage_name][key]
            y = stage_labels[stage_name]
            row["synthetic_direction_auroc"][stage_name] = auc_from_scores(y, x @ synthetic)
            fits[stage_name] = fit_logistic_probe(x, y)
            row["probe_within_auroc"][stage_name] = auc_from_scores(y, probe_scores(fits[stage_name], x))
            probe_dir = fits[stage_name].weights.astype(np.float32)
            row["probe_synthetic_cosine"][stage_name] = cosine(probe_dir, synthetic)

        for source_name in STAGE_SPECS:
            row["probe_cross_auroc"][source_name] = {}
            for target_name in STAGE_SPECS:
                x = stage_matrices[target_name][key]
                y = stage_labels[target_name]
                row["probe_cross_auroc"][source_name][target_name] = auc_from_scores(y, probe_scores(fits[source_name], x))

        row["probe_probe_cosine"]["stage1a_vs_stage1b"] = cosine(fits["stage1a"].weights, fits["stage1b"].weights)
        summary["layers"][key] = row

    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    best_stage1b = max(((payload["probe_within_auroc"]["stage1b"], layer) for layer, payload in summary["layers"].items()), key=lambda item: item[0])
    best_stage1a = max(((payload["probe_within_auroc"]["stage1a"], layer) for layer, payload in summary["layers"].items()), key=lambda item: item[0])
    print(f"Best stage1a within-probe AUROC: L{best_stage1a[1]} {best_stage1a[0]:.4f}")
    print(f"Best stage1b within-probe AUROC: L{best_stage1b[1]} {best_stage1b[0]:.4f}")


if __name__ == "__main__":
    main()
