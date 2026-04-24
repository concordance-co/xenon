from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from pipelines_v2.api import ModalVolumeStore, PostgresCatalog, PostgresSource, TokenPooling, TokenSelector, TransferPolicy
from pipelines_v2.operations.execution.common import feature_matrices
from pipelines_v2.storage.artifacts import CaptureArtifact, OperationArtifact, artifact_from_manifest


DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
ARTIFACT_ROOT = "/data/artifacts/morebench_phase_03_experiment02"
ARTIFACT_STORE_NAME = "xenon-data"
FEATURE_NAME = "generated_sequence_residual"
REPORT_DIR = Path("projects/MOREBENCH/phase_03/reports/experiment_02_pca_geometry")
LAYERS = [0, 4, 8, 16, 24, 32, 40, 44]


@dataclass(frozen=True, slots=True)
class FamilySpec:
    family: str
    capture_dataset_id: str
    capture_id: str


FAMILY_SPECS = [
    FamilySpec("description_only", "transform_1_4a60e2ca", "capture_1_34cdfd7923d9"),
    FamilySpec("alias_only", "transform_1_02d360bd", "capture_1_5be107b39b39"),
]


def _catalog() -> PostgresCatalog:
    return PostgresCatalog(source=PostgresSource.from_env(DB_ENV_VAR))


def _store() -> ModalVolumeStore:
    return ModalVolumeStore(
        name=ARTIFACT_STORE_NAME,
        root=ARTIFACT_ROOT,
        transfer_policy=TransferPolicy(allow_large_transfer=True),
    )


def _load_operation_artifact(artifact_id: str) -> OperationArtifact:
    manifest = _catalog().load_artifact(artifact_id)
    if manifest is None:
        raise RuntimeError(f"Could not load artifact manifest {artifact_id!r}")
    artifact = artifact_from_manifest(manifest, store=_store())
    if not isinstance(artifact, OperationArtifact):
        raise TypeError(f"Artifact {artifact_id!r} is not an operation artifact")
    return artifact


def _load_capture_artifact(artifact_id: str) -> CaptureArtifact:
    manifest = _catalog().load_artifact(artifact_id)
    if manifest is None:
        raise RuntimeError(f"Could not load artifact manifest {artifact_id!r}")
    artifact = artifact_from_manifest(manifest, store=_store())
    if not isinstance(artifact, CaptureArtifact):
        raise TypeError(f"Artifact {artifact_id!r} is not a capture artifact")
    return artifact


def _row_from_example(example: dict[str, Any], *, family_fallback: str) -> dict[str, Any]:
    labels = dict(example.get("labels", {}))
    alias_bank = str(labels.get("alias_bank") or "")
    group_id = str(labels.get("group_id") or "")
    slot_id = f"{group_id}::{alias_bank}" if alias_bank else group_id
    return {
        "example_key": str(example["key"]),
        "group_id": group_id,
        "slot_id": slot_id,
        "prime_condition": str(labels.get("prime_condition") or ""),
        "prime_family": str(labels.get("prime_family") or family_fallback),
        "alias_bank": alias_bank,
    }


def _tail_mean_payload(capture_artifact: CaptureArtifact, *, tail_fraction: float) -> dict[int, tuple[np.ndarray, list[str], dict[str, int]]]:
    payload = capture_artifact.feature(FEATURE_NAME).load()
    if payload.get("kind") != "residual":
        raise TypeError(f"Expected residual payload, got {payload.get('kind')!r}")

    matrices: dict[int, tuple[np.ndarray, list[str], dict[str, int]]] = {}
    for layer in sorted(int(layer) for layer in payload["layers"]):
        layer_payload = payload["layers"][str(layer)]
        example_keys = sorted(layer_payload)
        rows: list[np.ndarray] = []
        token_counts: dict[str, int] = {}
        for key in example_keys:
            values = np.asarray(layer_payload[key]["values"], dtype=np.float32)
            if values.ndim != 2:
                raise TypeError("Residual values must be rank-2")
            token_count = int(values.shape[0])
            start = max(0, min(token_count - 1, int(math.floor(token_count * (1.0 - tail_fraction)))))
            selected = values[start:]
            rows.append(selected.mean(axis=0))
            token_counts[key] = token_count
        matrices[layer] = (np.vstack(rows).astype(np.float32), example_keys, token_counts)
    return matrices


def _load_family_rows_and_features(
    spec: FamilySpec,
    *,
    tail_fraction: float = 0.25,
) -> tuple[list[dict[str, Any]], dict[str, dict[int, np.ndarray]]]:
    dataset_artifact = _load_operation_artifact(spec.capture_dataset_id)
    capture_artifact = _load_capture_artifact(spec.capture_id)
    payload = dataset_artifact.result()
    dataset_payload = payload.get("dataset") if isinstance(payload, dict) else None
    if not isinstance(dataset_payload, dict):
        raise TypeError(f"Capture dataset artifact {spec.capture_dataset_id!r} missing serialized dataset")

    rows = [_row_from_example(example, family_fallback=spec.family) for example in dataset_payload["examples"]]
    row_map = {row["example_key"]: row for row in rows}

    full_mats, full_keys = feature_matrices(
        capture_artifact.feature(FEATURE_NAME),
        token_selector=TokenSelector.full_sequence(),
        token_pooling=TokenPooling.mean(),
    )
    tail_mats = _tail_mean_payload(capture_artifact, tail_fraction=tail_fraction)

    full_features: dict[int, np.ndarray] = {}
    ordered_rows = [row_map[key] for key in full_keys]
    if set(row_map) != set(full_keys):
        raise RuntimeError(f"Feature/row mismatch for {spec.family}")
    for layer, matrix in sorted(full_mats.items()):
        full_features[layer] = matrix.astype(np.float32)

    tail_features: dict[int, np.ndarray] = {}
    for layer, (matrix, example_keys, _token_counts) in sorted(tail_mats.items()):
        if example_keys != full_keys:
            raise RuntimeError(f"Tail/full key mismatch for {spec.family} layer {layer}")
        tail_features[layer] = matrix.astype(np.float32)
    return ordered_rows, {"full_sequence": full_features, "tail_25": tail_features}


def _center_by_slot(X: np.ndarray, slot_ids: list[str]) -> np.ndarray:
    centered = np.zeros_like(X)
    unique_slots = sorted(set(slot_ids))
    for slot in unique_slots:
        idx = [i for i, item in enumerate(slot_ids) if item == slot]
        slot_mean = X[idx].mean(axis=0, keepdims=True)
        centered[idx] = X[idx] - slot_mean
    return centered


def _safe_silhouette(X: np.ndarray, labels: list[str]) -> float | None:
    unique = sorted(set(labels))
    if len(unique) < 2:
        return None
    counts = {label: labels.count(label) for label in unique}
    if min(counts.values()) < 2:
        return None
    return round(float(silhouette_score(X, labels)), 4)


def _pca_summary(X: np.ndarray, *, n_components: int = 5) -> dict[str, Any]:
    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X)
    return {
        "explained_variance_ratio": [round(float(x), 4) for x in pca.explained_variance_ratio_.tolist()],
        "coords": coords,
    }


def _centroid_rank_summary(X: np.ndarray, labels: list[str]) -> dict[str, Any]:
    classes = sorted(set(labels))
    centroids = np.vstack([X[np.asarray([label == c for label in labels])].mean(axis=0) for c in classes]).astype(np.float32)
    centroids = centroids - centroids.mean(axis=0, keepdims=True)
    u, s, _vh = np.linalg.svd(centroids, full_matrices=False)
    del u
    eig = s**2
    if float(eig.sum()) <= 0:
        evr = [0.0 for _ in eig]
    else:
        evr = (eig / eig.sum()).tolist()
    cosine = cosine_similarity(centroids)
    pairwise: dict[str, float] = {}
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            if j <= i:
                continue
            pairwise[f"{a}__{b}"] = round(float(cosine[i, j]), 4)
    return {
        "class_names": classes,
        "centroid_rank_explained_variance_ratio": [round(float(x), 4) for x in evr],
        "centroid_cosine_pairs": dict(sorted(pairwise.items())),
    }


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    slot_prime_counts: dict[str, int] = {}
    for row in rows:
        counts[row["slot_id"]] = counts.get(row["slot_id"], 0) + 1
    for slot, count in counts.items():
        slot_prime_counts[str(count)] = slot_prime_counts.get(str(count), 0) + 1
    complete = sum(1 for count in counts.values() if count >= 6)
    return {
        "row_count": len(rows),
        "slot_count": len(counts),
        "slot_prime_count_histogram": dict(sorted(slot_prime_counts.items(), key=lambda item: int(item[0]))),
        "complete_slot_count_ge_6": complete,
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, tuple[list[dict[str, Any]], dict[str, dict[int, np.ndarray]]]] = {}
    for spec in FAMILY_SPECS:
        loaded[spec.family] = _load_family_rows_and_features(spec)

    combined_rows = [*loaded["description_only"][0], *loaded["alias_only"][0]]
    combined_features: dict[str, dict[int, np.ndarray]] = {"full_sequence": {}, "tail_25": {}}
    for window in combined_features:
        for layer in LAYERS:
            combined_features[window][layer] = np.concatenate(
                [loaded["description_only"][1][window][layer], loaded["alias_only"][1][window][layer]],
                axis=0,
            ).astype(np.float32)
    loaded["combined"] = (combined_rows, combined_features)

    summary: dict[str, Any] = {"families": {}, "notes": []}
    summary["notes"].append(
        "Strict capture is incomplete after filtering, so per-dilemma 6-point PCA is not cleanly available. "
        "This analysis therefore emphasizes dilemma-slot-centered global PCA and class-centroid geometry."
    )

    for family_name, (rows, window_features) in loaded.items():
        family_payload: dict[str, Any] = {
            "coverage": _coverage(rows),
            "windows": {},
        }
        slot_ids = [row["slot_id"] for row in rows]
        prime_labels = [row["prime_condition"] for row in rows]
        prime_family_labels = [row["prime_family"] for row in rows]
        for window_name, layer_map in window_features.items():
            window_payload: dict[str, Any] = {}
            for layer, X in sorted(layer_map.items()):
                raw_pca = _pca_summary(X)
                centered = _center_by_slot(X, slot_ids)
                centered_pca = _pca_summary(centered)

                prime_sil_2 = _safe_silhouette(centered_pca["coords"][:, :2], prime_labels)
                prime_sil_4 = _safe_silhouette(centered_pca["coords"][:, : min(4, centered_pca["coords"].shape[1])], prime_labels)
                payload = {
                    "raw_row_pca_explained_variance_ratio": raw_pca["explained_variance_ratio"],
                    "centered_row_pca_explained_variance_ratio": centered_pca["explained_variance_ratio"],
                    "prime_condition_silhouette_pc2": prime_sil_2,
                    "prime_condition_silhouette_pc4": prime_sil_4,
                    **_centroid_rank_summary(centered, prime_labels),
                }
                if family_name == "combined":
                    family_sil_2 = _safe_silhouette(centered_pca["coords"][:, :2], prime_family_labels)
                    family_sil_4 = _safe_silhouette(
                        centered_pca["coords"][:, : min(4, centered_pca["coords"].shape[1])],
                        prime_family_labels,
                    )
                    payload["prime_family_silhouette_pc2"] = family_sil_2
                    payload["prime_family_silhouette_pc4"] = family_sil_4
                window_payload[str(layer)] = payload
            family_payload["windows"][window_name] = window_payload
        summary["families"][family_name] = family_payload

    output_path = REPORT_DIR / "pca_geometry_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: list[str] = [
        "# Experiment 2 PCA Geometry",
        "",
        "## Coverage",
    ]
    for family_name in ["description_only", "alias_only", "combined"]:
        coverage = summary["families"][family_name]["coverage"]
        lines.append(
            f"- `{family_name}`: {coverage['row_count']} rows, {coverage['slot_count']} dilemma-slots, "
            f"complete 6-row slots = {coverage['complete_slot_count_ge_6']}, "
            f"histogram = {coverage['slot_prime_count_histogram']}"
        )
    lines.extend(
        [
            "",
            "## Headline read",
            "",
            "- This is a slot-centered PCA because strict capture is incomplete; per-dilemma 6-point PCA is only clean for 3 description slots and 0 alias slots.",
            "- The right question here is therefore whether prime structure shows up after subtracting slot means, and whether that structure is low-rank or mostly family/topic-shaped.",
            "",
            "## Tail 25% focus",
        ]
    )

    for family_name in ["description_only", "alias_only", "combined"]:
        layer8 = summary["families"][family_name]["windows"]["tail_25"]["8"]
        layer44 = summary["families"][family_name]["windows"]["tail_25"]["44"]
        lines.append(
            f"- `{family_name}` layer 8 tail: centered row EVR={layer8['centered_row_pca_explained_variance_ratio'][:3]}, "
            f"centroid rank EVR={layer8['centroid_rank_explained_variance_ratio'][:3]}, "
            f"prime silhouette pc2={layer8['prime_condition_silhouette_pc2']}"
        )
        lines.append(
            f"- `{family_name}` layer 44 tail: centered row EVR={layer44['centered_row_pca_explained_variance_ratio'][:3]}, "
            f"centroid rank EVR={layer44['centroid_rank_explained_variance_ratio'][:3]}, "
            f"prime silhouette pc2={layer44['prime_condition_silhouette_pc2']}"
        )

    combined_tail8 = summary["families"]["combined"]["windows"]["tail_25"]["8"]
    strongest = sorted(
        combined_tail8["centroid_cosine_pairs"].items(),
        key=lambda item: item[1],
        reverse=True,
    )[:6]
    weakest = sorted(
        combined_tail8["centroid_cosine_pairs"].items(),
        key=lambda item: item[1],
    )[:6]
    lines.extend(
        [
            "",
            "## Combined Tail Layer 8 Centroid Cosines",
            "",
            "Most similar pairs:",
        ]
    )
    for name, value in strongest:
        lines.append(f"- `{name}`: {value}")
    lines.append("")
    lines.append("Least similar pairs:")
    for name, value in weakest:
        lines.append(f"- `{name}`: {value}")

    report_path = REPORT_DIR / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_path)
    print(report_path)


if __name__ == "__main__":
    main()
