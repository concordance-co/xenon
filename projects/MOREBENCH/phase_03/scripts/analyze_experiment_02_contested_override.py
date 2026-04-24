from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, silhouette_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipelines_v2.api import ModalVolumeStore, PostgresCatalog, PostgresSource, TransferPolicy
from pipelines_v2.storage.artifacts import CaptureArtifact, OperationArtifact, artifact_from_manifest


DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
FEATURE_NAME = "generated_sequence_residual"
ARTIFACT_STORE_NAME = "xenon-data"
LAYERS = [0, 4, 8, 16, 24, 32, 40, 44]
PCA_LAYERS = [0, 8, 44]
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "experiment_02_contested_override_analysis"
MANUAL_JUDGMENT_PATH = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "experiment_02_behavior_broad_llm_judged"
    / "manual_judgment_summary.json"
)
DEFECTION_TABLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "experiment_02_behavior_broad_llm_judged"
    / "defection_table.json"
)


@dataclass(frozen=True, slots=True)
class SourceSpec:
    name: str
    row_source_artifact_id: str
    row_source_kind: str
    row_source_root: str
    capture_id: str
    capture_root: str


BENCHMARK_MAIN = SourceSpec(
    name="benchmark_main_filtered",
    row_source_artifact_id="transform_1_4a60e2ca",
    row_source_kind="capture_dataset",
    row_source_root="/data/artifacts/morebench_phase_03_experiment02",
    capture_id="capture_1_34cdfd7923d9",
    capture_root="/data/artifacts/morebench_phase_03_experiment02",
)
BENCHMARK_MISSING = SourceSpec(
    name="benchmark_missing_replay",
    row_source_artifact_id="generation_run_1_3d4009fb21d8",
    row_source_kind="generation_rows",
    row_source_root="/data/artifacts/morebench_phase_03_experiment02",
    capture_id="capture_1_540167b2e849",
    capture_root="/data/artifacts/morebench_phase_03_experiment02_benchmark_missing_capture",
)
PUBLIC_CONFLICT = SourceSpec(
    name="public_conflict_replay",
    row_source_artifact_id="generation_run_1_82f18dea7736",
    row_source_kind="generation_rows",
    row_source_root="/data/artifacts/morebench_phase_03_experiment02",
    capture_id="capture_1_6f2cf8da3f25",
    capture_root="/data/artifacts/morebench_phase_03_experiment02_behavior_broad_capture",
)


def _catalog() -> PostgresCatalog:
    return PostgresCatalog(source=PostgresSource.from_env(DB_ENV_VAR))


def _store(root: str) -> ModalVolumeStore:
    return ModalVolumeStore(
        name=ARTIFACT_STORE_NAME,
        root=root,
        transfer_policy=TransferPolicy(allow_large_transfer=True),
    )


def _load_operation_artifact(artifact_id: str, *, root: str) -> OperationArtifact:
    manifest = _catalog().load_artifact(artifact_id)
    if manifest is None:
        raise RuntimeError(f"Could not load artifact manifest {artifact_id!r}")
    artifact = artifact_from_manifest(manifest, store=_store(root))
    if not isinstance(artifact, OperationArtifact):
        raise TypeError(f"Artifact {artifact_id!r} is not an operation artifact")
    return artifact


def _load_capture_artifact(artifact_id: str, *, root: str) -> CaptureArtifact:
    manifest = _catalog().load_artifact(artifact_id)
    if manifest is None:
        raise RuntimeError(f"Could not load artifact manifest {artifact_id!r}")
    artifact = artifact_from_manifest(manifest, store=_store(root))
    if not isinstance(artifact, CaptureArtifact):
        raise TypeError(f"Artifact {artifact_id!r} is not a capture artifact")
    return artifact


def _mean_feature_by_key(capture_id: str, *, root: str) -> dict[int, dict[str, np.ndarray]]:
    capture = _load_capture_artifact(capture_id, root=root)
    payload = capture.feature(FEATURE_NAME).load()
    if payload.get("kind") != "residual":
        raise TypeError(f"Expected residual feature payload, got {payload.get('kind')!r}")

    out: dict[int, dict[str, np.ndarray]] = {}
    for layer in LAYERS:
        layer_payload = payload["layers"][str(layer)]
        row_map: dict[str, np.ndarray] = {}
        for key, record in layer_payload.items():
            values = np.asarray(record["values"], dtype=np.float32)
            if values.ndim != 2:
                raise TypeError("Residual values must be rank-2")
            row_map[str(key)] = values.mean(axis=0).astype(np.float32)
        out[layer] = row_map
    return out


def _rows_from_capture_dataset(source: SourceSpec) -> list[dict[str, Any]]:
    op = _load_operation_artifact(source.row_source_artifact_id, root=source.row_source_root)
    dataset = op.result()["dataset"]["examples"]
    rows: list[dict[str, Any]] = []
    for example in dataset:
        labels = dict(example.get("labels", {}))
        rows.append(
            {
                "example_key": str(example["key"]),
                "group_id": str(labels.get("group_id") or ""),
                "prime_condition": str(labels.get("prime_condition") or ""),
                "generated_text": str(labels.get("generated_text") or ""),
                "source_batch": source.name,
                "capture_id": source.capture_id,
            }
        )
    return rows


def _rows_from_generation(source: SourceSpec) -> list[dict[str, Any]]:
    op = _load_operation_artifact(source.row_source_artifact_id, root=source.row_source_root)
    rows_payload = op.result()["rows"]
    rows: list[dict[str, Any]] = []
    for row in rows_payload:
        example = dict(row.get("example") or {})
        labels = dict(example.get("labels", {}))
        rows.append(
            {
                "example_key": str(row.get("example_key") or example.get("key") or ""),
                "group_id": str(labels.get("group_id") or ""),
                "prime_condition": str(labels.get("prime_condition") or ""),
                "generated_text": str(row.get("generated_text") or row.get("text") or ""),
                "source_batch": source.name,
                "capture_id": source.capture_id,
            }
        )
    return rows


def _load_rows(source: SourceSpec) -> list[dict[str, Any]]:
    if source.row_source_kind == "capture_dataset":
        return _rows_from_capture_dataset(source)
    if source.row_source_kind == "generation_rows":
        return _rows_from_generation(source)
    raise ValueError(f"Unknown row source kind: {source.row_source_kind}")


def _safe_silhouette(X: np.ndarray, labels: list[str]) -> float | None:
    unique = sorted(set(labels))
    if len(unique) < 2:
        return None
    counts = {label: labels.count(label) for label in unique}
    if min(counts.values()) < 2:
        return None
    return round(float(silhouette_score(X, labels)), 4)


def _pca_records(X: np.ndarray, *, n_components: int = 2) -> tuple[list[float], np.ndarray]:
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X)
    return [round(float(x), 4) for x in pca.explained_variance_ratio_.tolist()], coords


def _fit_predict_tfidf_logo(
    texts: list[str],
    y: np.ndarray,
    groups: list[str],
) -> np.ndarray:
    logo = LeaveOneGroupOut()
    probs = np.zeros(len(y), dtype=np.float32)
    for train_idx, test_idx in logo.split(texts, y, groups):
        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000, random_state=42),
        )
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        model.fit(train_texts, y[train_idx])
        probs[test_idx] = model.predict_proba(test_texts)[:, 1]
    return probs


def _fit_predict_length_logo(
    lengths: np.ndarray,
    y: np.ndarray,
    groups: list[str],
) -> np.ndarray:
    logo = LeaveOneGroupOut()
    probs = np.zeros(len(y), dtype=np.float32)
    X = lengths.reshape(-1, 1).astype(np.float32)
    for train_idx, test_idx in logo.split(X, y, groups):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000, random_state=42),
        )
        model.fit(X[train_idx], y[train_idx])
        probs[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return probs


def _fit_predict_prime_logo(
    primes: list[str],
    y: np.ndarray,
    groups: list[str],
) -> np.ndarray:
    logo = LeaveOneGroupOut()
    probs = np.zeros(len(y), dtype=np.float32)
    X = np.asarray(primes, dtype=object).reshape(-1, 1)
    for train_idx, test_idx in logo.split(X, y, groups):
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        train_x = enc.fit_transform(X[train_idx])
        test_x = enc.transform(X[test_idx])
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000, random_state=42)
        model.fit(train_x, y[train_idx])
        probs[test_idx] = model.predict_proba(test_x)[:, 1]
    return probs


def _fit_predict_probe_logo(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[str],
) -> np.ndarray:
    logo = LeaveOneGroupOut()
    probs = np.zeros(len(y), dtype=np.float32)
    for train_idx, test_idx in logo.split(X, y, groups):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000, random_state=42),
        )
        model.fit(X[train_idx], y[train_idx])
        probs[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return probs


def _metrics(y: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    preds = (probs >= 0.5).astype(np.int64)
    return {
        "positive_count": int(y.sum()),
        "negative_count": int((1 - y).sum()),
        "auroc": round(float(roc_auc_score(y, probs)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, preds)), 4),
    }


def _within_prime_metrics(
    rows: list[dict[str, Any]],
    features_by_layer: dict[int, np.ndarray],
    *,
    prime: str,
    target_field: str,
) -> dict[str, Any]:
    prime_rows = [row for row in rows if row["prime_condition"] == prime]
    y = np.asarray([int(row[target_field]) for row in prime_rows], dtype=np.int64)
    groups = [str(row["group_id"]) for row in prime_rows]
    texts = [str(row["generated_text"]) for row in prime_rows]
    lengths = np.asarray([len(text.split()) for text in texts], dtype=np.float32)
    text_probs = _fit_predict_tfidf_logo(texts, y, groups)
    length_probs = _fit_predict_length_logo(lengths, y, groups)

    out = {
        "prime": prime,
        "example_count": len(prime_rows),
        "positive_count": int(y.sum()),
        "group_count": len(sorted(set(groups))),
        "text_baseline": _metrics(y, text_probs),
        "length_baseline": _metrics(y, length_probs),
        "probe_by_layer": [],
    }
    for layer in LAYERS:
        prime_x = np.vstack([features_by_layer[layer][row["example_key"]] for row in prime_rows]).astype(np.float32)
        probs = _fit_predict_probe_logo(prime_x, y, groups)
        metrics = _metrics(y, probs)
        metrics["layer"] = layer
        metrics["probe_minus_text_auroc"] = round(metrics["auroc"] - out["text_baseline"]["auroc"], 4)
        out["probe_by_layer"].append(metrics)
    out["best_probe_layer"] = max(out["probe_by_layer"], key=lambda item: item["auroc"])
    return out


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    manual = json.loads(MANUAL_JUDGMENT_PATH.read_text(encoding="utf-8"))
    defection = json.loads(DEFECTION_TABLE_PATH.read_text(encoding="utf-8"))
    split_groups = {entry["group_id"]: entry for entry in manual["split_groups"]}
    shape_map = dict(defection["shape_map"])
    minority_map = {key: list(value) for key, value in defection["group_minority_map"].items()}

    all_target_groups = sorted(split_groups)
    majority_defined_groups = sorted(gid for gid, shape in shape_map.items() if shape != "3-3")
    tie_groups = sorted(gid for gid, shape in shape_map.items() if shape == "3-3")

    row_map: dict[str, dict[str, Any]] = {}
    feature_maps_by_layer: dict[int, dict[str, np.ndarray]] = {layer: {} for layer in LAYERS}
    batch_roots: dict[str, str] = {}

    for source in [BENCHMARK_MAIN, BENCHMARK_MISSING, PUBLIC_CONFLICT]:
        rows = _load_rows(source)
        features = _mean_feature_by_key(source.capture_id, root=source.capture_root)
        batch_roots[source.name] = source.capture_root
        available_keys = set(features[0])
        for row in rows:
            key = row["example_key"]
            if not key or key not in available_keys:
                continue
            if row["group_id"] not in all_target_groups:
                continue
            row_map[key] = row
        for layer in LAYERS:
            for key, vector in features[layer].items():
                if key in row_map:
                    feature_maps_by_layer[layer][key] = vector

    ordered_keys = sorted(row_map)
    rows = [row_map[key] for key in ordered_keys]

    for row in rows:
        group_id = str(row["group_id"])
        shape = str(shape_map[group_id])
        row["split_shape"] = shape
        row["subset"] = str(split_groups[group_id]["subset"])
        row["role_domain"] = str(split_groups[group_id]["role_domain"])
        row["context"] = str(split_groups[group_id]["context"])
        row["minority_primes"] = list(minority_map[group_id])
        row["is_tie_group"] = shape == "3-3"
        row["is_generic_reference"] = row["prime_condition"] == "generic_ethics_control"
        if shape == "3-3":
            row["defect_from_majority"] = None
            row["differs_from_generic"] = None
        else:
            minority_primes = set(minority_map[group_id])
            row["defect_from_majority"] = int(row["prime_condition"] in minority_primes)
            if row["prime_condition"] == "generic_ethics_control":
                row["differs_from_generic"] = 0
            else:
                generic_is_minority = "generic_ethics_control" in minority_primes
                if generic_is_minority:
                    row["differs_from_generic"] = int(row["prime_condition"] not in minority_primes)
                else:
                    row["differs_from_generic"] = int(row["prime_condition"] in minority_primes)
        if row["is_tie_group"]:
            row["override_status"] = "tie_group"
        elif row["is_generic_reference"]:
            row["override_status"] = "generic_reference"
        elif int(row["differs_from_generic"]) == 1:
            row["override_status"] = "differs_from_generic"
        else:
            row["override_status"] = "aligned_to_generic"

    if len(rows) != 132:
        raise RuntimeError(f"Expected 132 contested-case rows after merging captures, got {len(rows)}")

    full_features_by_layer = {
        layer: np.vstack([feature_maps_by_layer[layer][key] for key in ordered_keys]).astype(np.float32)
        for layer in LAYERS
    }

    generic_majority_agreement_groups = [
        gid for gid in majority_defined_groups if "generic_ethics_control" not in set(minority_map[gid])
    ]
    generic_defector_groups = [
        gid for gid in majority_defined_groups if "generic_ethics_control" in set(minority_map[gid])
    ]

    non_generic_majority_rows = [
        row for row in rows if row["group_id"] in majority_defined_groups and not row["is_generic_reference"]
    ]
    non_generic_keys = [str(row["example_key"]) for row in non_generic_majority_rows]
    non_generic_groups = [str(row["group_id"]) for row in non_generic_majority_rows]
    non_generic_texts = [str(row["generated_text"]) for row in non_generic_majority_rows]
    non_generic_lengths = np.asarray([len(text.split()) for text in non_generic_texts], dtype=np.float32)
    non_generic_primes = [str(row["prime_condition"]) for row in non_generic_majority_rows]

    pooled_results: dict[str, Any] = {}
    for target_field in ["differs_from_generic", "defect_from_majority"]:
        y = np.asarray([int(row[target_field]) for row in non_generic_majority_rows], dtype=np.int64)
        text_probs = _fit_predict_tfidf_logo(non_generic_texts, y, non_generic_groups)
        length_probs = _fit_predict_length_logo(non_generic_lengths, y, non_generic_groups)
        prime_probs = _fit_predict_prime_logo(non_generic_primes, y, non_generic_groups)
        target_summary = {
            "example_count": len(non_generic_majority_rows),
            "group_count": len(sorted(set(non_generic_groups))),
            "text_baseline": _metrics(y, text_probs),
            "length_baseline": _metrics(y, length_probs),
            "prime_only_baseline": _metrics(y, prime_probs),
            "probe_by_layer": [],
        }
        for layer in LAYERS:
            X = np.vstack([feature_maps_by_layer[layer][key] for key in non_generic_keys]).astype(np.float32)
            probs = _fit_predict_probe_logo(X, y, non_generic_groups)
            metrics = _metrics(y, probs)
            metrics["layer"] = layer
            metrics["probe_minus_text_auroc"] = round(metrics["auroc"] - target_summary["text_baseline"]["auroc"], 4)
            metrics["probe_minus_prime_only_auroc"] = round(
                metrics["auroc"] - target_summary["prime_only_baseline"]["auroc"],
                4,
            )
            target_summary["probe_by_layer"].append(metrics)
        target_summary["best_probe_layer"] = max(target_summary["probe_by_layer"], key=lambda item: item["auroc"])
        pooled_results[target_field] = target_summary

    within_prime = {}
    for target_field in ["differs_from_generic", "defect_from_majority"]:
        target_rows = [row for row in non_generic_majority_rows if row[target_field] is not None]
        within_prime[target_field] = {
            "deontology": _within_prime_metrics(target_rows, feature_maps_by_layer, prime="deontology", target_field=target_field),
            "utilitarian": _within_prime_metrics(target_rows, feature_maps_by_layer, prime="utilitarian", target_field=target_field),
        }

    pca_summary: dict[str, Any] = {}
    pca_records: dict[str, list[dict[str, Any]]] = {}
    for layer in PCA_LAYERS:
        X = full_features_by_layer[layer]
        evr, coords = _pca_records(X)
        batch_labels = [str(row["source_batch"]) for row in rows]
        prime_labels = [str(row["prime_condition"]) for row in rows]
        override_labels = [str(row["override_status"]) for row in rows]
        pca_summary[str(layer)] = {
            "explained_variance_ratio": evr,
            "silhouette_by_capture_batch": _safe_silhouette(X, batch_labels),
            "silhouette_by_prime": _safe_silhouette(X, prime_labels),
            "silhouette_by_override_status": _safe_silhouette(X, override_labels),
        }
        pca_records[str(layer)] = [
            {
                "example_key": str(row["example_key"]),
                "group_id": str(row["group_id"]),
                "prime_condition": str(row["prime_condition"]),
                "source_batch": str(row["source_batch"]),
                "subset": str(row["subset"]),
                "split_shape": str(row["split_shape"]),
                "override_status": str(row["override_status"]),
                "pc1": round(float(coords[idx, 0]), 4),
                "pc2": round(float(coords[idx, 1]), 4),
            }
            for idx, row in enumerate(rows)
        ]

    result = {
        "row_count_all_split_groups": len(rows),
        "group_count_all_split_groups": len(sorted({row["group_id"] for row in rows})),
        "row_count_majority_defined_non_generic": len(non_generic_majority_rows),
        "group_count_majority_defined": len(majority_defined_groups),
        "tie_group_count": len(tie_groups),
        "tie_groups": tie_groups,
        "generic_tracks_majority_group_count": len(generic_majority_agreement_groups),
        "generic_defector_group_count": len(generic_defector_groups),
        "generic_defector_groups": generic_defector_groups,
        "label_disagreement_group_count": len(generic_defector_groups),
        "label_disagreement_row_count": len(generic_defector_groups) * 5,
        "capture_batch_roots": batch_roots,
        "pooled": pooled_results,
        "within_prime": within_prime,
        "pca": pca_summary,
    }

    (REPORT_DIR / "contested_override_analysis.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "contested_override_pca_records.json").write_text(
        json.dumps(pca_records, indent=2) + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "contested_override_rows.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Contested Override Analysis",
        "",
        "## Coverage",
        f"- split-group rows merged across captures: `{len(rows)}`",
        f"- majority-defined non-generic rows used for pooled binary probes: `{len(non_generic_majority_rows)}`",
        f"- tie groups excluded from binary targets: `{len(tie_groups)}`",
        f"- generic tracks majority in `{len(generic_majority_agreement_groups)}/{len(majority_defined_groups)}` majority-defined groups",
        f"- generic defects in `{len(generic_defector_groups)}/{len(majority_defined_groups)}` groups: "
        + ", ".join(f"`{gid}`" for gid in generic_defector_groups),
        "",
        "## Pooled Results",
    ]
    for target_field in ["differs_from_generic", "defect_from_majority"]:
        pooled = pooled_results[target_field]
        best = pooled["best_probe_layer"]
        lines.extend(
            [
                f"### `{target_field}`",
                f"- text baseline AUROC: `{pooled['text_baseline']['auroc']}`",
                f"- length baseline AUROC: `{pooled['length_baseline']['auroc']}`",
                f"- prime-only baseline AUROC: `{pooled['prime_only_baseline']['auroc']}`",
                f"- best probe layer: `{best['layer']}`",
                f"- best probe AUROC: `{best['auroc']}`",
                f"- probe minus text AUROC: `{best['probe_minus_text_auroc']}`",
                f"- probe minus prime-only AUROC: `{best['probe_minus_prime_only_auroc']}`",
                "",
            ]
        )
    lines.append("## Within-Prime Results")
    for target_field in ["differs_from_generic", "defect_from_majority"]:
        lines.append(f"### `{target_field}`")
        for prime in ["deontology", "utilitarian"]:
            item = within_prime[target_field][prime]
            best = item["best_probe_layer"]
            lines.extend(
                [
                    f"- `{prime}`: text AUROC `{item['text_baseline']['auroc']}`, "
                    f"length AUROC `{item['length_baseline']['auroc']}`, "
                    f"best probe layer `{best['layer']}` AUROC `{best['auroc']}`, "
                    f"delta `{best['probe_minus_text_auroc']}`",
                ]
            )
        lines.append("")
    lines.append("## PCA Batch Check")
    for layer in PCA_LAYERS:
        item = pca_summary[str(layer)]
        lines.extend(
            [
                f"- layer `{layer}`: batch silhouette `{item['silhouette_by_capture_batch']}`, "
                f"prime silhouette `{item['silhouette_by_prime']}`, "
                f"override-status silhouette `{item['silhouette_by_override_status']}`",
            ]
        )
    (REPORT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_DIR / "report.md")


if __name__ == "__main__":
    main()
