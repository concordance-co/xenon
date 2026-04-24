from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, silhouette_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipelines_v2.api import ModalVolumeStore, PostgresCatalog, PostgresSource, TransferPolicy
from pipelines_v2.storage.artifacts import CaptureArtifact, artifact_from_manifest


DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
ARTIFACT_STORE_NAME = "xenon-data"
FEATURE_NAME = "generated_sequence_residual"
LAYERS = [0, 4, 8, 16, 24, 32, 40, 44]
PCA_LAYERS = [0, 8, 44]
TAIL_FRACTION = 0.25
BOOTSTRAP_SAMPLES = 2000

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "reports" / "experiment_02_contested_override_analysis"
REPORT_DIR = ROOT / "reports" / "experiment_02_contested_override_controls"

ROWS_PATH = INPUT_DIR / "contested_override_rows.json"
FIRST_PASS_PATH = INPUT_DIR / "contested_override_analysis.json"

SOURCE_SPECS = {
    "benchmark_main_filtered": {
        "capture_id": "capture_1_34cdfd7923d9",
        "capture_root": "/data/artifacts/morebench_phase_03_experiment02",
    },
    "benchmark_missing_replay": {
        "capture_id": "capture_1_540167b2e849",
        "capture_root": "/data/artifacts/morebench_phase_03_experiment02_benchmark_missing_capture",
    },
    "public_conflict_replay": {
        "capture_id": "capture_1_6f2cf8da3f25",
        "capture_root": "/data/artifacts/morebench_phase_03_experiment02_behavior_broad_capture",
    },
}


def _catalog() -> PostgresCatalog:
    return PostgresCatalog(source=PostgresSource.from_env(DB_ENV_VAR))


def _store(root: str) -> ModalVolumeStore:
    return ModalVolumeStore(
        name=ARTIFACT_STORE_NAME,
        root=root,
        transfer_policy=TransferPolicy(allow_large_transfer=True),
    )


def _load_capture_payload(*, capture_id: str, root: str) -> dict[str, Any]:
    manifest = _catalog().load_artifact(capture_id)
    if manifest is None:
        raise RuntimeError(f"Could not load capture manifest {capture_id!r}")
    artifact = artifact_from_manifest(manifest, store=_store(root))
    if not isinstance(artifact, CaptureArtifact):
        raise TypeError(f"Artifact {capture_id!r} is not a capture artifact")
    payload = artifact.feature(FEATURE_NAME).load()
    if payload.get("kind") != "residual":
        raise TypeError(f"Expected residual payload for {capture_id!r}")
    return payload


def _load_feature_maps() -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict[str, np.ndarray]]]:
    full_maps: dict[int, dict[str, np.ndarray]] = {layer: {} for layer in LAYERS}
    tail_maps: dict[int, dict[str, np.ndarray]] = {layer: {} for layer in LAYERS}
    for source in SOURCE_SPECS.values():
        payload = _load_capture_payload(capture_id=source["capture_id"], root=source["capture_root"])
        for layer in LAYERS:
            layer_payload = payload["layers"][str(layer)]
            for key, record in layer_payload.items():
                values = np.asarray(record["values"], dtype=np.float32)
                full_maps[layer][str(key)] = values.mean(axis=0).astype(np.float32)
                token_count = int(values.shape[0])
                start = max(0, min(token_count - 1, int(math.floor(token_count * (1.0 - TAIL_FRACTION)))))
                tail_maps[layer][str(key)] = values[start:].mean(axis=0).astype(np.float32)
    return full_maps, tail_maps


def _safe_silhouette(X: np.ndarray, labels: list[str]) -> float | None:
    unique = sorted(set(labels))
    if len(unique) < 2:
        return None
    counts = {label: labels.count(label) for label in unique}
    if min(counts.values()) < 2:
        return None
    return round(float(silhouette_score(X, labels)), 4)


def _pca_summary(X: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X)
    return {
        "explained_variance_ratio": [round(float(x), 4) for x in pca.explained_variance_ratio_.tolist()],
        "silhouette_by_capture_batch": _safe_silhouette(X, [str(row["source_batch"]) for row in rows]),
        "silhouette_by_prime": _safe_silhouette(X, [str(row["prime_condition"]) for row in rows]),
        "silhouette_by_override_status": _safe_silhouette(X, [str(row["override_status"]) for row in rows]),
    }


def _tail_text(text: str) -> str:
    pieces = text.split()
    if not pieces:
        return text
    start = min(len(pieces) - 1, max(0, int(math.floor(len(pieces) * (1.0 - TAIL_FRACTION)))))
    return " ".join(pieces[start:])


def _word_char_union() -> FeatureUnion:
    return FeatureUnion(
        transformer_list=[
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)),
        ]
    )


def _make_text_features(
    name: str,
    train_texts: list[str],
    test_texts: list[str],
    *,
    train_lengths: np.ndarray | None = None,
    test_lengths: np.ndarray | None = None,
    train_primes: list[str] | None = None,
    test_primes: list[str] | None = None,
) -> tuple[sparse.spmatrix | np.ndarray, sparse.spmatrix | np.ndarray]:
    if name == "length_only":
        return train_lengths.reshape(-1, 1).astype(np.float32), test_lengths.reshape(-1, 1).astype(np.float32)
    if name == "prime_only":
        enc = OneHotEncoder(handle_unknown="ignore")
        train_x = enc.fit_transform(np.asarray(train_primes, dtype=object).reshape(-1, 1))
        test_x = enc.transform(np.asarray(test_primes, dtype=object).reshape(-1, 1))
        return train_x, test_x
    if name == "prime_length":
        enc = OneHotEncoder(handle_unknown="ignore")
        train_prime = enc.fit_transform(np.asarray(train_primes, dtype=object).reshape(-1, 1))
        test_prime = enc.transform(np.asarray(test_primes, dtype=object).reshape(-1, 1))
        train_len = sparse.csr_matrix(train_lengths.reshape(-1, 1).astype(np.float32))
        test_len = sparse.csr_matrix(test_lengths.reshape(-1, 1).astype(np.float32))
        return sparse.hstack([train_prime, train_len]), sparse.hstack([test_prime, test_len])
    if name == "tfidf_word":
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        return vec.fit_transform(train_texts), vec.transform(test_texts)
    if name == "tfidf_char":
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        return vec.fit_transform(train_texts), vec.transform(test_texts)
    if name == "tfidf_word_char":
        union = _word_char_union()
        return union.fit_transform(train_texts), union.transform(test_texts)
    if name == "tfidf_word_char_length":
        union = _word_char_union()
        train_text_x = union.fit_transform(train_texts)
        test_text_x = union.transform(test_texts)
        train_len = sparse.csr_matrix(train_lengths.reshape(-1, 1).astype(np.float32))
        test_len = sparse.csr_matrix(test_lengths.reshape(-1, 1).astype(np.float32))
        return sparse.hstack([train_text_x, train_len]), sparse.hstack([test_text_x, test_len])
    if name == "tfidf_word_char_length_prime":
        union = _word_char_union()
        train_text_x = union.fit_transform(train_texts)
        test_text_x = union.transform(test_texts)
        train_len = sparse.csr_matrix(train_lengths.reshape(-1, 1).astype(np.float32))
        test_len = sparse.csr_matrix(test_lengths.reshape(-1, 1).astype(np.float32))
        enc = OneHotEncoder(handle_unknown="ignore")
        train_prime = enc.fit_transform(np.asarray(train_primes, dtype=object).reshape(-1, 1))
        test_prime = enc.transform(np.asarray(test_primes, dtype=object).reshape(-1, 1))
        return (
            sparse.hstack([train_text_x, train_len, train_prime]),
            sparse.hstack([test_text_x, test_len, test_prime]),
        )
    raise ValueError(f"Unknown baseline {name}")


def _fit_baseline_logo(
    *,
    baseline_name: str,
    texts: list[str],
    lengths: np.ndarray,
    primes: list[str] | None,
    y: np.ndarray,
    groups: list[str],
) -> np.ndarray:
    logo = LeaveOneGroupOut()
    probs = np.zeros(len(y), dtype=np.float32)
    for train_idx, test_idx in logo.split(np.zeros(len(y)), y, groups):
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        train_lengths = lengths[train_idx]
        test_lengths = lengths[test_idx]
        train_primes = [primes[i] for i in train_idx] if primes is not None else None
        test_primes = [primes[i] for i in test_idx] if primes is not None else None
        train_x, test_x = _make_text_features(
            baseline_name,
            train_texts,
            test_texts,
            train_lengths=train_lengths,
            test_lengths=test_lengths,
            train_primes=train_primes,
            test_primes=test_primes,
        )
        if baseline_name == "length_only":
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000, random_state=42),
            )
        else:
            model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000, random_state=42)
        model.fit(train_x, y[train_idx])
        probs[test_idx] = model.predict_proba(test_x)[:, 1]
    return probs


def _fit_probe_logo(X: np.ndarray, y: np.ndarray, groups: list[str]) -> np.ndarray:
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


def _fit_residualized_probe_logo(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    Z: np.ndarray,
) -> np.ndarray:
    logo = LeaveOneGroupOut()
    probs = np.zeros(len(y), dtype=np.float32)
    for train_idx, test_idx in logo.split(X, y, groups):
        resid_model = Ridge(alpha=1.0)
        resid_model.fit(Z[train_idx], X[train_idx])
        X_train = X[train_idx] - resid_model.predict(Z[train_idx])
        X_test = X[test_idx] - resid_model.predict(Z[test_idx])
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000, random_state=42),
        )
        model.fit(X_train, y[train_idx])
        probs[test_idx] = model.predict_proba(X_test)[:, 1]
    return probs


def _metrics(y: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    preds = (probs >= 0.5).astype(np.int64)
    return {
        "positive_count": int(y.sum()),
        "negative_count": int((1 - y).sum()),
        "auroc": round(float(roc_auc_score(y, probs)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, preds)), 4),
    }


def _bootstrap_delta_ci(
    *,
    y: np.ndarray,
    probe_probs: np.ndarray,
    baseline_probs: np.ndarray,
    groups: list[str],
    n_boot: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    unique_groups = sorted(set(groups))
    idx_by_group = {
        group: np.asarray([idx for idx, item in enumerate(groups) if item == group], dtype=np.int64)
        for group in unique_groups
    }
    deltas: list[float] = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([idx_by_group[group] for group in sampled_groups], axis=0)
        sample_y = y[idx]
        if len(np.unique(sample_y)) < 2:
            continue
        deltas.append(float(roc_auc_score(sample_y, probe_probs[idx]) - roc_auc_score(sample_y, baseline_probs[idx])))
    if not deltas:
        return {"n_boot_valid": 0, "delta_mean": None, "delta_ci95": None}
    arr = np.asarray(deltas, dtype=np.float32)
    return {
        "n_boot_valid": int(arr.shape[0]),
        "delta_mean": round(float(arr.mean()), 4),
        "delta_ci95": [round(float(x), 4) for x in np.quantile(arr, [0.025, 0.975]).tolist()],
    }


def _baseline_suite(
    *,
    texts: list[str],
    lengths: np.ndarray,
    primes: list[str] | None,
    y: np.ndarray,
    groups: list[str],
    include_prime: bool,
) -> dict[str, dict[str, Any]]:
    baseline_names = ["length_only", "tfidf_word", "tfidf_char", "tfidf_word_char", "tfidf_word_char_length"]
    if include_prime:
        baseline_names.extend(["prime_only", "prime_length", "tfidf_word_char_length_prime"])
    results: dict[str, dict[str, Any]] = {}
    for name in baseline_names:
        probs = _fit_baseline_logo(
            baseline_name=name,
            texts=texts,
            lengths=lengths,
            primes=primes,
            y=y,
            groups=groups,
        )
        results[name] = {"probs": probs, **_metrics(y, probs)}
    return results


def _zs_for_residualization(
    *,
    strongest_baseline_probs: np.ndarray,
    lengths: np.ndarray,
    primes: list[str] | None,
) -> np.ndarray:
    cols = [strongest_baseline_probs.reshape(-1, 1), lengths.reshape(-1, 1).astype(np.float32)]
    if primes is not None:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cols.append(enc.fit_transform(np.asarray(primes, dtype=object).reshape(-1, 1)).astype(np.float32))
    return np.concatenate(cols, axis=1).astype(np.float32)


def _analyze_target(
    *,
    rows: list[dict[str, Any]],
    full_feature_maps: dict[int, dict[str, np.ndarray]],
    tail_feature_maps: dict[int, dict[str, np.ndarray]],
    target_field: str,
    include_prime: bool,
) -> dict[str, Any]:
    texts = [str(row["generated_text"]) for row in rows]
    tail_texts = [_tail_text(text) for text in texts]
    lengths = np.asarray([len(text.split()) for text in texts], dtype=np.float32)
    tail_lengths = np.asarray([len(text.split()) for text in tail_texts], dtype=np.float32)
    groups = [str(row["group_id"]) for row in rows]
    primes = [str(row["prime_condition"]) for row in rows]
    y = np.asarray([int(row[target_field]) for row in rows], dtype=np.int64)
    keys = [str(row["example_key"]) for row in rows]

    full_baselines = _baseline_suite(
        texts=texts,
        lengths=lengths,
        primes=primes if include_prime else None,
        y=y,
        groups=groups,
        include_prime=include_prime,
    )
    strongest_name = max(full_baselines, key=lambda name: full_baselines[name]["auroc"])
    strongest_probs = full_baselines[strongest_name]["probs"]

    full_probes: list[dict[str, Any]] = []
    for layer in LAYERS:
        X = np.vstack([full_feature_maps[layer][key] for key in keys]).astype(np.float32)
        probs = _fit_probe_logo(X, y, groups)
        entry = {"layer": layer, "probs": probs, **_metrics(y, probs)}
        entry["delta_vs_strongest_baseline"] = round(entry["auroc"] - full_baselines[strongest_name]["auroc"], 4)
        entry["delta_ci_vs_strongest_baseline"] = _bootstrap_delta_ci(
            y=y,
            probe_probs=probs,
            baseline_probs=strongest_probs,
            groups=groups,
        )
        full_probes.append(entry)

    fixed_layer = next(item for item in full_probes if item["layer"] == 8)
    best_layer = max(full_probes, key=lambda item: item["auroc"])

    tail_baselines = _baseline_suite(
        texts=tail_texts,
        lengths=tail_lengths,
        primes=primes if include_prime else None,
        y=y,
        groups=groups,
        include_prime=include_prime,
    )
    tail_strongest_name = max(tail_baselines, key=lambda name: tail_baselines[name]["auroc"])
    tail_strongest_probs = tail_baselines[tail_strongest_name]["probs"]
    tail_probes: list[dict[str, Any]] = []
    for layer in LAYERS:
        X = np.vstack([tail_feature_maps[layer][key] for key in keys]).astype(np.float32)
        probs = _fit_probe_logo(X, y, groups)
        entry = {"layer": layer, "probs": probs, **_metrics(y, probs)}
        entry["delta_vs_strongest_baseline"] = round(entry["auroc"] - tail_baselines[tail_strongest_name]["auroc"], 4)
        entry["delta_ci_vs_strongest_baseline"] = _bootstrap_delta_ci(
            y=y,
            probe_probs=probs,
            baseline_probs=tail_strongest_probs,
            groups=groups,
        )
        tail_probes.append(entry)

    tail_fixed_layer = next(item for item in tail_probes if item["layer"] == 8)
    tail_best_layer = max(tail_probes, key=lambda item: item["auroc"])

    full_Z = _zs_for_residualization(
        strongest_baseline_probs=strongest_probs,
        lengths=lengths,
        primes=primes if include_prime else None,
    )
    tail_Z = _zs_for_residualization(
        strongest_baseline_probs=tail_strongest_probs,
        lengths=tail_lengths,
        primes=primes if include_prime else None,
    )

    residualized = {}
    for label, probe_info, fmap, Z in [
        ("full_best", best_layer, full_feature_maps, full_Z),
        ("full_layer8", fixed_layer, full_feature_maps, full_Z),
        ("tail_best", tail_best_layer, tail_feature_maps, tail_Z),
        ("tail_layer8", tail_fixed_layer, tail_feature_maps, tail_Z),
    ]:
        X = np.vstack([fmap[int(probe_info["layer"])][key] for key in keys]).astype(np.float32)
        probs = _fit_residualized_probe_logo(X, y, groups, Z)
        residualized[label] = {
            "layer": int(probe_info["layer"]),
            **_metrics(y, probs),
            "delta_vs_strongest_baseline": round(
                float(roc_auc_score(y, probs)) - (
                    full_baselines[strongest_name]["auroc"] if label.startswith("full_") else tail_baselines[tail_strongest_name]["auroc"]
                ),
                4,
            ),
            "delta_ci_vs_strongest_baseline": _bootstrap_delta_ci(
                y=y,
                probe_probs=probs,
                baseline_probs=strongest_probs if label.startswith("full_") else tail_strongest_probs,
                groups=groups,
            ),
        }

    def _strip_probs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for item in items:
            copy = dict(item)
            copy.pop("probs", None)
            out.append(copy)
        return out

    baseline_public = {name: {k: v for k, v in metrics.items() if k != "probs"} for name, metrics in full_baselines.items()}
    tail_baseline_public = {name: {k: v for k, v in metrics.items() if k != "probs"} for name, metrics in tail_baselines.items()}

    return {
        "example_count": len(rows),
        "group_count": len(sorted(set(groups))),
        "positive_count": int(y.sum()),
        "full_sequence": {
            "baselines": baseline_public,
            "strongest_baseline_name": strongest_name,
            "probe_by_layer": _strip_probs(full_probes),
            "fixed_layer_8": {k: v for k, v in fixed_layer.items() if k != "probs"},
            "best_raw_layer": {k: v for k, v in best_layer.items() if k != "probs"},
        },
        "tail_25": {
            "baselines": tail_baseline_public,
            "strongest_baseline_name": tail_strongest_name,
            "probe_by_layer": _strip_probs(tail_probes),
            "fixed_layer_8": {k: v for k, v in tail_fixed_layer.items() if k != "probs"},
            "best_raw_layer": {k: v for k, v in tail_best_layer.items() if k != "probs"},
        },
        "residualized": residualized,
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = json.loads(ROWS_PATH.read_text(encoding="utf-8"))
    first_pass = json.loads(FIRST_PASS_PATH.read_text(encoding="utf-8"))
    full_feature_maps, tail_feature_maps = _load_feature_maps()

    pca = {}
    ordered_rows = list(rows)
    ordered_keys = [str(row["example_key"]) for row in ordered_rows]
    for layer in PCA_LAYERS:
        X = np.vstack([full_feature_maps[layer][key] for key in ordered_keys]).astype(np.float32)
        pca[str(layer)] = _pca_summary(X, ordered_rows)

    majority_rows = [
        row for row in rows if (not bool(row["is_tie_group"])) and (not bool(row["is_generic_reference"]))
    ]
    pooled = {
        "differs_from_generic": _analyze_target(
            rows=majority_rows,
            full_feature_maps=full_feature_maps,
            tail_feature_maps=tail_feature_maps,
            target_field="differs_from_generic",
            include_prime=True,
        ),
        "defect_from_majority": _analyze_target(
            rows=majority_rows,
            full_feature_maps=full_feature_maps,
            tail_feature_maps=tail_feature_maps,
            target_field="defect_from_majority",
            include_prime=True,
        ),
    }

    within_prime = {}
    for target_field in ["differs_from_generic", "defect_from_majority"]:
        within_prime[target_field] = {}
        for prime in ["deontology", "utilitarian"]:
            subset = [row for row in majority_rows if str(row["prime_condition"]) == prime]
            within_prime[target_field][prime] = _analyze_target(
                rows=subset,
                full_feature_maps=full_feature_maps,
                tail_feature_maps=tail_feature_maps,
                target_field=target_field,
                include_prime=False,
            )

    result = {
        "first_pass_reference": first_pass,
        "coverage": {
            "row_count_all_split_groups": len(rows),
            "row_count_majority_defined_non_generic": len(majority_rows),
            "generic_tracks_majority_group_count": int(first_pass["generic_tracks_majority_group_count"]),
            "generic_defector_group_count": int(first_pass["generic_defector_group_count"]),
            "generic_defector_groups": list(first_pass["generic_defector_groups"]),
        },
        "pca_batch_check": pca,
        "pooled": pooled,
        "within_prime": within_prime,
        "notes": [
            "sentence-transformer baseline was not run because sentence_transformers was unavailable in the local env",
            "stronger text baseline here means word+char tf-idf, plus length and prime stacked variants",
            "all bootstrap CIs resample group_id, not row",
            "fixed-layer readout is layer 8; best-of-8 is still reported but should be read as exploratory",
        ],
    }

    (REPORT_DIR / "controls_analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Contested Override Controls",
        "",
        "## Coverage",
        f"- majority-defined non-generic rows: `{len(majority_rows)}`",
        f"- generic tracks majority in `{first_pass['generic_tracks_majority_group_count']}/20` groups",
        f"- generic defector groups: " + ", ".join(f"`{gid}`" for gid in first_pass["generic_defector_groups"]),
        "",
        "## PCA Batch Check",
    ]
    for layer in PCA_LAYERS:
        item = pca[str(layer)]
        lines.append(
            f"- layer `{layer}`: batch `{item['silhouette_by_capture_batch']}`, "
            f"prime `{item['silhouette_by_prime']}`, override `{item['silhouette_by_override_status']}`"
        )
    lines.append("")

    for scope_name, scope in [("Pooled", pooled), ("Within-Prime", within_prime)]:
        lines.append(f"## {scope_name}")
        if scope_name == "Pooled":
            iterator = [(target, scope[target]) for target in ["differs_from_generic", "defect_from_majority"]]
        else:
            iterator = []
            for target in ["differs_from_generic", "defect_from_majority"]:
                for prime in ["deontology", "utilitarian"]:
                    iterator.append((f"{target} / {prime}", scope[target][prime]))
        for label, item in iterator:
            full_strong = item["full_sequence"]["strongest_baseline_name"]
            tail_strong = item["tail_25"]["strongest_baseline_name"]
            full_fixed = item["full_sequence"]["fixed_layer_8"]
            full_best = item["full_sequence"]["best_raw_layer"]
            tail_fixed = item["tail_25"]["fixed_layer_8"]
            tail_best = item["tail_25"]["best_raw_layer"]
            lines.extend(
                [
                    f"### `{label}`",
                    f"- full strongest baseline: `{full_strong}` AUROC `{item['full_sequence']['baselines'][full_strong]['auroc']}`",
                    f"- full layer 8 probe AUROC: `{full_fixed['auroc']}` delta `{full_fixed['delta_vs_strongest_baseline']}` CI `{full_fixed['delta_ci_vs_strongest_baseline']['delta_ci95']}`",
                    f"- full best raw layer: `{full_best['layer']}` AUROC `{full_best['auroc']}` delta `{full_best['delta_vs_strongest_baseline']}` CI `{full_best['delta_ci_vs_strongest_baseline']['delta_ci95']}`",
                    f"- tail strongest baseline: `{tail_strong}` AUROC `{item['tail_25']['baselines'][tail_strong]['auroc']}`",
                    f"- tail layer 8 probe AUROC: `{tail_fixed['auroc']}` delta `{tail_fixed['delta_vs_strongest_baseline']}` CI `{tail_fixed['delta_ci_vs_strongest_baseline']['delta_ci95']}`",
                    f"- tail best raw layer: `{tail_best['layer']}` AUROC `{tail_best['auroc']}` delta `{tail_best['delta_vs_strongest_baseline']}` CI `{tail_best['delta_ci_vs_strongest_baseline']['delta_ci95']}`",
                    f"- residualized full best: layer `{item['residualized']['full_best']['layer']}` AUROC `{item['residualized']['full_best']['auroc']}` delta `{item['residualized']['full_best']['delta_vs_strongest_baseline']}`",
                    f"- residualized tail best: layer `{item['residualized']['tail_best']['layer']}` AUROC `{item['residualized']['tail_best']['auroc']}` delta `{item['residualized']['tail_best']['delta_vs_strongest_baseline']}`",
                    "",
                ]
            )

    (REPORT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_DIR / "report.md")


if __name__ == "__main__":
    main()
