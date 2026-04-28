"""Activation probe analysis for ethical-vs-self-advantage v2 capture."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from pipelines_v2.api import ModalVolumeStore, PostgresCatalog, PostgresSource, TransferPolicy
from pipelines_v2.storage.artifacts import CaptureArtifact, artifact_from_manifest


PHASE_ROOT = Path("projects/MOREBENCH/ethical_advantage_vectors/phase_01")
DEFAULT_GENERATION_ROWS = (
    PHASE_ROOT
    / "reports"
    / "v2_full_capture"
    / "report_12abda8dec51_9d7390f7"
    / "results"
    / "generate_v2_responses_results.json"
)
DEFAULT_ACTION_ROWS = PHASE_ROOT / "reports" / "behavior_smoke_analysis" / "v2_full40" / "scored_rows.jsonl"
DEFAULT_REPORT_DIR = PHASE_ROOT / "reports" / "activation_probe_analysis" / "v2_full40"

DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
ARTIFACT_STORE_NAME = "xenon-data"
ARTIFACT_ROOT = "/data/artifacts/morebench_ethical_advantage_vectors_phase01"
LAYERS = (16, 24, 32, 40)
SLICES = ("prompt_end", "generated_first_16", "generated_full")

ETHICAL_CONDITIONS = {"P_ethical_01", "P_ethical_02"}
NEGATIVE_CONDITIONS = {"P_self_serving_01", "P_self_serving_02", "P_exploit_01"}
PRIMARY_CONDITIONS = ETHICAL_CONDITIONS | NEGATIVE_CONDITIONS


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _catalog() -> PostgresCatalog:
    return PostgresCatalog(source=PostgresSource.from_env(DB_ENV_VAR))


def _store() -> ModalVolumeStore:
    return ModalVolumeStore(
        name=ARTIFACT_STORE_NAME,
        root=ARTIFACT_ROOT,
        transfer_policy=TransferPolicy(allow_large_transfer=True),
    )


def _load_capture(artifact_id: str) -> CaptureArtifact:
    manifest = _catalog().load_artifact(artifact_id)
    if manifest is None:
        raise RuntimeError(f"could not load capture artifact {artifact_id!r}")
    artifact = artifact_from_manifest(manifest, store=_store())
    if not isinstance(artifact, CaptureArtifact):
        raise TypeError(f"artifact {artifact_id!r} is not a capture artifact")
    return artifact


_FEATURE_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def _feature_payload(capture: CaptureArtifact, site: str) -> dict[str, Any]:
    key = (capture.id, site)
    if key not in _FEATURE_CACHE:
        payload = capture.feature(site).load()
        if not isinstance(payload, Mapping):
            raise TypeError(f"feature {site!r} payload is not a mapping")
        _FEATURE_CACHE[key] = dict(payload)
    return _FEATURE_CACHE[key]


def _slice_array(arr: np.ndarray, slice_name: str) -> np.ndarray:
    if arr.ndim == 1:
        return arr
    n = arr.shape[0]
    if n <= 0:
        return np.empty((0,), dtype=np.float32)
    if slice_name == "generated_full":
        return arr.mean(axis=0)
    if slice_name == "generated_first_16":
        return arr[: min(16, n)].mean(axis=0)
    raise ValueError(f"slice {slice_name!r} cannot be applied to sequence array")


def _feature_map(capture: CaptureArtifact, *, layer: int, slice_name: str) -> dict[str, np.ndarray]:
    if slice_name == "prompt_end":
        site = "prompt_end_residual"
    else:
        site = "generated_sequence_residual"
    payload = _feature_payload(capture, site)
    layer_payload = payload.get("layers", {}).get(str(layer))
    if not isinstance(layer_payload, Mapping):
        raise RuntimeError(f"missing {site} L{layer}")
    out: dict[str, np.ndarray] = {}
    for key, rec in layer_payload.items():
        if not isinstance(rec, Mapping):
            continue
        values = rec.get("values")
        if values is None:
            continue
        arr = np.asarray(values, dtype=np.float32)
        if slice_name == "prompt_end":
            vec = arr if arr.ndim == 1 else arr.mean(axis=0)
        else:
            vec = _slice_array(arr, slice_name)
        if vec.size:
            out[str(key)] = vec
    return out


def _rows_by_key(generation_rows_path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(generation_rows_path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise TypeError(f"{generation_rows_path} must contain a rows list")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        example = row.get("example") if isinstance(row.get("example"), Mapping) else {}
        labels = example.get("labels") if isinstance(example.get("labels"), Mapping) else {}
        key = str(row.get("example_key") or example.get("key") or "")
        if not key:
            continue
        out[key] = {
            "key": key,
            "dilemma_id": str(labels.get("dilemma_id") or ""),
            "condition_id": str(labels.get("condition_id") or ""),
            "sample_index": int(labels.get("sample_index") or 0),
            "pole": str(labels.get("pole") or ""),
            "generated_text": str(row.get("generated_text") or ""),
            "generated_token_count": len(row.get("generated_token_ids") or []),
        }
    return out


def _action_by_key(action_rows_path: Path) -> dict[str, str]:
    return {str(row["example_key"]): str(row["action_label"]) for row in _read_jsonl(action_rows_path)}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (test - mean) / std


def _cv_probe(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    if len(set(y.tolist())) < 2 or x.shape[0] < 8:
        return {"n": int(x.shape[0]), "positive": int(y.sum()), "auc_mean": float("nan"), "fold_aucs": []}

    unique_groups = np.unique(groups)
    if len(unique_groups) >= 5 and min(np.bincount(y.astype(int))) >= 5:
        splitter = GroupKFold(n_splits=5)
        splits = splitter.split(x, y, groups)
        split_kind = "group_kfold_dilemma"
    else:
        splitter = StratifiedKFold(n_splits=min(5, int(min(np.bincount(y.astype(int))))), shuffle=True, random_state=seed)
        splits = splitter.split(x, y)
        split_kind = "stratified_kfold_fallback"

    fold_aucs: list[float] = []
    fold_direction_cosines: list[float] = []
    fold_dirs: list[np.ndarray] = []
    for train_idx, test_idx in splits:
        if len(set(y[train_idx].tolist())) < 2 or len(set(y[test_idx].tolist())) < 2:
            continue
        x_train, x_test = _standardize(x[train_idx], x[test_idx])
        clf = LogisticRegression(
            C=0.1,
            solver="liblinear",
            class_weight="balanced",
            random_state=seed,
            max_iter=2000,
        )
        clf.fit(x_train, y[train_idx])
        score = clf.decision_function(x_test)
        fold_aucs.append(float(roc_auc_score(y[test_idx], score)))
        fold_dirs.append(clf.coef_[0].astype(np.float32))
    if len(fold_dirs) >= 2:
        ref = np.mean(np.stack(fold_dirs, axis=0), axis=0)
        fold_direction_cosines = [_cos(direction, ref) for direction in fold_dirs]
    return {
        "n": int(x.shape[0]),
        "positive": int(y.sum()),
        "negative": int(x.shape[0] - y.sum()),
        "split_kind": split_kind,
        "auc_mean": float(np.mean(fold_aucs)) if fold_aucs else float("nan"),
        "auc_std": float(np.std(fold_aucs)) if fold_aucs else float("nan"),
        "fold_aucs": fold_aucs,
        "fold_direction_cosine_median": float(np.median(fold_direction_cosines)) if fold_direction_cosines else float("nan"),
    }


def _centroid_cv(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    if len(set(y.tolist())) < 2 or len(np.unique(groups)) < 2:
        return {"auc_mean": float("nan"), "fold_aucs": [], "fold_direction_cosine_median": float("nan")}
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    fold_aucs: list[float] = []
    directions: list[np.ndarray] = []
    for train_idx, test_idx in splitter.split(x, y, groups):
        if len(set(y[train_idx].tolist())) < 2 or len(set(y[test_idx].tolist())) < 2:
            continue
        train = x[train_idx]
        pos = train[y[train_idx] == 1].mean(axis=0)
        neg = train[y[train_idx] == 0].mean(axis=0)
        direction = pos - neg
        score = x[test_idx] @ direction
        fold_aucs.append(float(roc_auc_score(y[test_idx], score)))
        directions.append(direction.astype(np.float32))
    if len(directions) >= 2:
        ref = np.mean(np.stack(directions, axis=0), axis=0)
        direction_cosines = [_cos(direction, ref) for direction in directions]
    else:
        direction_cosines = []
    return {
        "auc_mean": float(np.mean(fold_aucs)) if fold_aucs else float("nan"),
        "auc_std": float(np.std(fold_aucs)) if fold_aucs else float("nan"),
        "fold_aucs": fold_aucs,
        "fold_direction_cosine_median": float(np.median(direction_cosines)) if direction_cosines else float("nan"),
    }


def _split_half_distribution(deltas: np.ndarray, *, n_trials: int, seed: int) -> list[float]:
    if deltas.shape[0] < 4:
        return []
    rng = np.random.default_rng(seed)
    n = deltas.shape[0]
    half = n // 2
    vals: list[float] = []
    for _ in range(n_trials):
        order = rng.permutation(n)
        vals.append(_cos(deltas[order[:half]].mean(axis=0), deltas[order[half:]].mean(axis=0)))
    return vals


def _sign_flip_null_distribution(deltas: np.ndarray, *, n_trials: int, seed: int) -> list[float]:
    if deltas.shape[0] < 4:
        return []
    rng = np.random.default_rng(seed)
    n = deltas.shape[0]
    half = n // 2
    vals: list[float] = []
    for _ in range(n_trials):
        fake = deltas * rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n, 1))
        order = rng.permutation(n)
        vals.append(_cos(fake[order[:half]].mean(axis=0), fake[order[half:]].mean(axis=0)))
    return vals


def _paired_deltas(
    rows_by_key: dict[str, dict[str, Any]],
    feats: dict[str, np.ndarray],
    *,
    dilemmas: set[str] | None = None,
) -> np.ndarray:
    grouped: dict[tuple[str, int], dict[str, list[np.ndarray]]] = defaultdict(lambda: {"neg": [], "eth": []})
    for key, row in rows_by_key.items():
        if key not in feats:
            continue
        dilemma_id = row["dilemma_id"]
        if dilemmas is not None and dilemma_id not in dilemmas:
            continue
        cid = row["condition_id"]
        group = grouped[(dilemma_id, int(row["sample_index"]))]
        if cid in NEGATIVE_CONDITIONS:
            group["neg"].append(feats[key])
        elif cid in ETHICAL_CONDITIONS:
            group["eth"].append(feats[key])
    deltas: list[np.ndarray] = []
    for group in grouped.values():
        if group["neg"] and group["eth"]:
            deltas.append(np.stack(group["neg"]).mean(axis=0) - np.stack(group["eth"]).mean(axis=0))
    if not deltas:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(deltas, axis=0).astype(np.float32)


def _direction_stats(deltas: np.ndarray, *, trials: int, seed: int) -> dict[str, float]:
    if deltas.shape[0] < 4:
        return {
            "paired_delta_count": int(deltas.shape[0]),
            "real_median": float("nan"),
            "null_p95": float("nan"),
            "gap": float("nan"),
        }
    real = _split_half_distribution(deltas, n_trials=trials, seed=seed)
    null = _sign_flip_null_distribution(deltas, n_trials=trials, seed=seed + 10000)
    return {
        "paired_delta_count": int(deltas.shape[0]),
        "real_median": float(np.median(real)),
        "null_p95": float(np.percentile(null, 95)),
        "gap": float(np.median(real) - np.percentile(null, 95)),
    }


def _build_probe_dataset(
    rows_by_key: dict[str, dict[str, Any]],
    feats: dict[str, np.ndarray],
    action_by_key: dict[str, str],
    target: str,
    *,
    strict_gate_dilemmas: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    groups: list[str] = []
    for key, row in rows_by_key.items():
        if key not in feats:
            continue
        if strict_gate_dilemmas is not None and row["dilemma_id"] not in strict_gate_dilemmas:
            continue
        cid = row["condition_id"]
        label: int | None = None
        if target == "prompt_pole":
            if cid in NEGATIVE_CONDITIONS:
                label = 1
            elif cid in ETHICAL_CONDITIONS:
                label = 0
        elif target == "observed_action":
            action = action_by_key.get(key)
            if action == "self_advantage":
                label = 1
            elif action == "ethical":
                label = 0
        elif target == "observed_action_within_negative":
            if cid not in NEGATIVE_CONDITIONS:
                continue
            action = action_by_key.get(key)
            if action == "self_advantage":
                label = 1
            elif action == "ethical":
                label = 0
        elif target == "prompt_pole_on_gate":
            if cid in NEGATIVE_CONDITIONS:
                label = 1
            elif cid in ETHICAL_CONDITIONS:
                label = 0
        else:
            raise ValueError(f"unknown target {target!r}")
        if label is None:
            continue
        xs.append(feats[key])
        ys.append(label)
        groups.append(row["dilemma_id"])
    if not xs:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=np.int64), np.array([], dtype=object)
    return np.stack(xs, axis=0).astype(np.float32), np.asarray(ys, dtype=np.int64), np.asarray(groups, dtype=object)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.3f}"
    return str(value)


def _write_report(summary: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Ethical Advantage Activation Probe Analysis",
        "",
        f"- capture: `{summary['capture_artifact_id']}`",
        f"- generation rows: `{summary['generation_rows_path']}`",
        f"- strict behavior-gate dilemmas: `{len(summary['strict_gate_dilemmas'])}`",
        "",
        "## Prompt Pole Probe",
        "",
        "Target: negative short-term self-advantage prompts vs ethical prompts, grouped by dilemma.",
        "",
        "| slice | layer | n | positives | logistic AUROC | centroid AUROC | split gap | real median | null p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        if row["target"] != "prompt_pole":
            continue
        lines.append(
            f"| {row['slice']} | {row['layer']} | {row['n']} | {row['positive']} | "
            f"{_fmt(row['logistic_auc_mean'])} | {_fmt(row['centroid_auc_mean'])} | "
            f"{_fmt(row['direction_gap'])} | {_fmt(row['direction_real_median'])} | {_fmt(row['direction_null_p95'])} |"
        )
    lines.extend(
        [
            "",
            "## Observed Action Probe",
            "",
            "Target: regex-labeled self-advantage action vs ethical action across all conditions; unknown labels excluded.",
            f"Condition-only AUROC for this target: `{_fmt(summary['condition_only_action_auc'])}`.",
            "",
            "| slice | layer | n | positives | logistic AUROC | centroid AUROC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["rows"]:
        if row["target"] != "observed_action":
            continue
        lines.append(
            f"| {row['slice']} | {row['layer']} | {row['n']} | {row['positive']} | "
            f"{_fmt(row['logistic_auc_mean'])} | {_fmt(row['centroid_auc_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Observed Action Within Negative Prompts",
            "",
            "Target: among only self-serving/exploit prompt conditions, distinguish responses that actually chose self-advantage from responses that chose ethical action. Unknown labels excluded.",
            "",
            "| slice | layer | n | positives | logistic AUROC | centroid AUROC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["rows"]:
        if row["target"] != "observed_action_within_negative":
            continue
        lines.append(
            f"| {row['slice']} | {row['layer']} | {row['n']} | {row['positive']} | "
            f"{_fmt(row['logistic_auc_mean'])} | {_fmt(row['centroid_auc_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Strict-Gate Prompt Pole Probe",
            "",
            "Target: prompt pole restricted to dilemmas where behavior cleanly flipped in the smoke labeler.",
            "",
            "| slice | layer | n | positives | logistic AUROC | centroid AUROC | split gap |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["rows"]:
        if row["target"] != "prompt_pole_on_gate":
            continue
        lines.append(
            f"| {row['slice']} | {row['layer']} | {row['n']} | {row['positive']} | "
            f"{_fmt(row['logistic_auc_mean'])} | {_fmt(row['centroid_auc_mean'])} | {_fmt(row['direction_gap'])} |"
        )
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    *,
    capture_id: str,
    generation_rows_path: Path,
    action_rows_path: Path,
    report_dir: Path,
    trials: int,
) -> None:
    rows_by_key = _rows_by_key(generation_rows_path)
    action_by_key = _action_by_key(action_rows_path)
    condition_scores = {
        key: 1.0 if row["condition_id"] in NEGATIVE_CONDITIONS else 0.0
        for key, row in rows_by_key.items()
    }
    action_y: list[int] = []
    condition_score_y: list[float] = []
    for key, action in action_by_key.items():
        if key not in condition_scores:
            continue
        if action == "self_advantage":
            action_y.append(1)
            condition_score_y.append(condition_scores[key])
        elif action == "ethical":
            action_y.append(0)
            condition_score_y.append(condition_scores[key])
    condition_only_action_auc = (
        float(roc_auc_score(np.asarray(action_y), np.asarray(condition_score_y)))
        if len(set(action_y)) == 2
        else float("nan")
    )

    # Strict gate mirrors the behavior-smoke report: all negative prompts self-advantage and all ethical prompts ethical.
    by_dilemma: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"neg": [], "eth": []})
    for key, row in rows_by_key.items():
        action = action_by_key.get(key)
        if row["condition_id"] in NEGATIVE_CONDITIONS:
            by_dilemma[row["dilemma_id"]]["neg"].append(action or "missing")
        elif row["condition_id"] in ETHICAL_CONDITIONS:
            by_dilemma[row["dilemma_id"]]["eth"].append(action or "missing")
    strict_gate_dilemmas = {
        dilemma_id
        for dilemma_id, rec in by_dilemma.items()
        if rec["neg"] and rec["eth"] and all(x == "self_advantage" for x in rec["neg"]) and all(x == "ethical" for x in rec["eth"])
    }

    capture = _load_capture(capture_id)
    rows: list[dict[str, Any]] = []
    for layer in LAYERS:
        for slice_name in SLICES:
            feats = _feature_map(capture, layer=layer, slice_name=slice_name)
            deltas_all = _paired_deltas(rows_by_key, feats)
            deltas_gate = _paired_deltas(rows_by_key, feats, dilemmas=strict_gate_dilemmas)
            direction_all = _direction_stats(deltas_all, trials=trials, seed=1000 + layer)
            direction_gate = _direction_stats(deltas_gate, trials=trials, seed=2000 + layer)
            for target in ("prompt_pole", "observed_action", "observed_action_within_negative", "prompt_pole_on_gate"):
                gate = strict_gate_dilemmas if target == "prompt_pole_on_gate" else None
                x, y, groups = _build_probe_dataset(rows_by_key, feats, action_by_key, target, strict_gate_dilemmas=gate)
                logistic = _cv_probe(x, y, groups, seed=3000 + layer)
                centroid = _centroid_cv(x, y, groups)
                direction = direction_gate if target == "prompt_pole_on_gate" else direction_all
                rows.append(
                    {
                        "target": target,
                        "slice": slice_name,
                        "layer": layer,
                        "n": logistic.get("n"),
                        "positive": logistic.get("positive"),
                        "negative": logistic.get("negative"),
                        "logistic_auc_mean": logistic.get("auc_mean"),
                        "logistic_auc_std": logistic.get("auc_std"),
                        "logistic_fold_aucs": logistic.get("fold_aucs"),
                        "logistic_fold_direction_cosine_median": logistic.get("fold_direction_cosine_median"),
                        "centroid_auc_mean": centroid.get("auc_mean"),
                        "centroid_auc_std": centroid.get("auc_std"),
                        "centroid_fold_aucs": centroid.get("fold_aucs"),
                        "centroid_fold_direction_cosine_median": centroid.get("fold_direction_cosine_median"),
                        "direction_paired_delta_count": direction.get("paired_delta_count"),
                        "direction_real_median": direction.get("real_median"),
                        "direction_null_p95": direction.get("null_p95"),
                        "direction_gap": direction.get("gap"),
                    }
                )

    summary = {
        "capture_artifact_id": capture_id,
        "generation_rows_path": str(generation_rows_path),
        "action_rows_path": str(action_rows_path),
        "layers": list(LAYERS),
        "slices": list(SLICES),
        "trials": trials,
        "condition_only_action_auc": condition_only_action_auc,
        "strict_gate_dilemmas": sorted(strict_gate_dilemmas),
        "rows": rows,
    }
    _write_report(summary, report_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-id", default="capture_1_2461d8ccdc41")
    parser.add_argument("--generation-rows", type=Path, default=DEFAULT_GENERATION_ROWS)
    parser.add_argument("--action-rows", type=Path, default=DEFAULT_ACTION_ROWS)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--trials", type=int, default=128)
    args = parser.parse_args()
    analyze(
        capture_id=args.capture_id,
        generation_rows_path=args.generation_rows,
        action_rows_path=args.action_rows,
        report_dir=args.report_dir,
        trials=args.trials,
    )
    print(f"wrote {args.report_dir / 'report.md'}")


if __name__ == "__main__":
    main()
