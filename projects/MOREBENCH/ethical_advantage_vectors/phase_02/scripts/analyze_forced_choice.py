"""Analyze forced-choice ethical-vs-self-advantage generations and prompt residuals."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from projects.MOREBENCH.ethical_advantage_vectors.phase_01.scripts import analyze_activation_probes as phase01_probes


PHASE_ROOT = Path("projects/MOREBENCH/ethical_advantage_vectors/phase_02")
DEFAULT_REPORT_DIR = PHASE_ROOT / "reports" / "forced_choice_analysis"
LAYERS = (16, 24, 32, 40)
SITES = ("scenario_end_residual", "options_end_residual", "prompt_end_residual")
ETHICAL_CONDITIONS = {"P_ethical_choice_01"}
NEGATIVE_CONDITIONS = {"P_self_serving_choice_01", "P_exploit_choice_01"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_choice(text: str) -> str:
    match = re.search(r"\b([ABCD])\b", text.strip().upper())
    if match:
        return match.group(1)
    match = re.search(r"([ABCD])", text.strip().upper())
    return match.group(1) if match else "unknown"


def _generation_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise TypeError(f"{path} must contain a rows list")
    rows: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        if not isinstance(row, Mapping):
            continue
        example = row.get("example") if isinstance(row.get("example"), Mapping) else {}
        labels = example.get("labels") if isinstance(example.get("labels"), Mapping) else {}
        key = str(row.get("example_key") or example.get("key") or "")
        choice_letter = _extract_choice(str(row.get("generated_text") or ""))
        choice_type = str(labels.get(f"option_{choice_letter}_type") or "unknown")
        rows[key] = {
            "key": key,
            "dilemma_id": str(labels.get("dilemma_id") or ""),
            "condition_id": str(labels.get("condition_id") or ""),
            "pole": str(labels.get("pole") or ""),
            "option_order_index": int(labels.get("option_order_index") or 0),
            "ethical_letter": str(labels.get("ethical_letter") or ""),
            "self_advantage_letter": str(labels.get("self_advantage_letter") or ""),
            "choice_letter": choice_letter,
            "choice_type": choice_type,
            "generated_text": str(row.get("generated_text") or ""),
            "finish_reason": str(row.get("finish_reason") or ""),
        }
    return rows


def _feature_payload(capture: Any, site: str) -> dict[str, Any]:
    payload = capture.feature(site).load()
    if not isinstance(payload, Mapping):
        raise TypeError(f"feature {site!r} payload is not a mapping")
    return dict(payload)


def _feature_map(capture: Any, *, site: str, layer: int) -> dict[str, np.ndarray]:
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
        vec = arr if arr.ndim == 1 else arr.mean(axis=0)
        if vec.size:
            out[str(key)] = vec
    return out


def _local_feature_map(capture_dir: Path, *, site: str, layer: int) -> dict[str, np.ndarray]:
    metadata_path = capture_dir / "features" / f"{site}.metadata.json"
    tensor_path = capture_dir / "features" / "feature_tensors.safetensors"
    metadata = _read_json(metadata_path)
    tensors = load_file(str(tensor_path))
    layer_payload = metadata.get("layers", {}).get(str(layer))
    if not isinstance(layer_payload, Mapping):
        raise RuntimeError(f"missing {site} L{layer} in {metadata_path}")
    out: dict[str, np.ndarray] = {}
    for key, rec in layer_payload.items():
        if not isinstance(rec, Mapping):
            continue
        values = rec.get("values")
        if isinstance(values, Mapping) and "__tensor_key__" in values:
            arr = np.asarray(tensors[str(values["__tensor_key__"])], dtype=np.float32)
        else:
            arr = np.asarray(values, dtype=np.float32)
        vec = arr if arr.ndim == 1 else arr.mean(axis=0)
        if vec.size:
            out[str(key)] = vec
    return out


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (test - mean) / std


def _splits(y: np.ndarray, groups: np.ndarray, *, seed: int):
    if len(set(y.tolist())) < 2 or len(y) < 8:
        return "none", []
    unique_groups = np.unique(groups)
    counts = np.bincount(y.astype(int), minlength=2)
    if len(unique_groups) >= 5 and int(counts.min()) >= 5:
        splitter = GroupKFold(n_splits=5)
        return "group_kfold_dilemma", list(splitter.split(np.zeros((len(y), 1)), y, groups))
    n_splits = min(5, int(counts.min()))
    if n_splits < 2:
        return "none", []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return "stratified_kfold_fallback", list(splitter.split(np.zeros((len(y), 1)), y))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _probe(
    rows: list[dict[str, Any]],
    feats: dict[str, np.ndarray],
    *,
    target: str,
    seed: int,
) -> dict[str, Any]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    groups: list[str] = []
    for row in rows:
        key = row["key"]
        if key not in feats:
            continue
        label: int | None = None
        cid = row["condition_id"]
        if target == "prompt_pole":
            if cid in NEGATIVE_CONDITIONS:
                label = 1
            elif cid in ETHICAL_CONDITIONS:
                label = 0
        elif target == "chosen_self_vs_ethical":
            if row["choice_type"] == "self_advantage":
                label = 1
            elif row["choice_type"] == "ethical":
                label = 0
        elif target == "chosen_self_vs_ethical_neutral":
            if cid != "N_neutral_choice_01":
                continue
            if row["choice_type"] == "self_advantage":
                label = 1
            elif row["choice_type"] == "ethical":
                label = 0
        elif target == "chosen_self_vs_ethical_negative":
            if cid not in NEGATIVE_CONDITIONS:
                continue
            if row["choice_type"] == "self_advantage":
                label = 1
            elif row["choice_type"] == "ethical":
                label = 0
        else:
            raise ValueError(target)
        if label is None:
            continue
        xs.append(feats[key])
        ys.append(label)
        groups.append(row["dilemma_id"])
    if not xs:
        return {"n": 0, "positive": 0, "negative": 0, "auc_mean": float("nan"), "centroid_auc_mean": float("nan")}
    x = np.stack(xs, axis=0).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    group_arr = np.asarray(groups, dtype=object)
    split_kind, splits = _splits(y, group_arr, seed=seed)

    logit_aucs: list[float] = []
    centroid_aucs: list[float] = []
    directions: list[np.ndarray] = []
    for train_idx, test_idx in splits:
        if len(set(y[train_idx].tolist())) < 2 or len(set(y[test_idx].tolist())) < 2:
            continue
        x_train, x_test = _standardize(x[train_idx], x[test_idx])
        clf = LogisticRegression(C=0.1, solver="liblinear", class_weight="balanced", random_state=seed, max_iter=2000)
        clf.fit(x_train, y[train_idx])
        logit_aucs.append(float(roc_auc_score(y[test_idx], clf.decision_function(x_test))))
        pos = x[train_idx][y[train_idx] == 1].mean(axis=0)
        neg = x[train_idx][y[train_idx] == 0].mean(axis=0)
        direction = pos - neg
        centroid_aucs.append(float(roc_auc_score(y[test_idx], x[test_idx] @ direction)))
        directions.append(direction.astype(np.float32))
    if len(directions) >= 2:
        ref = np.mean(np.stack(directions, axis=0), axis=0)
        direction_cosines = [_cos(direction, ref) for direction in directions]
    else:
        direction_cosines = []
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "split_kind": split_kind,
        "auc_mean": float(np.mean(logit_aucs)) if logit_aucs else float("nan"),
        "auc_std": float(np.std(logit_aucs)) if logit_aucs else float("nan"),
        "centroid_auc_mean": float(np.mean(centroid_aucs)) if centroid_aucs else float("nan"),
        "centroid_auc_std": float(np.std(centroid_aucs)) if centroid_aucs else float("nan"),
        "fold_direction_cosine_median": float(np.median(direction_cosines)) if direction_cosines else float("nan"),
        "fold_aucs": logit_aucs,
        "fold_centroid_aucs": centroid_aucs,
    }


def _behavior_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, Counter[str]] = defaultdict(Counter)
    by_condition_letter: dict[str, Counter[str]] = defaultdict(Counter)
    by_order: dict[str, Counter[str]] = defaultdict(Counter)
    finish_reasons: Counter[str] = Counter()
    malformed = 0
    for row in rows:
        cid = row["condition_id"]
        by_condition[cid][row["choice_type"]] += 1
        by_condition_letter[cid][row["choice_letter"]] += 1
        by_order[str(row["option_order_index"])][row["choice_type"]] += 1
        finish_reasons[row["finish_reason"]] += 1
        if row["choice_letter"] == "unknown" or row["choice_type"] == "unknown":
            malformed += 1

    def rates(counter: Counter[str]) -> dict[str, float]:
        total = sum(counter.values()) or 1
        return {key: count / total for key, count in sorted(counter.items())}

    return {
        "row_count": len(rows),
        "malformed_choice_count": malformed,
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "choice_type_counts_by_condition": {condition: dict(counter) for condition, counter in sorted(by_condition.items())},
        "choice_type_rates_by_condition": {condition: rates(counter) for condition, counter in sorted(by_condition.items())},
        "choice_letter_counts_by_condition": {condition: dict(counter) for condition, counter in sorted(by_condition_letter.items())},
        "choice_type_counts_by_order": {order: dict(counter) for order, counter in sorted(by_order.items())},
    }


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
        "# Forced-Choice Ethical Advantage Analysis",
        "",
        f"- generation rows: `{summary['generation_rows_path']}`",
        f"- capture: `{summary['capture_artifact_id']}`",
        "",
        "## Behavior",
        "",
        f"- rows: `{summary['behavior']['row_count']}`",
        f"- malformed choices: `{summary['behavior']['malformed_choice_count']}`",
        "",
        "| condition | ethical | self_advantage | procedural | delay | unknown |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    rates_by_condition = summary["behavior"]["choice_type_rates_by_condition"]
    for condition, rates in rates_by_condition.items():
        lines.append(
            f"| {condition} | {_fmt(float(rates.get('ethical', 0.0)))} | {_fmt(float(rates.get('self_advantage', 0.0)))} | "
            f"{_fmt(float(rates.get('procedural', 0.0)))} | {_fmt(float(rates.get('delay', 0.0)))} | {_fmt(float(rates.get('unknown', 0.0)))} |"
        )

    lines.extend(
        [
            "",
            "## Residual Probes",
            "",
            "| target | site | layer | n | positives | logistic AUROC | centroid AUROC | dir cos median |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["probe_rows"]:
        lines.append(
            f"| {row['target']} | {row['site']} | {row['layer']} | {row['n']} | {row['positive']} | "
            f"{_fmt(row['auc_mean'])} | {_fmt(row['centroid_auc_mean'])} | {_fmt(row['fold_direction_cosine_median'])} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `prompt_pole` decodes negative prompt regimes vs ethical prompt regime; this is expected to carry instruction information.",
            "- `chosen_self_vs_ethical` is the cleaner target because the generated output is only a balanced option letter.",
            "- Letter balancing matters: inspect behavior rates by option-order variant before treating choice probes as action geometry.",
        ]
    )
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    *,
    generation_rows_path: Path,
    capture_id: str,
    capture_local_dir: Path | None,
    report_dir: Path,
    seed: int,
) -> None:
    rows_by_key = _generation_rows(generation_rows_path)
    rows = list(rows_by_key.values())
    capture = None if capture_local_dir is not None else phase01_probes._load_capture(capture_id)
    probe_rows: list[dict[str, Any]] = []
    for site in SITES:
        for layer in LAYERS:
            feats = (
                _local_feature_map(capture_local_dir, site=site, layer=layer)
                if capture_local_dir is not None
                else _feature_map(capture, site=site, layer=layer)
            )
            for target in (
                "prompt_pole",
                "chosen_self_vs_ethical",
                "chosen_self_vs_ethical_neutral",
                "chosen_self_vs_ethical_negative",
            ):
                probe_rows.append(
                    {
                        "target": target,
                        "site": site,
                        "layer": layer,
                        **_probe(rows, feats, target=target, seed=seed),
                    }
                )

    summary = {
        "generation_rows_path": str(generation_rows_path),
        "capture_artifact_id": capture_id,
        "capture_local_dir": str(capture_local_dir) if capture_local_dir is not None else None,
        "behavior": _behavior_summary(rows),
        "probe_rows": probe_rows,
    }
    _write_report(summary, report_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-rows", type=Path, required=True)
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--capture-local-dir", type=Path)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    analyze(
        generation_rows_path=args.generation_rows,
        capture_id=args.capture_id,
        capture_local_dir=args.capture_local_dir,
        report_dir=args.report_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
