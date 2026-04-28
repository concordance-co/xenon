"""Corrected paired/tail analysis for the all-theories pole pilot.

This script fixes two methodological problems in the first phase_02 analysis:

1. The design is paired by dilemma, but the first analysis treated positives and
   negatives as independent bags. Here each direction is the mean of per-dilemma
   deltas: activation(condition A, dilemma i) - activation(condition B, dilemma i).

2. The first "gap" compared a real split-half cosine to a null cosine between
   fake directions and the real full direction. Here the null recomputes the
   same split-half statistic after random sign flips of the paired deltas.

It also tests the terse-substrate diagnosis by recomputing the paired smoke
under response-length filters on the existing generation/capture artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from pipelines_v2.api import ModalVolumeStore, PostgresCatalog, PostgresSource, TransferPolicy
from pipelines_v2.storage.artifacts import CaptureArtifact, artifact_from_manifest


PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_02")
DEFAULT_GENERATION_ROWS = (
    PHASE_ROOT
    / "reports"
    / "all_theories_pole_pilot_report"
    / "report_0a6a1bb7ed4a_f2f091e0"
    / "results"
    / "generate_terse_responses_results.json"
)
DEFAULT_REPORT_DIR = PHASE_ROOT / "reports" / "all_theories_paired_tail_analysis"

DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
ARTIFACT_STORE_NAME = "xenon-data"
PILOT_ROOT = "/data/artifacts/morebench_theory_persona_vectors_phase02"

CAPTURED_LAYERS = (0, 4, 16, 24, 32, 40)
PRIMARY_LAYER = 32
PRIMARY_SITE = "generated_sequence_residual"
DIAGNOSTIC_SITE = "prompt_end_residual"

THEORIES = ("deontology", "utilitarian", "virtue_ethics", "contractualism")
THEORY_SHORT = {
    "deontology": "deont",
    "utilitarian": "util",
    "virtue_ethics": "virtue",
    "contractualism": "contract",
}
POSITIVE_PRIMARY = {
    "deontology": "P_deont_01",
    "utilitarian": "P_util_01",
    "virtue_ethics": "P_virtue_01",
    "contractualism": "P_contract_01",
}
ANTI_POLE = {
    "deontology": "N_anti_deont_01",
    "utilitarian": "N_anti_util_01",
    "virtue_ethics": "N_anti_virtue_01",
    "contractualism": "N_anti_contract_01",
}
NEUTRAL_SHORT = "N_neutral_01"
NEUTRAL_LM = "N_neutral_02"

CONSTRUCTION_ORDER = (
    "neutral_short",
    "neutral_length_matched",
    "anti",
    "alt_deont",
    "alt_util",
    "alt_virtue",
    "alt_contract",
)


def _catalog() -> PostgresCatalog:
    return PostgresCatalog(source=PostgresSource.from_env(DB_ENV_VAR))


def _store(root: str = PILOT_ROOT) -> ModalVolumeStore:
    return ModalVolumeStore(
        name=ARTIFACT_STORE_NAME,
        root=root,
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


def _capture_layer_features(capture: CaptureArtifact, *, site: str, layer: int) -> dict[str, np.ndarray]:
    payload = _feature_payload(capture, site)
    layer_payload = payload.get("layers", {}).get(str(layer))
    if not isinstance(layer_payload, Mapping):
        raise RuntimeError(f"site {site!r} missing layer {layer}")
    out: dict[str, np.ndarray] = {}
    for key, rec in layer_payload.items():
        values = rec.get("values") if isinstance(rec, Mapping) else None
        if values is None:
            continue
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 1:
            out[str(key)] = arr
        elif arr.shape[0] > 0:
            out[str(key)] = arr.mean(axis=0)
    return out


def _rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise TypeError(f"{path} must contain a rows list")
    return [r for r in rows if isinstance(r, Mapping)]


def _row_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        example = row.get("example") if isinstance(row.get("example"), Mapping) else {}
        labels = example.get("labels") if isinstance(example.get("labels"), Mapping) else {}
        dilemma_id = str(labels.get("dilemma_id") or "")
        condition_id = str(labels.get("condition_id") or "")
        if not dilemma_id or not condition_id:
            continue
        token_ids = row.get("generated_token_ids")
        token_count = len(token_ids) if isinstance(token_ids, list) else len(str(row.get("generated_text") or "").split())
        out[(dilemma_id, condition_id)] = {
            "key": str(row.get("example_key") or example.get("key") or ""),
            "text": str(row.get("generated_text") or ""),
            "token_count": int(token_count),
            "char_count": len(str(row.get("generated_text") or "")),
            "labels": dict(labels),
        }
    return out


def _contrast_pairs(theory: str) -> dict[str, tuple[str, str]]:
    short = THEORY_SHORT[theory]
    p = POSITIVE_PRIMARY[theory]
    pairs = {
        f"{short}_neutral_short": (p, NEUTRAL_SHORT),
        f"{short}_neutral_length_matched": (p, NEUTRAL_LM),
        f"{short}_anti": (p, ANTI_POLE[theory]),
    }
    for other in THEORIES:
        if other == theory:
            continue
        pairs[f"{short}_alt_{THEORY_SHORT[other]}"] = (p, POSITIVE_PRIMARY[other])
    return pairs


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _split_half_distribution(
    deltas: np.ndarray,
    *,
    n_trials: int,
    seed: int,
) -> list[float]:
    if deltas.shape[0] < 4:
        return []
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = deltas.shape[0]
    half = n // 2
    for _ in range(n_trials):
        order = rng.permutation(n)
        a = deltas[order[:half]].mean(axis=0)
        b = deltas[order[half:]].mean(axis=0)
        values.append(_cos(a, b))
    return values


def _sign_flip_null_distribution(
    deltas: np.ndarray,
    *,
    n_trials: int,
    seed: int,
) -> list[float]:
    if deltas.shape[0] < 4:
        return []
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = deltas.shape[0]
    half = n // 2
    for _ in range(n_trials):
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n, 1))
        fake = deltas * signs
        order = rng.permutation(n)
        a = fake[order[:half]].mean(axis=0)
        b = fake[order[half:]].mean(axis=0)
        values.append(_cos(a, b))
    return values


def _stats(values: list[float]) -> dict[str, float]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return {"mean": float("nan"), "median": float("nan"), "p05": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(clean)),
        "median": float(np.median(clean)),
        "p05": float(np.percentile(clean, 5)),
        "p95": float(np.percentile(clean, 95)),
    }


def _include_pair(pos: dict[str, Any], neg: dict[str, Any], *, min_tokens: int, mode: str) -> bool:
    if mode == "none":
        return True
    if mode == "pos":
        return pos["token_count"] >= min_tokens
    if mode == "both":
        return pos["token_count"] >= min_tokens and neg["token_count"] >= min_tokens
    if mode == "either":
        return pos["token_count"] >= min_tokens or neg["token_count"] >= min_tokens
    raise ValueError(f"unknown filter mode: {mode}")


def _paired_deltas(
    *,
    row_by_dilemma_condition: dict[tuple[str, str], dict[str, Any]],
    feats: dict[str, np.ndarray],
    pos_condition: str,
    neg_condition: str,
    min_tokens: int,
    filter_mode: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    deltas: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    dilemma_ids = sorted({d for d, _ in row_by_dilemma_condition})
    for dilemma_id in dilemma_ids:
        pos = row_by_dilemma_condition.get((dilemma_id, pos_condition))
        neg = row_by_dilemma_condition.get((dilemma_id, neg_condition))
        if not pos or not neg:
            continue
        if not _include_pair(pos, neg, min_tokens=min_tokens, mode=filter_mode):
            continue
        pos_key = pos["key"]
        neg_key = neg["key"]
        if pos_key not in feats or neg_key not in feats:
            continue
        deltas.append(feats[pos_key] - feats[neg_key])
        meta.append(
            {
                "dilemma_id": dilemma_id,
                "pos_tokens": pos["token_count"],
                "neg_tokens": neg["token_count"],
                "pos_chars": pos["char_count"],
                "neg_chars": neg["char_count"],
            }
        )
    if not deltas:
        return np.empty((0, 0), dtype=np.float32), meta
    return np.stack(deltas, axis=0), meta


def _analyze_filter(
    *,
    capture: CaptureArtifact,
    row_index: dict[tuple[str, str], dict[str, Any]],
    site: str,
    layer: int,
    min_tokens: int,
    filter_mode: str,
    n_trials: int,
) -> dict[str, Any]:
    feats = _capture_layer_features(capture, site=site, layer=layer)
    rows: list[dict[str, Any]] = []
    directions: dict[str, np.ndarray] = {}
    for theory in THEORIES:
        for construction, (pos, neg) in _contrast_pairs(theory).items():
            deltas, meta = _paired_deltas(
                row_by_dilemma_condition=row_index,
                feats=feats,
                pos_condition=pos,
                neg_condition=neg,
                min_tokens=min_tokens,
                filter_mode=filter_mode,
            )
            if deltas.shape[0] < 4:
                rows.append(
                    {
                        "theory": theory,
                        "construction": construction,
                        "n_pairs": int(deltas.shape[0]),
                        "status": "too_few_pairs",
                    }
                )
                continue
            direction = deltas.mean(axis=0)
            directions[construction] = direction
            real = _split_half_distribution(deltas, n_trials=n_trials, seed=1000 + layer)
            null = _sign_flip_null_distribution(deltas, n_trials=n_trials, seed=2000 + layer)
            real_stats = _stats(real)
            null_stats = _stats(null)
            rows.append(
                {
                    "theory": theory,
                    "construction": construction,
                    "n_pairs": int(deltas.shape[0]),
                    "direction_norm": float(np.linalg.norm(direction)),
                    "mean_pos_tokens": float(mean(m["pos_tokens"] for m in meta)),
                    "mean_neg_tokens": float(mean(m["neg_tokens"] for m in meta)),
                    "real_split_half": real_stats,
                    "sign_flip_null": null_stats,
                    "gap_median_minus_null_p95": real_stats["median"] - null_stats["p95"],
                    "gap_mean_minus_null_p95": real_stats["mean"] - null_stats["p95"],
                    "status": "ok",
                }
            )

    # Cross-theory cosine matrices for neutral constructions.
    cross_theory: dict[str, dict[str, dict[str, float]]] = {}
    for anchor in ("neutral_short", "neutral_length_matched", "anti"):
        mat_dirs: dict[str, np.ndarray] = {}
        for theory in THEORIES:
            short = THEORY_SHORT[theory]
            name = f"{short}_{anchor}"
            if name in directions:
                mat_dirs[short] = directions[name]
        matrix: dict[str, dict[str, float]] = {}
        for a, va in mat_dirs.items():
            matrix[a] = {b: _cos(va, vb) for b, vb in mat_dirs.items()}
        cross_theory[anchor] = matrix

    return {
        "site": site,
        "layer": layer,
        "min_tokens": min_tokens,
        "filter_mode": filter_mode,
        "n_trials": n_trials,
        "rows": rows,
        "cross_theory_cosines": cross_theory,
    }


def _token_distribution(row_index: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[int]] = defaultdict(list)
    all_tokens: list[int] = []
    for (_, condition), row in row_index.items():
        tok = int(row["token_count"])
        by_condition[condition].append(tok)
        all_tokens.append(tok)
    out = {
        "overall": {
            "n": len(all_tokens),
            "mean": float(mean(all_tokens)),
            "median": float(np.median(all_tokens)),
            "p10": float(np.percentile(all_tokens, 10)),
            "p90": float(np.percentile(all_tokens, 90)),
            "share_lt_10": float(np.mean(np.asarray(all_tokens) < 10)),
            "share_lt_20": float(np.mean(np.asarray(all_tokens) < 20)),
        },
        "by_condition": {},
    }
    for condition, toks in sorted(by_condition.items()):
        arr = np.asarray(toks)
        out["by_condition"][condition] = {
            "n": len(toks),
            "mean": float(mean(toks)),
            "median": float(np.median(arr)),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "share_ge_20": float(np.mean(arr >= 20)),
        }
    return out


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.3f}"
    return str(v)


def _write_report(summary: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines: list[str] = []
    lines.append("# All-Theories Paired Tail Analysis")
    lines.append("")
    lines.append(f"- capture artifact: `{summary['capture_artifact_id']}`")
    lines.append(f"- generation rows: `{summary['generation_rows_path']}`")
    lines.append(f"- primary site/layer: `{PRIMARY_SITE}` / L{PRIMARY_LAYER}")
    lines.append("")
    lines.append("## Response Token Distribution")
    lines.append("")
    overall = summary["token_distribution"]["overall"]
    lines.append(
        f"- n={overall['n']}, mean={_fmt(overall['mean'])}, median={_fmt(overall['median'])}, "
        f"share_lt_10={_fmt(overall['share_lt_10'])}, share_lt_20={_fmt(overall['share_lt_20'])}"
    )
    lines.append("")
    lines.append("| condition | n | mean | median | min | max | share_ge_20 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for condition, stats in summary["token_distribution"]["by_condition"].items():
        lines.append(
            f"| {condition} | {stats['n']} | {_fmt(stats['mean'])} | {_fmt(stats['median'])} | "
            f"{stats['min']} | {stats['max']} | {_fmt(stats['share_ge_20'])} |"
        )
    lines.append("")

    for block in summary["analyses"]:
        lines.append(f"## Paired Smoke: `{block['filter_mode']}` min_tokens={block['min_tokens']}")
        lines.append("")
        lines.append("| theory | construction | n | real_median | null_p95 | gap | pos_tok | neg_tok |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in block["rows"]:
            if row["status"] != "ok":
                lines.append(f"| {row['theory']} | {row['construction']} | {row['n_pairs']} | NA | NA | NA | NA | NA |")
                continue
            lines.append(
                f"| {row['theory']} | {row['construction']} | {row['n_pairs']} | "
                f"{_fmt(row['real_split_half']['median'])} | {_fmt(row['sign_flip_null']['p95'])} | "
                f"{_fmt(row['gap_median_minus_null_p95'])} | "
                f"{_fmt(row['mean_pos_tokens'])} | {_fmt(row['mean_neg_tokens'])} |"
            )
        lines.append("")
        lines.append("Cross-theory cosines:")
        lines.append("")
        for anchor, matrix in block["cross_theory_cosines"].items():
            if not matrix:
                continue
            names = sorted(matrix)
            lines.append(f"### {anchor}")
            lines.append("")
            lines.append("| | " + " | ".join(names) + " |")
            lines.append("|---" + "|---:" * len(names) + "|")
            for a in names:
                lines.append(f"| {a} | " + " | ".join(_fmt(matrix[a].get(b)) for b in names) + " |")
            lines.append("")

    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-id", default="capture_1_c2684db0530c")
    parser.add_argument("--generation-rows", default=str(DEFAULT_GENERATION_ROWS))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--trials", type=int, default=256)
    args = parser.parse_args()

    rows = _rows(Path(args.generation_rows))
    row_idx = _row_index(rows)
    capture = _load_capture(args.capture_id)

    analyses = []
    # all rows, then tail filters. "pos" asks whether theory-expressive positive
    # rows are the binding constraint; "both" is stricter but can get small.
    for filter_mode, min_tokens in (
        ("none", 0),
        ("pos", 10),
        ("pos", 20),
        ("both", 10),
        ("both", 20),
    ):
        analyses.append(
            _analyze_filter(
                capture=capture,
                row_index=row_idx,
                site=PRIMARY_SITE,
                layer=PRIMARY_LAYER,
                min_tokens=min_tokens,
                filter_mode=filter_mode,
                n_trials=args.trials,
            )
        )

    summary = {
        "capture_artifact_id": args.capture_id,
        "generation_rows_path": str(Path(args.generation_rows)),
        "primary_site": PRIMARY_SITE,
        "primary_layer": PRIMARY_LAYER,
        "n_trials": args.trials,
        "token_distribution": _token_distribution(row_idx),
        "analyses": analyses,
    }
    _write_report(summary, Path(args.report_dir))
    print(f"wrote {Path(args.report_dir) / 'report.md'}")


if __name__ == "__main__":
    main()
