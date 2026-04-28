"""Paired sign-flip analysis for phase 03 natural-prompt captures."""

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


PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_03")
DEFAULT_REPORT_ROOT = PHASE_ROOT / "reports" / "all_theories_natural_prompt_report"
DEFAULT_REPORT_DIR = PHASE_ROOT / "reports" / "all_theories_natural_prompt_paired_analysis"

DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
ARTIFACT_STORE_NAME = "xenon-data"
PILOT_ROOT = "/data/artifacts/morebench_theory_persona_vectors_phase03"

PRIMARY_SITE = "generated_sequence_residual"
PRIMARY_LAYER = 32
CAPTURED_LAYERS = (0, 4, 16, 24, 32, 40)

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
POSITIVE_VARIANT = {
    "deontology": "P_deont_02",
    "utilitarian": "P_util_02",
    "virtue_ethics": "P_virtue_02",
    "contractualism": "P_contract_02",
}
ANTI_POLE = {
    "deontology": "N_anti_deont_01",
    "utilitarian": "N_anti_util_01",
    "virtue_ethics": "N_anti_virtue_01",
    "contractualism": "N_anti_contract_01",
}
NEUTRAL_SHORT = "N_neutral_01"
NEUTRAL_LENGTH = "N_neutral_02"
GENERIC_MORAL = "N_generic_moral_01"


def _latest_generation_rows_path(report_root: Path) -> Path:
    candidates = sorted(
        report_root.glob("report_*/results/generate_natural_responses_results.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no generate_natural_responses result found under {report_root}")
    return candidates[0]


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
        f"{short}_neutral_length": (p, NEUTRAL_LENGTH),
        f"{short}_generic_moral": (p, GENERIC_MORAL),
        f"{short}_anti": (p, ANTI_POLE[theory]),
        f"{short}_positive_variant": (p, POSITIVE_VARIANT[theory]),
    }
    for other in THEORIES:
        if other == theory:
            continue
        pairs[f"{short}_alt_{THEORY_SHORT[other]}"] = (p, POSITIVE_PRIMARY[other])
    return pairs


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _project_out(vec: np.ndarray, basis: np.ndarray) -> np.ndarray:
    denom = float(np.dot(basis, basis))
    if denom < 1e-12:
        return vec
    return vec - (float(np.dot(vec, basis)) / denom) * basis


def _split_half_distribution(deltas: np.ndarray, *, n_trials: int, seed: int) -> list[float]:
    if deltas.shape[0] < 4:
        return []
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = deltas.shape[0]
    half = n // 2
    for _ in range(n_trials):
        order = rng.permutation(n)
        values.append(_cos(deltas[order[:half]].mean(axis=0), deltas[order[half:]].mean(axis=0)))
    return values


def _sign_flip_null_distribution(deltas: np.ndarray, *, n_trials: int, seed: int) -> list[float]:
    if deltas.shape[0] < 4:
        return []
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = deltas.shape[0]
    half = n // 2
    for _ in range(n_trials):
        fake = deltas * rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n, 1))
        order = rng.permutation(n)
        values.append(_cos(fake[order[:half]].mean(axis=0), fake[order[half:]].mean(axis=0)))
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


def _paired_deltas(
    *,
    row_index: dict[tuple[str, str], dict[str, Any]],
    feats: dict[str, np.ndarray],
    pos_condition: str,
    neg_condition: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    deltas: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for dilemma_id in sorted({d for d, _ in row_index}):
        pos = row_index.get((dilemma_id, pos_condition))
        neg = row_index.get((dilemma_id, neg_condition))
        if not pos or not neg:
            continue
        if pos["key"] not in feats or neg["key"] not in feats:
            continue
        deltas.append(feats[pos["key"]] - feats[neg["key"]])
        meta.append(
            {
                "dilemma_id": dilemma_id,
                "pos_tokens": pos["token_count"],
                "neg_tokens": neg["token_count"],
            }
        )
    if not deltas:
        return np.empty((0, 0), dtype=np.float32), meta
    return np.stack(deltas, axis=0), meta


def _analyze_site_layer(
    *,
    capture: CaptureArtifact,
    row_index: dict[tuple[str, str], dict[str, Any]],
    site: str,
    layer: int,
    n_trials: int,
) -> dict[str, Any]:
    feats = _capture_layer_features(capture, site=site, layer=layer)
    rows: list[dict[str, Any]] = []
    directions: dict[str, np.ndarray] = {}

    for theory in THEORIES:
        for construction, (pos, neg) in _contrast_pairs(theory).items():
            deltas, meta = _paired_deltas(
                row_index=row_index,
                feats=feats,
                pos_condition=pos,
                neg_condition=neg,
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
                    "status": "ok",
                }
            )

    cross_theory: dict[str, dict[str, dict[str, float]]] = {}
    for anchor in ("neutral_short", "neutral_length", "generic_moral", "anti"):
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

    generic_direction, _ = _paired_deltas(
        row_index=row_index,
        feats=feats,
        pos_condition=GENERIC_MORAL,
        neg_condition=NEUTRAL_SHORT,
    )
    generic_vec = generic_direction.mean(axis=0) if generic_direction.shape[0] >= 4 else np.empty((0,), dtype=np.float32)
    generic_alignment: dict[str, float] = {}
    residual_cross_theory: dict[str, dict[str, float]] = {}
    residual_dirs: dict[str, np.ndarray] = {}
    if generic_vec.size:
        for theory in THEORIES:
            short = THEORY_SHORT[theory]
            name = f"{short}_neutral_short"
            if name in directions:
                generic_alignment[short] = _cos(directions[name], generic_vec)
                residual_dirs[short] = _project_out(directions[name], generic_vec)
        for a, va in residual_dirs.items():
            residual_cross_theory[a] = {b: _cos(va, vb) for b, vb in residual_dirs.items()}

    return {
        "site": site,
        "layer": layer,
        "n_trials": n_trials,
        "rows": rows,
        "cross_theory_cosines": cross_theory,
        "generic_moral_direction": {
            "condition_a": GENERIC_MORAL,
            "condition_b": NEUTRAL_SHORT,
            "n_pairs": int(generic_direction.shape[0]),
            "norm": float(np.linalg.norm(generic_vec)) if generic_vec.size else float("nan"),
        },
        "cosine_to_generic_moral_minus_neutral": generic_alignment,
        "residual_after_projecting_generic_cross_theory": residual_cross_theory,
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

    primary = summary["primary_analysis"]
    lines: list[str] = []
    lines.append("# Natural-Prompt Paired Analysis")
    lines.append("")
    lines.append(f"- capture artifact: `{summary['capture_artifact_id']}`")
    lines.append(f"- generation rows: `{summary['generation_rows_path']}`")
    lines.append(f"- primary site/layer: `{PRIMARY_SITE}` / L{PRIMARY_LAYER}")
    lines.append("")
    overall = summary["token_distribution"]["overall"]
    lines.append("## Response Length")
    lines.append("")
    lines.append(
        f"- n={overall['n']}, mean tokens={_fmt(overall['mean'])}, median tokens={_fmt(overall['median'])}, "
        f"share_lt_10={_fmt(overall['share_lt_10'])}, share_lt_20={_fmt(overall['share_lt_20'])}"
    )
    lines.append("")
    lines.append("## Primary Paired Smoke")
    lines.append("")
    lines.append("| theory | construction | n | real_median | null_p95 | gap | pos_tok | neg_tok |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in primary["rows"]:
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
    lines.append("## Generic Moral Anchor")
    lines.append("")
    g = primary["generic_moral_direction"]
    lines.append(f"- `{g['condition_a']} - {g['condition_b']}`: n={g['n_pairs']}, norm={_fmt(g['norm'])}")
    lines.append("")
    lines.append("| theory | cos(theory-neutral, generic-neutral) |")
    lines.append("|---|---:|")
    for theory, value in sorted(primary["cosine_to_generic_moral_minus_neutral"].items()):
        lines.append(f"| {theory} | {_fmt(value)} |")
    lines.append("")
    lines.append("## Cross-Theory Cosines")
    for anchor, matrix in primary["cross_theory_cosines"].items():
        if not matrix:
            continue
        names = sorted(matrix)
        lines.append("")
        lines.append(f"### {anchor}")
        lines.append("")
        lines.append("| | " + " | ".join(names) + " |")
        lines.append("|---" + "|---:" * len(names) + "|")
        for a in names:
            lines.append(f"| {a} | " + " | ".join(_fmt(matrix[a].get(b)) for b in names) + " |")
    if primary["residual_after_projecting_generic_cross_theory"]:
        lines.append("")
        lines.append("## Cross-Theory Cosines After Projecting Generic Moral Direction")
        names = sorted(primary["residual_after_projecting_generic_cross_theory"])
        lines.append("")
        lines.append("| | " + " | ".join(names) + " |")
        lines.append("|---" + "|---:" * len(names) + "|")
        for a in names:
            row = primary["residual_after_projecting_generic_cross_theory"][a]
            lines.append(f"| {a} | " + " | ".join(_fmt(row.get(b)) for b in names) + " |")
    lines.append("")

    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--generation-rows", default=None)
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--trials", type=int, default=256)
    args = parser.parse_args()

    generation_rows = Path(args.generation_rows) if args.generation_rows else _latest_generation_rows_path(Path(args.report_root))
    rows = _rows(generation_rows)
    row_idx = _row_index(rows)
    capture = _load_capture(args.capture_id)
    primary = _analyze_site_layer(
        capture=capture,
        row_index=row_idx,
        site=PRIMARY_SITE,
        layer=PRIMARY_LAYER,
        n_trials=args.trials,
    )
    summary = {
        "capture_artifact_id": args.capture_id,
        "generation_rows_path": str(generation_rows),
        "primary_site": PRIMARY_SITE,
        "primary_layer": PRIMARY_LAYER,
        "n_trials": args.trials,
        "token_distribution": _token_distribution(row_idx),
        "primary_analysis": primary,
    }
    _write_report(summary, Path(args.report_dir))
    print(f"wrote {Path(args.report_dir) / 'report.md'}")


if __name__ == "__main__":
    main()
