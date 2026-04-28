"""Analyze the all-theories persona-vector pole pilot (phase 02).

Loops the phase_01 single-theory analysis over all four theories
(deontology, utilitarian, virtue_ethics, contractualism) on the same 30-dilemma
substrate. Produces a cross-theory smoke summary plus per-theory direction
tables.

Inputs:
  --capture-id       capture artifact id (capture_1_...)
  --generation-id    generation artifact id (generation_run_1_...)

Outputs:
  - projects/MOREBENCH/theory_persona_vectors/phase_02/reports/<run>/summary.json
  - projects/MOREBENCH/theory_persona_vectors/phase_02/reports/<run>/report.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from pipelines_v2.api import ModalVolumeStore, PostgresCatalog, PostgresSource, TransferPolicy
from pipelines_v2.storage.artifacts import (
    CaptureArtifact,
    OperationArtifact,
    artifact_from_manifest,
)


PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_02")
DILEMMAS_PATH = PHASE_ROOT / "outputs" / "all_theories_pole_pilot_synth_dilemmas.jsonl"
CONDITIONS_PATH = PHASE_ROOT / "specs" / "all_theories_pole_pilot_prompt_conditions.json"

DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
ARTIFACT_STORE_NAME = "xenon-data"
PILOT_ROOT = "/data/artifacts/morebench_theory_persona_vectors_phase02"

CAPTURED_LAYERS = (0, 4, 16, 24, 32, 40)
PRIMARY_LAYER = 32
# Persona-vectors method extracts from response-token activations
# (mean-pooled over generated tokens), not prompt-end. The pre-reg fixed
# prompt-end as primary because of prior MoReBench prompt-side results;
# that inheritance was the wrong choice for this method. See phase_02
# PHASE.md Corrections section.
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
NEUTRAL_LM = "N_neutral_02"

# Soft smoke gate: split_half - null_p95 >= 0.20 is "passes"; >= 0.10 is
# "marginal"; below 0.10 is "fail" (essentially indistinguishable from
# random label shuffles).
GAP_PASS = 0.20
GAP_MARGINAL = 0.10


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _catalog() -> PostgresCatalog:
    return PostgresCatalog(source=PostgresSource.from_env(DB_ENV_VAR))


def _store(root: str = PILOT_ROOT) -> ModalVolumeStore:
    return ModalVolumeStore(
        name=ARTIFACT_STORE_NAME,
        root=root,
        transfer_policy=TransferPolicy(allow_large_transfer=True),
    )


def _load_capture(artifact_id: str, *, root: str = PILOT_ROOT) -> CaptureArtifact:
    manifest = _catalog().load_artifact(artifact_id)
    if manifest is None:
        raise RuntimeError(f"could not load capture artifact {artifact_id!r}")
    artifact = artifact_from_manifest(manifest, store=_store(root))
    if not isinstance(artifact, CaptureArtifact):
        raise TypeError(f"artifact {artifact_id!r} is not a capture artifact")
    return artifact


def _load_operation(artifact_id: str, *, root: str = PILOT_ROOT) -> OperationArtifact:
    manifest = _catalog().load_artifact(artifact_id)
    if manifest is None:
        raise RuntimeError(f"could not load artifact {artifact_id!r}")
    artifact = artifact_from_manifest(manifest, store=_store(root))
    if not isinstance(artifact, OperationArtifact):
        raise TypeError(f"artifact {artifact_id!r} is not an operation artifact")
    return artifact


def _generation_rows(artifact_id: str) -> list[dict[str, Any]]:
    artifact = _load_operation(artifact_id)
    payload = artifact.result()
    rows = payload.get("rows") if isinstance(payload, Mapping) else []
    return [r for r in rows if isinstance(r, Mapping)]


_FEATURE_PAYLOAD_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def _feature_payload(capture: CaptureArtifact, site: str) -> dict[str, Any]:
    cache_key = (capture.id, site)
    cached = _FEATURE_PAYLOAD_CACHE.get(cache_key)
    if cached is not None:
        return cached
    payload = capture.feature(site).load()
    if not isinstance(payload, Mapping):
        raise TypeError(f"feature {site!r} payload is not a mapping")
    _FEATURE_PAYLOAD_CACHE[cache_key] = dict(payload)
    return _FEATURE_PAYLOAD_CACHE[cache_key]


def _capture_layer_features(capture: CaptureArtifact, *, layer: int, site: str) -> dict[str, np.ndarray]:
    payload = _feature_payload(capture, site)
    layer_key = str(layer)
    layer_payload = payload.get("layers", {}).get(layer_key)
    if not isinstance(layer_payload, Mapping):
        raise RuntimeError(f"site {site!r} missing layer {layer}")
    out: dict[str, np.ndarray] = {}
    for key, rec in layer_payload.items():
        values = rec.get("values") if isinstance(rec, Mapping) else None
        if values is None:
            continue
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 1:
            out[key] = arr
        elif arr.shape[0] == 0:
            continue
        else:
            out[key] = arr.mean(axis=0)
    return out


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return float("nan")
    if a.size == 0 or b.size == 0:
        return float("nan")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _difference_in_means(
    pos_keys: list[str],
    neg_keys: list[str],
    feats: dict[str, np.ndarray],
) -> np.ndarray:
    pos = [feats[k] for k in pos_keys if k in feats]
    neg = [feats[k] for k in neg_keys if k in feats]
    if not pos or not neg:
        return np.array([])
    return np.stack(pos, axis=0).mean(axis=0) - np.stack(neg, axis=0).mean(axis=0)


def _split_half_cosine(
    pos_keys: list[str],
    neg_keys: list[str],
    feats: dict[str, np.ndarray],
    *,
    seed: int = 7,
) -> float:
    rng = np.random.default_rng(seed)
    pos = list(pos_keys)
    neg = list(neg_keys)
    rng.shuffle(pos)
    rng.shuffle(neg)
    pos_a, pos_b = pos[: len(pos) // 2], pos[len(pos) // 2 :]
    neg_a, neg_b = neg[: len(neg) // 2], neg[len(neg) // 2 :]
    d_a = _difference_in_means(pos_a, neg_a, feats)
    d_b = _difference_in_means(pos_b, neg_b, feats)
    return _cos(d_a, d_b)


def _random_label_cosine(
    pos_keys: list[str],
    neg_keys: list[str],
    feats: dict[str, np.ndarray],
    real_direction: np.ndarray,
    *,
    n_trials: int = 64,
    seed: int = 11,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    all_keys = list(pos_keys) + list(neg_keys)
    n_pos = len(pos_keys)
    cosines: list[float] = []
    for _ in range(n_trials):
        perm = list(all_keys)
        rng.shuffle(perm)
        fake_pos, fake_neg = perm[:n_pos], perm[n_pos:]
        d = _difference_in_means(fake_pos, fake_neg, feats)
        cosines.append(abs(_cos(d, real_direction)))
    return {
        "mean_abs_cos": float(np.mean(cosines)) if cosines else float("nan"),
        "p95_abs_cos": float(np.percentile(cosines, 95)) if cosines else float("nan"),
        "max_abs_cos": float(np.max(cosines)) if cosines else float("nan"),
        "n_trials": n_trials,
    }


def _row_keys_by_condition(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    by_condition: dict[str, list[str]] = defaultdict(list)
    labels_by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        ex = row.get("example") if isinstance(row.get("example"), Mapping) else {}
        labels = dict(ex.get("labels") or {})
        key = str(row.get("example_key") or ex.get("key") or "")
        if not key:
            continue
        condition_id = str(labels.get("condition_id") or "")
        if not condition_id:
            continue
        by_condition[condition_id].append(key)
        labels_by_key[key] = labels
    return dict(by_condition), labels_by_key


def _normalize_recommendation(text: str, *, max_chars: int = 240) -> str:
    s = text.strip().lower()
    s = " ".join(s.split())
    return s[:max_chars]


def _ngrams(s: str, n: int = 3) -> set[str]:
    return {s[i : i + n] for i in range(max(0, len(s) - n + 1))}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _generation_sanity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition_lengths: dict[str, list[int]] = defaultdict(list)
    finish_reasons: Counter[str] = Counter()
    n_total = 0
    n_nonempty = 0
    for row in rows:
        ex = row.get("example") if isinstance(row.get("example"), Mapping) else {}
        labels = dict(ex.get("labels") or {})
        condition_id = str(labels.get("condition_id") or "")
        text = str(row.get("generated_text") or "")
        finish = str(row.get("finish_reason") or "")
        finish_reasons[finish] += 1
        n_total += 1
        if text.strip():
            n_nonempty += 1
        if condition_id:
            by_condition_lengths[condition_id].append(len(text))
    length_stats = {}
    for cond, lens in by_condition_lengths.items():
        length_stats[cond] = {
            "n": len(lens),
            "mean_chars": float(mean(lens)) if lens else 0.0,
            "median_chars": float(np.median(lens)) if lens else 0.0,
            "min_chars": int(min(lens)) if lens else 0,
            "max_chars": int(max(lens)) if lens else 0,
        }
    return {
        "row_count": n_total,
        "nonempty_count": n_nonempty,
        "nonempty_rate": n_nonempty / n_total if n_total else 0.0,
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "per_condition_response_length": length_stats,
    }


def _per_theory_pole_pairs(theory: str) -> dict[str, tuple[str, str]]:
    """Build the pole-pair set for one theory.

    Constructions:
      - {short}_neutral_short
      - {short}_neutral_length_matched
      - {short}_anti
      - {short}_alt_<other theory short> for each of the other 3 theories
      - p_variant duplicates of the four primary constructions
    """
    short = THEORY_SHORT[theory]
    p1 = POSITIVE_PRIMARY[theory]
    p2 = POSITIVE_VARIANT[theory]
    anti = ANTI_POLE[theory]
    pairs: dict[str, tuple[str, str]] = {
        f"{short}_neutral_short": (p1, NEUTRAL_SHORT),
        f"{short}_neutral_length_matched": (p1, NEUTRAL_LM),
        f"{short}_anti": (p1, anti),
        f"{short}_neutral_short_p_variant": (p2, NEUTRAL_SHORT),
        f"{short}_neutral_length_matched_p_variant": (p2, NEUTRAL_LM),
        f"{short}_anti_p_variant": (p2, anti),
    }
    for other in THEORIES:
        if other == theory:
            continue
        other_short = THEORY_SHORT[other]
        pairs[f"{short}_alt_{other_short}"] = (p1, POSITIVE_PRIMARY[other])
    return pairs


def _theory_direction_analysis(
    capture: CaptureArtifact,
    keys_by_condition: dict[str, list[str]],
    *,
    theory: str,
    site: str,
    layers: tuple[int, ...] = CAPTURED_LAYERS,
) -> dict[str, Any]:
    pole_pairs = _per_theory_pole_pairs(theory)
    out: dict[str, Any] = {"site": site, "theory": theory, "layers": {}}
    for layer in layers:
        try:
            feats = _capture_layer_features(capture, layer=layer, site=site)
        except RuntimeError:
            continue
        layer_record: dict[str, Any] = {
            "directions": {},
            "cosine_matrix": {},
            "stability": {},
        }
        directions: dict[str, np.ndarray] = {}
        for name, (pos, neg) in pole_pairs.items():
            pos_keys = keys_by_condition.get(pos, [])
            neg_keys = keys_by_condition.get(neg, [])
            d = _difference_in_means(pos_keys, neg_keys, feats)
            if d.size == 0:
                continue
            directions[name] = d
            layer_record["directions"][name] = {
                "pos_n": len(pos_keys),
                "neg_n": len(neg_keys),
                "norm": float(np.linalg.norm(d)),
            }
            split_half = _split_half_cosine(pos_keys, neg_keys, feats, seed=7 + layer)
            random_null = _random_label_cosine(
                pos_keys, neg_keys, feats, d, n_trials=64, seed=11 + layer
            )
            layer_record["stability"][name] = {
                "split_half_cosine": split_half,
                "random_label_null": random_null,
                "gap": (
                    split_half - random_null["p95_abs_cos"]
                    if not (math.isnan(split_half) or math.isnan(random_null["p95_abs_cos"]))
                    else float("nan")
                ),
            }
        primary_names = [n for n in directions if "p_variant" not in n]
        for a in primary_names:
            layer_record["cosine_matrix"][a] = {
                b: _cos(directions[a], directions[b]) for b in primary_names
            }
        # cross-positive cosine: P_01-derived vs P_02-derived for each negative anchor
        cross_pos = {}
        for anchor in ("neutral_short", "neutral_length_matched", "anti"):
            short = THEORY_SHORT[theory]
            a_name = f"{short}_{anchor}"
            b_name = f"{short}_{anchor}_p_variant"
            if a_name in directions and b_name in directions:
                cross_pos[anchor] = _cos(directions[a_name], directions[b_name])
        if cross_pos:
            layer_record["cross_positive_cosine"] = cross_pos
        out["layers"][str(layer)] = layer_record
    return out


def _cross_theory_cosine_matrix(
    capture: CaptureArtifact,
    keys_by_condition: dict[str, list[str]],
    *,
    site: str,
    layer: int,
    construction: str,
) -> dict[str, dict[str, float]]:
    """Cosine matrix of {theory}_{construction} directions across the 4 theories."""
    feats = _capture_layer_features(capture, layer=layer, site=site)
    directions: dict[str, np.ndarray] = {}
    for theory in THEORIES:
        short = THEORY_SHORT[theory]
        pairs = _per_theory_pole_pairs(theory)
        key = f"{short}_{construction}"
        if key not in pairs:
            continue
        pos, neg = pairs[key]
        pos_keys = keys_by_condition.get(pos, [])
        neg_keys = keys_by_condition.get(neg, [])
        d = _difference_in_means(pos_keys, neg_keys, feats)
        if d.size > 0:
            directions[short] = d
    matrix = {}
    for a, va in directions.items():
        matrix[a] = {b: _cos(va, vb) for b, vb in directions.items()}
    return matrix


def _behavioral_divergence_per_theory(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """For each theory, compute pair-divergence shares against neutral, anti, and other-theory positives."""
    by_dilemma_cond: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        ex = row.get("example") if isinstance(row.get("example"), Mapping) else {}
        labels = dict(ex.get("labels") or {})
        dilemma_id = str(labels.get("dilemma_id") or "")
        condition_id = str(labels.get("condition_id") or "")
        text = str(row.get("generated_text") or "")
        if not dilemma_id or not condition_id or not text.strip():
            continue
        by_dilemma_cond[dilemma_id][condition_id] = _normalize_recommendation(text)

    results: dict[str, dict[str, Any]] = {}
    for theory in THEORIES:
        short = THEORY_SHORT[theory]
        p1 = POSITIVE_PRIMARY[theory]
        anti = ANTI_POLE[theory]
        pair_defs = {
            f"{short}_vs_N_neutral_short": (p1, NEUTRAL_SHORT),
            f"{short}_vs_N_neutral_length_matched": (p1, NEUTRAL_LM),
            f"{short}_vs_N_anti": (p1, anti),
        }
        for other in THEORIES:
            if other == theory:
                continue
            other_short = THEORY_SHORT[other]
            pair_defs[f"{short}_vs_alt_{other_short}"] = (p1, POSITIVE_PRIMARY[other])
        pair_defs[f"{short}_p1_vs_p2"] = (p1, POSITIVE_VARIANT[theory])

        per_pair: dict[str, dict[str, float]] = {}
        for pair_name, (left, right) in pair_defs.items():
            diverged = 0
            total = 0
            jaccards: list[float] = []
            for did, conds in by_dilemma_cond.items():
                if left not in conds or right not in conds:
                    continue
                total += 1
                j = _jaccard(_ngrams(conds[left]), _ngrams(conds[right]))
                jaccards.append(j)
                if j < 0.6 and conds[left] != conds[right]:
                    diverged += 1
            per_pair[pair_name] = {
                "diverged_share": diverged / total if total else float("nan"),
                "avg_jaccard": float(mean(jaccards)) if jaccards else float("nan"),
                "n": total,
            }
        results[theory] = {"pairs": per_pair}
    return results


def _format_metric(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.3f}"
    return str(v)


def _gap_label(gap: float) -> str:
    if math.isnan(gap):
        return "nan"
    if gap >= GAP_PASS:
        return "pass"
    if gap >= GAP_MARGINAL:
        return "marginal"
    return "fail"


def _build_smoke_summary(
    summary: dict[str, Any],
    *,
    site: str,
    layer: int,
) -> list[dict[str, Any]]:
    rows = []
    per_theory = summary.get("per_theory") or {}
    for theory in THEORIES:
        short = THEORY_SHORT[theory]
        site_block = (per_theory.get(theory) or {}).get(site) or {}
        layer_block = (site_block.get("layers") or {}).get(str(layer)) or {}
        stab = layer_block.get("stability") or {}
        for construction in (
            f"{short}_neutral_short",
            f"{short}_neutral_length_matched",
            f"{short}_anti",
        ):
            srec = stab.get(construction) or {}
            null = srec.get("random_label_null") or {}
            split_half = srec.get("split_half_cosine")
            null_p95 = null.get("p95_abs_cos")
            gap = srec.get("gap")
            rows.append(
                {
                    "theory": theory,
                    "construction": construction,
                    "split_half_cos": split_half,
                    "null_p95": null_p95,
                    "gap": gap,
                    "verdict": _gap_label(gap if isinstance(gap, float) else float("nan")),
                }
            )
        for other in THEORIES:
            if other == theory:
                continue
            other_short = THEORY_SHORT[other]
            cname = f"{short}_alt_{other_short}"
            srec = stab.get(cname) or {}
            null = srec.get("random_label_null") or {}
            split_half = srec.get("split_half_cosine")
            null_p95 = null.get("p95_abs_cos")
            gap = srec.get("gap")
            rows.append(
                {
                    "theory": theory,
                    "construction": cname,
                    "split_half_cos": split_half,
                    "null_p95": null_p95,
                    "gap": gap,
                    "verdict": _gap_label(gap if isinstance(gap, float) else float("nan")),
                }
            )
    return rows


def _write_report(summary: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    lines: list[str] = []
    lines.append("# All-Theories Persona-Vector Pole Pilot Report (Phase 02)")
    lines.append("")
    lines.append(f"- generation artifact: `{summary.get('generation_artifact_id')}`")
    lines.append(f"- capture artifact: `{summary.get('capture_artifact_id')}`")
    lines.append(f"- captured layers: `{list(CAPTURED_LAYERS)}`")
    lines.append(f"- primary layer: `{PRIMARY_LAYER}` / primary site: `{PRIMARY_SITE}`")
    lines.append(f"- theories: `{list(THEORIES)}`")
    lines.append("")

    # Headline smoke at primary locus
    lines.append(f"## Headline smoke @ L{PRIMARY_LAYER} {PRIMARY_SITE}")
    lines.append("")
    lines.append(
        f"Pass criterion (soft): split_half − null_p95 ≥ {GAP_PASS}; "
        f"marginal in [{GAP_MARGINAL}, {GAP_PASS}); fail < {GAP_MARGINAL}."
    )
    lines.append("")
    smoke = summary.get("smoke_table_primary") or []
    if smoke:
        lines.append("| theory | construction | split_half | null_p95 | gap | verdict |")
        lines.append("|---|---|---:|---:|---:|---|")
        for row in smoke:
            lines.append(
                f"| {row['theory']} | {row['construction']} | "
                f"{_format_metric(row['split_half_cos'])} | "
                f"{_format_metric(row['null_p95'])} | "
                f"{_format_metric(row['gap'])} | "
                f"{row['verdict']} |"
            )
        lines.append("")

    # Cross-theory cosine matrix at primary locus
    cross = summary.get("cross_theory_cosines_primary") or {}
    if cross:
        lines.append(f"## Cross-theory direction cosines @ L{PRIMARY_LAYER} {PRIMARY_SITE}")
        lines.append("")
        for construction, matrix in cross.items():
            if not matrix:
                continue
            lines.append(f"### {construction}")
            lines.append("")
            theories_present = sorted(matrix.keys())
            lines.append("| | " + " | ".join(theories_present) + " |")
            lines.append("|---" + "|---:" * len(theories_present) + "|")
            for a in theories_present:
                row = [_format_metric(matrix[a].get(b)) for b in theories_present]
                lines.append(f"| {a} | " + " | ".join(row) + " |")
            lines.append("")

    # Generation sanity
    gen = summary.get("generation_sanity") or {}
    lines.append("## Generation sanity")
    lines.append("")
    lines.append(f"- rows: `{gen.get('row_count')}`")
    lines.append(
        f"- nonempty: `{gen.get('nonempty_count')}` ({_format_metric(gen.get('nonempty_rate'))})"
    )
    lines.append(f"- finish reasons: `{gen.get('finish_reason_counts')}`")
    lines.append("")
    plen = gen.get("per_condition_response_length") or {}
    if plen:
        lines.append("Response length by condition:")
        lines.append("")
        lines.append("| condition | n | mean_chars | median_chars | min | max |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for cond, stats in sorted(plen.items()):
            lines.append(
                f"| {cond} | {stats.get('n')} | "
                f"{_format_metric(stats.get('mean_chars'))} | "
                f"{_format_metric(stats.get('median_chars'))} | "
                f"{stats.get('min_chars')} | {stats.get('max_chars')} |"
            )
        lines.append("")

    # Behavioral divergence per theory
    div = summary.get("behavioral_divergence_per_theory") or {}
    if div:
        lines.append("## Behavioral divergence per theory (diagnostic)")
        lines.append("")
        for theory in THEORIES:
            block = div.get(theory) or {}
            pairs = block.get("pairs") or {}
            if not pairs:
                continue
            lines.append(f"### {theory}")
            lines.append("")
            lines.append("| pair | n | diverged_share | avg_jaccard |")
            lines.append("|---|---:|---:|---:|")
            for pair_name, stats in sorted(pairs.items()):
                lines.append(
                    f"| {pair_name} | {stats.get('n')} | "
                    f"{_format_metric(stats.get('diverged_share'))} | "
                    f"{_format_metric(stats.get('avg_jaccard'))} |"
                )
            lines.append("")

    # Per-theory direction tables
    per_theory = summary.get("per_theory") or {}
    for theory in THEORIES:
        block = per_theory.get(theory) or {}
        if not block:
            continue
        lines.append(f"## Per-theory directions: {theory}")
        lines.append("")
        for site_label in (PRIMARY_SITE, DIAGNOSTIC_SITE):
            site_block = block.get(site_label) or {}
            layers_block = site_block.get("layers") or {}
            if not layers_block:
                continue
            lines.append(f"### Site `{site_label}`")
            lines.append("")
            for layer_str, rec in sorted(layers_block.items(), key=lambda kv: int(kv[0])):
                lines.append(f"#### Layer {layer_str}")
                lines.append("")
                stab = rec.get("stability") or {}
                if stab:
                    lines.append("Direction stability:")
                    lines.append("")
                    lines.append("| construction | split_half | null_p95 | null_max | gap |")
                    lines.append("|---|---:|---:|---:|---:|")
                    for name in sorted(stab):
                        srec = stab[name]
                        null = srec.get("random_label_null") or {}
                        lines.append(
                            f"| {name} | {_format_metric(srec.get('split_half_cosine'))} | "
                            f"{_format_metric(null.get('p95_abs_cos'))} | "
                            f"{_format_metric(null.get('max_abs_cos'))} | "
                            f"{_format_metric(srec.get('gap'))} |"
                        )
                    lines.append("")
                cm = rec.get("cosine_matrix") or {}
                if cm:
                    lines.append("Pole-construction cosine matrix:")
                    lines.append("")
                    names = sorted(cm.keys())
                    lines.append("| | " + " | ".join(names) + " |")
                    lines.append("|---" + "|---:" * len(names) + "|")
                    for a in names:
                        row = [_format_metric(cm[a].get(b)) for b in names]
                        lines.append(f"| {a} | " + " | ".join(row) + " |")
                    lines.append("")
                cp = rec.get("cross_positive_cosine") or {}
                if cp:
                    lines.append("Cross-positive (P_01 vs P_02) cosines per anchor:")
                    for k, v in cp.items():
                        lines.append(f"- `{k}`: `{_format_metric(v)}`")
                    lines.append("")

    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument(
        "--report-dir",
        default=str(PHASE_ROOT / "reports" / "all_theories_pole_pilot_analysis"),
    )
    args = parser.parse_args()

    print(f"[load] generation artifact={args.generation_id}")
    rows = _generation_rows(args.generation_id)
    print(f"[load] generation rows={len(rows)}")

    keys_by_condition, _ = _row_keys_by_condition(rows)
    print(f"[load] conditions: {sorted(keys_by_condition)}")
    for cond, keys in sorted(keys_by_condition.items()):
        print(f"[load]   {cond}: {len(keys)} rows")

    print(f"[load] capture artifact={args.capture_id}")
    capture = _load_capture(args.capture_id)

    summary: dict[str, Any] = {
        "generation_artifact_id": args.generation_id,
        "capture_artifact_id": args.capture_id,
        "primary_site": PRIMARY_SITE,
        "primary_layer": PRIMARY_LAYER,
        "captured_layers": list(CAPTURED_LAYERS),
        "theories": list(THEORIES),
    }

    print("[analyze] generation sanity")
    summary["generation_sanity"] = _generation_sanity(rows)
    print("[analyze] behavioral divergence per theory")
    summary["behavioral_divergence_per_theory"] = _behavioral_divergence_per_theory(rows)

    per_theory: dict[str, Any] = {}
    for theory in THEORIES:
        print(f"[analyze] theory={theory} site={PRIMARY_SITE}")
        prim = _theory_direction_analysis(
            capture, keys_by_condition, theory=theory, site=PRIMARY_SITE
        )
        print(f"[analyze] theory={theory} site={DIAGNOSTIC_SITE}")
        diag = _theory_direction_analysis(
            capture, keys_by_condition, theory=theory, site=DIAGNOSTIC_SITE
        )
        per_theory[theory] = {PRIMARY_SITE: prim, DIAGNOSTIC_SITE: diag}
    summary["per_theory"] = per_theory

    print(f"[analyze] cross-theory cosines @ L{PRIMARY_LAYER} {PRIMARY_SITE}")
    cross_theory: dict[str, Any] = {}
    for construction in ("neutral_short", "neutral_length_matched", "anti"):
        cross_theory[construction] = _cross_theory_cosine_matrix(
            capture, keys_by_condition, site=PRIMARY_SITE, layer=PRIMARY_LAYER, construction=construction
        )
    summary["cross_theory_cosines_primary"] = cross_theory

    print(f"[analyze] smoke table @ L{PRIMARY_LAYER} {PRIMARY_SITE}")
    summary["smoke_table_primary"] = _build_smoke_summary(
        summary, site=PRIMARY_SITE, layer=PRIMARY_LAYER
    )

    report_dir = Path(args.report_dir)
    _write_report(summary, report_dir)
    print(f"[done] wrote {report_dir / 'report.md'}")


if __name__ == "__main__":
    main()
