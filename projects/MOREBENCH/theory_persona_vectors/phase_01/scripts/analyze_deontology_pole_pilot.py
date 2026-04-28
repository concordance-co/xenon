"""Analyze the deontology persona-vector pole pilot.

Inputs:
  - capture artifact id (positional argument or --capture-id flag)
  - generation artifact id (positional argument or --generation-id flag)
  - dilemmas + prompt-conditions specs from phase_01

Reports:
  - generation completion sanity
  - behavioral divergence diagnostic
  - direction stability (split-half within construction, cross-positive, cross-neutral)
  - pole-construction cosine matrix (deont_neutral_short / _length_matched / _anti / _util)
  - transfer to existing MoReBench prompt-final captures (if available)
  - existing-contested-capture diagnostic (if available)

Outputs:
  - projects/MOREBENCH/theory_persona_vectors/phase_01/reports/<run>/summary.json
  - projects/MOREBENCH/theory_persona_vectors/phase_01/reports/<run>/report.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

from pipelines_v2.api import ModalVolumeStore, PostgresCatalog, PostgresSource, TransferPolicy
from pipelines_v2.storage.artifacts import (
    CaptureArtifact,
    OperationArtifact,
    artifact_from_manifest,
)


PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_01")
DILEMMAS_PATH = PHASE_ROOT / "outputs" / "deontology_pole_pilot_synth_dilemmas.jsonl"
CONDITIONS_PATH = PHASE_ROOT / "specs" / "deontology_pole_pilot_prompt_conditions.json"

DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
ARTIFACT_STORE_NAME = "xenon-data"
PILOT_ROOT = "/data/artifacts/morebench_theory_persona_vectors_phase01"

CAPTURED_LAYERS = (0, 4, 16, 24, 32, 40)
PRIMARY_LAYER = 32
PRIMARY_SITE = "prompt_end_residual"
DIAGNOSTIC_SITE = "generated_sequence_residual"


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
    """Return mean-pooled per-example feature vectors for the given site/layer.

    For prompt_end (single-token section), pooling is identity.
    For generated_sequence_residual, mean-pools over the section tokens.
    """
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


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
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
    pos = np.stack([feats[k] for k in pos_keys if k in feats], axis=0)
    neg = np.stack([feats[k] for k in neg_keys if k in feats], axis=0)
    if pos.size == 0 or neg.size == 0:
        return np.array([])
    return pos.mean(axis=0) - neg.mean(axis=0)


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
    all_keys = pos_keys + neg_keys
    n_pos = len(pos_keys)
    cosines: list[float] = []
    for _ in range(n_trials):
        perm = list(all_keys)
        rng.shuffle(perm)
        fake_pos, fake_neg = perm[:n_pos], perm[n_pos:]
        d = _difference_in_means(fake_pos, fake_neg, feats)
        cosines.append(abs(_cos(d, real_direction)))
    return {
        "mean_abs_cos": float(np.mean(cosines)),
        "p95_abs_cos": float(np.percentile(cosines, 95)),
        "max_abs_cos": float(np.max(cosines)),
        "n_trials": n_trials,
    }


def _row_keys_by_condition(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    """Group example_keys by condition_id; also return labels-by-key."""
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
    # Crude normalization: collapse whitespace, strip punctuation tail.
    s = " ".join(s.split())
    return s[:max_chars]


def _behavioral_divergence(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-dilemma: do conditions converge on the same recommendation?

    Two heuristics:
      - exact-string convergence on normalized first 240 chars
      - char-3gram Jaccard >= 0.6 between condition recommendations
    """
    by_dilemma: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        ex = row.get("example") if isinstance(row.get("example"), Mapping) else {}
        labels = dict(ex.get("labels") or {})
        dilemma_id = str(labels.get("dilemma_id") or "")
        condition_id = str(labels.get("condition_id") or "")
        text = str(row.get("generated_text") or "")
        if not dilemma_id or not condition_id:
            continue
        if not text.strip():
            continue
        by_dilemma[dilemma_id][condition_id].append(_normalize_recommendation(text))

    def _ngrams(s: str, n: int = 3) -> set[str]:
        return {s[i : i + n] for i in range(max(0, len(s) - n + 1))}

    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    pair_diverged_share: dict[str, float] = {}
    pair_avg_jaccard: dict[str, float] = {}
    pair_definitions = {
        "P_deont_vs_N_neutral_short": ("P_deont_01", "N_neutral_01"),
        "P_deont_vs_N_neutral_length_matched": ("P_deont_01", "N_neutral_02"),
        "P_deont_vs_N_anti": ("P_deont_01", "N_anti_01"),
        "P_deont_vs_N_alt_util": ("P_deont_01", "N_alt_util_01"),
        "P_deont_01_vs_P_deont_02": ("P_deont_01", "P_deont_02"),
    }
    for pair_name, (left, right) in pair_definitions.items():
        diverged = 0
        total = 0
        jaccards: list[float] = []
        for dilemma_id, conds in by_dilemma.items():
            if left not in conds or right not in conds:
                continue
            total += 1
            l_text = conds[left][0] if conds[left] else ""
            r_text = conds[right][0] if conds[right] else ""
            j = _jaccard(_ngrams(l_text), _ngrams(r_text))
            jaccards.append(j)
            if j < 0.6 and l_text != r_text:
                diverged += 1
        pair_diverged_share[pair_name] = diverged / total if total else float("nan")
        pair_avg_jaccard[pair_name] = float(np.mean(jaccards)) if jaccards else float("nan")

    all_converge = 0
    all_diverge = 0
    cond_count_per_dilemma = []
    for dilemma_id, conds in by_dilemma.items():
        cond_count_per_dilemma.append(len(conds))
        first_texts = [c[0] for c in conds.values() if c]
        if not first_texts:
            continue
        ngrams = [_ngrams(t) for t in first_texts]
        # all converge if pairwise jaccard >= 0.7
        pairwise = [
            _jaccard(ngrams[i], ngrams[j])
            for i in range(len(ngrams))
            for j in range(i + 1, len(ngrams))
        ]
        if pairwise and min(pairwise) >= 0.7:
            all_converge += 1
        elif pairwise and max(pairwise) < 0.4:
            all_diverge += 1

    return {
        "n_dilemmas_evaluated": len(by_dilemma),
        "all_converge_count": all_converge,
        "all_diverge_count": all_diverge,
        "avg_conditions_per_dilemma": float(mean(cond_count_per_dilemma)) if cond_count_per_dilemma else 0.0,
        "pair_diverged_share": pair_diverged_share,
        "pair_avg_jaccard": pair_avg_jaccard,
    }


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


def _direction_analysis(
    capture: CaptureArtifact,
    keys_by_condition: dict[str, list[str]],
    *,
    site: str,
    layers: tuple[int, ...] = CAPTURED_LAYERS,
) -> dict[str, Any]:
    """Compute persona-vector style directions and stability metrics per layer."""
    pole_pairs = {
        "deont_neutral_short": ("P_deont_01", "N_neutral_01"),
        "deont_neutral_length_matched": ("P_deont_01", "N_neutral_02"),
        "deont_anti": ("P_deont_01", "N_anti_01"),
        "deont_util": ("P_deont_01", "N_alt_util_01"),
        "deont_neutral_short_p_variant": ("P_deont_02", "N_neutral_01"),
        "deont_neutral_length_matched_p_variant": ("P_deont_02", "N_neutral_02"),
    }

    out: dict[str, Any] = {"site": site, "layers": {}}
    for layer in layers:
        feats = _capture_layer_features(capture, layer=layer, site=site)
        layer_record: dict[str, Any] = {"directions": {}, "cosine_matrix": {}, "stability": {}}
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
            random_null = _random_label_cosine(pos_keys, neg_keys, feats, d, n_trials=64, seed=11 + layer)
            layer_record["stability"][name] = {
                "split_half_cosine": split_half,
                "random_label_null": random_null,
            }
        # cosine matrix between primary directions (not p_variant duplicates)
        primary_names = [n for n in directions if "p_variant" not in n]
        for a in primary_names:
            layer_record["cosine_matrix"][a] = {b: _cos(directions[a], directions[b]) for b in primary_names}
        # cross-positive cosine (P_deont_01 vs P_deont_02 against N_neutral_short and N_neutral_length_matched)
        cross_pos = {}
        if "deont_neutral_short" in directions and "deont_neutral_short_p_variant" in directions:
            cross_pos["neutral_short_p1_vs_p2"] = _cos(
                directions["deont_neutral_short"], directions["deont_neutral_short_p_variant"]
            )
        if "deont_neutral_length_matched" in directions and "deont_neutral_length_matched_p_variant" in directions:
            cross_pos["neutral_length_matched_p1_vs_p2"] = _cos(
                directions["deont_neutral_length_matched"], directions["deont_neutral_length_matched_p_variant"]
            )
        if cross_pos:
            layer_record["cross_positive_cosine"] = cross_pos
        # cross-neutral (length asymmetry control)
        if "deont_neutral_short" in directions and "deont_neutral_length_matched" in directions:
            layer_record["cross_neutral_cosine"] = _cos(
                directions["deont_neutral_short"], directions["deont_neutral_length_matched"]
            )
        out["layers"][str(layer)] = layer_record
    return out


def _format_metric(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.3f}"
    return str(v)


def _write_report(summary: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Deontology Persona-Vector Pole Pilot Report")
    lines.append("")
    lines.append(f"- generation artifact: `{summary.get('generation_artifact_id')}`")
    lines.append(f"- capture artifact: `{summary.get('capture_artifact_id')}`")
    lines.append(f"- captured layers: `{list(CAPTURED_LAYERS)}`")
    lines.append(f"- primary layer: `{PRIMARY_LAYER}` / primary site: `{PRIMARY_SITE}`")
    lines.append("")

    # Generation sanity
    gen = summary.get("generation_sanity") or {}
    lines.append("## Generation sanity")
    lines.append("")
    lines.append(f"- rows: `{gen.get('row_count')}`")
    lines.append(f"- nonempty: `{gen.get('nonempty_count')}` ({_format_metric(gen.get('nonempty_rate'))})")
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
                f"| {cond} | {stats.get('n')} | {_format_metric(stats.get('mean_chars'))} | "
                f"{_format_metric(stats.get('median_chars'))} | {stats.get('min_chars')} | {stats.get('max_chars')} |"
            )
        lines.append("")

    # Behavioral divergence
    div = summary.get("behavioral_divergence") or {}
    lines.append("## Behavioral divergence (diagnostic only)")
    lines.append("")
    lines.append(f"- dilemmas evaluated: `{div.get('n_dilemmas_evaluated')}`")
    lines.append(f"- all-converge dilemmas (pairwise jaccard ≥ 0.7): `{div.get('all_converge_count')}`")
    lines.append(f"- all-diverge dilemmas (max pairwise jaccard < 0.4): `{div.get('all_diverge_count')}`")
    lines.append(
        f"- avg conditions per dilemma: `{_format_metric(div.get('avg_conditions_per_dilemma'))}`"
    )
    lines.append("")
    pair_div = div.get("pair_diverged_share") or {}
    if pair_div:
        lines.append("Pair-level divergence share (jaccard < 0.6 over condition recommendations):")
        lines.append("")
        lines.append("| pair | diverged_share | avg_jaccard |")
        lines.append("|---|---:|---:|")
        for pair, share in pair_div.items():
            lines.append(
                f"| {pair} | {_format_metric(share)} | "
                f"{_format_metric(div.get('pair_avg_jaccard', {}).get(pair))} |"
            )
        lines.append("")

    # Directions per site
    for site_label, site_results in (summary.get("directions") or {}).items():
        lines.append(f"## Directions @ site `{site_label}`")
        lines.append("")
        for layer_str, rec in sorted(site_results.get("layers", {}).items(), key=lambda kv: int(kv[0])):
            lines.append(f"### Layer {layer_str}")
            lines.append("")
            stab = rec.get("stability") or {}
            if stab:
                lines.append("Direction stability:")
                lines.append("")
                lines.append("| construction | split_half_cos | null_p95 | null_max |")
                lines.append("|---|---:|---:|---:|")
                for name, srec in sorted(stab.items()):
                    null = srec.get("random_label_null") or {}
                    lines.append(
                        f"| {name} | {_format_metric(srec.get('split_half_cosine'))} | "
                        f"{_format_metric(null.get('p95_abs_cos'))} | "
                        f"{_format_metric(null.get('max_abs_cos'))} |"
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
                lines.append("Cross-positive (P_deont_01 vs P_deont_02) cosines:")
                for k, v in cp.items():
                    lines.append(f"- `{k}`: `{_format_metric(v)}`")
                lines.append("")
            cn = rec.get("cross_neutral_cosine")
            if cn is not None:
                lines.append(
                    f"Cross-neutral cosine (deont_neutral_short vs deont_neutral_length_matched): `{_format_metric(cn)}`"
                )
                lines.append("")

    transfer = summary.get("transfer_to_morebench")
    if transfer:
        lines.append("## Transfer to existing MoReBench captures (diagnostic)")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(transfer, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")

    contested = summary.get("contested_capture_diagnostic")
    if contested:
        lines.append("## Existing contested-capture diagnostic")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(contested, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")

    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-id", required=True, help="Capture artifact id (capture_1_...)")
    parser.add_argument("--generation-id", required=True, help="Generation artifact id (generation_run_1_...)")
    parser.add_argument(
        "--report-dir",
        default=str(PHASE_ROOT / "reports" / "deontology_pole_pilot_analysis"),
    )
    args = parser.parse_args()

    print(f"[load] generation artifact={args.generation_id}")
    rows = _generation_rows(args.generation_id)
    print(f"[load] generation rows={len(rows)}")

    keys_by_condition, labels_by_key = _row_keys_by_condition(rows)
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
    }

    print("[analyze] generation sanity")
    summary["generation_sanity"] = _generation_sanity(rows)

    print("[analyze] behavioral divergence")
    summary["behavioral_divergence"] = _behavioral_divergence(rows)

    print("[analyze] direction analysis @ prompt_end")
    primary_dir = _direction_analysis(capture, keys_by_condition, site=PRIMARY_SITE)
    print("[analyze] direction analysis @ generated mean")
    diag_dir = _direction_analysis(capture, keys_by_condition, site=DIAGNOSTIC_SITE)
    summary["directions"] = {
        PRIMARY_SITE: primary_dir,
        DIAGNOSTIC_SITE: diag_dir,
    }

    report_dir = Path(args.report_dir)
    _write_report(summary, report_dir)
    print(f"[done] wrote {report_dir / 'report.md'}")


if __name__ == "__main__":
    main()
