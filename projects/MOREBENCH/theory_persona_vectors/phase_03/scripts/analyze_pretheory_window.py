"""Pre-theory-token window check for phase 03 generated directions.

This tests whether L32 first-16 generated-token directions survive after
dropping same-dilemma pairs where theory-canonical vocabulary appears in the
first 16 whitespace words of either response. It is an intentionally cheap
analysis-side discriminator, not a perfect token-alignment claim.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import analyze_generated_slices as slices
from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import analyze_natural_prompt_paired as paired


PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_03")
DEFAULT_REPORT_ROOT = PHASE_ROOT / "reports" / "all_theories_brief_recommendation_report"
DEFAULT_REPORT_DIR = PHASE_ROOT / "reports" / "all_theories_brief_recommendation_pretheory_window"

THEORY_PATTERNS = {
    "deont": re.compile(
        r"\b(duty|duties|right|rights|promise|promises|obligation|obligations|constraint|constraints)\b",
        re.IGNORECASE,
    ),
    "util": re.compile(
        r"\b(welfare|maximize|maximizes|maximizing|benefit|benefits|outcome|outcomes|consequence|consequences|aggregate)\b",
        re.IGNORECASE,
    ),
    "virtue": re.compile(
        r"\b(virtue|virtues|character|honest|honesty|courage|courageous|prudent|prudence|integrity)\b",
        re.IGNORECASE,
    ),
    "contract": re.compile(
        r"\b(justified|justify|justifiable|reject|rejected|reasonable|reasonably|agree|agreement)\b",
        re.IGNORECASE,
    ),
}

CONDITIONS = {
    "deont": ("P_deont_01", "N_neutral_01"),
    "util": ("P_util_01", "N_neutral_01"),
    "virtue": ("P_virtue_01", "N_neutral_01"),
    "contract": ("P_contract_01", "N_neutral_01"),
}


def _word_index_of_match(text: str, pattern: re.Pattern[str]) -> int | None:
    match = pattern.search(text)
    if not match:
        return None
    prefix = text[: match.start()]
    return len(re.findall(r"\b\w+\b", prefix)) + 1


def _rows_by_pair(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        example = row.get("example") if isinstance(row.get("example"), Mapping) else {}
        labels = example.get("labels") if isinstance(example.get("labels"), Mapping) else {}
        dilemma = str(labels.get("dilemma_id") or "")
        condition = str(labels.get("condition_id") or "")
        if dilemma and condition:
            out[(dilemma, condition)] = row
    return out


def _passes_pretheory(row: dict[str, Any], pattern: re.Pattern[str], cutoff_words: int) -> bool:
    text = str(row.get("generated_text") or "")
    idx = _word_index_of_match(text, pattern)
    return idx is None or idx > cutoff_words


def _paired_filtered_direction(
    *,
    row_index: dict[tuple[str, str], dict[str, Any]],
    raw_rows: dict[tuple[str, str], dict[str, Any]],
    feats: dict[str, np.ndarray],
    pos_condition: str,
    neg_condition: str,
    pattern: re.Pattern[str],
    cutoff_words: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    deltas: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for dilemma_id in sorted({d for d, _ in row_index}):
        pos_meta = row_index.get((dilemma_id, pos_condition))
        neg_meta = row_index.get((dilemma_id, neg_condition))
        pos_raw = raw_rows.get((dilemma_id, pos_condition))
        neg_raw = raw_rows.get((dilemma_id, neg_condition))
        if not pos_meta or not neg_meta or not pos_raw or not neg_raw:
            continue
        if not _passes_pretheory(pos_raw, pattern, cutoff_words):
            continue
        if not _passes_pretheory(neg_raw, pattern, cutoff_words):
            continue
        if pos_meta["key"] not in feats or neg_meta["key"] not in feats:
            continue
        deltas.append(feats[pos_meta["key"]] - feats[neg_meta["key"]])
        meta.append(
            {
                "dilemma_id": dilemma_id,
                "pos_first_word": _word_index_of_match(str(pos_raw.get("generated_text") or ""), pattern),
                "neg_first_word": _word_index_of_match(str(neg_raw.get("generated_text") or ""), pattern),
            }
        )
    if not deltas:
        return np.empty((0,), dtype=np.float32), np.empty((0, 0), dtype=np.float32), meta
    stacked = np.stack(deltas, axis=0)
    return stacked.mean(axis=0), stacked, meta


def _paired_removed_direction(
    *,
    row_index: dict[tuple[str, str], dict[str, Any]],
    raw_rows: dict[tuple[str, str], dict[str, Any]],
    feats: dict[str, np.ndarray],
    pos_condition: str,
    neg_condition: str,
    pattern: re.Pattern[str],
    cutoff_words: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    deltas: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for dilemma_id in sorted({d for d, _ in row_index}):
        pos_meta = row_index.get((dilemma_id, pos_condition))
        neg_meta = row_index.get((dilemma_id, neg_condition))
        pos_raw = raw_rows.get((dilemma_id, pos_condition))
        neg_raw = raw_rows.get((dilemma_id, neg_condition))
        if not pos_meta or not neg_meta or not pos_raw or not neg_raw:
            continue
        pos_passes = _passes_pretheory(pos_raw, pattern, cutoff_words)
        neg_passes = _passes_pretheory(neg_raw, pattern, cutoff_words)
        if pos_passes and neg_passes:
            continue
        if pos_meta["key"] not in feats or neg_meta["key"] not in feats:
            continue
        deltas.append(feats[pos_meta["key"]] - feats[neg_meta["key"]])
        meta.append(
            {
                "dilemma_id": dilemma_id,
                "pos_first_word": _word_index_of_match(str(pos_raw.get("generated_text") or ""), pattern),
                "neg_first_word": _word_index_of_match(str(neg_raw.get("generated_text") or ""), pattern),
            }
        )
    if not deltas:
        return np.empty((0,), dtype=np.float32), np.empty((0, 0), dtype=np.float32), meta
    stacked = np.stack(deltas, axis=0)
    return stacked.mean(axis=0), stacked, meta


def _gap(deltas: np.ndarray, *, trials: int, layer: int) -> dict[str, float]:
    if deltas.shape[0] < 4:
        return {"real_median": float("nan"), "null_p95": float("nan"), "gap": float("nan")}
    real = paired._split_half_distribution(deltas, n_trials=trials, seed=1000 + layer)
    null = paired._sign_flip_null_distribution(deltas, n_trials=trials, seed=2000 + layer)
    return {
        "real_median": float(np.median(real)),
        "null_p95": float(np.percentile(null, 95)),
        "gap": float(np.median(real) - np.percentile(null, 95)),
    }


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.3f}"
    return str(v)


def _write_report(summary: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Pre-Theory-Token Window Check",
        "",
        f"- capture artifact: `{summary['capture_artifact_id']}`",
        f"- generation rows: `{summary['generation_rows_path']}`",
        f"- layer: L{summary['layer']}",
        f"- filter: drop pairs where matched theory-canonical vocabulary appears in first {summary['cutoff_words']} whitespace words of either response",
        "",
        "| theory | all n | removed n | removed rate | all gap | kept gap | removed gap | cos(kept, all) | cos(removed, all) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['theory']} | {row['all_n']} | {row['filter_removed_n']} | "
            f"{_fmt(row['filter_removed_rate'])} | {_fmt(row['all_gap'])} | "
            f"{_fmt(row['filtered_gap'])} | {_fmt(row['removed_gap'])} | "
            f"{_fmt(row['cos_filtered_to_all'])} | {_fmt(row['cos_removed_to_all'])} |"
        )
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--generation-rows", default=None)
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--cutoff-words", type=int, default=16)
    parser.add_argument("--trials", type=int, default=128)
    args = parser.parse_args()

    generation_rows = Path(args.generation_rows) if args.generation_rows else paired._latest_generation_rows_path(Path(args.report_root))
    rows = paired._rows(generation_rows)
    raw_rows = _rows_by_pair(rows)
    row_index = paired._row_index(rows)
    capture = paired._load_capture(args.capture_id)
    feats = slices._feature_slice_map(
        capture,
        site="generated_sequence_residual",
        layer=args.layer,
        slice_name="first_16",
    )

    out_rows: list[dict[str, Any]] = []
    for theory, (pos, neg) in CONDITIONS.items():
        pattern = THEORY_PATTERNS[theory]
        all_direction, all_deltas = slices._paired_direction(
            row_index=row_index,
            feats=feats,
            pos_condition=pos,
            neg_condition=neg,
        )
        filtered_direction, filtered_deltas, meta = _paired_filtered_direction(
            row_index=row_index,
            raw_rows=raw_rows,
            feats=feats,
            pos_condition=pos,
            neg_condition=neg,
            pattern=pattern,
            cutoff_words=args.cutoff_words,
        )
        removed_direction, removed_deltas, removed_meta = _paired_removed_direction(
            row_index=row_index,
            raw_rows=raw_rows,
            feats=feats,
            pos_condition=pos,
            neg_condition=neg,
            pattern=pattern,
            cutoff_words=args.cutoff_words,
        )
        all_gap = _gap(all_deltas, trials=args.trials, layer=args.layer)
        filtered_gap = _gap(filtered_deltas, trials=args.trials, layer=args.layer)
        removed_gap = _gap(removed_deltas, trials=args.trials, layer=args.layer)
        out_rows.append(
            {
                "theory": theory,
                "positive_condition": pos,
                "negative_condition": neg,
                "pattern": pattern.pattern,
                "all_n": int(all_deltas.shape[0]),
                "all_gap": all_gap["gap"],
                "filtered_n": int(filtered_deltas.shape[0]),
                "filter_removed_n": int(removed_deltas.shape[0]),
                "filter_removed_rate": float(removed_deltas.shape[0] / all_deltas.shape[0]) if all_deltas.shape[0] else float("nan"),
                "filtered_gap": filtered_gap["gap"],
                "filtered_real_median": filtered_gap["real_median"],
                "filtered_null_p95": filtered_gap["null_p95"],
                "removed_gap": removed_gap["gap"],
                "removed_real_median": removed_gap["real_median"],
                "removed_null_p95": removed_gap["null_p95"],
                "cos_filtered_to_all": _cos(filtered_direction, all_direction),
                "cos_removed_to_all": _cos(removed_direction, all_direction),
                "cos_filtered_to_removed": _cos(filtered_direction, removed_direction),
                "kept_dilemmas": meta,
                "removed_dilemmas": removed_meta,
            }
        )

    summary = {
        "capture_artifact_id": args.capture_id,
        "generation_rows_path": str(generation_rows),
        "layer": args.layer,
        "cutoff_words": args.cutoff_words,
        "trials": args.trials,
        "rows": out_rows,
    }
    _write_report(summary, Path(args.report_dir))
    print(f"wrote {Path(args.report_dir) / 'report.md'}")


if __name__ == "__main__":
    main()
