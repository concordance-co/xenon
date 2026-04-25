#!/usr/bin/env python3
"""Preflight for dilemma-family keyword variant labels.

For each of the 4 labels, runs:
  - within-variant CV char-TFIDF (each variant's label ceiling on dilemma text)
  - cross-variant char-TFIDF transfer for all 6 ordered (variant_src, variant_dst) pairs
  - pairwise label agreement across variants (exact agreement rate on all 500 rows)

Exit gate per skill:
  PASS: cross-variant BA <= 0.65 for all pairs AND label agreement >= 0.70
  ITERATE: cross-variant BA > 0.65 (variants not disjoint enough)
  FAIL: cross-variant BA >= 0.75 (mark shortcut-dominated)

Writes:
  phase_02/reports/dilemma_family_keyword_variant_preflight/report.md
  phase_02/reports/dilemma_family_keyword_variant_preflight/summary.json
"""

from __future__ import annotations

import json
from collections import Counter
from itertools import product
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "projects/MOREBENCH/phase_02/specs/dilemma_family_keyword_variants.json"
LABELS = ROOT / "projects/MOREBENCH/phase_02/outputs/dilemma_family_keyword_variant_labels.jsonl"
GENERATION_RESULT = ROOT / "artifacts/_modal_cache/generation_run_1_d6e12a467208/result.json"
REPORT_DIR = ROOT / "projects/MOREBENCH/phase_02/reports/dilemma_family_keyword_variant_preflight"

RANDOM_STATE = 7
MIN_TRAIN_POSITIVES = 10
MIN_TEST_POSITIVES = 3


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(
            str(item.get("content", item)) if isinstance(item, dict) else str(item)
            for item in value
        )
    return str(value)


def load_dilemma_texts() -> dict[str, str]:
    generation = json.loads(GENERATION_RESULT.read_text())
    texts: dict[str, str] = {}
    for item in generation["rows"]:
        ex = item["example"]
        row_id = ex["key"]
        metadata = dict(ex.get("metadata", {}))
        dilemma_text = normalize_text(metadata.get("dilemma_text", ""))
        if not dilemma_text:
            prompt_text = normalize_text(ex.get("prompt", ""))
            dilemma_text = (
                prompt_text.split("DILEMMA:", 1)[1].strip()
                if "DILEMMA:" in prompt_text
                else prompt_text
            )
        texts[row_id] = dilemma_text
    return texts


def load_variant_rows() -> list[dict[str, Any]]:
    with LABELS.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def char_tfidf_pipeline() -> Any:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=12000)
    clf = LinearSVC(class_weight="balanced", dual="auto", random_state=RANDOM_STATE)
    return make_pipeline(vec, clf)


def decision_score(clf: Any, x: list[str]) -> np.ndarray:
    raw = clf.decision_function(x)
    if raw.ndim == 1:
        return raw
    return raw[:, 1] if raw.shape[1] == 2 else raw


def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(set(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return None


def within_variant_cv(
    texts_by_rowid: dict[str, str],
    rows: list[dict[str, Any]],
    label: str,
    variant: str,
    n_splits: int = 5,
) -> dict[str, Any]:
    y = np.array([1 if row[f"{label}__{variant}"] else 0 for row in rows])
    if sum(y) < MIN_TRAIN_POSITIVES or sum(1 - y) < MIN_TRAIN_POSITIVES:
        return {"status": "insufficient_support", "positives": int(sum(y)), "negatives": int(sum(1 - y))}
    texts = [texts_by_rowid[row["row_id"]] for row in rows]

    positives = int(sum(y))
    effective_splits = min(n_splits, positives, int(sum(1 - y)))
    if effective_splits < 2:
        return {"status": "insufficient_support", "positives": positives, "negatives": int(sum(1 - y))}
    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=RANDOM_STATE)

    preds = np.zeros_like(y)
    scores_all = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
        clf = char_tfidf_pipeline()
        clf.fit([texts[i] for i in train_idx], y[train_idx])
        preds[test_idx] = clf.predict([texts[i] for i in test_idx])
        scores_all[test_idx] = decision_score(clf, [texts[i] for i in test_idx])

    return {
        "status": "ok",
        "positives": int(sum(y)),
        "negatives": int(sum(1 - y)),
        "folds": effective_splits,
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "auroc": safe_auroc(y, scores_all),
    }


def cross_variant_transfer(
    texts_by_rowid: dict[str, str],
    rows: list[dict[str, Any]],
    label: str,
    src_variant: str,
    dst_variant: str,
) -> dict[str, Any]:
    """Train char-TFIDF using src_variant labels on all rows; test predictions against dst_variant labels."""
    y_src = np.array([1 if row[f"{label}__{src_variant}"] else 0 for row in rows])
    y_dst = np.array([1 if row[f"{label}__{dst_variant}"] else 0 for row in rows])
    texts = [texts_by_rowid[row["row_id"]] for row in rows]

    # Need enough src positives to train, and enough dst positives+negatives for eval.
    if sum(y_src) < MIN_TRAIN_POSITIVES or sum(1 - y_src) < MIN_TRAIN_POSITIVES:
        return {"status": "insufficient_src_support", "src_positives": int(sum(y_src))}
    if sum(y_dst) < MIN_TEST_POSITIVES or sum(1 - y_dst) < MIN_TEST_POSITIVES:
        return {"status": "insufficient_dst_support", "dst_positives": int(sum(y_dst))}

    # Stratified out-of-fold predictions on src label to avoid training on test rows.
    effective_splits = min(5, int(sum(y_src)), int(sum(1 - y_src)))
    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=RANDOM_STATE)
    preds_src = np.zeros_like(y_src)
    scores_src = np.zeros_like(y_src, dtype=float)
    for train_idx, test_idx in skf.split(np.zeros(len(y_src)), y_src):
        clf = char_tfidf_pipeline()
        clf.fit([texts[i] for i in train_idx], y_src[train_idx])
        preds_src[test_idx] = clf.predict([texts[i] for i in test_idx])
        scores_src[test_idx] = decision_score(clf, [texts[i] for i in test_idx])

    return {
        "status": "ok",
        "src_positives": int(sum(y_src)),
        "dst_positives": int(sum(y_dst)),
        "balanced_accuracy_vs_dst": float(balanced_accuracy_score(y_dst, preds_src)),
        "auroc_vs_dst": safe_auroc(y_dst, scores_src),
        "agreement_rate": float(np.mean(y_src == y_dst)),
    }


def label_agreement(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    variants = ["variant_a", "variant_b", "variant_c"]
    matrix: dict[str, dict[str, float]] = {}
    for a in variants:
        matrix[a] = {}
        for b in variants:
            a_vals = np.array([1 if row[f"{label}__{a}"] else 0 for row in rows])
            b_vals = np.array([1 if row[f"{label}__{b}"] else 0 for row in rows])
            matrix[a][b] = float(np.mean(a_vals == b_vals))
    a_counts = {v: int(sum(1 for row in rows if row[f"{label}__{v}"])) for v in variants}
    a_or = int(sum(1 for row in rows if any(row[f"{label}__{v}"] for v in variants)))
    a_and = int(sum(1 for row in rows if all(row[f"{label}__{v}"] for v in variants)))
    return {
        "counts_true": a_counts,
        "union_count": a_or,
        "intersection_count": a_and,
        "pairwise_agreement": matrix,
    }


def triage(cross_variant_results: dict[tuple[str, str], dict[str, Any]], agreement: dict[str, Any]) -> str:
    ordered_pair_bas: list[float] = []
    for (src, dst), result in cross_variant_results.items():
        if src == dst:
            continue
        ba = result.get("balanced_accuracy_vs_dst")
        if ba is not None:
            ordered_pair_bas.append(ba)
    if not ordered_pair_bas:
        return "insufficient_support"

    max_ba = max(ordered_pair_bas)
    mean_ba = mean(ordered_pair_bas)

    agreement_values: list[float] = []
    pairwise = agreement["pairwise_agreement"]
    for a, row in pairwise.items():
        for b, val in row.items():
            if a != b:
                agreement_values.append(val)
    min_agreement = min(agreement_values) if agreement_values else 0.0

    if max_ba >= 0.75:
        return "shortcut_dominated"
    if max_ba > 0.65:
        return "iterate_variant_design"
    if min_agreement < 0.70:
        return "variants_incoherent"
    return "pass"


def format_metric(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def write_report(records: dict[str, dict[str, Any]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORT_DIR / "summary.json").open("w") as f:
        json.dump(records, f, indent=2, sort_keys=True)

    lines = [
        "# Dilemma Family Keyword Variant Preflight",
        "",
        "Phase-02 shortcut-stress-test preflight for four dilemma-family labels.",
        "Variants are applied to dilemma text only. Char-TFIDF transfer is the exit gate.",
        "",
        "## Exit Gate Outcomes",
        "",
    ]
    for label, record in records.items():
        lines.append(f"- `{label}`: **{record['triage']}**")
    lines.extend(["", "## Per-Label Results", ""])

    for label, record in records.items():
        lines.append(f"### `{label}`")
        lines.append("")
        lines.append(
            f"construct: {record['construct']}"
        )
        lines.append("")
        counts = record["agreement"]["counts_true"]
        union = record["agreement"]["union_count"]
        inter = record["agreement"]["intersection_count"]
        lines.append(
            f"- True counts: variant_a={counts['variant_a']}, "
            f"variant_b={counts['variant_b']}, variant_c={counts['variant_c']} "
            f"(union={union}, intersection={inter})"
        )
        lines.append("")

        lines.append("**Within-variant CV (char-TFIDF on dilemma text):**")
        lines.append("")
        lines.append("| variant | status | pos | neg | BA | AUROC |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for variant in ["variant_a", "variant_b", "variant_c"]:
            entry = record["within_variant"].get(variant, {"status": "missing"})
            status = entry.get("status", "")
            pos = entry.get("positives", "")
            neg = entry.get("negatives", "")
            ba = format_metric(entry.get("balanced_accuracy"))
            auroc = format_metric(entry.get("auroc"))
            lines.append(f"| {variant} | {status} | {pos} | {neg} | {ba} | {auroc} |")
        lines.append("")

        lines.append("**Cross-variant char-TFIDF transfer (train on src label, predict dst label):**")
        lines.append("")
        lines.append("| src | dst | status | src_pos | dst_pos | BA (dst) | AUROC (dst) | agreement |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        for pair_key, entry in record["cross_variant"].items():
            src, dst = pair_key.split("->")
            if src == dst:
                continue
            status = entry.get("status", "")
            src_pos = entry.get("src_positives", "")
            dst_pos = entry.get("dst_positives", "")
            ba = format_metric(entry.get("balanced_accuracy_vs_dst"))
            auroc = format_metric(entry.get("auroc_vs_dst"))
            agree = format_metric(entry.get("agreement_rate"))
            lines.append(
                f"| {src} | {dst} | {status} | {src_pos} | {dst_pos} | {ba} | {auroc} | {agree} |"
            )
        lines.append("")

        lines.append("**Pairwise label agreement matrix:**")
        lines.append("")
        lines.append("| | variant_a | variant_b | variant_c |")
        lines.append("|---|---:|---:|---:|")
        for a in ["variant_a", "variant_b", "variant_c"]:
            row = record["agreement"]["pairwise_agreement"][a]
            lines.append(
                f"| {a} | {format_metric(row['variant_a'])} | {format_metric(row['variant_b'])} | {format_metric(row['variant_c'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Triage Legend",
            "",
            "- `pass`: cross-variant char-TFIDF BA <= 0.65 for every ordered pair AND pairwise agreement >= 0.70.",
            "- `iterate_variant_design`: cross-variant BA above 0.65 but below 0.75 — variants not disjoint enough.",
            "- `shortcut_dominated`: cross-variant BA >= 0.75 — char-TFIDF transfers; mark in known_bugs.",
            "- `variants_incoherent`: pairwise agreement < 0.70 — variants don't measure the same construct.",
            "- `insufficient_support`: class support too small to run transfer reliably.",
        ]
    )

    (REPORT_DIR / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    spec = json.loads(SPEC.read_text())
    labels_spec = spec["labels"]
    rows = load_variant_rows()
    texts_by_rowid = load_dilemma_texts()

    variants = ["variant_a", "variant_b", "variant_c"]
    records: dict[str, dict[str, Any]] = {}

    for label, block in labels_spec.items():
        print(f"[{label}] running preflight...")
        within = {}
        for variant in variants:
            within[variant] = within_variant_cv(texts_by_rowid, rows, label, variant)

        cross: dict[tuple[str, str], dict[str, Any]] = {}
        for src, dst in product(variants, repeat=2):
            cross[(src, dst)] = cross_variant_transfer(texts_by_rowid, rows, label, src, dst)

        agreement = label_agreement(rows, label)
        verdict = triage(cross, agreement)

        records[label] = {
            "construct": block.get("construct", ""),
            "within_variant": within,
            "cross_variant": {f"{src}->{dst}": v for (src, dst), v in cross.items()},
            "agreement": agreement,
            "triage": verdict,
        }
        print(f"[{label}] triage: {verdict}")

    write_report(records)
    print(f"wrote {REPORT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
