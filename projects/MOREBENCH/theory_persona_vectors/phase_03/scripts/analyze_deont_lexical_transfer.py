"""Deontology lexical-transfer diagnostic for phase 03 brief-recommendation captures.

This is a targeted confound-mitigation check inspired by transfer-style tests:
learn a direction from one lexical realization of deontology, then ask whether
it transfers to a different realization and survives cheap lexical suppression.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import analyze_generated_slices as slices
from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import analyze_natural_prompt_paired as paired
from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import plot_generated_first16_auroc_curve as auroc_curve


PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_03")
DEFAULT_REPORT_ROOT = PHASE_ROOT / "reports" / "all_theories_brief_recommendation_report"
DEFAULT_REPORT_DIR = PHASE_ROOT / "reports" / "deont_lexical_transfer"
DEFAULT_CAPTURE_ID = "capture_1_1d7271d73617"
DEFAULT_SITE = "generated_sequence_residual"
DEFAULT_SLICE = "first_16"
DEFAULT_LAYERS = (16, 32, 40)

PRIMARY_LEX_PATTERN = re.compile(
    r"\b(duty|duties|right|rights|promise|promises|obligation|obligations|constraint|constraints)\b",
    re.IGNORECASE,
)
VARIANT_LEX_PATTERN = re.compile(
    r"\b(commitment|commitments|boundary|boundaries|must not|mustn't|forbidden|off-limits|line(?:s)? not to cross)\b",
    re.IGNORECASE,
)

TRAIN_SPECS = (
    ("deont_primary", "P_deont_01", "N_neutral_01"),
    ("generic_moral", "N_generic_moral_01", "N_neutral_01"),
    ("neutral_length", "N_neutral_02", "N_neutral_01"),
)
EVAL_SPECS = (
    ("deont_primary", "P_deont_01", "N_neutral_01"),
    ("deont_variant", "P_deont_02", "N_neutral_01"),
    ("generic_moral", "N_generic_moral_01", "N_neutral_01"),
    ("neutral_length", "N_neutral_02", "N_neutral_01"),
    ("anti_deont", "N_anti_deont_01", "N_neutral_01"),
)


FilterFn = Callable[[dict[str, Any], dict[str, Any]], bool]


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


def _contains_pattern(text: str, pattern: re.Pattern[str], *, first_words: int | None = None) -> bool:
    if first_words is not None:
        words = re.findall(r"\S+", text)
        text = " ".join(words[:first_words])
    return bool(pattern.search(text))


def _passes_source_lex_absent_first_n(pos_row: dict[str, Any], neg_row: dict[str, Any], cutoff_words: int) -> bool:
    pos_text = str(pos_row.get("generated_text") or "")
    neg_text = str(neg_row.get("generated_text") or "")
    return not _contains_pattern(pos_text, PRIMARY_LEX_PATTERN, first_words=cutoff_words) and not _contains_pattern(
        neg_text,
        PRIMARY_LEX_PATTERN,
        first_words=cutoff_words,
    )


def _passes_source_lex_absent_full(pos_row: dict[str, Any], neg_row: dict[str, Any]) -> bool:
    pos_text = str(pos_row.get("generated_text") or "")
    neg_text = str(neg_row.get("generated_text") or "")
    return not _contains_pattern(pos_text, PRIMARY_LEX_PATTERN) and not _contains_pattern(neg_text, PRIMARY_LEX_PATTERN)


def _passes_variant_lex_present(pos_row: dict[str, Any], _neg_row: dict[str, Any]) -> bool:
    pos_text = str(pos_row.get("generated_text") or "")
    return _contains_pattern(pos_text, VARIANT_LEX_PATTERN)


def _filter_all(_pos_row: dict[str, Any], _neg_row: dict[str, Any]) -> bool:
    return True


def _filter_named(name: str, cutoff_words: int) -> FilterFn:
    if name == "all":
        return _filter_all
    if name == "source_lex_absent_first_n":
        return lambda pos, neg: _passes_source_lex_absent_first_n(pos, neg, cutoff_words)
    if name == "source_lex_absent_full":
        return _passes_source_lex_absent_full
    if name == "source_lex_absent_first_n_and_variant_present":
        return lambda pos, neg: _passes_source_lex_absent_first_n(pos, neg, cutoff_words) and _passes_variant_lex_present(pos, neg)
    raise ValueError(f"unknown filter {name!r}")


def _collect_pairs(
    *,
    dilemma_ids: Sequence[str],
    row_index: dict[tuple[str, str], dict[str, Any]],
    raw_rows: dict[tuple[str, str], dict[str, Any]],
    feats: dict[str, np.ndarray],
    pos_condition: str,
    neg_condition: str,
    pair_filter: FilterFn,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    deltas: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for dilemma_id in dilemma_ids:
        pos_meta = row_index.get((dilemma_id, pos_condition))
        neg_meta = row_index.get((dilemma_id, neg_condition))
        pos_raw = raw_rows.get((dilemma_id, pos_condition))
        neg_raw = raw_rows.get((dilemma_id, neg_condition))
        if not pos_meta or not neg_meta or not pos_raw or not neg_raw:
            continue
        if pos_meta["key"] not in feats or neg_meta["key"] not in feats:
            continue
        if not pair_filter(pos_raw, neg_raw):
            continue
        pos_vec = feats[pos_meta["key"]]
        neg_vec = feats[neg_meta["key"]]
        deltas.append(pos_vec - neg_vec)
        meta.append(
            {
                "dilemma_id": dilemma_id,
                "pos_key": pos_meta["key"],
                "neg_key": neg_meta["key"],
                "pos_text": str(pos_raw.get("generated_text") or ""),
                "neg_text": str(neg_raw.get("generated_text") or ""),
            }
        )
    return deltas, meta


def _projection_stats(scores: list[float], labels: list[int]) -> dict[str, float]:
    if not scores or len(scores) != len(labels):
        return {"auroc": float("nan"), "mean_margin": float("nan"), "n_examples": 0}
    score_arr = np.asarray(scores, dtype=np.float64)
    label_arr = np.asarray(labels, dtype=np.int32)
    pos = score_arr[label_arr == 1]
    neg = score_arr[label_arr == 0]
    return {
        "auroc": auroc_curve._auroc(score_arr, label_arr),
        "mean_margin": float(pos.mean() - neg.mean()) if pos.size and neg.size else float("nan"),
        "n_examples": int(score_arr.size),
    }


def _balanced_accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    preds = (scores >= 0.0).astype(np.int32)
    pos = labels == 1
    neg = labels == 0
    pos_acc = float(np.mean(preds[pos] == 1)) if np.any(pos) else float("nan")
    neg_acc = float(np.mean(preds[neg] == 0)) if np.any(neg) else float("nan")
    if math.isnan(pos_acc) or math.isnan(neg_acc):
        return float("nan")
    return 0.5 * (pos_acc + neg_acc)


def _loo_transfer(
    *,
    dilemma_ids: Sequence[str],
    row_index: dict[tuple[str, str], dict[str, Any]],
    raw_rows: dict[tuple[str, str], dict[str, Any]],
    feats: dict[str, np.ndarray],
    train_pos: str,
    train_neg: str,
    eval_pos: str,
    eval_neg: str,
    train_filter: FilterFn,
    eval_filter: FilterFn,
) -> dict[str, Any]:
    scores: list[float] = []
    labels: list[int] = []
    heldout_meta: list[dict[str, Any]] = []
    skipped_train = 0
    skipped_eval = 0
    for heldout in dilemma_ids:
        train_ids = [d for d in dilemma_ids if d != heldout]
        train_deltas, train_meta = _collect_pairs(
            dilemma_ids=train_ids,
            row_index=row_index,
            raw_rows=raw_rows,
            feats=feats,
            pos_condition=train_pos,
            neg_condition=train_neg,
            pair_filter=train_filter,
        )
        eval_deltas, eval_meta = _collect_pairs(
            dilemma_ids=[heldout],
            row_index=row_index,
            raw_rows=raw_rows,
            feats=feats,
            pos_condition=eval_pos,
            neg_condition=eval_neg,
            pair_filter=eval_filter,
        )
        if not train_deltas:
            skipped_train += 1
            continue
        if not eval_meta:
            skipped_eval += 1
            continue
        direction = np.stack(train_deltas, axis=0).mean(axis=0)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            skipped_train += 1
            continue
        unit = direction / norm
        pos_key = eval_meta[0]["pos_key"]
        neg_key = eval_meta[0]["neg_key"]
        pos_score = float(np.dot(feats[pos_key], unit))
        neg_score = float(np.dot(feats[neg_key], unit))
        scores.extend([pos_score, neg_score])
        labels.extend([1, 0])
        heldout_meta.append(
            {
                "dilemma_id": heldout,
                "train_pairs": len(train_meta),
                "pos_score": pos_score,
                "neg_score": neg_score,
                "margin": pos_score - neg_score,
            }
        )

    projection = _projection_stats(scores, labels)
    margins = [row["margin"] for row in heldout_meta]
    return {
        **projection,
        "n_pairs": len(heldout_meta),
        "skipped_train": skipped_train,
        "skipped_eval": skipped_eval,
        "median_margin": float(np.median(margins)) if margins else float("nan"),
        "heldout_rows": heldout_meta,
    }


def _pair_count(
    *,
    dilemma_ids: Sequence[str],
    row_index: dict[tuple[str, str], dict[str, Any]],
    raw_rows: dict[tuple[str, str], dict[str, Any]],
    feats: dict[str, np.ndarray],
    pos_condition: str,
    neg_condition: str,
    pair_filter: FilterFn,
) -> int:
    _, meta = _collect_pairs(
        dilemma_ids=dilemma_ids,
        row_index=row_index,
        raw_rows=raw_rows,
        feats=feats,
        pos_condition=pos_condition,
        neg_condition=neg_condition,
        pair_filter=pair_filter,
    )
    return len(meta)


def _char_ngrams(text: str, ngram_range: tuple[int, int]) -> Counter[str]:
    lowered = f" {text.lower()} "
    grams: Counter[str] = Counter()
    for n in range(ngram_range[0], ngram_range[1] + 1):
        if len(lowered) < n:
            continue
        for i in range(len(lowered) - n + 1):
            grams[lowered[i : i + n]] += 1
    return grams


def _build_char_vocab(texts: list[str], *, max_features: int, min_df: int, ngram_range: tuple[int, int]) -> list[str]:
    df: Counter[str] = Counter()
    tf: Counter[str] = Counter()
    for text in texts:
        grams = _char_ngrams(text, ngram_range)
        tf.update(grams)
        df.update(set(grams))
    grams = [gram for gram, count in df.items() if count >= min_df]
    grams.sort(key=lambda gram: (tf[gram], df[gram], gram), reverse=True)
    return grams[:max_features]


def _char_tfidf_matrix(texts: list[str], vocab: list[str], *, ngram_range: tuple[int, int]) -> np.ndarray:
    if not vocab:
        return np.zeros((len(texts), 0), dtype=np.float32)
    index = {term: i for i, term in enumerate(vocab)}
    df = np.zeros(len(vocab), dtype=np.float32)
    counts_by_doc: list[Counter[str]] = []
    for text in texts:
        counts = Counter({gram: count for gram, count in _char_ngrams(text, ngram_range).items() if gram in index})
        counts_by_doc.append(counts)
        for gram in counts:
            df[index[gram]] += 1
    n_docs = len(texts)
    idf = np.log((1 + n_docs) / (1 + df)) + 1.0
    x = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for row, counts in enumerate(counts_by_doc):
        total = sum(counts.values()) or 1
        for gram, count in counts.items():
            x[row, index[gram]] = (count / total) * idf[index[gram]]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return x / norms


def _ridge_binary_scores(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, *, alpha: float) -> np.ndarray:
    y_signed = np.where(y_train == 1, 1.0, -1.0).astype(np.float32)
    train = np.column_stack([np.ones(x_train.shape[0], dtype=np.float64), x_train.astype(np.float64, copy=False)])
    test = np.column_stack([np.ones(x_test.shape[0], dtype=np.float64), x_test.astype(np.float64, copy=False)])
    reg = np.eye(train.shape[1], dtype=np.float64) * alpha
    reg[0, 0] = 0.0
    gram = train.T @ train + reg
    rhs = train.T @ y_signed.astype(np.float64, copy=False)
    try:
        beta = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(gram + (1e-6 * np.eye(gram.shape[0], dtype=np.float64))) @ rhs
    return test @ beta


def _collect_text_examples(
    *,
    dilemma_ids: Sequence[str],
    raw_rows: dict[tuple[str, str], dict[str, Any]],
    pos_condition: str,
    neg_condition: str,
    pair_filter: FilterFn,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dilemma_id in dilemma_ids:
        pos_row = raw_rows.get((dilemma_id, pos_condition))
        neg_row = raw_rows.get((dilemma_id, neg_condition))
        if not pos_row or not neg_row:
            continue
        if not pair_filter(pos_row, neg_row):
            continue
        rows.append({"dilemma_id": dilemma_id, "label": 1, "text": str(pos_row.get("generated_text") or "")})
        rows.append({"dilemma_id": dilemma_id, "label": 0, "text": str(neg_row.get("generated_text") or "")})
    return rows


def _loo_text_classification(
    *,
    dilemma_ids: Sequence[str],
    raw_rows: dict[tuple[str, str], dict[str, Any]],
    pos_condition: str,
    neg_condition: str,
    pair_filter: FilterFn,
    max_features: int,
    min_df: int,
    alpha: float,
    ngram_range: tuple[int, int],
) -> dict[str, Any]:
    scores: list[float] = []
    labels: list[int] = []
    eval_dilemmas = 0
    skipped = 0
    for heldout in dilemma_ids:
        train_ids = [d for d in dilemma_ids if d != heldout]
        train_rows = _collect_text_examples(
            dilemma_ids=train_ids,
            raw_rows=raw_rows,
            pos_condition=pos_condition,
            neg_condition=neg_condition,
            pair_filter=pair_filter,
        )
        test_rows = _collect_text_examples(
            dilemma_ids=[heldout],
            raw_rows=raw_rows,
            pos_condition=pos_condition,
            neg_condition=neg_condition,
            pair_filter=pair_filter,
        )
        if len(train_rows) < 4 or len(test_rows) != 2:
            skipped += 1
            continue
        vocab = _build_char_vocab(
            [row["text"] for row in train_rows],
            max_features=max_features,
            min_df=min_df,
            ngram_range=ngram_range,
        )
        if not vocab:
            skipped += 1
            continue
        x_train = _char_tfidf_matrix([row["text"] for row in train_rows], vocab, ngram_range=ngram_range)
        x_test = _char_tfidf_matrix([row["text"] for row in test_rows], vocab, ngram_range=ngram_range)
        test_scores = _ridge_binary_scores(
            x_train,
            np.asarray([row["label"] for row in train_rows], dtype=np.int32),
            x_test,
            alpha=alpha,
        )
        scores.extend(test_scores.tolist())
        labels.extend([int(row["label"]) for row in test_rows])
        eval_dilemmas += 1
    if not scores:
        return {
            "auroc": float("nan"),
            "balanced_accuracy": float("nan"),
            "n_examples": 0,
            "n_pairs": 0,
            "skipped_eval": skipped,
        }
    score_arr = np.asarray(scores, dtype=np.float64)
    label_arr = np.asarray(labels, dtype=np.int32)
    return {
        "auroc": auroc_curve._auroc(score_arr, label_arr),
        "balanced_accuracy": _balanced_accuracy(score_arr, label_arr),
        "n_examples": int(score_arr.size),
        "n_pairs": eval_dilemmas,
        "skipped_eval": skipped,
    }


def _loo_text_transfer(
    *,
    dilemma_ids: Sequence[str],
    raw_rows: dict[tuple[str, str], dict[str, Any]],
    train_pos_condition: str,
    train_neg_condition: str,
    eval_pos_condition: str,
    eval_neg_condition: str,
    train_filter: FilterFn,
    eval_filter: FilterFn,
    max_features: int,
    min_df: int,
    alpha: float,
    ngram_range: tuple[int, int],
) -> dict[str, Any]:
    scores: list[float] = []
    labels: list[int] = []
    eval_dilemmas = 0
    skipped_train = 0
    skipped_eval = 0
    for heldout in dilemma_ids:
        train_ids = [d for d in dilemma_ids if d != heldout]
        train_rows = _collect_text_examples(
            dilemma_ids=train_ids,
            raw_rows=raw_rows,
            pos_condition=train_pos_condition,
            neg_condition=train_neg_condition,
            pair_filter=train_filter,
        )
        test_rows = _collect_text_examples(
            dilemma_ids=[heldout],
            raw_rows=raw_rows,
            pos_condition=eval_pos_condition,
            neg_condition=eval_neg_condition,
            pair_filter=eval_filter,
        )
        if len(train_rows) < 4:
            skipped_train += 1
            continue
        if len(test_rows) != 2:
            skipped_eval += 1
            continue
        vocab = _build_char_vocab(
            [row["text"] for row in train_rows],
            max_features=max_features,
            min_df=min_df,
            ngram_range=ngram_range,
        )
        if not vocab:
            skipped_train += 1
            continue
        x_train = _char_tfidf_matrix([row["text"] for row in train_rows], vocab, ngram_range=ngram_range)
        x_test = _char_tfidf_matrix([row["text"] for row in test_rows], vocab, ngram_range=ngram_range)
        test_scores = _ridge_binary_scores(
            x_train,
            np.asarray([row["label"] for row in train_rows], dtype=np.int32),
            x_test,
            alpha=alpha,
        )
        scores.extend(test_scores.tolist())
        labels.extend([int(row["label"]) for row in test_rows])
        eval_dilemmas += 1
    if not scores:
        return {
            "auroc": float("nan"),
            "balanced_accuracy": float("nan"),
            "n_examples": 0,
            "n_pairs": 0,
            "skipped_train": skipped_train,
            "skipped_eval": skipped_eval,
        }
    score_arr = np.asarray(scores, dtype=np.float64)
    label_arr = np.asarray(labels, dtype=np.int32)
    return {
        "auroc": auroc_curve._auroc(score_arr, label_arr),
        "balanced_accuracy": _balanced_accuracy(score_arr, label_arr),
        "n_examples": int(score_arr.size),
        "n_pairs": eval_dilemmas,
        "skipped_train": skipped_train,
        "skipped_eval": skipped_eval,
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.3f}"
    return str(value)


def _write_report(summary: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Deontology Lexical Transfer",
        "",
        f"- capture artifact: `{summary['capture_artifact_id']}`",
        f"- generation rows: `{summary['generation_rows_path']}`",
        f"- site: `{summary['site']}`",
        f"- slice: `{summary['slice_name']}`",
        f"- layers: `{', '.join(str(layer) for layer in summary['layers'])}`",
        f"- source lexical family: `{summary['source_lexical_family']}`",
        f"- variant lexical family: `{summary['variant_lexical_family']}`",
        "",
        "## Pair Availability",
        "",
        "| eval pair | filter | available pairs |",
        "|---|---|---:|",
    ]
    for row in summary["availability_rows"]:
        lines.append(f"| {row['eval_name']} | {row['filter_name']} | {row['n_pairs']} |")

    lines.extend(
        [
            "",
            "## Transfer Results",
            "",
            "| layer | train pair | train filter | eval pair | eval filter | eval pairs | LOO AUROC | mean margin | median margin |",
            "|---:|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["results"]:
        lines.append(
            f"| {row['layer']} | {row['train_name']} | {row['train_filter']} | {row['eval_name']} | "
            f"{row['eval_filter']} | {row['n_pairs']} | {_fmt(row['auroc'])} | "
            f"{_fmt(row['mean_margin'])} | {_fmt(row['median_margin'])} |"
        )

    lines.extend(
        [
            "",
            "## Text-Only Companion Check",
            "",
            f"- model: `char TF-IDF {summary['text_model']['ngram_range'][0]}-{summary['text_model']['ngram_range'][1]} + ridge classifier`",
            f"- max features: `{summary['text_model']['max_features']}`",
            f"- min df: `{summary['text_model']['min_df']}`",
            "",
            "| eval pair | filter | eval pairs | text BA | text AUROC |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in summary["text_results"]:
        lines.append(
            f"| {row['eval_name']} | {row['eval_filter']} | {row['n_pairs']} | "
            f"{_fmt(row['balanced_accuracy'])} | {_fmt(row['auroc'])} |"
        )
    lines.extend(
        [
            "",
            "## Direct `01 vs 02` Text Check",
            "",
            "| task | filter | eval pairs | text BA | text AUROC |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in summary["text_variant_vs_primary"]:
        lines.append(
            f"| {row['task']} | {row['eval_filter']} | {row['n_pairs']} | "
            f"{_fmt(row['balanced_accuracy'])} | {_fmt(row['auroc'])} |"
        )
    lines.extend(
        [
            "",
            "## Transfer-Style Text Baselines",
            "",
            "| train task | eval task | eval filter | eval pairs | text BA | text AUROC |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in summary["text_transfer_results"]:
        lines.append(
            f"| {row['train_task']} | {row['eval_task']} | {row['eval_filter']} | {row['n_pairs']} | "
            f"{_fmt(row['balanced_accuracy'])} | {_fmt(row['auroc'])} |"
        )

    lines.extend(
        [
            "",
            "## Readout Heuristic",
            "",
            "- Good evidence for this strategy is: `P_deont_01 -> P_deont_02` stays clearly above chance, survives the source-lex suppression filters, and remains better aligned than generic or neutral control directions.",
            "- Weak evidence is: transfer collapses to chance once source-family words are filtered, or generic-moral training transfers just as well as the deont-primary direction.",
            "- Diagnostic caution: this is still a transfer-style confound audit on existing generations, not a claim that the direction is a clean latent deontology feature.",
            "",
        ]
    )

    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-id", default=DEFAULT_CAPTURE_ID)
    parser.add_argument("--generation-rows", default=None)
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--site", default=DEFAULT_SITE)
    parser.add_argument("--slice-name", default=DEFAULT_SLICE)
    parser.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    parser.add_argument("--cutoff-words", type=int, default=16)
    parser.add_argument("--text-max-features", type=int, default=4000)
    parser.add_argument("--text-min-df", type=int, default=2)
    parser.add_argument("--text-alpha", type=float, default=1.0)
    args = parser.parse_args()

    generation_rows = Path(args.generation_rows) if args.generation_rows else paired._latest_generation_rows_path(Path(args.report_root))
    rows = paired._rows(generation_rows)
    row_index = paired._row_index(rows)
    raw_rows = _rows_by_pair(rows)
    dilemma_ids = sorted({d for d, _ in row_index})
    capture = paired._load_capture(args.capture_id)

    filter_names = (
        "all",
        "source_lex_absent_first_n",
        "source_lex_absent_full",
        "source_lex_absent_first_n_and_variant_present",
    )

    results: list[dict[str, Any]] = []
    availability: list[dict[str, Any]] = []
    text_results: list[dict[str, Any]] = []
    text_variant_vs_primary: list[dict[str, Any]] = []
    text_transfer_results: list[dict[str, Any]] = []
    for layer in args.layers:
        feats = slices._feature_slice_map(capture, site=args.site, layer=layer, slice_name=args.slice_name)

        if layer == args.layers[0]:
            for eval_name, eval_pos, eval_neg in EVAL_SPECS:
                for filter_name in filter_names:
                    availability.append(
                        {
                            "eval_name": eval_name,
                            "filter_name": filter_name,
                            "n_pairs": _pair_count(
                                dilemma_ids=dilemma_ids,
                                row_index=row_index,
                                raw_rows=raw_rows,
                                feats=feats,
                                pos_condition=eval_pos,
                                neg_condition=eval_neg,
                                pair_filter=_filter_named(filter_name, args.cutoff_words),
                            ),
                        }
                    )

            text_eval_rows = [
                ("deont_primary", "all"),
                ("deont_variant", "all"),
                ("deont_variant", "source_lex_absent_first_n"),
                ("deont_variant", "source_lex_absent_full"),
                ("deont_variant", "source_lex_absent_first_n_and_variant_present"),
            ]
            for eval_name, eval_filter_name in text_eval_rows:
                eval_pos, eval_neg = {name: (pos, neg) for name, pos, neg in EVAL_SPECS}[eval_name]
                text_results.append(
                    {
                        "eval_name": eval_name,
                        "eval_filter": eval_filter_name,
                        **_loo_text_classification(
                            dilemma_ids=dilemma_ids,
                            raw_rows=raw_rows,
                            pos_condition=eval_pos,
                            neg_condition=eval_neg,
                            pair_filter=_filter_named(eval_filter_name, args.cutoff_words),
                            max_features=args.text_max_features,
                            min_df=args.text_min_df,
                            alpha=args.text_alpha,
                            ngram_range=(3, 5),
                        ),
                    }
                )
            direct_text_rows = [
                ("primary_vs_variant", "all"),
                ("primary_vs_variant", "source_lex_absent_first_n"),
                ("primary_vs_variant", "source_lex_absent_full"),
                ("primary_vs_variant", "source_lex_absent_first_n_and_variant_present"),
            ]
            for task_name, eval_filter_name in direct_text_rows:
                text_variant_vs_primary.append(
                    {
                        "task": task_name,
                        "eval_filter": eval_filter_name,
                        **_loo_text_classification(
                            dilemma_ids=dilemma_ids,
                            raw_rows=raw_rows,
                            pos_condition="P_deont_01",
                            neg_condition="P_deont_02",
                            pair_filter=_filter_named(eval_filter_name, args.cutoff_words),
                            max_features=args.text_max_features,
                            min_df=args.text_min_df,
                            alpha=args.text_alpha,
                            ngram_range=(3, 5),
                        ),
                    }
                )
            transfer_rows = [
                ("deont01_vs_generic", "deont02_vs_generic", "all", "P_deont_01", "N_generic_moral_01", "P_deont_02", "N_generic_moral_01"),
                ("deont01_vs_generic", "deont02_vs_generic", "source_lex_absent_first_n", "P_deont_01", "N_generic_moral_01", "P_deont_02", "N_generic_moral_01"),
                ("deont01_vs_generic", "deont02_vs_generic", "source_lex_absent_first_n_and_variant_present", "P_deont_01", "N_generic_moral_01", "P_deont_02", "N_generic_moral_01"),
                ("deont01_vs_neutral", "deont02_vs_neutral", "all", "P_deont_01", "N_neutral_01", "P_deont_02", "N_neutral_01"),
                ("deont01_vs_neutral", "deont02_vs_neutral", "source_lex_absent_first_n", "P_deont_01", "N_neutral_01", "P_deont_02", "N_neutral_01"),
                ("deont01_vs_neutral", "deont02_vs_neutral", "source_lex_absent_first_n_and_variant_present", "P_deont_01", "N_neutral_01", "P_deont_02", "N_neutral_01"),
            ]
            for (
                train_task,
                eval_task,
                eval_filter_name,
                train_pos,
                train_neg,
                eval_pos,
                eval_neg,
            ) in transfer_rows:
                text_transfer_results.append(
                    {
                        "train_task": train_task,
                        "eval_task": eval_task,
                        "eval_filter": eval_filter_name,
                        **_loo_text_transfer(
                            dilemma_ids=dilemma_ids,
                            raw_rows=raw_rows,
                            train_pos_condition=train_pos,
                            train_neg_condition=train_neg,
                            eval_pos_condition=eval_pos,
                            eval_neg_condition=eval_neg,
                            train_filter=_filter_all,
                            eval_filter=_filter_named(eval_filter_name, args.cutoff_words),
                            max_features=args.text_max_features,
                            min_df=args.text_min_df,
                            alpha=args.text_alpha,
                            ngram_range=(3, 5),
                        ),
                    }
                )

        row_specs = [
            ("deont_primary", "all", "deont_primary", "all"),
            ("deont_primary", "all", "deont_variant", "all"),
            ("deont_primary", "all", "deont_variant", "source_lex_absent_first_n"),
            ("deont_primary", "all", "deont_variant", "source_lex_absent_full"),
            ("deont_primary", "all", "deont_variant", "source_lex_absent_first_n_and_variant_present"),
            ("deont_primary", "source_lex_absent_first_n", "deont_variant", "source_lex_absent_first_n"),
            ("deont_primary", "source_lex_absent_full", "deont_variant", "source_lex_absent_full"),
            ("deont_primary", "all", "generic_moral", "all"),
            ("deont_primary", "all", "neutral_length", "all"),
            ("deont_primary", "all", "anti_deont", "all"),
            ("generic_moral", "all", "deont_variant", "all"),
            ("neutral_length", "all", "deont_variant", "all"),
        ]
        train_lookup = {name: (pos, neg) for name, pos, neg in TRAIN_SPECS}
        eval_lookup = {name: (pos, neg) for name, pos, neg in EVAL_SPECS}

        for train_name, train_filter_name, eval_name, eval_filter_name in row_specs:
            train_pos, train_neg = train_lookup[train_name]
            eval_pos, eval_neg = eval_lookup[eval_name]
            transfer = _loo_transfer(
                dilemma_ids=dilemma_ids,
                row_index=row_index,
                raw_rows=raw_rows,
                feats=feats,
                train_pos=train_pos,
                train_neg=train_neg,
                eval_pos=eval_pos,
                eval_neg=eval_neg,
                train_filter=_filter_named(train_filter_name, args.cutoff_words),
                eval_filter=_filter_named(eval_filter_name, args.cutoff_words),
            )
            results.append(
                {
                    "layer": layer,
                    "train_name": train_name,
                    "train_filter": train_filter_name,
                    "eval_name": eval_name,
                    "eval_filter": eval_filter_name,
                    **transfer,
                }
            )

    summary = {
        "capture_artifact_id": args.capture_id,
        "generation_rows_path": str(generation_rows),
        "site": args.site,
        "slice_name": args.slice_name,
        "layers": args.layers,
        "cutoff_words": args.cutoff_words,
        "text_model": {
            "alpha": args.text_alpha,
            "max_features": args.text_max_features,
            "min_df": args.text_min_df,
            "ngram_range": [3, 5],
        },
        "source_lexical_family": PRIMARY_LEX_PATTERN.pattern,
        "variant_lexical_family": VARIANT_LEX_PATTERN.pattern,
        "availability_rows": availability,
        "text_results": text_results,
        "text_variant_vs_primary": text_variant_vs_primary,
        "text_transfer_results": text_transfer_results,
        "results": results,
    }
    _write_report(summary, Path(args.report_dir))
    print(f"wrote {Path(args.report_dir) / 'report.md'}")


if __name__ == "__main__":
    main()
