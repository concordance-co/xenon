"""Full-run text leakage and transfer analysis for controlled deontology isolation."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import plot_generated_first16_auroc_curve as auroc_curve


PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_03")
DEFAULT_REPORT_ROOT = PHASE_ROOT / "reports" / "deont_prompt_isolation_report"
DEFAULT_OUTPUT_DIR = PHASE_ROOT / "reports" / "deont_prompt_isolation_text_analysis"

BANNED_PATTERN = re.compile(
    r"\b(duty|duties|right|rights|promise|promises|obligation|obligations|constraint|constraints|commitment|commitments|boundary|boundaries|forbidden)\b",
    re.IGNORECASE,
)

DIRECT_TASKS = (
    ("deont01_vs_deont02", "P_deont_iso_01", "P_deont_iso_02"),
    ("deont01_vs_generic", "P_deont_iso_01", "N_generic_moral_iso_01"),
    ("deont02_vs_generic", "P_deont_iso_02", "N_generic_moral_iso_01"),
    ("deont01_vs_neutral", "P_deont_iso_01", "N_neutral_iso_01"),
    ("deont02_vs_neutral", "P_deont_iso_02", "N_neutral_iso_01"),
)
TRANSFER_TASKS = (
    ("deont01_vs_generic", "deont02_vs_generic", "P_deont_iso_01", "N_generic_moral_iso_01", "P_deont_iso_02", "N_generic_moral_iso_01"),
    ("deont01_vs_neutral", "deont02_vs_neutral", "P_deont_iso_01", "N_neutral_iso_01", "P_deont_iso_02", "N_neutral_iso_01"),
)


def _latest_generation_rows_path(report_root: Path) -> Path:
    candidates = sorted(
        report_root.glob("report_*/results/generate_natural_responses_results.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no generate_natural_responses result found under {report_root}")
    return candidates[0]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise TypeError(f"{path} must contain a rows list")
    return rows


def _row_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for row in rows:
        example = row.get("example") or {}
        labels = example.get("labels") or {}
        dilemma_id = str(labels.get("dilemma_id") or "")
        condition_id = str(labels.get("condition_id") or "")
        if dilemma_id and condition_id:
            out[(dilemma_id, condition_id)] = str(row.get("generated_text") or "")
    return out


def _char_ngrams(text: str, ngram_range: tuple[int, int]) -> Counter[str]:
    lowered = f" {text.lower()} "
    grams: Counter[str] = Counter()
    for n in range(ngram_range[0], ngram_range[1] + 1):
        if len(lowered) < n:
            continue
        for i in range(len(lowered) - n + 1):
            grams[lowered[i : i + n]] += 1
    return grams


def _build_vocab(texts: list[str], *, max_features: int, min_df: int, ngram_range: tuple[int, int]) -> list[str]:
    df: Counter[str] = Counter()
    tf: Counter[str] = Counter()
    for text in texts:
        grams = _char_ngrams(text, ngram_range)
        tf.update(grams)
        df.update(set(grams))
    vocab = [gram for gram, count in df.items() if count >= min_df]
    vocab.sort(key=lambda gram: (tf[gram], df[gram], gram), reverse=True)
    return vocab[:max_features]


def _tfidf(texts: list[str], vocab: list[str], *, ngram_range: tuple[int, int]) -> np.ndarray:
    if not vocab:
        return np.zeros((len(texts), 0), dtype=np.float64)
    idx = {term: i for i, term in enumerate(vocab)}
    df = np.zeros(len(vocab), dtype=np.float64)
    counts_by_doc: list[Counter[str]] = []
    for text in texts:
        counts = Counter({gram: count for gram, count in _char_ngrams(text, ngram_range).items() if gram in idx})
        counts_by_doc.append(counts)
        for gram in counts:
            df[idx[gram]] += 1
    idf = np.log((1 + len(texts)) / (1 + df)) + 1.0
    x = np.zeros((len(texts), len(vocab)), dtype=np.float64)
    for row, counts in enumerate(counts_by_doc):
        total = sum(counts.values()) or 1
        for gram, count in counts.items():
            x[row, idx[gram]] = (count / total) * idf[idx[gram]]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return x / norms


def _ridge_scores(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, *, alpha: float) -> np.ndarray:
    y_signed = np.where(y_train == 1, 1.0, -1.0).astype(np.float64)
    train = np.column_stack([np.ones(x_train.shape[0], dtype=np.float64), x_train])
    test = np.column_stack([np.ones(x_test.shape[0], dtype=np.float64), x_test])
    reg = np.eye(train.shape[1], dtype=np.float64) * alpha
    reg[0, 0] = 0.0
    gram = train.T @ train + reg
    rhs = train.T @ y_signed
    try:
        beta = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(gram + 1e-6 * np.eye(gram.shape[0], dtype=np.float64)) @ rhs
    return test @ beta


def _balanced_accuracy(scores: Sequence[float], labels: Sequence[int]) -> float:
    score_arr = np.asarray(scores, dtype=np.float64)
    label_arr = np.asarray(labels, dtype=np.int32)
    preds = (score_arr >= 0.0).astype(np.int32)
    pos = label_arr == 1
    neg = label_arr == 0
    pos_acc = float(np.mean(preds[pos] == 1)) if np.any(pos) else float("nan")
    neg_acc = float(np.mean(preds[neg] == 0)) if np.any(neg) else float("nan")
    if math.isnan(pos_acc) or math.isnan(neg_acc):
        return float("nan")
    return 0.5 * (pos_acc + neg_acc)


def _collect_same_pair(index: dict[tuple[str, str], str], dilemmas: Sequence[str], pos: str, neg: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dilemma_id in dilemmas:
        pos_text = index.get((dilemma_id, pos))
        neg_text = index.get((dilemma_id, neg))
        if pos_text is None or neg_text is None:
            continue
        rows.append({"dilemma_id": dilemma_id, "label": 1, "text": pos_text})
        rows.append({"dilemma_id": dilemma_id, "label": 0, "text": neg_text})
    return rows


def _loo_same(index: dict[tuple[str, str], str], dilemmas: Sequence[str], pos: str, neg: str, *, max_features: int, min_df: int, alpha: float) -> dict[str, Any]:
    scores: list[float] = []
    labels: list[int] = []
    for heldout in dilemmas:
        train_rows = _collect_same_pair(index, [d for d in dilemmas if d != heldout], pos, neg)
        test_rows = _collect_same_pair(index, [heldout], pos, neg)
        if len(train_rows) < 4 or len(test_rows) != 2:
            continue
        vocab = _build_vocab([row["text"] for row in train_rows], max_features=max_features, min_df=min_df, ngram_range=(3, 5))
        if not vocab:
            continue
        x_train = _tfidf([row["text"] for row in train_rows], vocab, ngram_range=(3, 5))
        x_test = _tfidf([row["text"] for row in test_rows], vocab, ngram_range=(3, 5))
        test_scores = _ridge_scores(x_train, np.asarray([row["label"] for row in train_rows], dtype=np.int32), x_test, alpha=alpha)
        scores.extend(test_scores.tolist())
        labels.extend([int(row["label"]) for row in test_rows])
    if not scores:
        return {"n_pairs": 0, "auroc": float("nan"), "balanced_accuracy": float("nan")}
    return {
        "n_pairs": len(scores) // 2,
        "auroc": float(auroc_curve._auroc(np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int32))),
        "balanced_accuracy": _balanced_accuracy(scores, labels),
    }


def _loo_transfer(
    index: dict[tuple[str, str], str],
    dilemmas: Sequence[str],
    train_pos: str,
    train_neg: str,
    eval_pos: str,
    eval_neg: str,
    *,
    max_features: int,
    min_df: int,
    alpha: float,
) -> dict[str, Any]:
    scores: list[float] = []
    labels: list[int] = []
    for heldout in dilemmas:
        train_rows = _collect_same_pair(index, [d for d in dilemmas if d != heldout], train_pos, train_neg)
        test_rows = _collect_same_pair(index, [heldout], eval_pos, eval_neg)
        if len(train_rows) < 4 or len(test_rows) != 2:
            continue
        vocab = _build_vocab([row["text"] for row in train_rows], max_features=max_features, min_df=min_df, ngram_range=(3, 5))
        if not vocab:
            continue
        x_train = _tfidf([row["text"] for row in train_rows], vocab, ngram_range=(3, 5))
        x_test = _tfidf([row["text"] for row in test_rows], vocab, ngram_range=(3, 5))
        test_scores = _ridge_scores(x_train, np.asarray([row["label"] for row in train_rows], dtype=np.int32), x_test, alpha=alpha)
        scores.extend(test_scores.tolist())
        labels.extend([int(row["label"]) for row in test_rows])
    if not scores:
        return {"n_pairs": 0, "auroc": float("nan"), "balanced_accuracy": float("nan")}
    return {
        "n_pairs": len(scores) // 2,
        "auroc": float(auroc_curve._auroc(np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int32))),
        "balanced_accuracy": _balanced_accuracy(scores, labels),
    }


def _recommendation_line(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("Recommendation:"):
            return line
    return ""


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.3f}"
    return str(value)


def _write_report(summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Deont Prompt Isolation Text Analysis",
        "",
        f"- generation rows: `{summary['generation_rows_path']}`",
        f"- text model: `char TF-IDF 3-5 + ridge classifier`",
        "",
        "## Compliance",
        "",
        f"- total rows: `{summary['compliance']['total_rows']}`",
        f"- exact 3-line format: `{summary['compliance']['format_ok']}`",
        f"- banned-word leaks: `{summary['compliance']['banned_leaks']}`",
        "",
        "## Recommendation Overlap",
        "",
        "| pair | same recommendation count | total | rate |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["overlap_rows"]:
        lines.append(f"| {row['pair']} | {row['same_recommendation_count']} | {row['n_pairs']} | {_fmt(row['rate'])} |")
    lines.extend(
        [
            "",
            "## Direct Text Readouts",
            "",
            "| task | pairs | BA | AUROC |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summary["direct_rows"]:
        lines.append(f"| {row['task']} | {row['n_pairs']} | {_fmt(row['balanced_accuracy'])} | {_fmt(row['auroc'])} |")
    lines.extend(
        [
            "",
            "## Transfer-Style Text Readouts",
            "",
            "| train task | eval task | pairs | BA | AUROC |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in summary["transfer_rows"]:
        lines.append(
            f"| {row['train_task']} | {row['eval_task']} | {row['n_pairs']} | "
            f"{_fmt(row['balanced_accuracy'])} | {_fmt(row['auroc'])} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-rows", default=None)
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-features", type=int, default=4000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    generation_rows = Path(args.generation_rows) if args.generation_rows else _latest_generation_rows_path(Path(args.report_root))
    rows = _load_rows(generation_rows)
    index = _row_index(rows)
    dilemmas = sorted({d for d, _ in index})

    compliance = {
        "total_rows": len(rows),
        "format_ok": 0,
        "banned_leaks": 0,
    }
    for row in rows:
        txt = str(row.get("generated_text") or "")
        lines = [line for line in txt.strip().splitlines() if line.strip()]
        compliance["format_ok"] += int(
            len(lines) == 3
            and lines[0].startswith("Recommendation:")
            and lines[1].startswith("Reason:")
            and lines[2].startswith("Caveat:")
        )
        compliance["banned_leaks"] += int(bool(BANNED_PATTERN.search(txt)))

    overlap_rows: list[dict[str, Any]] = []
    for name, a, b in (
        ("deont01_vs_deont02", "P_deont_iso_01", "P_deont_iso_02"),
        ("deont01_vs_generic", "P_deont_iso_01", "N_generic_moral_iso_01"),
        ("deont02_vs_generic", "P_deont_iso_02", "N_generic_moral_iso_01"),
    ):
        same = 0
        n_pairs = 0
        for dilemma_id in dilemmas:
            a_txt = index.get((dilemma_id, a))
            b_txt = index.get((dilemma_id, b))
            if a_txt is None or b_txt is None:
                continue
            n_pairs += 1
            same += int(_recommendation_line(a_txt) == _recommendation_line(b_txt))
        overlap_rows.append(
            {
                "pair": name,
                "same_recommendation_count": same,
                "n_pairs": n_pairs,
                "rate": (same / n_pairs) if n_pairs else float("nan"),
            }
        )

    direct_rows = [
        {
            "task": task,
            **_loo_same(index, dilemmas, pos, neg, max_features=args.max_features, min_df=args.min_df, alpha=args.alpha),
        }
        for task, pos, neg in DIRECT_TASKS
    ]
    transfer_rows = [
        {
            "train_task": train_task,
            "eval_task": eval_task,
            **_loo_transfer(
                index,
                dilemmas,
                train_pos,
                train_neg,
                eval_pos,
                eval_neg,
                max_features=args.max_features,
                min_df=args.min_df,
                alpha=args.alpha,
            ),
        }
        for train_task, eval_task, train_pos, train_neg, eval_pos, eval_neg in TRANSFER_TASKS
    ]

    summary = {
        "generation_rows_path": str(generation_rows),
        "compliance": compliance,
        "overlap_rows": overlap_rows,
        "direct_rows": direct_rows,
        "transfer_rows": transfer_rows,
    }
    _write_report(summary, Path(args.output_dir))
    print(f"wrote {Path(args.output_dir) / 'report.md'}")


if __name__ == "__main__":
    main()
