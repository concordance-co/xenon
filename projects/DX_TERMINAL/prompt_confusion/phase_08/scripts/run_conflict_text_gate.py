from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline


DEFAULT_INPUT = Path(
    "projects/DX_TERMINAL/prompt_confusion/phase_08/outputs/phase_08_dataset/phase_08_dataset.jsonl"
)
DEFAULT_OUTPUT = Path(
    "projects/DX_TERMINAL/prompt_confusion/phase_08/reports/conflict_text_gate.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 08 raw-text conflict gate.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-balanced-accuracy", type=float, default=0.55)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    if not rows:
        raise SystemExit(f"No rows loaded from {path}")
    return rows


def _fit_and_score(texts_train: list[str], y_train: np.ndarray, texts_test: list[str], y_test: np.ndarray) -> float:
    pipe = make_pipeline(
        CountVectorizer(ngram_range=(1, 2), min_df=1),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )
    pipe.fit(texts_train, y_train)
    pred = pipe.predict(texts_test)
    return float(balanced_accuracy_score(y_test, pred))


def _score_binary(rows: list[dict], label_key: str) -> dict[str, float | int]:
    train_rows = [row for row in rows if str(row["lexical_split"]) == "train"]
    test_rows = [row for row in rows if str(row["lexical_split"]) == "test"]
    texts_train = [str(row["user_text"]) for row in train_rows]
    texts_test = [str(row["user_text"]) for row in test_rows]
    y_train = np.asarray([1 if bool(row[label_key]) else 0 for row in train_rows], dtype=np.int64)
    y_test = np.asarray([1 if bool(row[label_key]) else 0 for row in test_rows], dtype=np.int64)
    score = _fit_and_score(texts_train, y_train, texts_test, y_test)
    return {
        "balanced_accuracy": round(score, 4),
        "chance_baseline": 0.5,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    canonical_rows = [row for row in rows if row["conflict_present"] is not None]
    if not canonical_rows:
        raise SystemExit("No canonical rows found with conflict_present != null.")

    pooled = _score_binary(canonical_rows, "conflict_present")
    by_dimension: dict[str, dict[str, float | int]] = {}
    for dim in sorted({str(row["target_dimension"]) for row in canonical_rows}):
        by_dimension[dim] = _score_binary([row for row in canonical_rows if str(row["target_dimension"]) == dim], "conflict_present")

    payload = {
        "input": str(args.input),
        "n_rows": len(rows),
        "n_canonical_rows": len(canonical_rows),
        "max_balanced_accuracy": args.max_balanced_accuracy,
        "pooled_conflict_present": {
            **pooled,
            "passes_gate": bool(float(pooled["balanced_accuracy"]) <= args.max_balanced_accuracy),
        },
        "per_dimension_conflict_present": {
            dim: {
                **metrics,
                "passes_gate": bool(float(metrics["balanced_accuracy"]) <= args.max_balanced_accuracy),
            }
            for dim, metrics in by_dimension.items()
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if float(pooled["balanced_accuracy"]) > args.max_balanced_accuracy:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
