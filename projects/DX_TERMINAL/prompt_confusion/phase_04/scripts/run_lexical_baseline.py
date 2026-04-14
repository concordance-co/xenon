from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline


DEFAULT_INPUT = Path(
    "projects/DX_TERMINAL/prompt_confusion/phase_04/outputs/phase_04_dataset/phase_04_dataset.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lexical baseline on the Phase 04 dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--target", choices=["conflict_present"], default="conflict_present")
    parser.add_argument("--text-field", choices=["user_text", "system_plus_user"], default="user_text")
    parser.add_argument("--group-column", default="matched_pair_id")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def build_text(row: dict, text_field: str) -> str:
    if text_field == "system_plus_user":
        return f"{row['system_text']}\n\n{row['user_text']}"
    return str(row["user_text"])


def evaluate(
    texts: list[str],
    y: np.ndarray,
    *,
    n_folds: int,
    seed: int,
    groups: np.ndarray | None = None,
) -> dict[str, float | int | str | None]:
    pipe = make_pipeline(
        CountVectorizer(ngram_range=(1, 2), min_df=1),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    X = np.array(texts, dtype=object)

    split_mode = "stratified_kfold"
    split_kwargs: dict[str, object] = {}
    if groups is not None:
        unique_groups = len(set(groups.tolist()))
        actual_folds = max(2, min(n_folds, unique_groups))
        cv = StratifiedGroupKFold(n_splits=actual_folds, shuffle=True, random_state=seed)
        split_kwargs["groups"] = groups
        split_mode = "stratified_group_kfold"
        n_groups: int | None = unique_groups
    else:
        min_class_count = int(np.bincount(y).min())
        actual_folds = max(2, min(n_folds, len(y), min_class_count))
        cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=seed)
        n_groups = None

    accs: list[float] = []
    bals: list[float] = []
    for train_idx, test_idx in cv.split(X, y, **split_kwargs):
        pipe.fit(X[train_idx].tolist(), y[train_idx])
        pred = pipe.predict(X[test_idx].tolist())
        accs.append(float(accuracy_score(y[test_idx], pred)))
        bals.append(float(balanced_accuracy_score(y[test_idx], pred)))

    return {
        "split_mode": split_mode,
        "n_folds": actual_folds,
        "n_examples": int(len(y)),
        "n_groups": n_groups,
        "accuracy_mean": round(float(np.mean(accs)), 4),
        "accuracy_std": round(float(np.std(accs)), 4),
        "balanced_accuracy_mean": round(float(np.mean(bals)), 4),
        "balanced_accuracy_std": round(float(np.std(bals)), 4),
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    texts = [build_text(row, args.text_field) for row in rows]
    y = np.array([1 if bool(row[args.target]) else 0 for row in rows], dtype=np.int64)

    row_level = evaluate(texts, y, n_folds=args.n_folds, seed=args.seed, groups=None)
    grouped = evaluate(
        texts,
        y,
        n_folds=args.n_folds,
        seed=args.seed,
        groups=np.array([str(row[args.group_column]) for row in rows], dtype=object),
    )
    payload = {
        "input": str(args.input),
        "target": args.target,
        "text_field": args.text_field,
        "group_column": args.group_column,
        "row_level": row_level,
        "grouped": grouped,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
