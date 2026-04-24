from __future__ import annotations

import csv
import io
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path


ROOT = Path("/Users/trentelmore/Projects/concordance/xenon/projects/MOREBENCH")
OUTPUT_DIR = ROOT / "phase_03" / "outputs"

PUBLIC_URL = "https://huggingface.co/datasets/morebench/morebench/resolve/main/morebench_public.csv"
THEORY_URL = "https://huggingface.co/datasets/morebench/morebench/resolve/main/morebench_theory.csv"

SELECTED_POOL_INDICES = [
    81,
    113,
    311,
    241,
    424,
    167,
    313,
    396,
    371,
    378,
    159,
    149,
    279,
    198,
    94,
    362,
    286,
    27,
    444,
    217,
    128,
    426,
    28,
    335,
    192,
    41,
    417,
    394,
    36,
    120,
    303,
    127,
    374,
    124,
    227,
    115,
    229,
    471,
    215,
    203,
    478,
    405,
    415,
    89,
    221,
    386,
    206,
    341,
    17,
    52,
    22,
    339,
    293,
    256,
    175,
    384,
    20,
    291,
    211,
    213,
]


def fetch_rows(url: str) -> list[dict[str, str]]:
    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    public_rows = fetch_rows(PUBLIC_URL)
    theory_rows = fetch_rows(THEORY_URL)

    theory_dilemmas = {normalize_text(row["DILEMMA"]) for row in theory_rows}
    non_overlap_rows = [
        row for row in public_rows if normalize_text(row["DILEMMA"]) not in theory_dilemmas
    ]

    selected_rows: list[dict[str, object]] = []
    for selected_rank, pool_index in enumerate(SELECTED_POOL_INDICES, start=1):
        row = dict(non_overlap_rows[pool_index])
        row["extension_split"] = "public_conflict_extension"
        row["selection_protocol"] = "manual_structural_screen_v1"
        row["selection_rank"] = selected_rank
        row["pool_index"] = pool_index
        row["normalized_dilemma"] = normalize_text(row["DILEMMA"])
        selected_rows.append(row)

    source_counts = Counter(row["DILEMMA_SOURCE"] for row in selected_rows)
    role_counts = Counter(row["ROLE_DOMAIN"] for row in selected_rows)
    type_counts = Counter(row["DILEMMA_TYPE"] for row in selected_rows)
    context_counts = Counter(row["CONTEXT"] for row in selected_rows)

    summary = {
        "selection_protocol": "manual_structural_screen_v1",
        "selection_count": len(selected_rows),
        "source_counts": dict(sorted(source_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "type_counts": dict(sorted(type_counts.items())),
        "context_counts": dict(sorted(context_counts.items())),
        "selected_pool_indices": SELECTED_POOL_INDICES,
        "non_overlap_public_pool_size": len(non_overlap_rows),
        "excluded_exact_theory_overlap_count": len(public_rows) - len(non_overlap_rows),
    }

    write_jsonl(OUTPUT_DIR / "experiment_02_public_conflict_extension.jsonl", selected_rows)
    write_json(
        OUTPUT_DIR / "experiment_02_public_conflict_extension_summary.json",
        summary,
    )


if __name__ == "__main__":
    main()
