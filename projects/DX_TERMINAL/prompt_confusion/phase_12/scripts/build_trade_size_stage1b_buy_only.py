from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from projects.DX_TERMINAL.prompt_confusion.paths import phase_root

ROOT = phase_root("phase_12", __file__)
REAL_DATASET = ROOT / "outputs" / "real_complaint_transfer" / "real_complaint_transfer_dataset.jsonl"
SOURCE_DATASET = ROOT / "outputs" / "transfer_bridge" / "trade_size_stage1b_adapter_strict.jsonl"
OUTPUT_DATASET = ROOT / "outputs" / "transfer_bridge" / "trade_size_stage1b_adapter_strict_buy_only.jsonl"
OUTPUT_SUMMARY = ROOT / "outputs" / "transfer_bridge" / "trade_size_stage1b_adapter_strict_buy_only_summary.json"

SELLISH_PATTERNS = (
    "sell",
    "liquidate",
    "trim",
    "exit",
    "take profit",
    "stop loss",
    "cut losing",
    "convert all",
    "reduce exposure",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def main() -> None:
    real_rows = {row["example_id"]: row for row in load_jsonl(REAL_DATASET)}
    kept: list[dict[str, Any]] = []
    counts = Counter()

    for row in load_jsonl(SOURCE_DATASET):
        source_example_id = str(row["source_example_id"])
        strategies_text = str(real_rows[source_example_id].get("strategies_text") or "").lower()
        if any(pattern in strategies_text for pattern in SELLISH_PATTERNS):
            continue
        kept.append(row)
        counts[row["adapter_alignment_label"]] += 1
        counts[(row["strategy_size_preference"], row["slider_size_bucket"], row["adapter_alignment_label"])] += 1

    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_DATASET.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    summary = {
        "rows": len(kept),
        "label_counts": {key if isinstance(key, str) else str(key): value for key, value in counts.items()},
        "output_dataset": str(OUTPUT_DATASET),
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
