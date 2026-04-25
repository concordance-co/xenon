from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any

from projects.DX_TERMINAL.prompt_confusion.neon import connect_neon, ensure_schema, validate_table_name
from projects.DX_TERMINAL.prompt_confusion.paths import phase_root
from projects.DX_TERMINAL.prompt_confusion.phase_12.scripts.transfer_bridge_neon import replace_transfer_table

ROOT = phase_root("phase_12", __file__)
OUTPUT_SUMMARY = ROOT / "outputs" / "transfer_bridge" / "trade_size_stage1b_adapter_strict_buy_only_summary.json"
SOURCE_TABLE = os.environ.get("TRADE_SIZE_STAGE1B_STRICT_TABLE", "dx_terminal_trade_size_stage1b_adapter_strict_v1")
REAL_TICK_TABLE = os.environ.get("REAL_COMPLAINT_TRANSFER_TABLE", "dx_terminal_real_complaint_transfer_ticks_v1")
OUTPUT_TABLE = os.environ.get(
    "TRADE_SIZE_STAGE1B_STRICT_BUY_ONLY_TABLE",
    "dx_terminal_trade_size_stage1b_adapter_strict_buy_only_v1",
)

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


def load_source_rows(conn: Any) -> list[dict[str, Any]]:
    table = validate_table_name(SOURCE_TABLE)
    rows = conn.execute(f"SELECT * FROM {table} ORDER BY example_id").fetchall()
    return [dict(row) for row in rows]


def load_real_strategy_text(conn: Any) -> dict[str, str]:
    table = validate_table_name(REAL_TICK_TABLE)
    rows = conn.execute(
        f"""
        SELECT DISTINCT ON (example_id)
            example_id,
            strategies_text
        FROM {table}
        ORDER BY example_id, tick_index
        """
    ).fetchall()
    return {str(row["example_id"]): str(row.get("strategies_text") or "") for row in rows}


def main() -> None:
    with connect_neon(autocommit=True) as conn:
        ensure_schema(conn)
        strategy_text_by_source = load_real_strategy_text(conn)
        source_rows = load_source_rows(conn)

    kept: list[dict[str, Any]] = []
    counts = Counter()

    for row in source_rows:
        source_example_id = str(row["source_example_id"])
        strategies_text = strategy_text_by_source.get(source_example_id, "").lower()
        if any(pattern in strategies_text for pattern in SELLISH_PATTERNS):
            continue
        kept.append(row)
        counts[row["adapter_alignment_label"]] += 1
        counts[(row["strategy_size_preference"], row["slider_size_bucket"], row["adapter_alignment_label"])] += 1

    with connect_neon(autocommit=True) as conn:
        ensure_schema(conn)
        table_summary = replace_transfer_table(conn, OUTPUT_TABLE, kept)

    summary = {
        "rows": len(kept),
        "label_counts": {key if isinstance(key, str) else str(key): value for key, value in counts.items()},
        "dataset_table": OUTPUT_TABLE,
        "table_summary": table_summary,
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
