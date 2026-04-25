from __future__ import annotations

import json
import os
from typing import Any

from projects.DX_TERMINAL.prompt_confusion.neon import connect_neon, ensure_schema, validate_table_name
from projects.DX_TERMINAL.prompt_confusion.paths import phase_root
from projects.DX_TERMINAL.prompt_confusion.phase_12.scripts.transfer_bridge_neon import (
    replace_transfer_table,
    table_columns,
    table_exists,
)

OUTPUT_DIR = phase_root("phase_12", __file__) / "outputs" / "transfer_bridge"
OUTPUT_SUMMARY = OUTPUT_DIR / "trade_size_stage1a_template_control_summary.json"
REAL_TICK_TABLE = os.environ.get("REAL_COMPLAINT_TRANSFER_TABLE", "dx_terminal_real_complaint_transfer_ticks_v1")
SYNTHETIC_TABLE_CANDIDATES = (
    os.environ.get("TRADE_SIZE_SYNTHETIC_SOURCE_TABLE", "conflict_probe_examples_v5"),
    "conflict_probe_examples_v4",
)
OUTPUT_TABLE = os.environ.get("TRADE_SIZE_STAGE1A_TABLE", "dx_terminal_trade_size_stage1a_template_control_v1")

MARKET_START_MARKER = "## MARKET SNAPSHOT"


def first_real_template(conn: Any) -> tuple[str, str]:
    table = validate_table_name(REAL_TICK_TABLE)
    row = conn.execute(
        f"""
        SELECT prompt_messages_json
        FROM {table}
        WHERE prompt_messages_json IS NOT NULL
          AND jsonb_typeof(prompt_messages_json) = 'array'
        ORDER BY example_id
        LIMIT 1
        """
    ).fetchone()
    if not row:
        raise ValueError(f"No prompt_messages_json rows found in Neon table {table}")
    messages = row["prompt_messages_json"]
    if isinstance(messages, str):
        messages = json.loads(messages)
    system_text = next(message["content"] for message in messages if message["role"] == "system")
    user_text = next(message["content"] for message in messages if message["role"] == "user")
    prefix_end = user_text.find(MARKET_START_MARKER)
    if prefix_end < 0:
        raise ValueError("Could not find market marker in representative real prompt")
    return system_text, user_text[:prefix_end].rstrip()


def _direction_from_setting(value: int | None) -> str | None:
    if value in (1, 2):
        return "small"
    if value in (4, 5):
        return "large"
    return "balanced" if value == 3 else None


def _synthetic_rows_sql(table: str, columns: set[str]) -> str:
    if "target_dimension" in columns:
        filters = ["target_dimension = 'trade_size'"]
        if "edge_conflict" in columns:
            filters.append("COALESCE(edge_conflict, false) = false")
        if "main_benchmark_row" in columns:
            filters.append("COALESCE(main_benchmark_row, true) = true")
        return f"SELECT * FROM {table} WHERE {' AND '.join(filters)} ORDER BY example_id"
    if "setting_family" in columns:
        return f"SELECT * FROM {table} WHERE setting_family = 'trade_size' ORDER BY example_id"
    raise ValueError(f"Neon table {table} does not look like a supported synthetic trade-size source")


def load_synthetic_trade_size_rows(conn: Any) -> list[dict[str, Any]]:
    for raw_table in SYNTHETIC_TABLE_CANDIDATES:
        table = validate_table_name(str(raw_table))
        if not table_exists(conn, table):
            continue
        columns = table_columns(conn, table)
        rows = conn.execute(_synthetic_rows_sql(table, columns)).fetchall()
        if rows:
            return [normalize_synthetic_row(dict(row)) for row in rows]
    raise ValueError("No trade-size synthetic rows found in the configured Neon source tables")


def normalize_synthetic_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("target_dimension") == "trade_size":
        return row
    setting_value = row.get("setting_value")
    if setting_value is not None:
        setting_value = int(setting_value)
    row["target_dimension"] = "trade_size"
    row["main_benchmark_row"] = True
    row["edge_conflict"] = False
    row["strategy_direction"] = row.get("strategy_expected_size")
    row["setting_implied_direction"] = _direction_from_setting(setting_value)
    row["settings_snapshot_json"] = row.get("settings_snapshot_json") or {
        "activity_value": 3,
        "risk_value": 3,
        "size_value": setting_value,
        "holding_value": 3,
        "diversification_value": 3,
    }
    return row


def extract_synthetic_section(user_text: str, marker: str) -> str:
    start = user_text.find(marker)
    if start < 0:
        return ""
    content_start = start + len(marker)
    remainder = user_text[content_start:]
    next_markers = [m for m in ("\nACTIVE SETTINGS\n", "\nPORTFOLIO\n", "\nMARKET\n") if m != marker]
    end = len(remainder)
    for next_marker in next_markers:
        idx = remainder.find(next_marker)
        if idx >= 0:
            end = min(end, idx)
    return remainder[:end].strip()


def render_real_style_user(prefix: str, synthetic_user_text: str) -> str:
    strategy = extract_synthetic_section(synthetic_user_text, "STRATEGY\n")
    settings = extract_synthetic_section(synthetic_user_text, "ACTIVE SETTINGS\n")
    portfolio = extract_synthetic_section(synthetic_user_text, "PORTFOLIO\n")
    market = extract_synthetic_section(synthetic_user_text, "MARKET\n")
    return (
        f"{prefix}\n\n"
        "## MARKET SNAPSHOT\n\n"
        f"{market}\n\n"
        "------------------------------\n\n"
        "## ACTIVE STRATEGIES\n\n"
        f"{strategy}\n\n"
        "------------------------------\n\n"
        "## ACTIVE SETTINGS\n\n"
        f"{settings}\n\n"
        "------------------------------\n\n"
        "PORTFOLIO CONTEXT\n\n"
        f"{portfolio}\n"
    ).strip()


def build_rows(conn: Any) -> list[dict[str, Any]]:
    system_text, real_prefix = first_real_template(conn)
    rows: list[dict[str, Any]] = []
    for row in load_synthetic_trade_size_rows(conn):
        if row.get("target_dimension") != "trade_size":
            continue
        if not row.get("main_benchmark_row"):
            continue
        if bool(row.get("edge_conflict")):
            continue

        user_text = render_real_style_user(real_prefix, str(row["user_text"]))
        prompt_messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        settings_snapshot = row.get("settings_snapshot_json") or {}
        if isinstance(settings_snapshot, str):
            settings_snapshot = json.loads(settings_snapshot)

        rows.append(
            {
                "example_id": f"trade_size_stage1a:{row['example_id']}",
                "trace_id": row["example_id"],
                "source_example_id": row["example_id"],
                "prompt_messages_json": prompt_messages,
                "prompt_text": f"[system] {system_text}\n\n[user] {user_text}",
                "prompt_message_count": 2,
                "label": "synthetic",
                "fault": "synthetic",
                "root_cause": "template_control",
                "agent_was_correct": None,
                "severity": None,
                "confidence": None,
                "urgency": None,
                "complaint_type": "TEMPLATE_CONTROL",
                "complaint_text": None,
                "referenced_tokens": [],
                "has_strategy": True,
                "slider_ta": settings_snapshot.get("activity_value", 3),
                "slider_arp": settings_snapshot.get("risk_value", 3),
                "slider_ts": settings_snapshot.get("size_value", row.get("setting_value")),
                "slider_hs": settings_snapshot.get("holding_value", 3),
                "slider_div": settings_snapshot.get("diversification_value", 3),
                "n_relevant_ticks": None,
                "n_ticks_attached": None,
                "tick_index": 0,
                "created_at": None,
                "minute_key": None,
                "tool": None,
                "llm_model": None,
                "size_relevant_complaint": True,
                "activity_relevant_complaint": False,
                "config_conflict_like": bool(row["conflict_present"]),
                "system_fault": None,
                "transfer_stage": "stage1a",
                "transfer_family": "trade_size",
                "transfer_format": "synthetic_content_in_real_template",
                "adapter_alignment_label": "conflict" if bool(row["conflict_present"]) else "aligned",
                "strategy_size_preference": row.get("strategy_direction"),
                "slider_size_bucket": row.get("setting_implied_direction"),
                "target_dimension": row.get("target_dimension"),
                "synthetic_conflict_present": bool(row["conflict_present"]),
            }
        )
    return rows


def main() -> None:
    with connect_neon(autocommit=True) as conn:
        ensure_schema(conn)
        rows = build_rows(conn)
        table_summary = replace_transfer_table(conn, OUTPUT_TABLE, rows)
    summary = {
        "dataset_table": OUTPUT_TABLE,
        "rows": len(rows),
        "aligned": sum(1 for row in rows if row["adapter_alignment_label"] == "aligned"),
        "conflict": sum(1 for row in rows if row["adapter_alignment_label"] == "conflict"),
        "table_summary": table_summary,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
