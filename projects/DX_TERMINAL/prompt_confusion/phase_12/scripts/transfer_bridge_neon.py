from __future__ import annotations

import json
from typing import Any

from projects.DX_TERMINAL.prompt_confusion.neon import validate_table_name


TRANSFER_ROW_COLUMNS: tuple[str, ...] = (
    "example_id",
    "trace_id",
    "source_example_id",
    "prompt_messages_json",
    "prompt_text",
    "prompt_message_count",
    "label",
    "fault",
    "root_cause",
    "agent_was_correct",
    "severity",
    "confidence",
    "urgency",
    "complaint_type",
    "complaint_text",
    "referenced_tokens",
    "has_strategy",
    "slider_ta",
    "slider_arp",
    "slider_ts",
    "slider_hs",
    "slider_div",
    "n_relevant_ticks",
    "n_ticks_attached",
    "tick_index",
    "created_at",
    "minute_key",
    "tool",
    "llm_model",
    "size_relevant_complaint",
    "activity_relevant_complaint",
    "config_conflict_like",
    "system_fault",
    "transfer_stage",
    "transfer_family",
    "transfer_format",
    "adapter_alignment_label",
    "strategy_size_preference",
    "slider_size_bucket",
    "target_dimension",
    "synthetic_conflict_present",
    "extracted_portfolio_present",
    "extracted_market_present",
)

JSON_COLUMNS = {"prompt_messages_json", "referenced_tokens"}


def table_exists(conn: Any, table_name: str) -> bool:
    table_name = validate_table_name(table_name)
    row = conn.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = %s
        ) AS exists
        """,
        (table_name,),
    ).fetchone()
    return bool(row["exists"])


def table_columns(conn: Any, table_name: str) -> set[str]:
    table_name = validate_table_name(table_name)
    rows = conn.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
        """,
        (table_name,),
    ).fetchall()
    return {str(row["column_name"]) for row in rows}


def fetch_table_rows(conn: Any, table_name: str) -> list[dict[str, Any]]:
    table_name = validate_table_name(table_name)
    rows = conn.execute(f"SELECT * FROM {table_name} ORDER BY example_id").fetchall()
    return [dict(row) for row in rows]


def _as_copy_value(column: str, value: Any) -> Any:
    if column not in JSON_COLUMNS:
        return value
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        json.loads(text)
        return text
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def replace_transfer_table(conn: Any, table_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    table_name = validate_table_name(table_name)
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(
        f"""
        CREATE TABLE {table_name} (
            example_id TEXT PRIMARY KEY,
            trace_id TEXT,
            source_example_id TEXT,
            prompt_messages_json JSONB NOT NULL,
            prompt_text TEXT,
            prompt_message_count BIGINT,
            label TEXT,
            fault TEXT,
            root_cause TEXT,
            agent_was_correct BOOLEAN,
            severity BIGINT,
            confidence DOUBLE PRECISION,
            urgency BIGINT,
            complaint_type TEXT,
            complaint_text TEXT,
            referenced_tokens JSONB,
            has_strategy BOOLEAN,
            slider_ta BIGINT,
            slider_arp BIGINT,
            slider_ts BIGINT,
            slider_hs BIGINT,
            slider_div BIGINT,
            n_relevant_ticks BIGINT,
            n_ticks_attached BIGINT,
            tick_index BIGINT,
            created_at TEXT,
            minute_key TEXT,
            tool TEXT,
            llm_model TEXT,
            size_relevant_complaint BOOLEAN,
            activity_relevant_complaint BOOLEAN,
            config_conflict_like BOOLEAN,
            system_fault BOOLEAN,
            transfer_stage TEXT NOT NULL,
            transfer_family TEXT NOT NULL,
            transfer_format TEXT NOT NULL,
            adapter_alignment_label TEXT,
            strategy_size_preference TEXT,
            slider_size_bucket TEXT,
            target_dimension TEXT,
            synthetic_conflict_present BOOLEAN,
            extracted_portfolio_present BOOLEAN,
            extracted_market_present BOOLEAN,
            built_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    column_list = ", ".join(TRANSFER_ROW_COLUMNS)
    with conn.cursor().copy(f"COPY {table_name} ({column_list}) FROM STDIN") as copy:
        for row in rows:
            copy.write_row(tuple(_as_copy_value(column, row.get(column)) for column in TRANSFER_ROW_COLUMNS))

    conn.execute(f"CREATE INDEX {table_name}_alignment_idx ON {table_name} (adapter_alignment_label)")
    conn.execute(f"CREATE INDEX {table_name}_trace_id_idx ON {table_name} (trace_id)")
    conn.execute(f"CREATE INDEX {table_name}_source_example_id_idx ON {table_name} (source_example_id)")

    summary = conn.execute(
        f"""
        SELECT
            COUNT(*) AS rows,
            COUNT(*) FILTER (WHERE adapter_alignment_label = 'aligned') AS aligned,
            COUNT(*) FILTER (WHERE adapter_alignment_label = 'conflict') AS conflict
        FROM {table_name}
        """
    ).fetchone()
    return dict(summary)
