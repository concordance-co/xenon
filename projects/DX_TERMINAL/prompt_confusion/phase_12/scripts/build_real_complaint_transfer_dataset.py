from __future__ import annotations

"""Build the real complaint transfer tick table in Neon.

The source complaint export is stored as one row per complaint, with recent
agent ticks nested under the `ticks` JSONB column. This script materializes the
workflow-ready tick table in Neon so capture workflows can use
`Dataset.from_postgres(...)` directly.
"""

import argparse
import json
from typing import Any

from projects.DX_TERMINAL.prompt_confusion.neon import connect_neon, ensure_schema, validate_table_name


DEFAULT_SOURCE_TABLE = "dx_terminal_complaint_dataset_enriched_v1"
DEFAULT_DEST_TABLE = "dx_terminal_real_complaint_transfer_ticks_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the DX Terminal real complaint transfer tick table in Neon.")
    parser.add_argument("--source-table", default=DEFAULT_SOURCE_TABLE)
    parser.add_argument("--dest-table", default=DEFAULT_DEST_TABLE)
    parser.add_argument("--limit", type=int, default=None, help="Optional complaint-row limit for smoke builds.")
    return parser.parse_args()


def _source_relation(table_name: str, limit: int | None) -> str:
    table_name = validate_table_name(table_name)
    if limit is None:
        return f"SELECT * FROM {table_name}"
    if limit <= 0:
        raise ValueError("--limit must be positive when provided")
    return f"SELECT * FROM {table_name} ORDER BY trace_id LIMIT {int(limit)}"


def build_tick_table(conn: Any, *, source_table: str, dest_table: str, limit: int | None = None) -> dict[str, Any]:
    source_table = validate_table_name(source_table)
    dest_table = validate_table_name(dest_table)
    source_sql = _source_relation(source_table, limit)

    conn.execute(f"DROP TABLE IF EXISTS {dest_table}")
    conn.execute(
        f"""
        CREATE TABLE {dest_table} AS
        WITH source_rows AS (
            {source_sql}
        )
        SELECT
            c.trace_id || '::tick_' || (expanded.ordinality - 1)::text AS example_id,
            c.trace_id,
            (expanded.ordinality - 1)::bigint AS tick_index,
            expanded.tick ->> 'created_at' AS created_at,
            expanded.tick ->> 'minute_key' AS minute_key,
            c.vault_address,
            c.person_id,
            c.label,
            c.fault,
            c.root_cause,
            c.agent_was_correct,
            c.severity,
            c.confidence,
            c.urgency,
            c.complaint_text,
            c.complaint_type,
            c.referenced_tokens,
            c.has_strategy,
            c.strategies_text,
            c.slider_ta,
            c.slider_arp,
            c.slider_ts,
            c.slider_hs,
            c.slider_div,
            c.n_relevant_ticks,
            c.n_ticks_attached,
            expanded.tick ->> 'tool' AS tool,
            expanded.tick -> 'tool_args' AS tool_args_json,
            expanded.tick ->> 'reasoning' AS reasoning,
            NULLIF(expanded.tick ->> 'inference_duration_ms', '')::bigint AS inference_duration_ms,
            expanded.tick #>> '{{llm_request_payload,model}}' AS llm_model,
            expanded.tick #> '{{llm_request_payload,options}}' AS request_options_json,
            expanded.tick #> '{{snapshot,agent,options}}' AS snapshot_agent_options_json,
            expanded.tick -> 'snapshot' AS snapshot_json,
            expanded.tick #> '{{llm_request_payload,llm_input,messages}}' AS prompt_messages_json,
            (
                SELECT string_agg(
                    '[' || COALESCE(message ->> 'role', '') || '] ' ||
                    CASE jsonb_typeof(message -> 'content')
                        WHEN 'string' THEN message ->> 'content'
                        WHEN 'array' THEN (
                            SELECT string_agg(COALESCE(content_item ->> 'text', content_item::text), ' ')
                            FROM jsonb_array_elements(message -> 'content') AS content_items(content_item)
                        )
                        ELSE COALESCE((message -> 'content')::text, '')
                    END,
                    E'\\n\\n'
                    ORDER BY message_ordinality
                )
                FROM jsonb_array_elements(expanded.tick #> '{{llm_request_payload,llm_input,messages}}')
                    WITH ORDINALITY AS messages(message, message_ordinality)
            ) AS prompt_text,
            jsonb_array_length(expanded.tick #> '{{llm_request_payload,llm_input,messages}}')::bigint AS prompt_message_count,
            expanded.tick #>> '{{llm_completion_payload,choices,0,message,content}}' AS completion_content,
            expanded.tick #>> '{{llm_completion_payload,choices,0,message,reasoning}}' AS completion_reasoning,
            COALESCE(expanded.tick #> '{{llm_completion_payload,choices,0,message,tool_calls}}', '[]'::jsonb) AS completion_tool_calls_json,
            expanded.tick -> 'llm_request_payload' AS raw_request_payload_json,
            expanded.tick -> 'llm_completion_payload' AS raw_completion_payload_json,
            (c.complaint_type = 'WRONG_SIZE') AS size_relevant_complaint,
            (c.complaint_type IN ('NOT_TRADING', 'OVERTRADING', 'HOLDING_VIOLATION')) AS activity_relevant_complaint,
            (c.root_cause IN ('USER_CONFIG_CONFLICT', 'STRATEGY_SLIDER_LOCKOUT')) AS config_conflict_like,
            (c.label = 'true_confusion') AS system_fault,
            c.raw_row_json,
            now() AS built_at
        FROM source_rows c
        CROSS JOIN LATERAL jsonb_array_elements(c.ticks) WITH ORDINALITY AS expanded(tick, ordinality)
        WHERE jsonb_typeof(c.ticks) = 'array'
          AND jsonb_typeof(expanded.tick #> '{{llm_request_payload,llm_input,messages}}') = 'array'
          AND jsonb_array_length(expanded.tick #> '{{llm_request_payload,llm_input,messages}}') > 0
        """
    )
    conn.execute(f"ALTER TABLE {dest_table} ADD PRIMARY KEY (example_id)")
    conn.execute(f"CREATE INDEX {dest_table}_trace_id_idx ON {dest_table} (trace_id)")
    conn.execute(f"CREATE INDEX {dest_table}_label_idx ON {dest_table} (label)")
    conn.execute(f"CREATE INDEX {dest_table}_complaint_type_idx ON {dest_table} (complaint_type)")
    conn.execute(f"CREATE INDEX {dest_table}_root_cause_idx ON {dest_table} (root_cause)")
    conn.execute(f"CREATE INDEX {dest_table}_slider_ts_idx ON {dest_table} (slider_ts)")

    summary = conn.execute(
        f"""
        SELECT
            COUNT(*) AS output_examples,
            COUNT(DISTINCT trace_id) AS distinct_traces,
            COUNT(DISTINCT lower(vault_address)) AS distinct_vaults,
            SUM(CASE WHEN size_relevant_complaint THEN 1 ELSE 0 END) AS size_relevant_examples,
            SUM(CASE WHEN activity_relevant_complaint THEN 1 ELSE 0 END) AS activity_relevant_examples,
            SUM(CASE WHEN config_conflict_like THEN 1 ELSE 0 END) AS config_conflict_like_examples,
            SUM(CASE WHEN system_fault THEN 1 ELSE 0 END) AS system_fault_examples
        FROM {dest_table}
        """
    ).fetchone()
    label_counts = conn.execute(
        f"SELECT label, COUNT(*) AS n FROM {dest_table} GROUP BY label ORDER BY n DESC, label"
    ).fetchall()
    complaint_type_counts = conn.execute(
        f"SELECT complaint_type, COUNT(*) AS n FROM {dest_table} GROUP BY complaint_type ORDER BY n DESC, complaint_type LIMIT 10"
    ).fetchall()
    return {
        "source_table": source_table,
        "dest_table": dest_table,
        "source_limit": limit,
        **dict(summary),
        "label_counts": {row["label"]: row["n"] for row in label_counts},
        "complaint_type_counts_top10": {row["complaint_type"]: row["n"] for row in complaint_type_counts},
    }


def main() -> None:
    args = parse_args()
    with connect_neon(autocommit=True) as conn:
        ensure_schema(conn)
        summary = build_tick_table(
            conn,
            source_table=args.source_table,
            dest_table=args.dest_table,
            limit=args.limit,
        )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
