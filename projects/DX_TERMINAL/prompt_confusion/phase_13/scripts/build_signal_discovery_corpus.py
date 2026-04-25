from __future__ import annotations

"""Build the Phase 13 real signal-discovery corpus in Neon.

The output table is intentionally the canonical dataset source for the phase.
Local files are not used as durable dataset artifacts.
"""

import argparse
import copy
import datetime as dt
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal
from typing import Any

from psycopg.types.json import Jsonb

from projects.DX_TERMINAL.prompt_confusion.neon import connect_neon, validate_table_name


DEFAULT_OUTPUT_TABLE = "dx_terminal_signal_discovery_phase13_v1"
STRICT_ANCHOR_TABLE = "dx_terminal_trade_size_stage1b_adapter_strict_v1"
STRICT_BUY_ONLY_ANCHOR_TABLE = "dx_terminal_trade_size_stage1b_adapter_strict_buy_only_v1"
COMPLAINT_TICK_TABLE = "dx_terminal_real_complaint_transfer_ticks_v1"
STRUCTURE_CONTROL_TABLE = "dx_terminal_trade_size_stage1a_template_control_v1"
PROMPT_TIERS = ("full", "light", "aggressive")

SIZE_ALLOCATION_RISK_RE = re.compile(
    r"\b("
    r"size|sizing|sized|allocation|allocate|allocated|risk|risky|"
    r"too\s+much|too\s+little|all\s+in|max|maximum|min|minimal|"
    r"small|large|bigger|smaller|percent|percentage|spend_pct|trade_size"
    r")\b",
    re.IGNORECASE,
)
HIGH_SIGNAL_COMPLAINT_TYPES = {
    "WRONG_SIZE",
    "UNWANTED_BUY",
    "UNWANTED_SELL",
    "STRATEGY_IGNORED",
    "HOLDING_VIOLATION",
}
LOW_SIGNAL_ROOT_CAUSES = {
    "USER_EXPECTATION_MISMATCH",
    "CORRECT_BEHAVIOR",
    "MARKET_LEGITIMATE",
}

DROP_LIGHT_HEADINGS = re.compile(
    r"^\s*(recent activity|activity log|tick history|recent ticks|tool schema|tools?|"
    r"available tools|recent trading activity)\b",
    re.IGNORECASE,
)
SECTION_HEADING_RE = re.compile(r"^\s*([A-Z][A-Z0-9 _/-]{2,}):?\s*$")
AGGRESSIVE_SECTION_PATTERNS = {
    "strategies": re.compile(r"\b(active\s+strategies|strategies?|strategy)\b", re.IGNORECASE),
    "settings": re.compile(r"\b(active\s+settings|settings?|configuration|risk settings)\b", re.IGNORECASE),
    "portfolio": re.compile(r"\b(portfolio|holdings|positions)\b", re.IGNORECASE),
    "market": re.compile(r"\b(market|market snapshot|prices?|tokens?)\b", re.IGNORECASE),
}

COMMON_COLUMNS = [
    "trace_id",
    "source_example_id",
    "vault_address",
    "person_id",
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
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(item) for item in value]
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _messages_from_row(row: Mapping[str, Any]) -> list[dict[str, str]]:
    raw = row.get("prompt_messages_json")
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes | bytearray):
        raise ValueError("prompt_messages_json must be a JSON array")
    messages: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role", "user"))
        content = "" if item.get("content") is None else str(item.get("content"))
        messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError("prompt_messages_json resolved to no usable messages")
    return messages


def _render_prompt_text(messages: Sequence[Mapping[str, str]]) -> str:
    return "\n\n".join(
        f"[{message.get('role', 'user')}]\n{message.get('content', '')}".rstrip()
        for message in messages
    )


def _system_message(messages: Sequence[Mapping[str, str]]) -> dict[str, str] | None:
    for message in messages:
        if str(message.get("role", "")).lower() == "system":
            return {"role": "system", "content": str(message.get("content", ""))}
    return None


def _user_content(messages: Sequence[Mapping[str, str]]) -> str:
    return "\n\n".join(
        str(message.get("content", ""))
        for message in messages
        if str(message.get("role", "")).lower() == "user"
    )


def _light_prune_text(text: str) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    dropping = False
    for line in lines:
        if DROP_LIGHT_HEADINGS.search(line):
            dropping = True
            continue
        if dropping and SECTION_HEADING_RE.match(line):
            dropping = False
        if not dropping:
            kept.append(line)
    pruned = "\n".join(kept).strip()
    return pruned or text


def _heading_sections(text: str) -> dict[str, str]:
    lines = text.splitlines()
    sections: list[tuple[str, int]] = []
    for index, line in enumerate(lines):
        match = SECTION_HEADING_RE.match(line)
        if match:
            heading = match.group(1).strip().lower()
            sections.append((heading, index))

    payload: dict[str, str] = {}
    for i, (heading, start_index) in enumerate(sections):
        end_index = sections[i + 1][1] if i + 1 < len(sections) else len(lines)
        body = "\n".join(lines[start_index:end_index]).strip()
        for section_name, pattern in AGGRESSIVE_SECTION_PATTERNS.items():
            if section_name not in payload and pattern.search(heading):
                payload[section_name] = body
                break
    return payload


def _fallback_section(text: str, section_name: str, max_chars: int) -> str:
    pattern = AGGRESSIVE_SECTION_PATTERNS[section_name]
    lines = [line for line in text.splitlines() if pattern.search(line)]
    if not lines and section_name == "settings":
        lines = [
            line
            for line in text.splitlines()
            if re.search(r"\b(trade size|risk|diversification|holding style|trading activity)\b", line, re.IGNORECASE)
        ]
    joined = "\n".join(lines).strip()
    if not joined:
        return f"{section_name.upper()}\nUnavailable in source prompt."
    return joined[:max_chars].strip()


def _aggressive_user_text(messages: Sequence[Mapping[str, str]]) -> str:
    text = _user_content(messages)
    sections = _heading_sections(text)
    strategy = sections.get("strategies") or _fallback_section(text, "strategies", 2500)
    settings = sections.get("settings") or _fallback_section(text, "settings", 1800)
    portfolio = sections.get("portfolio") or _fallback_section(text, "portfolio", 1800)
    market = sections.get("market") or _fallback_section(text, "market", 2200)
    return "\n\n".join(
        [
            "TASK\nChoose exactly one action for this tick.",
            f"STRATEGIES\n{strategy}",
            f"SETTINGS\n{settings}",
            f"PORTFOLIO\n{portfolio}",
            f"MARKET\n{market}",
        ]
    )


def _tier_messages(messages: Sequence[Mapping[str, str]], tier: str) -> list[dict[str, str]]:
    copied = copy.deepcopy(list(messages))
    if tier == "full":
        return [{"role": str(m.get("role", "user")), "content": str(m.get("content", ""))} for m in copied]

    system = _system_message(copied)
    if tier == "light":
        return [
            {"role": str(message.get("role", "user")), "content": _light_prune_text(str(message.get("content", "")))}
            if str(message.get("role", "")).lower() == "user"
            else {"role": str(message.get("role", "system")), "content": str(message.get("content", ""))}
            for message in copied
        ]
    if tier == "aggressive":
        result: list[dict[str, str]] = []
        if system is not None:
            result.append(system)
        result.append({"role": "user", "content": _aggressive_user_text(copied)})
        return result
    raise ValueError(f"Unsupported prompt tier: {tier}")


def _is_size_allocation_risk_row(row: Mapping[str, Any]) -> bool:
    if bool(row.get("size_relevant_complaint")):
        return True
    complaint_type = str(row.get("complaint_type") or "")
    if complaint_type in HIGH_SIGNAL_COMPLAINT_TYPES:
        return True
    root_cause = str(row.get("root_cause") or "")
    if root_cause in {"USER_CONFIG_CONFLICT", "STRATEGY_SLIDER_LOCKOUT"}:
        return True
    text = " ".join(
        str(row.get(column) or "")
        for column in ("complaint_text", "strategies_text", "reasoning", "prompt_text")
    )
    return bool(SIZE_ALLOCATION_RISK_RE.search(text))


def _source_row_json(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable(value) for key, value in row.items() if key != "prompt_messages_json"}


def _output_row(
    *,
    source_table: str,
    source_row: Mapping[str, Any],
    stratum: str,
    stratum_detail: str,
    tier: str,
) -> dict[str, Any]:
    source_example_id = str(source_row.get("example_id") or source_row.get("source_example_id") or source_row.get("trace_id"))
    messages = _tier_messages(_messages_from_row(source_row), tier)
    example_id = f"phase13:{stratum}:{source_table}:{source_example_id}:{tier}"
    prompt_text = _render_prompt_text(messages)
    payload = {
        "example_id": example_id,
        "source_table": source_table,
        "source_example_id": source_example_id,
        "stratum": stratum,
        "stratum_detail": stratum_detail,
        "prompt_tier": tier,
        "prompt_messages_json": Jsonb(_jsonable(messages)),
        "prompt_text": prompt_text,
        "prompt_message_count": len(messages),
        "prompt_char_count": len(prompt_text),
        "size_allocation_risk_candidate": _is_size_allocation_risk_row(source_row),
        "source_row_json": Jsonb(_source_row_json(source_row)),
    }
    for column in COMMON_COLUMNS:
        value = source_row.get(column)
        payload[column] = Jsonb(_jsonable(value)) if isinstance(value, Mapping | Sequence) and not isinstance(value, str | bytes | bytearray) else value
    if not payload.get("source_example_id") and source_row.get("source_example_id"):
        payload["source_example_id"] = str(source_row["source_example_id"])
    return payload


def _table_exists(conn: Any, table_name: str) -> bool:
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


def _fetch_rows(conn: Any, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def _anchor_rows(conn: Any, table_name: str) -> list[dict[str, Any]]:
    table_name = validate_table_name(table_name)
    return _fetch_rows(
        conn,
        f"""
        SELECT *
        FROM {table_name}
        WHERE prompt_messages_json IS NOT NULL
        ORDER BY example_id
        """,
    )


def _limited_rows(rows: Sequence[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return list(rows)
    if int(limit) < 0:
        raise ValueError("Limits must be non-negative")
    return list(rows)[: int(limit)]


def _complaint_rows(conn: Any, table_name: str, limit: int) -> list[dict[str, Any]]:
    table_name = validate_table_name(table_name)
    return _fetch_rows(
        conn,
        f"""
        WITH candidates AS (
          SELECT
            *,
            CASE
              WHEN size_relevant_complaint THEN 4
              WHEN complaint_type = ANY(%s) THEN 3
              WHEN root_cause IN ('USER_CONFIG_CONFLICT', 'STRATEGY_SLIDER_LOCKOUT') THEN 2
              WHEN complaint_text ~* %s THEN 1
              ELSE 0
            END AS sampling_priority,
            row_number() OVER (
              PARTITION BY COALESCE(vault_address, trace_id)
              ORDER BY md5(example_id)
            ) AS vault_rank
          FROM {table_name}
          WHERE prompt_messages_json IS NOT NULL
            AND label <> 'market'
            AND (root_cause IS NULL OR root_cause <> ALL(%s))
        )
        SELECT *
        FROM candidates
        ORDER BY sampling_priority DESC, vault_rank ASC, md5(example_id)
        LIMIT %s
        """,
        (
            sorted(HIGH_SIGNAL_COMPLAINT_TYPES),
            SIZE_ALLOCATION_RISK_RE.pattern,
            sorted(LOW_SIGNAL_ROOT_CAUSES),
            int(limit),
        ),
    )


def _generic_rows(conn: Any, table_name: str, limit: int) -> list[dict[str, Any]]:
    table_name = validate_table_name(table_name)
    return _fetch_rows(
        conn,
        f"""
        SELECT *
        FROM {table_name}
        WHERE prompt_messages_json IS NOT NULL
        ORDER BY md5(COALESCE(example_id, trace_id, prompt_text))
        LIMIT %s
        """,
        (int(limit),),
    )


def _aligned_structure_control_rows(conn: Any, table_name: str, limit: int) -> list[dict[str, Any]]:
    table_name = validate_table_name(table_name)
    return _fetch_rows(
        conn,
        f"""
        SELECT *
        FROM {table_name}
        WHERE prompt_messages_json IS NOT NULL
          AND adapter_alignment_label = 'aligned'
        ORDER BY md5(COALESCE(example_id, trace_id, prompt_text))
        LIMIT %s
        """,
        (int(limit),),
    )


def _build_rows(
    conn: Any,
    *,
    anchor_limit: int | None,
    buy_only_anchor_limit: int | None,
    complaint_limit: int,
    structure_control_table: str | None,
    structure_control_limit: int,
    baseline_table: str | None,
    baseline_limit: int,
    obvious_aligned_table: str | None,
    obvious_aligned_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "output_tiers": list(PROMPT_TIERS),
        "source_counts": {},
        "missing_sources": [],
        "warnings": [],
    }

    for source_table, stratum, detail, source_limit in (
        (STRICT_ANCHOR_TABLE, "anchor_positive", "stage1b_strict", anchor_limit),
        (STRICT_BUY_ONLY_ANCHOR_TABLE, "anchor_positive_buy_only", "stage1b_strict_buy_only", buy_only_anchor_limit),
    ):
        if not _table_exists(conn, source_table):
            summary["missing_sources"].append(source_table)
            continue
        source_rows = _limited_rows(_anchor_rows(conn, source_table), source_limit)
        summary["source_counts"][source_table] = len(source_rows)
        for source_row in source_rows:
            for tier in PROMPT_TIERS:
                rows.append(
                    _output_row(
                        source_table=source_table,
                        source_row=source_row,
                        stratum=stratum,
                        stratum_detail=detail,
                        tier=tier,
                    )
                )

    if not _table_exists(conn, COMPLAINT_TICK_TABLE):
        summary["missing_sources"].append(COMPLAINT_TICK_TABLE)
    else:
        source_rows = _complaint_rows(conn, COMPLAINT_TICK_TABLE, complaint_limit)
        summary["source_counts"][COMPLAINT_TICK_TABLE] = len(source_rows)
        for source_row in source_rows:
            detail = "size_allocation_risk_oversample" if _is_size_allocation_risk_row(source_row) else "complaint_background"
            for tier in PROMPT_TIERS:
                rows.append(
                    _output_row(
                        source_table=COMPLAINT_TICK_TABLE,
                        source_row=source_row,
                        stratum="complaint",
                        stratum_detail=detail,
                        tier=tier,
                    )
            )

    if structure_control_table:
        structure_control_table = validate_table_name(structure_control_table)
        if not _table_exists(conn, structure_control_table):
            summary["missing_sources"].append(structure_control_table)
        else:
            source_rows = _aligned_structure_control_rows(conn, structure_control_table, structure_control_limit)
            summary["source_counts"][structure_control_table] = len(source_rows)
            for source_row in source_rows:
                for tier in PROMPT_TIERS:
                    rows.append(
                        _output_row(
                            source_table=structure_control_table,
                            source_row=source_row,
                            stratum="structure_matched_control",
                            stratum_detail="stage1a_real_template_aligned",
                            tier=tier,
                        )
                    )
    else:
        summary["warnings"].append("structure_matched_control skipped: no structure-control source table was provided")

    if baseline_table:
        baseline_table = validate_table_name(baseline_table)
        if not _table_exists(conn, baseline_table):
            summary["missing_sources"].append(baseline_table)
        else:
            source_rows = _generic_rows(conn, baseline_table, baseline_limit)
            summary["source_counts"][baseline_table] = len(source_rows)
            for source_row in source_rows:
                for tier in PROMPT_TIERS:
                    rows.append(
                        _output_row(
                            source_table=baseline_table,
                            source_row=source_row,
                            stratum="baseline_control",
                            stratum_detail="non_complaint_tick",
                            tier=tier,
                        )
                    )
    else:
        summary["warnings"].append("baseline_control skipped: no baseline source table was provided")

    if obvious_aligned_table:
        obvious_aligned_table = validate_table_name(obvious_aligned_table)
        if not _table_exists(conn, obvious_aligned_table):
            summary["missing_sources"].append(obvious_aligned_table)
        else:
            source_rows = _generic_rows(conn, obvious_aligned_table, obvious_aligned_limit)
            summary["source_counts"][obvious_aligned_table] = len(source_rows)
            for source_row in source_rows:
                for tier in PROMPT_TIERS:
                    rows.append(
                        _output_row(
                            source_table=obvious_aligned_table,
                            source_row=source_row,
                            stratum="obvious_aligned",
                            stratum_detail="source_provided",
                            tier=tier,
                        )
                    )
    else:
        summary["warnings"].append("obvious_aligned skipped: no source table was provided")

    summary["output_rows"] = len(rows)
    summary["base_prompt_rows"] = len(rows) // len(PROMPT_TIERS) if rows else 0
    return rows, summary


def _create_table(conn: Any, table_name: str) -> None:
    table_name = validate_table_name(table_name)
    conn.execute(
        f"""
        DROP TABLE IF EXISTS {table_name};
        CREATE TABLE {table_name} (
          example_id text PRIMARY KEY,
          source_table text NOT NULL,
          source_example_id text,
          trace_id text,
          vault_address text,
          person_id text,
          stratum text NOT NULL,
          stratum_detail text NOT NULL,
          prompt_tier text NOT NULL,
          prompt_messages_json jsonb NOT NULL,
          prompt_text text NOT NULL,
          prompt_message_count integer NOT NULL,
          prompt_char_count integer NOT NULL,
          label text,
          fault text,
          root_cause text,
          agent_was_correct boolean,
          severity integer,
          confidence double precision,
          urgency integer,
          complaint_type text,
          complaint_text text,
          referenced_tokens jsonb,
          has_strategy boolean,
          slider_ta integer,
          slider_arp integer,
          slider_ts integer,
          slider_hs integer,
          slider_div integer,
          n_relevant_ticks integer,
          n_ticks_attached integer,
          tick_index integer,
          created_at text,
          minute_key text,
          tool text,
          llm_model text,
          size_relevant_complaint boolean,
          activity_relevant_complaint boolean,
          config_conflict_like boolean,
          system_fault boolean,
          transfer_stage text,
          transfer_family text,
          transfer_format text,
          adapter_alignment_label text,
          strategy_size_preference text,
          slider_size_bucket text,
          target_dimension text,
          synthetic_conflict_present boolean,
          extracted_portfolio_present boolean,
          extracted_market_present boolean,
          size_allocation_risk_candidate boolean NOT NULL,
          source_row_json jsonb NOT NULL,
          built_at timestamptz NOT NULL DEFAULT now()
        );
        CREATE INDEX {table_name}_stratum_idx ON {table_name} (stratum, prompt_tier);
        CREATE INDEX {table_name}_source_idx ON {table_name} (source_table, source_example_id);
        CREATE INDEX {table_name}_vault_idx ON {table_name} (vault_address);
        """
    )


def _insert_rows(conn: Any, table_name: str, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    table_name = validate_table_name(table_name)
    columns = [
        "example_id",
        "source_table",
        "source_example_id",
        "trace_id",
        "vault_address",
        "person_id",
        "stratum",
        "stratum_detail",
        "prompt_tier",
        "prompt_messages_json",
        "prompt_text",
        "prompt_message_count",
        "prompt_char_count",
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
        "size_allocation_risk_candidate",
        "source_row_json",
    ]
    placeholders = ", ".join(["%s"] * len(columns))
    sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
    values = [tuple(row.get(column) for column in columns) for row in rows]
    with conn.cursor() as cur:
        cur.executemany(sql, values)


def _write_summary(conn: Any, table_name: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    table_name = validate_table_name(table_name)
    counts = conn.execute(
        f"""
        SELECT stratum, prompt_tier, count(*) AS n
        FROM {table_name}
        GROUP BY stratum, prompt_tier
        ORDER BY stratum, prompt_tier
        """
    ).fetchall()
    vaults = conn.execute(
        f"""
        SELECT stratum, count(DISTINCT vault_address) AS n_vaults
        FROM {table_name}
        WHERE vault_address IS NOT NULL
        GROUP BY stratum
        ORDER BY stratum
        """
    ).fetchall()
    payload = dict(summary)
    payload["stratum_tier_counts"] = [dict(row) for row in counts]
    payload["vault_counts"] = [dict(row) for row in vaults]
    return payload


def build_corpus(
    *,
    output_table: str = DEFAULT_OUTPUT_TABLE,
    anchor_limit: int | None = None,
    buy_only_anchor_limit: int | None = None,
    complaint_limit: int = 500,
    structure_control_table: str | None = STRUCTURE_CONTROL_TABLE,
    structure_control_limit: int = 300,
    baseline_table: str | None = None,
    baseline_limit: int = 300,
    obvious_aligned_table: str | None = None,
    obvious_aligned_limit: int = 100,
    dry_run: bool = False,
) -> dict[str, Any]:
    output_table = validate_table_name(output_table)
    with connect_neon(autocommit=False) as conn:
        rows, summary = _build_rows(
            conn,
            anchor_limit=anchor_limit,
            buy_only_anchor_limit=buy_only_anchor_limit,
            complaint_limit=complaint_limit,
            structure_control_table=structure_control_table,
            structure_control_limit=structure_control_limit,
            baseline_table=baseline_table,
            baseline_limit=baseline_limit,
            obvious_aligned_table=obvious_aligned_table,
            obvious_aligned_limit=obvious_aligned_limit,
        )
        if dry_run:
            conn.rollback()
            return summary
        _create_table(conn, output_table)
        _insert_rows(conn, output_table, rows)
        payload = _write_summary(conn, output_table, summary)
        conn.commit()
        return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-table", default=DEFAULT_OUTPUT_TABLE)
    parser.add_argument("--anchor-limit", type=int, default=None)
    parser.add_argument("--buy-only-anchor-limit", type=int, default=None)
    parser.add_argument("--complaint-limit", type=int, default=500)
    parser.add_argument("--structure-control-table", default=STRUCTURE_CONTROL_TABLE)
    parser.add_argument("--structure-control-limit", type=int, default=300)
    parser.add_argument("--baseline-table", default=None)
    parser.add_argument("--baseline-limit", type=int, default=300)
    parser.add_argument("--obvious-aligned-table", default=None)
    parser.add_argument("--obvious-aligned-limit", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    summary = build_corpus(
        output_table=args.output_table,
        anchor_limit=args.anchor_limit,
        buy_only_anchor_limit=args.buy_only_anchor_limit,
        complaint_limit=args.complaint_limit,
        structure_control_table=args.structure_control_table,
        structure_control_limit=args.structure_control_limit,
        baseline_table=args.baseline_table,
        baseline_limit=args.baseline_limit,
        obvious_aligned_table=args.obvious_aligned_table,
        obvious_aligned_limit=args.obvious_aligned_limit,
        dry_run=args.dry_run,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
