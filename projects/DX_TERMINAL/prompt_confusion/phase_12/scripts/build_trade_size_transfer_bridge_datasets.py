from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from projects.DX_TERMINAL.prompt_confusion.neon import connect_neon, ensure_schema, validate_table_name
from projects.DX_TERMINAL.prompt_confusion.paths import phase_root
from projects.DX_TERMINAL.prompt_confusion.phase_12.scripts.transfer_bridge_neon import (
    replace_transfer_table,
    table_exists,
)

PHASE_12_ROOT = phase_root("phase_12", __file__)
OUTPUT_DIR = PHASE_12_ROOT / "outputs" / "transfer_bridge"

STRICT_SUMMARY = OUTPUT_DIR / "trade_size_stage1b_adapter_strict_summary.json"
LOOSE_SUMMARY = OUTPUT_DIR / "trade_size_stage1b_adapter_loose_summary.json"
REAL_TICK_TABLE = os.environ.get("REAL_COMPLAINT_TRANSFER_TABLE", "dx_terminal_real_complaint_transfer_ticks_v1")
SYNTHETIC_TABLE_CANDIDATES = (
    os.environ.get("TRADE_SIZE_SYNTHETIC_SOURCE_TABLE", "conflict_probe_examples_v5"),
    "conflict_probe_examples_v4",
)
STRICT_TABLE = os.environ.get("TRADE_SIZE_STAGE1B_STRICT_TABLE", "dx_terminal_trade_size_stage1b_adapter_strict_v1")
LOOSE_TABLE = os.environ.get("TRADE_SIZE_STAGE1B_LOOSE_TABLE", "dx_terminal_trade_size_stage1b_adapter_loose_v1")

MARKET_START_MARKER = "## MARKET SNAPSHOT"
ACTIVE_STRATEGIES_MARKER = "## ACTIVE STRATEGIES"
ACTIVE_SETTINGS_MARKER = "## ACTIVE SETTINGS"
PORTFOLIO_MARKER = "PORTFOLIO CONTEXT"
CONSTRAINTS_MARKER = "## CONSTRAINTS"


def first_synthetic_system_text(conn: Any) -> str:
    for raw_table in SYNTHETIC_TABLE_CANDIDATES:
        table = validate_table_name(str(raw_table))
        if not table_exists(conn, table):
            continue
        row = conn.execute(
            f"""
            SELECT system_text
            FROM {table}
            WHERE system_text IS NOT NULL
              AND length(system_text) > 0
            ORDER BY example_id
            LIMIT 1
            """
        ).fetchone()
        if row and row.get("system_text"):
            return str(row["system_text"])
    raise ValueError("No synthetic system_text found in the configured Neon synthetic source tables")


def load_real_tick_rows(conn: Any) -> list[dict[str, Any]]:
    table = validate_table_name(REAL_TICK_TABLE)
    rows = conn.execute(
        f"""
        SELECT
            example_id,
            trace_id,
            prompt_messages_json,
            label,
            fault,
            root_cause,
            agent_was_correct,
            severity,
            confidence,
            urgency,
            complaint_type,
            complaint_text,
            referenced_tokens,
            has_strategy,
            strategies_text,
            slider_ta,
            slider_arp,
            slider_ts,
            slider_hs,
            slider_div,
            n_relevant_ticks,
            n_ticks_attached,
            tick_index,
            created_at,
            minute_key,
            tool,
            llm_model,
            size_relevant_complaint,
            activity_relevant_complaint,
            config_conflict_like,
            system_fault
        FROM {table}
        WHERE prompt_messages_json IS NOT NULL
          AND jsonb_typeof(prompt_messages_json) = 'array'
        ORDER BY trace_id, tick_index
        """
    ).fetchall()
    if not rows:
        raise ValueError(f"No rows found in Neon table {table}")
    return [dict(row) for row in rows]


def unique_trace_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first_by_trace: dict[str, dict[str, Any]] = {}
    for row in rows:
        trace_id = str(row["trace_id"])
        if trace_id not in first_by_trace:
            first_by_trace[trace_id] = row
    return list(first_by_trace.values())


def parse_prompt_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("prompt_messages_json")
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, list):
        return raw
    return []


def message_content(messages: list[dict[str, Any]], role: str) -> str:
    for message in messages:
        if str(message.get("role")) == role:
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def extract_section(text: str, start_marker: str, end_marker: str | None) -> str | None:
    start = text.find(start_marker)
    if start < 0:
        return None
    content_start = start + len(start_marker)
    end = len(text) if end_marker is None else text.find(end_marker, content_start)
    if end < 0:
        end = len(text)
    section = text[start:end].strip()
    return section or None


def extract_market_block(user_text: str) -> str:
    section = extract_section(user_text, MARKET_START_MARKER, ACTIVE_STRATEGIES_MARKER)
    if section:
        return section
    return "No market block was recovered from the source prompt."


def extract_portfolio_block(user_text: str) -> str:
    section = extract_section(user_text, PORTFOLIO_MARKER, CONSTRAINTS_MARKER)
    if section:
        return section
    return "No portfolio block was recovered from the source prompt."


def render_trade_size_setting(value: int | None) -> str:
    if value == 1:
        return "up to ~15% of available capital per trade"
    if value == 2:
        return "up to ~30% of available capital per trade"
    if value == 3:
        return "up to ~50% of available capital per trade"
    if value == 4:
        return "up to ~70% of available capital per trade"
    if value == 5:
        return "up to ~90% of available capital per trade"
    return "balanced / unspecified trade sizing"


def render_activity_setting(value: int | None) -> str:
    mapping = {
        1: "very patient; trade rarely",
        2: "selective; only clear setups",
        3: "balanced activity",
        4: "active when fresh edge appears",
        5: "very active when edge exists",
    }
    return mapping.get(value, "balanced / unspecified activity")


def render_risk_setting(value: int | None) -> str:
    mapping = {
        1: "prefer lowest volatility names",
        2: "lean conservative on token selection",
        3: "balanced risk appetite",
        4: "lean aggressive on token selection",
        5: "embrace high volatility names",
    }
    return mapping.get(value, "balanced / unspecified risk preference")


def render_holding_setting(value: int | None) -> str:
    mapping = {
        1: "short minimum hold",
        2: "roughly ~1 hour minimum hold",
        3: "hold for hours",
        4: "hold for many hours",
        5: "hold for days",
    }
    return mapping.get(value, "balanced / unspecified holding style")


def render_diversification_setting(value: int | None) -> str:
    mapping = {
        1: "concentrated in 1-2 names",
        2: "focused 2-3 names",
        3: "balanced 3-5 names",
        4: "spread 4-6 names",
        5: "wide 5+ names",
    }
    return mapping.get(value, "balanced / unspecified diversification")


def normalize_strategy_lines(strategies_text: str) -> list[str]:
    lines = [line.strip() for line in strategies_text.splitlines() if line.strip()]
    if not lines:
        return ["- No explicit strategy text recovered."]
    return [line if line.startswith("- ") else f"- {line}" for line in lines]


def build_synthetic_shape_user_text(row: dict[str, Any]) -> str:
    messages = parse_prompt_messages(row)
    user_text = message_content(messages, "user")
    market_block = extract_market_block(user_text)
    portfolio_block = extract_portfolio_block(user_text)
    strategy_lines = normalize_strategy_lines(str(row.get("strategies_text") or ""))

    block = [
        "TASK",
        "Choose exactly one action for this tick.",
        "",
        "STRATEGY",
        *strategy_lines,
        "Apply the strategy only within what ACTIVE SETTINGS allow.",
        "",
        "ACTIVE SETTINGS",
        f"- Trading Activity: {row.get('slider_ta')}/5 — {render_activity_setting(row.get('slider_ta'))}.",
        f"- Trade Size: {row.get('slider_ts')}/5 — {render_trade_size_setting(row.get('slider_ts'))}.",
        f"- Asset Risk Preference: {row.get('slider_arp')}/5 — {render_risk_setting(row.get('slider_arp'))}.",
        f"- Holding Style: {row.get('slider_hs')}/5 — {render_holding_setting(row.get('slider_hs'))}.",
        f"- Diversification: {row.get('slider_div')}/5 — {render_diversification_setting(row.get('slider_div'))}.",
        "",
        "PORTFOLIO",
        portfolio_block,
        "",
        "MARKET",
        market_block,
    ]
    return "\n".join(block).strip()


def strategy_size_preference(strategies_text: str) -> str | None:
    text = (strategies_text or "").lower()
    percents = [int(match) for match in re.findall(r"(\d{1,3})%", text)]

    if "10-15%" in text or "10–15%" in text or "15% of vault" in text or "small opportunistic" in text:
        return "small"
    if any(
        phrase in text
        for phrase in (
            "80-90%",
            "80–90%",
            "allocate 80%",
            "allocate 90%",
            "90% of eth",
            "90% weth allocation",
            "full deployment",
        )
    ):
        return "large"
    if percents:
        max_pct = max(percents)
        if max_pct <= 20:
            return "small"
        if max_pct >= 70:
            return "large"
    if "small" in text and "large" not in text:
        return "small"
    if "large" in text and "small" not in text:
        return "large"
    return None


def slider_bucket(value: int | None, *, strict: bool) -> str | None:
    if strict:
        if value == 1:
            return "small"
        if value == 5:
            return "large"
        return None
    if value in (1, 2):
        return "small"
    if value in (4, 5):
        return "large"
    return None


def stable_sort_key(row: dict[str, Any]) -> str:
    return hashlib.sha256(str(row["trace_id"]).encode("utf-8")).hexdigest()


def build_adapter_rows(conn: Any, *, strict: bool) -> list[dict[str, Any]]:
    system_text = first_synthetic_system_text(conn)
    base_rows = unique_trace_rows(load_real_tick_rows(conn))
    adapter_rows: list[dict[str, Any]] = []

    for row in sorted(base_rows, key=stable_sort_key):
        preference = strategy_size_preference(str(row.get("strategies_text") or ""))
        bucket = slider_bucket(row.get("slider_ts"), strict=strict)
        if preference is None or bucket is None:
            continue

        alignment_label = "aligned" if preference == bucket else "conflict"
        adapter_user_text = build_synthetic_shape_user_text(row)
        prompt_messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": adapter_user_text},
        ]

        adapter_rows.append(
            {
                "example_id": f"trade_size_stage1b:{'strict' if strict else 'loose'}:{row['trace_id']}",
                "trace_id": row["trace_id"],
                "source_example_id": row["example_id"],
                "prompt_messages_json": prompt_messages,
                "prompt_text": f"[system] {system_text}\n\n[user] {adapter_user_text}",
                "prompt_message_count": 2,
                "label": row.get("label"),
                "fault": row.get("fault"),
                "root_cause": row.get("root_cause"),
                "agent_was_correct": row.get("agent_was_correct"),
                "severity": row.get("severity"),
                "confidence": row.get("confidence"),
                "urgency": row.get("urgency"),
                "complaint_type": row.get("complaint_type"),
                "complaint_text": row.get("complaint_text"),
                "referenced_tokens": row.get("referenced_tokens") or [],
                "has_strategy": row.get("has_strategy"),
                "slider_ta": row.get("slider_ta"),
                "slider_arp": row.get("slider_arp"),
                "slider_ts": row.get("slider_ts"),
                "slider_hs": row.get("slider_hs"),
                "slider_div": row.get("slider_div"),
                "n_relevant_ticks": row.get("n_relevant_ticks"),
                "n_ticks_attached": row.get("n_ticks_attached"),
                "tick_index": row.get("tick_index"),
                "created_at": row.get("created_at"),
                "minute_key": row.get("minute_key"),
                "tool": row.get("tool"),
                "llm_model": row.get("llm_model"),
                "size_relevant_complaint": row.get("size_relevant_complaint"),
                "activity_relevant_complaint": row.get("activity_relevant_complaint"),
                "config_conflict_like": row.get("config_conflict_like"),
                "system_fault": row.get("system_fault"),
                "transfer_stage": "stage1b",
                "transfer_family": "trade_size",
                "transfer_format": "real_content_in_synthetic_shape",
                "adapter_alignment_label": alignment_label,
                "strategy_size_preference": preference,
                "slider_size_bucket": bucket,
                "extracted_portfolio_present": PORTFOLIO_MARKER in message_content(parse_prompt_messages(row), "user"),
                "extracted_market_present": MARKET_START_MARKER in message_content(parse_prompt_messages(row), "user"),
            }
        )
    return adapter_rows


def write_dataset(conn: Any, rows: list[dict[str, Any]], *, table_name: str, summary_path: Path, strict: bool) -> None:
    table_name = validate_table_name(table_name)
    table_summary = replace_transfer_table(conn, table_name, rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(row["adapter_alignment_label"] for row in rows)
    complaint_counts = defaultdict(Counter)
    root_counts = defaultdict(Counter)
    for row in rows:
        complaint_counts[row["adapter_alignment_label"]][str(row.get("complaint_type"))] += 1
        root_counts[row["adapter_alignment_label"]][str(row.get("root_cause"))] += 1

    summary = {
        "dataset_table": table_name,
        "strict_slider_extremes": strict,
        "rows": len(rows),
        "table_summary": table_summary,
        "label_counts": dict(counts),
        "complaint_type_counts_by_label": {label: dict(counter.most_common(10)) for label, counter in complaint_counts.items()},
        "root_cause_counts_by_label": {label: dict(counter.most_common(10)) for label, counter in root_counts.items()},
        "example_ids_preview": [row["example_id"] for row in rows[:10]],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    with connect_neon(autocommit=True) as conn:
        ensure_schema(conn)
        strict_rows = build_adapter_rows(conn, strict=True)
        loose_rows = build_adapter_rows(conn, strict=False)
        write_dataset(conn, strict_rows, table_name=STRICT_TABLE, summary_path=STRICT_SUMMARY, strict=True)
        write_dataset(conn, loose_rows, table_name=LOOSE_TABLE, summary_path=LOOSE_SUMMARY, strict=False)
    print(f"Wrote {len(strict_rows)} strict rows to Neon table {STRICT_TABLE}")
    print(f"Wrote {len(loose_rows)} loose rows to Neon table {LOOSE_TABLE}")


if __name__ == "__main__":
    main()
