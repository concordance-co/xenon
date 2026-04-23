from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PHASE_09_DATASET = Path(
    "/Users/trentelmore/Projects/concordance/xenon-dashboard/projects/DX_TERMINAL/prompt_confusion/phase_09/outputs/phase_09_dataset/phase_09_dataset.jsonl"
)
REAL_DATASET = Path(
    "/Users/trentelmore/Projects/concordance/xenon-dashboard/projects/DX_TERMINAL/prompt_confusion/phase_12/outputs/real_complaint_transfer/real_complaint_transfer_dataset.jsonl"
)
OUTPUT_DIR = Path(
    "/Users/trentelmore/Projects/concordance/xenon-dashboard/projects/DX_TERMINAL/prompt_confusion/phase_12/outputs/transfer_bridge"
)
OUTPUT_JSONL = OUTPUT_DIR / "trade_size_stage1a_template_control.jsonl"
OUTPUT_SUMMARY = OUTPUT_DIR / "trade_size_stage1a_template_control_summary.json"

MARKET_START_MARKER = "## MARKET SNAPSHOT"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def first_real_template() -> tuple[str, str]:
    row = load_jsonl(REAL_DATASET)[0]
    messages = json.loads(row["prompt_messages_json"])
    system_text = next(message["content"] for message in messages if message["role"] == "system")
    user_text = next(message["content"] for message in messages if message["role"] == "user")
    prefix_end = user_text.find(MARKET_START_MARKER)
    if prefix_end < 0:
        raise ValueError("Could not find market marker in representative real prompt")
    return system_text, user_text[:prefix_end].rstrip()


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


def build_rows() -> list[dict[str, Any]]:
    system_text, real_prefix = first_real_template()
    rows: list[dict[str, Any]] = []
    for row in load_jsonl(PHASE_09_DATASET):
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
        settings_snapshot = row["settings_snapshot_json"]
        if isinstance(settings_snapshot, str):
            settings_snapshot = json.loads(settings_snapshot)

        rows.append(
            {
                "example_id": f"trade_size_stage1a:{row['example_id']}",
                "trace_id": row["example_id"],
                "source_example_id": row["example_id"],
                "prompt_messages_json": json.dumps(prompt_messages, ensure_ascii=False),
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
                "slider_ta": settings_snapshot["activity_value"],
                "slider_arp": settings_snapshot["risk_value"],
                "slider_ts": settings_snapshot["size_value"],
                "slider_hs": settings_snapshot["holding_value"],
                "slider_div": settings_snapshot["diversification_value"],
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
    rows = build_rows()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    summary = {
        "rows": len(rows),
        "aligned": sum(1 for row in rows if row["adapter_alignment_label"] == "aligned"),
        "conflict": sum(1 for row in rows if row["adapter_alignment_label"] == "conflict"),
        "output_jsonl": str(OUTPUT_JSONL),
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
