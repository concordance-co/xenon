from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


SOURCE_PATH = Path(
    "/Users/trentelmore/Projects/concordance/xenon/projects/DX_TERMINAL/dataset_exports/complaint_dataset_enriched.parquet"
)
OUTPUT_DIR = Path(
    "/Users/trentelmore/Projects/concordance/xenon-dashboard/projects/DX_TERMINAL/prompt_confusion/phase_12/outputs/real_complaint_transfer"
)
OUTPUT_JSONL = OUTPUT_DIR / "real_complaint_transfer_dataset.jsonl"
OUTPUT_SUMMARY = OUTPUT_DIR / "summary.json"


def _safe_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    return table.to_pylist()


def _message_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", ""))
        content = message.get("content", "")
        if isinstance(content, list):
            text = " ".join(str(item.get("text", item)) for item in content)
        else:
            text = str(content)
        parts.append(f"[{role}] {text}")
    return "\n\n".join(parts)


def _tool_calls(choice_message: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = choice_message.get("tool_calls") or []
    normalized: list[dict[str, Any]] = []
    for call in tool_calls:
        function = call.get("function") or {}
        normalized.append(
            {
                "id": call.get("id"),
                "type": call.get("type"),
                "name": function.get("name"),
                "arguments": function.get("arguments"),
            }
        )
    return normalized


def _normalize_tick(row: dict[str, Any], tick: dict[str, Any], tick_index: int) -> dict[str, Any] | None:
    request_payload = tick.get("llm_request_payload") or {}
    llm_input = request_payload.get("llm_input") or {}
    messages = llm_input.get("messages") or []
    if not messages:
        return None

    completion_payload = tick.get("llm_completion_payload") or {}
    choices = completion_payload.get("choices") or []
    first_choice = choices[0] if choices else {}
    choice_message = first_choice.get("message") or {}

    trace_id = str(row["trace_id"])
    complaint_label = str(row["label"])
    root_cause = str(row["root_cause"])
    complaint_type = str(row["complaint_type"])

    snapshot = tick.get("snapshot") or {}
    agent = snapshot.get("agent") or {}
    options = agent.get("options") or {}

    return {
        "example_id": f"{trace_id}::tick_{tick_index}",
        "trace_id": trace_id,
        "tick_index": tick_index,
        "created_at": tick.get("created_at"),
        "minute_key": tick.get("minute_key"),
        "vault_address": row.get("vault_address"),
        "person_id": row.get("person_id"),
        "label": complaint_label,
        "fault": row.get("fault"),
        "root_cause": root_cause,
        "agent_was_correct": row.get("agent_was_correct"),
        "severity": row.get("severity"),
        "confidence": row.get("confidence"),
        "urgency": row.get("urgency"),
        "complaint_text": row.get("complaint_text"),
        "complaint_type": complaint_type,
        "referenced_tokens": row.get("referenced_tokens") or [],
        "has_strategy": row.get("has_strategy"),
        "strategies_text": row.get("strategies_text"),
        "slider_ta": row.get("slider_ta"),
        "slider_arp": row.get("slider_arp"),
        "slider_ts": row.get("slider_ts"),
        "slider_hs": row.get("slider_hs"),
        "slider_div": row.get("slider_div"),
        "n_relevant_ticks": row.get("n_relevant_ticks"),
        "n_ticks_attached": row.get("n_ticks_attached"),
        "tool": tick.get("tool"),
        "tool_args_json": _safe_json_dumps(tick.get("tool_args")),
        "reasoning": tick.get("reasoning"),
        "inference_duration_ms": tick.get("inference_duration_ms"),
        "llm_model": request_payload.get("model"),
        "request_options_json": _safe_json_dumps(request_payload.get("options")),
        "snapshot_agent_options_json": _safe_json_dumps(options),
        "snapshot_json": _safe_json_dumps(snapshot),
        "prompt_messages_json": _safe_json_dumps(messages),
        "prompt_text": _message_text(messages),
        "prompt_message_count": len(messages),
        "completion_content": choice_message.get("content"),
        "completion_reasoning": choice_message.get("reasoning"),
        "completion_tool_calls_json": _safe_json_dumps(_tool_calls(choice_message)),
        "raw_request_payload_json": _safe_json_dumps(request_payload),
        "raw_completion_payload_json": _safe_json_dumps(completion_payload),
        # Useful coarse slices for first real-data transfer passes.
        "size_relevant_complaint": complaint_type == "WRONG_SIZE",
        "activity_relevant_complaint": complaint_type in {"NOT_TRADING", "OVERTRADING", "HOLDING_VIOLATION"},
        "config_conflict_like": root_cause in {"USER_CONFIG_CONFLICT", "STRATEGY_SLIDER_LOCKOUT"},
        "system_fault": complaint_label == "true_confusion",
    }


def build_dataset() -> dict[str, Any]:
    rows = _load_rows(SOURCE_PATH)
    examples: list[dict[str, Any]] = []

    trace_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    root_cause_counter: Counter[str] = Counter()
    complaint_type_counter: Counter[str] = Counter()
    tool_counter: Counter[str] = Counter()
    model_counter: Counter[str] = Counter()

    skipped_no_ticks = 0
    skipped_no_messages = 0

    for row in rows:
        trace_counter[str(row["trace_id"])] += 1
        ticks_raw = row.get("ticks")
        if not ticks_raw:
            skipped_no_ticks += 1
            continue
        ticks = json.loads(ticks_raw)
        if not ticks:
            skipped_no_ticks += 1
            continue

        for tick_index, tick in enumerate(ticks):
            example = _normalize_tick(row, tick, tick_index)
            if example is None:
                skipped_no_messages += 1
                continue
            examples.append(example)
            label_counter.update([str(example["label"])])
            root_cause_counter.update([str(example["root_cause"])])
            complaint_type_counter.update([str(example["complaint_type"])])
            tool_counter.update([str(example["tool"])])
            model_counter.update([str(example["llm_model"])])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(_safe_json_dumps(example))
            handle.write("\n")

    summary = {
        "source_path": str(SOURCE_PATH),
        "output_jsonl": str(OUTPUT_JSONL),
        "source_rows": len(rows),
        "output_examples": len(examples),
        "distinct_traces": len({example["trace_id"] for example in examples}),
        "distinct_vaults": len({str(example["vault_address"]).lower() for example in examples}),
        "skipped_no_ticks": skipped_no_ticks,
        "skipped_no_messages": skipped_no_messages,
        "avg_examples_per_trace": round(len(examples) / max(1, len({example["trace_id"] for example in examples})), 4),
        "label_counts": dict(label_counter),
        "root_cause_counts_top10": dict(root_cause_counter.most_common(10)),
        "complaint_type_counts_top10": dict(complaint_type_counter.most_common(10)),
        "tool_counts": dict(tool_counter),
        "model_counts": dict(model_counter),
        "size_relevant_examples": sum(1 for example in examples if bool(example["size_relevant_complaint"])),
        "activity_relevant_examples": sum(1 for example in examples if bool(example["activity_relevant_complaint"])),
        "config_conflict_like_examples": sum(1 for example in examples if bool(example["config_conflict_like"])),
        "system_fault_examples": sum(1 for example in examples if bool(example["system_fault"])),
    }
    OUTPUT_SUMMARY.write_text(_safe_json_dumps(summary), encoding="utf-8")
    return summary


if __name__ == "__main__":
    summary = build_dataset()
    print(json.dumps(summary, indent=2))
