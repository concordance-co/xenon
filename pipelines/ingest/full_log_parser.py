from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ParsedFullLog:
    prompt_text: str | None
    completion_text: str | None
    reasoning_content: str | None
    tool_calls_json: str | None
    llm_model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    total_tokens: int | None
    parse_error: str | None


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def parse_full_log(payload: dict[str, Any], include_reasoning: bool = True) -> ParsedFullLog:
    try:
        llm_request_payload = (payload.get("llm_request_payload") or {}) if payload else {}
        llm_input = llm_request_payload.get("llm_input") or {}
        messages = llm_input.get("messages") or []

        prompt_parts: list[str] = []
        for message in messages:
            role = message.get("role") or "unknown"
            content = _stringify(message.get("content"))
            prompt_parts.append(f"[{role}]\n{content}".strip())
        prompt_text = "\n\n".join(part for part in prompt_parts if part).strip() or None

        llm_completion_payload = payload.get("llm_completion_payload") or {}
        choices = llm_completion_payload.get("choices") or []
        first_choice = choices[0] if choices else {}
        message = (first_choice.get("message") or {}) if isinstance(first_choice, dict) else {}

        completion_text = _stringify(message.get("content")).strip() or None
        reasoning_content = None
        if include_reasoning:
            reasoning_content = _stringify(message.get("reasoning_content")).strip() or None

        tool_calls = message.get("tool_calls")
        tool_calls_json = None
        if tool_calls is not None:
            tool_calls_json = json.dumps(tool_calls, ensure_ascii=True, separators=(",", ":"))

        usage = llm_completion_payload.get("usage") or {}
        llm_model = llm_request_payload.get("model") or llm_completion_payload.get("model")

        return ParsedFullLog(
            prompt_text=prompt_text,
            completion_text=completion_text,
            reasoning_content=reasoning_content,
            tool_calls_json=tool_calls_json,
            llm_model=llm_model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=usage.get("reasoning_tokens"),
            total_tokens=usage.get("total_tokens"),
            parse_error=None,
        )
    except Exception as exc:  # pragma: no cover - defensive parse guard
        return ParsedFullLog(
            prompt_text=None,
            completion_text=None,
            reasoning_content=None,
            tool_calls_json=None,
            llm_model=None,
            prompt_tokens=None,
            completion_tokens=None,
            reasoning_tokens=None,
            total_tokens=None,
            parse_error=f"{type(exc).__name__}: {exc}",
        )

