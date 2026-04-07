"""Shared tool schemas for capture and generation flows.

This module stores the supported function-calling tool definitions used by
trading-agent prompts and exposes helpers for selecting a schema by name and
building constrained offline-generation schemas.
"""

from __future__ import annotations

import copy
from typing import Any


TRADING_DECISION_TOOLS_V1: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "buy_token",
            "parameters": {
                "type": "object",
                "required": ["token", "spend_pct"],
                "properties": {
                    "token": {
                        "type": "string",
                        "description": "Counterparty token symbol or address.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Short description of your reasoning for this trade",
                    },
                    "strategy": {
                        "type": "string",
                        "description": (
                            "Optional. If this action follows an active strategy from the "
                            'ACTIVE STRATEGIES section, provide its label (e.g. "strategy1"). '
                            "Omit if no active strategies exist or if this action is not strategy-driven."
                        ),
                    },
                    "spend_pct": {
                        "type": "number",
                        "description": "Percent (0-100] of the source balance to allocate to this trade.",
                    },
                },
            },
            "description": "Buy a token using ETH with a percentage of the available balance.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sell_token",
            "parameters": {
                "type": "object",
                "required": ["token", "spend_pct"],
                "properties": {
                    "token": {
                        "type": "string",
                        "description": "Counterparty token symbol or address.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Short description of your reasoning for this trade",
                    },
                    "strategy": {
                        "type": "string",
                        "description": (
                            "Optional. If this action follows an active strategy from the "
                            'ACTIVE STRATEGIES section, provide its label (e.g. "strategy1"). '
                            "Omit if no active strategies exist or if this action is not strategy-driven."
                        ),
                    },
                    "spend_pct": {
                        "type": "number",
                        "description": "Percent (0-100] of the source balance to allocate to this trade.",
                    },
                },
            },
            "description": "Sell a token back into ETH using a percentage of the token balance.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_observation",
            "parameters": {
                "type": "object",
                "required": ["content"],
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Body of the message to save",
                    },
                    "strategy": {
                        "type": "string",
                        "description": (
                            "Optional. If this action follows an active strategy from the "
                            'ACTIVE STRATEGIES section, provide its label (e.g. "strategy1"). '
                            "Omit if no active strategies exist or if this action is not strategy-driven."
                        ),
                    },
                },
            },
            "description": "Save a short note about your current market observations to aid you in future trades.",
        },
    },
]


def resolve_tool_schema_mode(mode: str) -> list[dict[str, Any]] | None:
    normalized = mode.strip().lower()
    if not normalized or normalized == "none":
        return None
    if normalized in {"trading_v1", "decision_v1"}:
        return TRADING_DECISION_TOOLS_V1
    raise ValueError(f"Unknown tool schema mode: {mode}")


def _extract_forced_tool_name(tool_choice: str | dict[str, Any] | None) -> str | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        normalized = tool_choice.strip()
        if not normalized or normalized in {"none", "auto", "required"}:
            return None
        return normalized

    if not isinstance(tool_choice, dict):
        return None

    function = tool_choice.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()

    name = tool_choice.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def build_structured_tool_call_schema(
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Build a JSON schema for constrained offline tool calls.

    vLLM's OpenAI serving layer converts required/named tool calls into
    structured-output constraints. Our offline path needs to do the same
    manually when calling ``LLM.generate`` directly.
    """

    if not tools or tool_choice in (None, "none", "auto"):
        return None

    forced_tool_name = _extract_forced_tool_name(tool_choice)
    selected_tools: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function", {})
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        if forced_tool_name is not None and name != forced_tool_name:
            continue
        params = copy.deepcopy(function.get("parameters") or {"type": "object", "properties": {}})
        selected_tools.append(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": [name]},
                    "arguments": params,
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
        )

    if not selected_tools:
        if forced_tool_name is not None:
            raise ValueError(f"Tool '{forced_tool_name}' was not found in tools.")
        return None

    if len(selected_tools) == 1:
        return selected_tools[0]

    return {
        "type": "object",
        "anyOf": selected_tools,
    }
