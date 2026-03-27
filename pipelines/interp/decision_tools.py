from __future__ import annotations

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
