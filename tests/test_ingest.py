"""Tests for the ingest pipeline.

Covers: full_log_parser, CLI arg parsing, API types, helper functions.
DB tests are skipped unless XENON_NEON_DATABASE_URL is set (requires live Neon).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from pipelines.ingest.api import RetryPolicy, TerminalApiError
from pipelines.ingest.cli import _build_parser
from pipelines.ingest.db import _as_bool
from pipelines.ingest.full_log_parser import ParsedFullLog, parse_full_log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_full_log_payload(
    *,
    messages: list[dict[str, str]] | None = None,
    completion: str = "I recommend buying ETH.",
    reasoning: str = "ETH looks strong.",
    tool_calls: list[dict[str, Any]] | None = None,
    model: str = "qwen3-8b",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> dict[str, Any]:
    if messages is None:
        messages = [
            {"role": "system", "content": "You are a trading agent."},
            {"role": "user", "content": "Analyze ETH price action."},
        ]
    first_choice: dict[str, Any] = {
        "message": {
            "content": completion,
            "reasoning_content": reasoning,
        }
    }
    if tool_calls is not None:
        first_choice["message"]["tool_calls"] = tool_calls
    return {
        "llm_request_payload": {
            "model": model,
            "llm_input": {
                "messages": messages,
            },
        },
        "llm_completion_payload": {
            "choices": [first_choice],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        },
    }


# ---------------------------------------------------------------------------
# full_log_parser tests
# ---------------------------------------------------------------------------


class TestParseFullLog:
    def test_parses_complete_payload(self) -> None:
        payload = _make_full_log_payload()
        result = parse_full_log(payload)

        assert result.parse_error is None
        assert result.llm_model == "qwen3-8b"
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150
        assert "trading agent" in (result.prompt_text or "")
        assert result.completion_text == "I recommend buying ETH."
        assert result.reasoning_content == "ETH looks strong."

    def test_extracts_prompt_from_messages(self) -> None:
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "User message."},
        ]
        payload = _make_full_log_payload(messages=messages)
        result = parse_full_log(payload)

        assert result.prompt_text is not None
        assert "[system]" in result.prompt_text
        assert "System prompt." in result.prompt_text
        assert "[user]" in result.prompt_text
        assert "User message." in result.prompt_text

    def test_extracts_tool_calls(self) -> None:
        tool_calls = [
            {
                "function": {
                    "name": "buy_token",
                    "arguments": '{"token":"ETH","spend_pct":"0.1"}',
                }
            }
        ]
        payload = _make_full_log_payload(tool_calls=tool_calls)
        result = parse_full_log(payload)

        assert result.tool_calls_json is not None
        parsed = json.loads(result.tool_calls_json)
        assert parsed[0]["function"]["name"] == "buy_token"

    def test_excludes_reasoning_when_disabled(self) -> None:
        payload = _make_full_log_payload(reasoning="secret reasoning")
        result = parse_full_log(payload, include_reasoning=False)

        assert result.reasoning_content is None

    def test_handles_empty_payload(self) -> None:
        result = parse_full_log({})

        assert result.parse_error is None
        assert result.prompt_text is None
        assert result.completion_text is None

    def test_handles_missing_choices(self) -> None:
        payload = {
            "llm_request_payload": {"llm_input": {"messages": []}},
            "llm_completion_payload": {},
        }
        result = parse_full_log(payload)
        assert result.parse_error is None
        assert result.completion_text is None

    def test_model_fallback_to_completion_payload(self) -> None:
        payload = {
            "llm_request_payload": {"llm_input": {"messages": []}},
            "llm_completion_payload": {"model": "fallback-model", "choices": []},
        }
        result = parse_full_log(payload)
        assert result.llm_model == "fallback-model"


# ---------------------------------------------------------------------------
# _as_bool tests
# ---------------------------------------------------------------------------


class TestAsBool:
    def test_none(self) -> None:
        assert _as_bool(None) is None

    def test_true(self) -> None:
        assert _as_bool(True) is True

    def test_false(self) -> None:
        assert _as_bool(False) is False

    def test_truthy_string(self) -> None:
        assert _as_bool("yes") is True

    def test_zero(self) -> None:
        assert _as_bool(0) is False


# ---------------------------------------------------------------------------
# RetryPolicy / TerminalApiError tests
# ---------------------------------------------------------------------------


class TestApiTypes:
    def test_retry_policy_defaults(self) -> None:
        policy = RetryPolicy()
        assert policy.max_attempts == 6
        assert policy.initial_backoff_s == 1.0
        assert policy.max_backoff_s == 60.0

    def test_terminal_api_error_is_runtime_error(self) -> None:
        err = TerminalApiError("test error")
        assert isinstance(err, RuntimeError)
        assert "test error" in str(err)


# ---------------------------------------------------------------------------
# CLI arg parsing
# ---------------------------------------------------------------------------


class TestIngestCLI:
    def test_defaults(self) -> None:
        args = _build_parser().parse_args([])
        assert args.base_url == "https://api.terminal.markets/api/v1"
        assert args.top_n == 3
        assert args.leaderboard_sort_by == "total_pnl_usd"
        assert args.request_limit == 50
        assert args.request_concurrency == 10
        assert args.timeout_s == 30
        assert args.retry_max_attempts == 6
        assert args.max_logs_per_vault is None
        assert args.max_full_logs_per_vault is None
        assert args.max_swaps_per_vault is None
        assert args.exclude_reasoning is False

    def test_all_flags(self) -> None:
        args = _build_parser().parse_args([
            "--base-url", "http://localhost:8000",
            "--top-n", "10",
            "--leaderboard-sort-by", "realized_pnl_usd",
            "--request-limit", "25",
            "--request-concurrency", "5",
            "--timeout-s", "60",
            "--retry-max-attempts", "3",
            "--max-logs-per-vault", "100",
            "--max-full-logs-per-vault", "50",
            "--max-swaps-per-vault", "200",
            "--exclude-reasoning",
        ])
        assert args.base_url == "http://localhost:8000"
        assert args.top_n == 10
        assert args.request_limit == 25
        assert args.max_logs_per_vault == 100
        assert args.max_full_logs_per_vault == 50
        assert args.max_swaps_per_vault == 200
        assert args.exclude_reasoning is True
