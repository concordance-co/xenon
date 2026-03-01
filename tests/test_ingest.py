"""Tests for the ingest pipeline.

Covers: full_log_parser, payload_store, db schema + upserts, CLI arg parsing.
No external API calls — all tests use local fixtures.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from pipelines.ingest.api import RetryPolicy, TerminalApiError
from pipelines.ingest.cli import _build_parser
from pipelines.ingest.db import FullLogRecord, IngestDatabase, _as_bool_int
from pipelines.ingest.full_log_parser import ParsedFullLog, parse_full_log
from pipelines.ingest.payload_store import RawPayloadMetadata, RawPayloadStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro: Any) -> Any:
    """Run an async coroutine synchronously for tests."""
    return asyncio.run(coro)


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


SAMPLE_VAULT_LEADERBOARD = {
    "vaultAddress": "0xabc123",
    "ownerAddress": "0xowner1",
    "nftId": "42",
    "nftName": "TestVault",
    "rank": 1,
    "totalPnlUsd": 12345.67,
    "realizedPnlUsd": 10000.0,
    "unrealizedPnlUsd": 2345.67,
}

SAMPLE_VAULT_CONFIG = {
    "vaultAddress": "0xabc123",
    "ownerAddress": "0xowner1",
    "nftId": "42",
    "nftName": "TestVault",
    "persona": {"style": "aggressive"},
    "tradeSize": 3,
    "tradingActivity": 4,
    "holdingStyle": 2,
    "diversification": 1,
    "assetRiskPreference": 5,
    "maxTradeAmount": "1000000",
    "slippageBps": "50",
    "paused": False,
    "state": "active",
    "createdBlock": 100,
    "updatedBlock": 200,
}

SAMPLE_INFERENCE_LOG = {
    "id": 1001,
    "cursor": "cur_1001",
    "vault_address": "0xabc123",
    "request_id": "req_001",
    "execution_key": "exec_001",
    "tool": "buy_token",
    "tool_args": {"token": "ETH", "spend_pct": "0.1"},
    "strategyId": "strat_1",
    "status": "success",
    "inference_duration_ms": 1500,
    "error": None,
    "transactionHash": "0xtx001",
    "created_at": "2025-01-01T00:00:00Z",
    "completed_at": "2025-01-01T00:00:01Z",
}

SAMPLE_SWAP = {
    "transactionHash": "0xtx001",
    "blockNumber": 12345,
    "logIndex": 0,
    "timestamp": 1700000000,
    "poolId": "pool_1",
    "tokenAddress": "0xtoken1",
    "tokenName": "TestToken",
    "tokenSymbol": "TT",
    "vaultAddress": "0xabc123",
    "isReapTwap": False,
    "side": "buy",
    "tokenAmount": "1000",
    "ethAmount": "0.5",
    "ethPriceUsd": "3000",
    "effectivePriceEth": "0.0005",
    "effectivePriceUsd": "1.50",
    "logId": 1001,
    "strategyId": "strat_1",
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
# payload_store tests
# ---------------------------------------------------------------------------


class TestPayloadStore:
    def test_write_and_read_round_trip(self, tmp_path: Path) -> None:
        store = RawPayloadStore(tmp_path / "full_logs")
        payload = {"key": "value", "nested": {"a": 1}}

        meta = store.write(42, payload)

        assert meta.payload_size_bytes > 0
        assert len(meta.payload_sha256) == 64
        assert "42.json.gz" in meta.payload_path

        # Read back and verify
        with gzip.open(meta.payload_path, "rt", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == payload

    def test_sharding(self, tmp_path: Path) -> None:
        store = RawPayloadStore(tmp_path / "full_logs")

        meta_0 = store.write(0, {"id": 0})
        meta_999 = store.write(999, {"id": 999})
        meta_1000 = store.write(1000, {"id": 1000})
        meta_5432 = store.write(5432, {"id": 5432})

        assert "/000000/" in meta_0.payload_path
        assert "/000000/" in meta_999.payload_path
        assert "/000001/" in meta_1000.payload_path
        assert "/000005/" in meta_5432.payload_path

    def test_overwrite_existing(self, tmp_path: Path) -> None:
        store = RawPayloadStore(tmp_path / "full_logs")

        meta1 = store.write(42, {"version": 1})
        meta2 = store.write(42, {"version": 2})

        # SHA should differ
        assert meta1.payload_sha256 != meta2.payload_sha256

        # File should contain version 2
        with gzip.open(meta2.payload_path, "rt", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["version"] == 2

    def test_deterministic_sha256(self, tmp_path: Path) -> None:
        store = RawPayloadStore(tmp_path / "full_logs")
        payload = {"deterministic": True}

        meta1 = store.write(1, payload)
        meta2 = store.write(2, payload)

        assert meta1.payload_sha256 == meta2.payload_sha256


# ---------------------------------------------------------------------------
# db tests
# ---------------------------------------------------------------------------


class TestIngestDatabase:
    def _make_db(self, tmp_path: Path) -> IngestDatabase:
        db = IngestDatabase(tmp_path / "test.db")
        _run(db.connect())
        _run(db.init_schema())
        return db

    def _query_sync(self, db_path: Path, sql: str) -> list[dict[str, Any]]:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql).fetchall()
        result = [dict(r) for r in rows]
        conn.close()
        return result

    def test_schema_creates_all_tables(self, tmp_path: Path) -> None:
        db = self._make_db(tmp_path)
        try:
            rows = self._query_sync(
                db.db_path,
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
            )
            table_names = {r["name"] for r in rows}
            assert "vaults" in table_names
            assert "strategies" in table_names
            assert "inference_logs" in table_names
            assert "full_logs" in table_names
            assert "swaps" in table_names
        finally:
            _run(db.close())

    def test_schema_creates_indexes(self, tmp_path: Path) -> None:
        db = self._make_db(tmp_path)
        try:
            rows = self._query_sync(
                db.db_path,
                "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY name",
            )
            index_names = {r["name"] for r in rows}
            assert "idx_inference_logs_vault" in index_names
            assert "idx_swaps_log_id" in index_names
            assert "idx_swaps_vault" in index_names
            assert "idx_swaps_token_timestamp" in index_names
        finally:
            _run(db.close())

    def test_schema_idempotent(self, tmp_path: Path) -> None:
        db = self._make_db(tmp_path)
        try:
            # Running init_schema twice should not raise
            _run(db.init_schema())
        finally:
            _run(db.close())

    def test_upsert_vault(self, tmp_path: Path) -> None:
        db = self._make_db(tmp_path)
        try:
            _run(db.upsert_vault(SAMPLE_VAULT_LEADERBOARD, SAMPLE_VAULT_CONFIG))
            rows = self._query_sync(db.db_path, "SELECT * FROM vaults")
            assert len(rows) == 1
            row = rows[0]
            assert row["vault_address"] == "0xabc123"
            assert row["leaderboard_rank"] == 1
            assert row["total_pnl_usd"] == 12345.67
            assert row["trade_size"] == 3
            assert row["paused"] == 0
            assert row["state"] == "active"
            persona = json.loads(row["persona_json"])
            assert persona["style"] == "aggressive"
        finally:
            _run(db.close())

    def test_upsert_vault_idempotent(self, tmp_path: Path) -> None:
        db = self._make_db(tmp_path)
        try:
            _run(db.upsert_vault(SAMPLE_VAULT_LEADERBOARD, SAMPLE_VAULT_CONFIG))
            _run(db.upsert_vault(SAMPLE_VAULT_LEADERBOARD, SAMPLE_VAULT_CONFIG))
            rows = self._query_sync(db.db_path, "SELECT * FROM vaults")
            assert len(rows) == 1
        finally:
            _run(db.close())

    def test_upsert_strategies(self, tmp_path: Path) -> None:
        db = self._make_db(tmp_path)
        try:
            _run(db.upsert_vault(SAMPLE_VAULT_LEADERBOARD, SAMPLE_VAULT_CONFIG))
            strategies = [
                {
                    "strategyId": "1",
                    "vaultOwnerAddress": "0xowner1",
                    "content": "Buy low sell high",
                    "expiry": 0,
                    "enabled": True,
                    "strategyPriority": "high",
                    "createdBlock": 100,
                    "updatedBlock": 200,
                },
                {
                    "strategyId": "2",
                    "vaultOwnerAddress": "0xowner1",
                    "content": "HODL everything",
                    "expiry": 0,
                    "enabled": False,
                    "strategyPriority": "low",
                    "createdBlock": 100,
                    "updatedBlock": 150,
                },
            ]
            _run(db.upsert_strategies("0xabc123", strategies))
            rows = self._query_sync(db.db_path, "SELECT * FROM strategies ORDER BY strategy_id")
            assert len(rows) == 2
            assert rows[0]["content"] == "Buy low sell high"
            assert rows[0]["enabled"] == 1
            assert rows[1]["content"] == "HODL everything"
            assert rows[1]["enabled"] == 0
        finally:
            _run(db.close())

    def test_upsert_inference_logs(self, tmp_path: Path) -> None:
        db = self._make_db(tmp_path)
        try:
            _run(db.upsert_vault(SAMPLE_VAULT_LEADERBOARD, SAMPLE_VAULT_CONFIG))
            _run(db.upsert_inference_logs([SAMPLE_INFERENCE_LOG]))
            rows = self._query_sync(db.db_path, "SELECT * FROM inference_logs")
            assert len(rows) == 1
            row = rows[0]
            assert row["id"] == 1001
            assert row["vault_address"] == "0xabc123"
            assert row["tool"] == "buy_token"
            args = json.loads(row["tool_args_json"])
            assert args["token"] == "ETH"
            assert row["transaction_hash"] == "0xtx001"
        finally:
            _run(db.close())

    def test_upsert_swaps(self, tmp_path: Path) -> None:
        db = self._make_db(tmp_path)
        try:
            _run(db.upsert_vault(SAMPLE_VAULT_LEADERBOARD, SAMPLE_VAULT_CONFIG))
            _run(db.upsert_swaps([SAMPLE_SWAP]))
            rows = self._query_sync(db.db_path, "SELECT * FROM swaps")
            assert len(rows) == 1
            row = rows[0]
            assert row["transaction_hash"] == "0xtx001"
            assert row["side"] == "buy"
            assert row["token_symbol"] == "TT"
            assert row["log_id"] == 1001
        finally:
            _run(db.close())

    def test_upsert_full_log(self, tmp_path: Path) -> None:
        db = self._make_db(tmp_path)
        try:
            _run(db.upsert_vault(SAMPLE_VAULT_LEADERBOARD, SAMPLE_VAULT_CONFIG))
            _run(db.upsert_inference_logs([SAMPLE_INFERENCE_LOG]))

            record = FullLogRecord(
                log_id=1001,
                vault_address="0xabc123",
                payload_meta=RawPayloadMetadata(
                    payload_path="/tmp/test/1001.json.gz",
                    payload_sha256="abc123hash",
                    payload_size_bytes=4096,
                ),
                parsed=ParsedFullLog(
                    prompt_text="[system]\nYou are a trading agent.",
                    completion_text="Buy ETH.",
                    reasoning_content="ETH is strong.",
                    tool_calls_json=None,
                    llm_model="qwen3-8b",
                    prompt_tokens=100,
                    completion_tokens=50,
                    reasoning_tokens=20,
                    total_tokens=170,
                    parse_error=None,
                ),
            )
            _run(db.upsert_full_log(record))

            rows = self._query_sync(db.db_path, "SELECT * FROM full_logs")
            assert len(rows) == 1
            row = rows[0]
            assert row["log_id"] == 1001
            assert row["vault_address"] == "0xabc123"
            assert row["payload_sha256"] == "abc123hash"
            assert row["prompt_tokens"] == 100
            assert row["llm_model"] == "qwen3-8b"
            assert row["parse_error"] is None
        finally:
            _run(db.close())

    def test_fetch_existing_full_log_ids(self, tmp_path: Path) -> None:
        db = self._make_db(tmp_path)
        try:
            _run(db.upsert_vault(SAMPLE_VAULT_LEADERBOARD, SAMPLE_VAULT_CONFIG))
            _run(db.upsert_inference_logs([SAMPLE_INFERENCE_LOG]))

            record = FullLogRecord(
                log_id=1001,
                vault_address="0xabc123",
                payload_meta=RawPayloadMetadata(
                    payload_path="/tmp/test.gz",
                    payload_sha256="hash",
                    payload_size_bytes=100,
                ),
                parsed=ParsedFullLog(
                    prompt_text=None, completion_text=None,
                    reasoning_content=None, tool_calls_json=None,
                    llm_model=None, prompt_tokens=None,
                    completion_tokens=None, reasoning_tokens=None,
                    total_tokens=None, parse_error=None,
                ),
            )
            _run(db.upsert_full_log(record))

            existing = _run(db.fetch_existing_full_log_ids([1001, 1002, 1003]))
            assert existing == {1001}

            empty = _run(db.fetch_existing_full_log_ids([]))
            assert empty == set()
        finally:
            _run(db.close())

    def test_upsert_empty_lists(self, tmp_path: Path) -> None:
        """Upserting empty lists should not raise."""
        db = self._make_db(tmp_path)
        try:
            _run(db.upsert_strategies("0xabc123", []))
            _run(db.upsert_inference_logs([]))
            _run(db.upsert_swaps([]))
        finally:
            _run(db.close())


# ---------------------------------------------------------------------------
# _as_bool_int tests
# ---------------------------------------------------------------------------


class TestAsBoolInt:
    def test_none(self) -> None:
        assert _as_bool_int(None) is None

    def test_true(self) -> None:
        assert _as_bool_int(True) == 1

    def test_false(self) -> None:
        assert _as_bool_int(False) == 0

    def test_truthy_string(self) -> None:
        assert _as_bool_int("yes") == 1

    def test_zero(self) -> None:
        assert _as_bool_int(0) == 0


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
        assert args.db_path == Path("data/terminal_ingest.db")
        assert args.raw_payload_dir == Path("data/full_logs")
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
            "--db-path", "/tmp/test.db",
            "--raw-payload-dir", "/tmp/payloads",
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
        assert args.db_path == Path("/tmp/test.db")
        assert args.top_n == 10
        assert args.request_limit == 25
        assert args.max_logs_per_vault == 100
        assert args.max_full_logs_per_vault == 50
        assert args.max_swaps_per_vault == 200
        assert args.exclude_reasoning is True
