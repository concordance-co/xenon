from __future__ import annotations

import asyncio

from pipelines.interp.modal_analysis import _resolve_blocking_result


def test_resolve_blocking_result_returns_plain_value():
    value = [{"processed": 3}]
    assert _resolve_blocking_result(value) == value


def test_resolve_blocking_result_runs_awaitable():
    async def _coro():
        return {"ok": True}

    assert _resolve_blocking_result(_coro()) == {"ok": True}
