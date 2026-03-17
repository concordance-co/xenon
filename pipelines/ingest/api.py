from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class TerminalApiError(RuntimeError):
    pass


@dataclass(slots=True)
class RetryPolicy:
    max_attempts: int = 6
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 60.0


class _RateLimiter:
    """Token-bucket rate limiter: allows *rate* requests per second."""

    def __init__(self, rate: float) -> None:
        self._rate = rate
        self._min_interval = 1.0 / rate
        self._lock = asyncio.Lock()
        self._last: float = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._min_interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = asyncio.get_event_loop().time()


class TerminalMarketsApiClient:
    def __init__(
        self,
        base_url: str,
        concurrency: int = 10,
        requests_per_second: float = 2.0,
        timeout_s: int = 30,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.retry_policy = retry_policy or RetryPolicy()
        self.semaphore = asyncio.Semaphore(concurrency)
        self._rate_limiter = _RateLimiter(requests_per_second)
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "TerminalMarketsApiClient":
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers={"User-Agent": "xenon-terminal-ingest/0.1"},
        )
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.session is not None:
            await self.session.close()
            self.session = None

    async def request_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | list[Any]:
        if self.session is None:
            raise RuntimeError("API client not initialized — use as async context manager")
        url = f"{self.base_url}{path}"
        delay = self.retry_policy.initial_backoff_s

        for attempt in range(1, self.retry_policy.max_attempts + 1):
            try:
                await self._rate_limiter.acquire()
                async with self.semaphore:
                    async with self.session.get(url, params=params) as response:
                        if response.status in {429, 500, 502, 503, 504}:
                            body = await response.text()
                            status = response.status
                        elif response.status >= 400:
                            body = await response.text()
                            raise TerminalApiError(f"API request failed: {url} status={response.status} body={body}")
                        else:
                            return await response.json()
                # Semaphore released — now handle retryable errors
                if attempt == self.retry_policy.max_attempts:
                    raise TerminalApiError(
                        f"API request failed after retries: {url} status={status} body={body}"
                    )
                # Back off longer on rate limits
                retry_delay = delay * 3 if status == 429 else delay
                jittered_delay = retry_delay * (1 + random.uniform(0, 0.3))
                logger.warning(
                    "Retryable %d on attempt %d/%d for %s, sleeping %.1fs",
                    status, attempt, self.retry_policy.max_attempts, url, jittered_delay,
                )
                await asyncio.sleep(jittered_delay)
                delay = min(delay * 2, self.retry_policy.max_backoff_s)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning(
                    "Connection error on attempt %d/%d for %s: %s",
                    attempt, self.retry_policy.max_attempts, url, exc,
                )
                if attempt == self.retry_policy.max_attempts:
                    raise TerminalApiError(f"API request failed after retries: {url} error={exc}") from exc
                jittered_delay = delay * (1 + random.uniform(0, 0.3))
                await asyncio.sleep(jittered_delay)
                delay = min(delay * 2, self.retry_policy.max_backoff_s)

        raise TerminalApiError(f"Unreachable retry state for {url}")

    async def get_leaderboard_page(
        self,
        *,
        limit: int,
        sort_by: str,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit, "sortBy": sort_by}
        if cursor:
            params["cursor"] = cursor
        result = await self.request_json("/leaderboard", params=params)
        if not isinstance(result, dict):
            raise TerminalApiError("Unexpected leaderboard response type")
        return result

    async def get_vault(self, vault_address: str) -> dict[str, Any]:
        result = await self.request_json("/vault", params={"vaultAddress": vault_address})
        if not isinstance(result, dict):
            raise TerminalApiError("Unexpected vault response type")
        return result

    async def get_strategies(self, vault_address: str) -> list[dict[str, Any]]:
        result = await self.request_json(f"/strategies/{vault_address}")
        if not isinstance(result, list):
            raise TerminalApiError("Unexpected strategies response type")
        return [item for item in result if isinstance(item, dict)]

    async def get_logs_page(
        self,
        vault_address: str,
        *,
        limit: int,
        order: str = "asc",
        cursor: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit, "order": order}
        if cursor:
            params["cursor"] = cursor
        result = await self.request_json(f"/logs/{vault_address}", params=params)
        if not isinstance(result, dict):
            raise TerminalApiError("Unexpected logs response type")
        return result

    async def get_full_log(self, log_id: int) -> dict[str, Any]:
        result = await self.request_json(f"/full-log/{log_id}")
        if not isinstance(result, dict):
            raise TerminalApiError("Unexpected full-log response type")
        return result

    async def get_swaps_page(
        self,
        vault_address: str,
        *,
        limit: int,
        order: str = "asc",
        cursor: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "vaultAddress": vault_address,
            "limit": limit,
            "order": order,
        }
        if cursor:
            params["cursor"] = cursor
        result = await self.request_json("/swaps", params=params)
        if not isinstance(result, dict):
            raise TerminalApiError("Unexpected swaps response type")
        return result

    async def get_candles(
        self,
        token_address: str,
        *,
        timeframe: str = "1h",
        from_ts: int | None = None,
        to_ts: int | None = None,
        countback: int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"timeframe": timeframe}
        if to_ts is not None:
            params["to"] = to_ts
        if from_ts is not None:
            params["from"] = from_ts
        if countback is not None:
            params["countback"] = countback
        result = await self.request_json(f"/candles/{token_address}", params=params)
        if not isinstance(result, dict):
            raise TerminalApiError("Unexpected candles response type")
        return result

