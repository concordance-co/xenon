from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import aiohttp


class TerminalApiError(RuntimeError):
    pass


@dataclass(slots=True)
class RetryPolicy:
    max_attempts: int = 6
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 60.0


class TerminalMarketsApiClient:
    def __init__(
        self,
        base_url: str,
        concurrency: int = 10,
        timeout_s: int = 30,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.retry_policy = retry_policy or RetryPolicy()
        self.semaphore = asyncio.Semaphore(concurrency)
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
        assert self.session is not None
        url = f"{self.base_url}{path}"
        delay = self.retry_policy.initial_backoff_s

        for attempt in range(1, self.retry_policy.max_attempts + 1):
            try:
                async with self.semaphore:
                    async with self.session.get(url, params=params) as response:
                        if response.status in {429, 500, 502, 503, 504}:
                            body = await response.text()
                            if attempt == self.retry_policy.max_attempts:
                                raise TerminalApiError(
                                    f"API request failed after retries: {url} status={response.status} body={body}"
                                )
                            await asyncio.sleep(delay)
                            delay = min(delay * 2, self.retry_policy.max_backoff_s)
                            continue

                        if response.status >= 400:
                            body = await response.text()
                            raise TerminalApiError(f"API request failed: {url} status={response.status} body={body}")

                        return await response.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt == self.retry_policy.max_attempts:
                    raise TerminalApiError(f"API request failed after retries: {url} error={exc}") from exc
                await asyncio.sleep(delay)
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

