"""Dashboard catalog wiring.

Reuses the same env-var convention as `pipelines_v2 workflow run`: an optional
`--catalog-postgres-env` flag names an environment variable whose value is a
Postgres connection spec. The result is a `CompositeCatalog` with the local
file catalog first and the optional Postgres catalog second, matching the
ordering used during workflow execution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipelines_v2.core.config import load_workspace_config
from pipelines_v2.core.paths import pipelines_v2_catalog_root
from pipelines_v2.storage.composite import CompositeCatalog
from pipelines_v2.storage.local import FileCatalog
from pipelines_v2.storage.postgres import PostgresCatalog
from pipelines_v2.data.sources import PostgresSource
from pipelines_v2.dashboard.caching import CachedCatalog
from pipelines_v2.dashboard.pg import DashboardPg, build_pg, resolve_pg_conninfo


@dataclass(frozen=True, slots=True)
class DashboardCatalog:
    """Merged catalog view used by all dashboard endpoints.

    `composite` is the publicly-exposed read catalog — it's a `CachedCatalog`
    wrapping the underlying `CompositeCatalog` so repeat Postgres reads are
    coalesced. `raw` is the unwrapped catalog (used for tests + cache
    invalidation diagnostics). `local` is the file catalog underlying both.
    `pg` is an optional dashboard-owned connection pool that powers the
    aggregated fast paths; None when no Postgres catalog is attached.
    """

    local: FileCatalog
    composite: Any            # CachedCatalog in production; a raw catalog in tests
    raw: CompositeCatalog
    local_root: Path
    postgres_env: str | None
    pg: DashboardPg | None = None

    def identity(self) -> dict[str, Any]:
        payload = {
            "local_root": str(self.local_root),
            "postgres_env": self.postgres_env,
            "composite": self.composite.identity(),
        }
        if self.pg is not None:
            payload["pg"] = self.pg.stats()
        return payload

    def close(self) -> None:
        if self.pg is not None:
            self.pg.close()


def build_catalog(
    *,
    local_root: Path | str | None = None,
    catalog_postgres_env: str | None = None,
    list_ttl: float = 15.0,
    hot_ttl: float = 15.0,
    cold_ttl: float = 3600.0,
    cache_maxsize: int = 2048,
) -> DashboardCatalog:
    """Construct the dashboard catalog.

    Parameters mirror the `pipelines_v2 workflow run` CLI flags so the
    dashboard reads exactly the same state as execution. The read view is
    wrapped in a TTL cache to protect Postgres from repeat queries — tune
    via the `*_ttl` kwargs or the CLI flags.
    """
    config = load_workspace_config()
    resolved_local_root = (
        Path(local_root).expanduser()
        if local_root is not None
        else config.dashboard_local_catalog_root() or pipelines_v2_catalog_root()
    )
    resolved_catalog_env = catalog_postgres_env
    if resolved_catalog_env is None:
        configured = config.dashboard_catalog_postgres_env()
        if configured and os.environ.get(configured):
            resolved_catalog_env = configured

    root = resolved_local_root.resolve()
    local = FileCatalog(root=root)
    catalogs: tuple[Any, ...] = (local,)
    postgres_catalog: PostgresCatalog | None = None
    if resolved_catalog_env:
        postgres_catalog = PostgresCatalog(source=PostgresSource.from_env(resolved_catalog_env))
        catalogs = (local, postgres_catalog)
    composite = CompositeCatalog(catalogs=catalogs)
    conninfo = resolve_pg_conninfo(
        postgres_env=resolved_catalog_env,
        postgres_catalog=postgres_catalog,
    )
    pg = build_pg(conninfo)
    cached = CachedCatalog(
        composite,
        pg=pg,
        local=local,
        list_ttl=list_ttl,
        hot_ttl=hot_ttl,
        cold_ttl=cold_ttl,
        maxsize=cache_maxsize,
    )
    return DashboardCatalog(
        local=local,
        composite=cached,
        raw=composite,
        local_root=root,
        postgres_env=resolved_catalog_env,
        pg=pg,
    )
