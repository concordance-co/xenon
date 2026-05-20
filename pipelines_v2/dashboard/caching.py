"""Dashboard-local caching wrapper around a read catalog.

Wraps a `CompositeCatalog` (or any catalog implementing the subset of methods
the dashboard reads) with a TTL-tiered in-memory cache so repeat requests
don't round-trip to Postgres. Lives entirely in `pipelines_v2.dashboard` —
core `pipelines_v2.storage` code is untouched.

Caching strategy
----------------
- `list_workflow_runs(...)` → short TTL (`list_ttl`). New runs can appear.
- `load_workflow_run(run_id)` → TTL depends on status: terminal runs
  (completed / failed / cancelled) are effectively immutable, cached for
  `cold_ttl` seconds. Running / pending runs are cached for `hot_ttl`.
- `list_workflow_steps(run_id)` → same tiered TTL based on the run's status
  (looked up through the same cache).
- `load_artifact(artifact_id)` → always cached for `cold_ttl` (artifact
  manifests are write-once).
- `find_artifact_for_workflow_step(...)` → hit cached for `cold_ttl`; miss
  cached briefly for `hot_ttl` so we don't re-query Postgres every request
  for an in-flight step.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from pipelines_v2.storage.artifacts import ArtifactManifest
from pipelines_v2.workflow.records import WorkflowRunRecord, WorkflowStepRecord


TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})

_MISS = object()  # sentinel distinct from None so cached-None is a real hit


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "invalidations": self.invalidations,
        }


@dataclass
class _TTLCache:
    """Small LRU with per-entry TTLs. Thread-safe, no external deps."""

    maxsize: int = 2048
    _data: "OrderedDict[Any, tuple[float, Any]]" = field(
        default_factory=OrderedDict, repr=False
    )
    _lock: Lock = field(default_factory=Lock, repr=False)
    stats: CacheStats = field(default_factory=CacheStats)

    def get(self, key: Any) -> Any:
        """Return the cached value, or the `_MISS` sentinel if absent/expired.

        Callers must check `value is _MISS` — the cached value may itself be
        `None` (used to remember a negative lookup).
        """
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.stats.misses += 1
                return _MISS
            expiry, value = entry
            if expiry < now:
                self._data.pop(key, None)
                self.stats.misses += 1
                return _MISS
            self._data.move_to_end(key)
            self.stats.hits += 1
            return value

    def put(self, key: Any, value: Any, ttl: float) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)
                self.stats.evictions += 1

    def pop(self, key: Any) -> None:
        with self._lock:
            if self._data.pop(key, None) is not None:
                self.stats.invalidations += 1

    def clear(self) -> None:
        with self._lock:
            self.stats.invalidations += len(self._data)
            self._data.clear()


def _merge_runs(primary, fallback, limit):
    """Dedupe-by-run_id + newest-first merge used when combining pg with
    a local FileCatalog listing."""
    seen = {r.run_id for r in primary}
    merged = list(primary)
    for r in fallback:
        if r.run_id in seen:
            continue
        seen.add(r.run_id)
        merged.append(r)
    merged.sort(key=lambda r: (r.started_at, r.run_id), reverse=True)
    if limit is not None:
        return merged[:limit]
    return merged


class CachedCatalog:
    """Read-through cache over a catalog.

    Only the read methods the dashboard uses are wrapped. Writes (which the
    dashboard never performs) invalidate related entries if called.
    """

    def __init__(
        self,
        inner: Any,
        *,
        pg: Any = None,
        local: Any = None,
        list_ttl: float = 15.0,
        hot_ttl: float = 15.0,
        cold_ttl: float = 3600.0,
        miss_ttl: float = 15.0,
        maxsize: int = 2048,
    ) -> None:
        self._inner = inner
        # Pooled pg path: when set, single-row reads prefer it over the
        # composite. The composite still backs writes and any method pg
        # doesn't implement.
        self._pg = pg
        # Local-only fallback: when pg is attached and returns nothing, we
        # don't want to re-hit Postgres through the composite. A direct
        # FileCatalog lookup here keeps the miss path cheap.
        self._local = local
        self._list_ttl = list_ttl
        self._hot_ttl = hot_ttl
        self._cold_ttl = cold_ttl
        self._miss_ttl = miss_ttl
        self._cache = _TTLCache(maxsize=maxsize)

    # --- passthrough helpers ------------------------------------------------

    @property
    def inner(self) -> Any:
        return self._inner

    def identity(self) -> dict[str, Any]:
        base = {
            "kind": "cached",
            "list_ttl": self._list_ttl,
            "hot_ttl": self._hot_ttl,
            "cold_ttl": self._cold_ttl,
            "stats": self._cache.stats.to_dict(),
        }
        if hasattr(self._inner, "identity"):
            base["inner"] = self._inner.identity()
        return base

    def invalidate_all(self) -> None:
        self._cache.clear()

    def stats(self) -> dict[str, int]:
        return self._cache.stats.to_dict()

    # --- read methods used by the dashboard --------------------------------

    def list_workflow_runs(
        self,
        *,
        workflow_name: str | None = None,
        workflow_hash: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[WorkflowRunRecord]:
        key = ("list_workflow_runs", workflow_name, workflow_hash, status, limit)
        cached = self._cache.get(key)
        if cached is not _MISS:
            return cached
        # When pg is attached, merge pooled Postgres + local FileCatalog
        # directly so we don't open a fresh psycopg connection via the
        # composite's PostgresCatalog child.
        if self._pg is not None:
            try:
                pg_runs = self._pg.list_workflow_runs(
                    workflow_name=workflow_name,
                    workflow_hash=workflow_hash,
                    status=status,
                    limit=limit,
                )
            except Exception:
                pg_runs = None
            if pg_runs is not None:
                local_runs = []
                if self._local is not None:
                    try:
                        local_runs = self._local.list_workflow_runs(
                            workflow_name=workflow_name,
                            workflow_hash=workflow_hash,
                            status=status,
                            limit=limit,
                        )
                    except Exception:
                        local_runs = []
                value = _merge_runs(pg_runs, local_runs, limit)
            else:
                value = self._inner.list_workflow_runs(
                    workflow_name=workflow_name,
                    workflow_hash=workflow_hash,
                    status=status,
                    limit=limit,
                )
        else:
            value = self._inner.list_workflow_runs(
                workflow_name=workflow_name,
                workflow_hash=workflow_hash,
                status=status,
                limit=limit,
            )
        self._cache.put(key, value, self._list_ttl)
        for record in value:
            self._cache.put(
                ("load_workflow_run", record.run_id),
                record,
                self._ttl_for_run_status(record.status),
            )
        return value

    def load_workflow_run(self, run_id: str) -> WorkflowRunRecord | None:
        key = ("load_workflow_run", run_id)
        cached = self._cache.get(key)
        if cached is not _MISS:
            return cached
        # Try local first — disk reads are ~50ms vs ~13s from Neon for runs
        # with large embedded datasets. Also: after report generation, the
        # run is mirrored locally and local is the most up-to-date source.
        value = None
        if self._local is not None:
            value = self._local.load_workflow_run(run_id)
        if value is None and self._pg is not None:
            try:
                value = self._pg.load_workflow_run(run_id)
            except Exception:
                value = None
        if value is None:
            value = self._inner.load_workflow_run(run_id)
        ttl = self._ttl_for_run_status(value.status) if value is not None else self._miss_ttl
        self._cache.put(key, value, ttl)
        return value

    def list_workflow_steps(self, run_id: str) -> list[WorkflowStepRecord]:
        key = ("list_workflow_steps", run_id)
        cached = self._cache.get(key)
        if cached is not _MISS:
            return cached
        value: list[WorkflowStepRecord] = []
        # Try local FIRST — after report generation the dashboard mirrors
        # step records locally with updated artifact_ids. The local version
        # is strictly more up-to-date than Postgres for any mirrored run.
        if self._local is not None:
            try:
                value = self._local.list_workflow_steps(run_id)
            except Exception:
                value = []
        if not value and self._pg is not None:
            try:
                value = self._pg.list_workflow_steps(run_id)
            except Exception:
                value = []
        if not value:
            value = self._inner.list_workflow_steps(run_id)
        run = self._cache.get(("load_workflow_run", run_id))
        if run is _MISS or run is None:
            ttl = self._hot_ttl
        else:
            ttl = self._ttl_for_run_status(run.status)
        self._cache.put(key, value, ttl)
        return value

    def load_artifact(self, artifact_id: str) -> ArtifactManifest | None:
        key = ("load_artifact", artifact_id)
        cached = self._cache.get(key)
        if cached is not _MISS:
            return cached
        # Local first — generated report artifacts are only in the local
        # FileCatalog, not in Postgres.
        value = None
        if self._local is not None:
            value = self._local.load_artifact(artifact_id)
        if value is None and self._pg is not None:
            try:
                value = self._pg.load_artifact(artifact_id)
            except Exception:
                value = None
        if value is None:
            value = self._inner.load_artifact(artifact_id)
        ttl = self._cold_ttl if value is not None else self._miss_ttl
        self._cache.put(key, value, ttl)
        return value

    def find_artifact_for_workflow_step(
        self,
        *,
        run_id: str,
        workflow_step_key: str,
    ) -> ArtifactManifest | None:
        key = ("find_artifact_for_workflow_step", run_id, workflow_step_key)
        cached = self._cache.get(key)
        if cached is not _MISS:
            return cached
        value = None
        if self._pg is not None:
            try:
                value = self._pg.find_artifact_for_workflow_step(
                    run_id=run_id, workflow_step_key=workflow_step_key
                )
            except Exception:
                value = None
        if value is None and self._local is not None:
            finder = getattr(self._local, "find_artifact_for_workflow_step", None)
            if callable(finder):
                try:
                    value = finder(run_id=run_id, workflow_step_key=workflow_step_key)
                except Exception:
                    value = None
        if value is None and self._pg is None:
            value = self._inner.find_artifact_for_workflow_step(
                run_id=run_id, workflow_step_key=workflow_step_key
            )
        if value is None:
            self._cache.put(key, value, self._miss_ttl)
        else:
            self._cache.put(key, value, self._cold_ttl)
            self._cache.put(
                ("load_artifact", value.artifact_id), value, self._cold_ttl
            )
        return value

    # Pass-through for any catalog method we don't wrap (defensive — the
    # dashboard shouldn't be calling writes, but this keeps the duck-type).
    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    # --- helpers -----------------------------------------------------------

    def _ttl_for_run_status(self, status: str | None) -> float:
        if status and status.lower() in TERMINAL_STATUSES:
            return self._cold_ttl
        return self._hot_ttl
