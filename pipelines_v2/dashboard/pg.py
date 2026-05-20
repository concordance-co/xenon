"""Dashboard-owned Postgres fast paths.

Core `pipelines_v2.storage.postgres.PostgresCatalog` uses one fresh
`psycopg.connect(url)` per method call. On a remote DB that's ~50-200 ms
of connect+auth overhead dominated by TCP/TLS setup — murder for the
/runs page, which previously emitted 1+N step-list queries.

This module owns its own `psycopg_pool.ConnectionPool` for the duration
of the FastAPI process and provides batch queries tailored to what the
dashboard actually needs:

- `step_status_counts(run_ids)` — one aggregated SELECT returning
  per-run status histograms, replacing N × list_workflow_steps on
  /api/runs.
- `batch_load_artifacts(artifact_ids)` — one SELECT returning all
  requested artifact manifests, replacing N × find_artifact_for_workflow_step
  on /api/runs/{id}/steps-detail.

When the dashboard isn't attached to Postgres (or when psycopg_pool isn't
importable), `build_pg()` returns None and callers fall back to the
cached composite catalog. Nothing here touches core storage code.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
from typing import Any, Iterable

from pipelines_v2.storage.artifacts import ArtifactManifest

logger = logging.getLogger(__name__)


class DashboardPg:
    """Owns a psycopg connection pool and exposes dashboard-specific
    aggregated queries."""

    def __init__(self, conninfo: str, *, min_size: int = 1, max_size: int = 6) -> None:
        from psycopg_pool import ConnectionPool  # lazy import

        # `open=True` eagerly opens the pool so the first request doesn't pay
        # the handshake. Kwargs keep min/max small — the dashboard is a single
        # process with low concurrency.
        self._pool = ConnectionPool(
            conninfo=conninfo,
            min_size=min_size,
            max_size=max_size,
            kwargs={"autocommit": True, "connect_timeout": 5},
            open=True,
            check=ConnectionPool.check_connection,
            name="pipelines_v2_dashboard",
            timeout=5.0,
            max_idle=300.0,
            reconnect_timeout=30.0,
        )

    def close(self) -> None:
        try:
            self._pool.close()
        except Exception:  # pragma: no cover - defensive close
            logger.exception("dashboard pg pool close failed")

    # ------------------------------------------------------------------
    # Aggregated reads
    # ------------------------------------------------------------------

    @contextmanager
    def _connection(self):
        try:
            with self._pool.connection() as conn:
                yield conn
                return
        except Exception as exc:
            if not _is_retryable_connection_error(exc):
                raise
            logger.warning("dashboard pg transient connection failure; retrying once: %s", exc)
        with self._pool.connection() as conn:
            yield conn

    def _fetchall(self, sql: str, params: tuple[Any, ...]) -> list[Any]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return list(cur.fetchall())

    def _fetchone(self, sql: str, params: tuple[Any, ...]) -> Any | None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchone()

    def step_status_counts(
        self, run_ids: Iterable[str]
    ) -> dict[str, dict[str, int]]:
        """Return `{run_id: {status: count, ...}, ...}` for the requested runs.

        Runs that have no rows in `pipelines_v2_workflow_steps` are absent
        from the result — callers should treat that as "unknown / use
        fallback" rather than "zero".
        """
        ids = [str(rid) for rid in run_ids]
        if not ids:
            return {}
        sql = """
            SELECT run_id, status, COUNT(*)::int
            FROM pipelines_v2_workflow_steps
            WHERE run_id = ANY(%s)
            GROUP BY run_id, status
        """
        out: dict[str, dict[str, int]] = {}
        for run_id, status, n in self._fetchall(sql, (ids,)):
            out.setdefault(run_id, {})[str(status)] = int(n)
        return out

    def batch_load_artifacts(
        self, artifact_ids: Iterable[str]
    ) -> dict[str, ArtifactManifest]:
        """Return `{artifact_id: ArtifactManifest}` for the requested ids.

        Missing ids are absent from the result. One round-trip regardless
        of how many ids are requested.
        """
        ids = [aid for aid in (str(x) for x in artifact_ids) if aid]
        if not ids:
            return {}
        sql = """
            SELECT artifact_id, manifest
            FROM pipelines_v2_artifacts
            WHERE artifact_id = ANY(%s)
        """
        out: dict[str, ArtifactManifest] = {}
        for artifact_id, manifest in self._fetchall(sql, (ids,)):
            out[str(artifact_id)] = ArtifactManifest.from_dict(manifest)
        return out

    # ------------------------------------------------------------------
    # Single-row reads via the pool (drop-in replacements for
    # PostgresCatalog's per-call `psycopg.connect()` methods)
    # ------------------------------------------------------------------

    def load_workflow_run_light(self, run_id: str):
        """Load a workflow run with the payload stripped of embedded dataset
        examples — the multi-MB part. Keeps spec.kind, depends_on, sites,
        tokens, labels, and all other fields the dashboard needs for the DAG,
        overview, and spec summary. Transfers ~50KB instead of ~50MB.

        Dataset/prompt previews use the full payload via separate endpoints.
        """
        from pipelines_v2.workflow.records import WorkflowRunRecord

        sql = """
            SELECT
                run_id, workflow_name, workflow_hash, workflow_spec_hash,
                status, started_at, parent_run_id, finished_at, error,
                jsonb_set(
                    workflow_payload,
                    '{steps}',
                    COALESCE(
                        (SELECT jsonb_agg(
                            CASE
                                WHEN step->'spec'->'dataset'->'examples' IS NOT NULL
                                THEN jsonb_set(
                                    step,
                                    '{spec,dataset}',
                                    (step->'spec'->'dataset') - 'examples'
                                )
                                ELSE step
                            END
                        ) FROM jsonb_array_elements(
                            workflow_payload->'steps'
                        ) AS step),
                        '[]'::jsonb
                    )
                ) AS workflow_payload
            FROM pipelines_v2_workflow_runs
            WHERE run_id = %s
        """
        row = self._fetchone(sql, (run_id,))
        if row is None:
            return None
        return WorkflowRunRecord.from_dict(
            {
                "run_id": row[0],
                "workflow_name": row[1],
                "workflow_hash": row[2],
                "workflow_spec_hash": row[3],
                "status": row[4],
                "started_at": _iso(row[5]),
                "parent_run_id": row[6],
                "finished_at": _iso(row[7]),
                "error": row[8],
                "workflow_payload": row[9],
            }
        )

    def load_workflow_run(self, run_id: str):
        from pipelines_v2.workflow.records import WorkflowRunRecord

        sql = """
            SELECT
                run_id, workflow_name, workflow_hash, workflow_spec_hash,
                status, started_at, parent_run_id, finished_at, error,
                workflow_payload
            FROM pipelines_v2_workflow_runs
            WHERE run_id = %s
        """
        row = self._fetchone(sql, (run_id,))
        if row is None:
            return None
        return WorkflowRunRecord.from_dict(
            {
                "run_id": row[0],
                "workflow_name": row[1],
                "workflow_hash": row[2],
                "workflow_spec_hash": row[3],
                "status": row[4],
                "started_at": _iso(row[5]),
                "parent_run_id": row[6],
                "finished_at": _iso(row[7]),
                "error": row[8],
                "workflow_payload": row[9],
            }
        )

    def list_workflow_steps(self, run_id: str):
        from pipelines_v2.workflow.records import WorkflowStepRecord

        sql = """
            SELECT
                run_id, workflow_hash, workflow_step_key, step_name,
                step_index, runner, status, step_semantic_hash,
                step_spec_hash, input_artifact_refs, artifact_id,
                artifact_kind, started_at, finished_at, runtime_app_id,
                reused_from_run_id, reused_from_artifact_id
            FROM pipelines_v2_workflow_steps
            WHERE run_id = %s
            ORDER BY step_index ASC, step_name ASC
        """
        out = []
        for row in self._fetchall(sql, (run_id,)):
            out.append(
                WorkflowStepRecord.from_dict(
                    {
                        "run_id": row[0],
                        "workflow_hash": row[1],
                        "workflow_step_key": row[2],
                        "step_name": row[3],
                        "step_index": row[4],
                        "runner": row[5],
                        "status": row[6],
                        "step_semantic_hash": row[7],
                        "step_spec_hash": row[8],
                        "input_artifact_refs": row[9],
                        "artifact_id": row[10],
                        "artifact_kind": row[11],
                        "started_at": _iso(row[12]),
                        "finished_at": _iso(row[13]),
                        "runtime_app_id": row[14],
                        "reused_from_run_id": row[15],
                        "reused_from_artifact_id": row[16],
                    }
                )
            )
        return out

    def load_artifact(self, artifact_id: str) -> ArtifactManifest | None:
        sql = "SELECT manifest FROM pipelines_v2_artifacts WHERE artifact_id = %s"
        row = self._fetchone(sql, (artifact_id,))
        if row is None:
            return None
        return ArtifactManifest.from_dict(row[0])

    def find_artifact_for_workflow_step(
        self, *, run_id: str, workflow_step_key: str
    ) -> ArtifactManifest | None:
        sql = """
            WITH target_step AS (
                SELECT workflow_step_id
                FROM pipelines_v2_workflow_steps
                WHERE run_id = %s
                  AND workflow_step_key = %s
                LIMIT 1
            )
            SELECT a.manifest
            FROM pipelines_v2_artifacts a
            LEFT JOIN target_step step ON TRUE
            WHERE (
                step.workflow_step_id IS NOT NULL
                AND a.produced_by_step_id = step.workflow_step_id
            )
            OR (
                a.produced_by_step_id IS NULL
                AND (
                    (
                        a.produced_by_run_id = %s
                        AND a.produced_by_workflow_step_key = %s
                    )
                    OR (
                        a.produced_by_run_id IS NULL
                        AND a.produced_by_workflow_step_key IS NULL
                        AND a.manifest->'workflow_context'->>'run_id' = %s
                        AND a.manifest->'workflow_context'->>'workflow_step_key' = %s
                    )
                )
            )
            ORDER BY created_at DESC
            LIMIT 1
        """
        row = self._fetchone(
            sql,
            (run_id, workflow_step_key, run_id, workflow_step_key, run_id, workflow_step_key),
        )
        if row is None:
            return None
        return ArtifactManifest.from_dict(row[0])

    def list_workflow_runs(
        self,
        *,
        workflow_name: str | None = None,
        workflow_hash: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ):
        """Full list with the workflow_payload attached (for rare callers
        that actually need it). Prefer `list_workflow_runs_light` for the
        /api/runs index page — it avoids transferring multi-MB payloads."""
        from pipelines_v2.workflow.records import WorkflowRunRecord

        predicates, params = self._run_filter_clauses(
            workflow_name=workflow_name, workflow_hash=workflow_hash, status=status
        )
        where = f"WHERE {' AND '.join(predicates)}" if predicates else ""
        limit_clause = "LIMIT %s" if limit is not None else ""
        if limit is not None:
            params.append(int(limit))
        sql = f"""
            SELECT
                run_id, workflow_name, workflow_hash, workflow_spec_hash,
                status, started_at, parent_run_id, finished_at, error,
                workflow_payload
            FROM pipelines_v2_workflow_runs
            {where}
            ORDER BY started_at DESC, run_id DESC
            {limit_clause}
        """
        out = []
        for row in self._fetchall(sql, tuple(params)):
            out.append(
                WorkflowRunRecord.from_dict(
                    {
                        "run_id": row[0],
                        "workflow_name": row[1],
                        "workflow_hash": row[2],
                        "workflow_spec_hash": row[3],
                        "status": row[4],
                        "started_at": _iso(row[5]),
                        "parent_run_id": row[6],
                        "finished_at": _iso(row[7]),
                        "error": row[8],
                        "workflow_payload": row[9],
                    }
                )
            )
        return out

    def list_workflow_runs_light(
        self,
        *,
        workflow_name: str | None = None,
        workflow_hash: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Summary rows for /api/runs.

        Skips `workflow_payload` entirely (it can be multi-MB per run when
        datasets are materialized into the payload). `has_report` is
        computed on the Postgres side, but only after the candidate run set
        has been ordered + limited. This avoids forcing Postgres to inspect
        large workflow payload JSONB blobs for every historical row when the
        UI only needs the newest N runs.
        """
        predicates, params = self._run_filter_clauses(
            workflow_name=workflow_name, workflow_hash=workflow_hash, status=status
        )
        where = f"WHERE {' AND '.join(predicates)}" if predicates else ""
        limit_clause = "LIMIT %s" if limit is not None else ""
        if limit is not None:
            params.append(int(limit))
        sql = f"""
            WITH recent_runs AS MATERIALIZED (
                SELECT
                    run_id,
                    workflow_name,
                    workflow_hash,
                    workflow_spec_hash,
                    status,
                    started_at,
                    parent_run_id,
                    finished_at,
                    error,
                    workflow_payload
                FROM pipelines_v2_workflow_runs
                {where}
                ORDER BY started_at DESC, run_id DESC
                {limit_clause}
            )
            SELECT
                run_id,
                workflow_name,
                workflow_hash,
                workflow_spec_hash,
                status,
                started_at,
                parent_run_id,
                finished_at,
                error,
                COALESCE(
                    jsonb_path_exists(
                        workflow_payload,
                        '$.steps[*] ? (@.spec.kind == "report")'
                    ),
                    FALSE
                ) AS has_report
            FROM recent_runs
            ORDER BY started_at DESC, run_id DESC
        """
        out: list[dict[str, Any]] = []
        for row in self._fetchall(sql, tuple(params)):
            out.append(
                {
                    "run_id": row[0],
                    "workflow_name": row[1],
                    "workflow_hash": row[2],
                    "workflow_spec_hash": row[3],
                    "status": row[4],
                    "started_at": _iso(row[5]),
                    "parent_run_id": row[6],
                    "finished_at": _iso(row[7]),
                    "error": row[8],
                    "has_report": bool(row[9]),
                }
            )
        return out

    @staticmethod
    def _run_filter_clauses(
        *,
        workflow_name: str | None,
        workflow_hash: str | None,
        status: str | None,
    ) -> tuple[list[str], list[Any]]:
        predicates: list[str] = []
        params: list[Any] = []
        if workflow_name is not None:
            predicates.append("workflow_name = %s")
            params.append(workflow_name)
        if workflow_hash is not None:
            predicates.append("workflow_hash = %s")
            params.append(workflow_hash)
        if status is not None:
            predicates.append("status = %s")
            params.append(status)
        return predicates, params

    def report_artifact_ids_for_runs(
        self, run_ids: Iterable[str]
    ) -> dict[str, str]:
        """Return `{run_id: artifact_id}` for any run that has a completed
        report step with an artifact. One query for all runs."""
        ids = [str(rid) for rid in run_ids]
        if not ids:
            return {}
        sql = """
            SELECT DISTINCT ON (run_id) run_id, artifact_id
            FROM pipelines_v2_workflow_steps
            WHERE run_id = ANY(%s)
              AND artifact_kind = 'report'
              AND artifact_id IS NOT NULL
            ORDER BY run_id, step_index ASC
        """
        out: dict[str, str] = {}
        for run_id, artifact_id in self._fetchall(sql, (ids,)):
            out[str(run_id)] = str(artifact_id)
        return out

    def find_artifacts_for_run(
        self, run_id: str
    ) -> dict[str, ArtifactManifest]:
        """Return every artifact whose manifest has this run_id in its
        workflow_context, keyed by `workflow_step_key` (most-recent
        artifact per step if multiple exist).
        """
        sql = """
            SELECT
                COALESCE(
                    s.workflow_step_key,
                    a.produced_by_workflow_step_key,
                    a.manifest->'workflow_context'->>'workflow_step_key'
                ) AS key,
                a.manifest,
                a.created_at
            FROM pipelines_v2_artifacts a
            LEFT JOIN pipelines_v2_workflow_steps s
              ON s.workflow_step_id = a.produced_by_step_id
            WHERE s.run_id = %s
               OR (
                    a.produced_by_step_id IS NULL
                    AND a.produced_by_run_id = %s
               )
               OR (
                    a.produced_by_step_id IS NULL
                    AND a.produced_by_run_id IS NULL
                    AND a.manifest->'workflow_context'->>'run_id' = %s
               )
            ORDER BY created_at DESC
        """
        out: dict[str, ArtifactManifest] = {}
        for key, manifest, _created in self._fetchall(sql, (run_id, run_id, run_id)):
            if key is None:
                continue
            out.setdefault(str(key), ArtifactManifest.from_dict(manifest))
        return out

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        try:
            return {
                "pool": {
                    "size": self._pool.get_stats().get("pool_size", 0),
                    "available": self._pool.get_stats().get("pool_available", 0),
                    "requests_queued": self._pool.get_stats().get("requests_queued", 0),
                    "requests_num": self._pool.get_stats().get("requests_num", 0),
                    "requests_wait_ms_avg": self._pool.get_stats().get(
                        "requests_wait_ms", 0
                    ),
                }
            }
        except Exception:
            return {"pool": "unavailable"}


def _iso(value: Any) -> Any:
    """Coerce Postgres timestamp rows into ISO strings (matches PostgresCatalog)."""
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _is_retryable_connection_error(exc: Exception) -> bool:
    module = type(exc).__module__
    name = type(exc).__name__
    if module.startswith("psycopg") and name in {"OperationalError", "InterfaceError"}:
        return True
    if module.startswith("psycopg_pool") and name == "PoolTimeout":
        return True
    message = str(exc).lower()
    return "discarding closed connection" in message or "connection is closed" in message


def build_pg(conninfo: str | None) -> DashboardPg | None:
    """Construct a `DashboardPg` when a conninfo is available.

    Returns None when either no Postgres URL is configured or the
    `psycopg_pool` package isn't installed. Never raises on import
    failure — the dashboard should degrade to fallback code paths.
    """
    if not conninfo:
        return None
    try:
        return DashboardPg(conninfo)
    except ImportError as exc:  # psycopg_pool missing
        logger.warning("psycopg_pool not installed; dashboard fast paths disabled: %s", exc)
        return None
    except Exception:
        logger.exception("failed to open dashboard pg pool; falling back to per-call connects")
        return None


def resolve_pg_conninfo(
    *,
    postgres_env: str | None,
    postgres_catalog: Any | None,
) -> str | None:
    """Resolve a Postgres connection string.

    Prefers pulling it directly from the already-built `PostgresCatalog` (so
    the dashboard uses the exact same URL as execution), otherwise falls back
    to reading the env var directly.
    """
    if postgres_catalog is not None:
        source = getattr(postgres_catalog, "source", None)
        if source is not None:
            # PostgresSource may expose `connection_url` or `_resolved_url`.
            for attr in ("connection_url", "_resolved_url"):
                fn = getattr(source, attr, None)
                if callable(fn):
                    try:
                        url = fn()
                        if url:
                            return str(url)
                    except Exception:
                        logger.exception("failed to resolve pg conninfo via %s", attr)
    if postgres_env:
        import os

        value = os.environ.get(postgres_env)
        if value:
            return str(value)
    return None
