"""Postgres-backed catalog descriptor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipelines_v2.data.sources import PostgresSource
from pipelines_v2.storage.artifacts import ArtifactManifest
from pipelines_v2.workflow.records import WorkflowRunRecord, WorkflowStepRecord


@dataclass(frozen=True, slots=True)
class PostgresCatalog:
    """Catalog that upserts artifact manifests into Postgres."""
    source: Any

    kind: str = "postgres"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PostgresCatalog":
        return cls(source=PostgresSource.from_dict(dict(payload["source"])))

    def identity(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source": self.source.identity(),
        }

    def record_artifact(self, manifest: ArtifactManifest) -> None:
        import json

        import psycopg

        payload = manifest.to_dict()
        with psycopg.connect(self.source.connection_url()) as conn:
            with conn.cursor() as cur:
                self._ensure_schema(cur)
                cur.execute(
                    """
                    INSERT INTO pipelines_v2_artifacts (
                        artifact_id,
                        artifact_kind,
                        operation_spec_hash,
                        operation_semantic_hash,
                        created_at,
                        manifest
                    ) VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (artifact_id) DO UPDATE SET
                        artifact_kind = EXCLUDED.artifact_kind,
                        operation_spec_hash = EXCLUDED.operation_spec_hash,
                        operation_semantic_hash = EXCLUDED.operation_semantic_hash,
                        created_at = EXCLUDED.created_at,
                        manifest = EXCLUDED.manifest
                    """,
                    (
                        manifest.artifact_id,
                        manifest.artifact_kind,
                        manifest.operation_spec_hash,
                        manifest.operation_semantic_hash,
                        manifest.created_at,
                        json.dumps(payload, sort_keys=True),
                    ),
                )
            conn.commit()

    def load_artifact(self, artifact_id: str) -> ArtifactManifest | None:
        import psycopg

        with psycopg.connect(self.source.connection_url()) as conn:
            with conn.cursor() as cur:
                self._ensure_schema(cur)
                cur.execute(
                    "SELECT manifest FROM pipelines_v2_artifacts WHERE artifact_id = %s",
                    (artifact_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return ArtifactManifest.from_dict(row[0])

    def find_artifact_for_workflow_step(
        self,
        *,
        run_id: str,
        workflow_step_key: str,
    ) -> ArtifactManifest | None:
        import psycopg

        with psycopg.connect(self.source.connection_url()) as conn:
            with conn.cursor() as cur:
                self._ensure_schema(cur)
                cur.execute(
                    """
                    SELECT manifest
                    FROM pipelines_v2_artifacts
                    WHERE manifest->'workflow_context'->>'run_id' = %s
                      AND manifest->'workflow_context'->>'workflow_step_key' = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (run_id, workflow_step_key),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return ArtifactManifest.from_dict(row[0])

    def record_workflow_run(self, record: WorkflowRunRecord) -> None:
        import json

        import psycopg

        with psycopg.connect(self.source.connection_url()) as conn:
            with conn.cursor() as cur:
                self._ensure_schema(cur)
                cur.execute(
                    """
                    INSERT INTO pipelines_v2_workflow_runs (
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
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (run_id) DO UPDATE SET
                        workflow_name = EXCLUDED.workflow_name,
                        workflow_hash = EXCLUDED.workflow_hash,
                        workflow_spec_hash = EXCLUDED.workflow_spec_hash,
                        status = EXCLUDED.status,
                        started_at = EXCLUDED.started_at,
                        parent_run_id = EXCLUDED.parent_run_id,
                        finished_at = EXCLUDED.finished_at,
                        error = EXCLUDED.error,
                        workflow_payload = EXCLUDED.workflow_payload
                    """,
                    (
                        record.run_id,
                        record.workflow_name,
                        record.workflow_hash,
                        record.workflow_spec_hash,
                        record.status,
                        record.started_at,
                        record.parent_run_id,
                        record.finished_at,
                        record.error,
                        json.dumps(record.to_dict()["workflow_payload"], sort_keys=True),
                    ),
                )
            conn.commit()

    def load_workflow_run(self, run_id: str) -> WorkflowRunRecord | None:
        import psycopg

        with psycopg.connect(self.source.connection_url()) as conn:
            with conn.cursor() as cur:
                self._ensure_schema(cur)
                cur.execute(
                    """
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
                    WHERE run_id = %s
                    """,
                    (run_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return WorkflowRunRecord.from_dict(
            {
                "run_id": row[0],
                "workflow_name": row[1],
                "workflow_hash": row[2],
                "workflow_spec_hash": row[3],
                "status": row[4],
                "started_at": row[5].isoformat() if hasattr(row[5], "isoformat") else row[5],
                "parent_run_id": row[6],
                "finished_at": row[7].isoformat() if row[7] is not None and hasattr(row[7], "isoformat") else row[7],
                "error": row[8],
                "workflow_payload": row[9],
            }
        )

    def list_workflow_runs(
        self,
        *,
        workflow_name: str | None = None,
        workflow_hash: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[WorkflowRunRecord]:
        import psycopg

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
        where = f"WHERE {' AND '.join(predicates)}" if predicates else ""
        limit_clause = "LIMIT %s" if limit is not None else ""
        if limit is not None:
            params.append(int(limit))
        query = f"""
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
        """
        with psycopg.connect(self.source.connection_url()) as conn:
            with conn.cursor() as cur:
                self._ensure_schema(cur)
                cur.execute(query, tuple(params))
                rows = cur.fetchall()
        records: list[WorkflowRunRecord] = []
        for row in rows:
            records.append(
                WorkflowRunRecord.from_dict(
                    {
                        "run_id": row[0],
                        "workflow_name": row[1],
                        "workflow_hash": row[2],
                        "workflow_spec_hash": row[3],
                        "status": row[4],
                        "started_at": row[5].isoformat() if hasattr(row[5], "isoformat") else row[5],
                        "parent_run_id": row[6],
                        "finished_at": row[7].isoformat() if row[7] is not None and hasattr(row[7], "isoformat") else row[7],
                        "error": row[8],
                        "workflow_payload": row[9],
                    }
                )
            )
        return records

    def record_workflow_step(self, record: WorkflowStepRecord) -> None:
        import json

        import psycopg

        with psycopg.connect(self.source.connection_url()) as conn:
            with conn.cursor() as cur:
                self._ensure_schema(cur)
                cur.execute(
                    """
                    INSERT INTO pipelines_v2_workflow_steps (
                        run_id,
                        step_name,
                        workflow_hash,
                        workflow_step_key,
                        step_index,
                        runner,
                        status,
                        step_semantic_hash,
                        step_spec_hash,
                        input_artifact_refs,
                        artifact_id,
                        artifact_kind,
                        started_at,
                        finished_at,
                        runtime_app_id,
                        reused_from_run_id,
                        reused_from_artifact_id
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, step_name) DO UPDATE SET
                        workflow_hash = EXCLUDED.workflow_hash,
                        workflow_step_key = EXCLUDED.workflow_step_key,
                        step_index = EXCLUDED.step_index,
                        runner = EXCLUDED.runner,
                        status = EXCLUDED.status,
                        step_semantic_hash = EXCLUDED.step_semantic_hash,
                        step_spec_hash = EXCLUDED.step_spec_hash,
                        input_artifact_refs = EXCLUDED.input_artifact_refs,
                        artifact_id = EXCLUDED.artifact_id,
                        artifact_kind = EXCLUDED.artifact_kind,
                        started_at = EXCLUDED.started_at,
                        finished_at = EXCLUDED.finished_at,
                        runtime_app_id = EXCLUDED.runtime_app_id,
                        reused_from_run_id = EXCLUDED.reused_from_run_id,
                        reused_from_artifact_id = EXCLUDED.reused_from_artifact_id
                    """,
                    (
                        record.run_id,
                        record.step_name,
                        record.workflow_hash,
                        record.workflow_step_key,
                        record.step_index,
                        record.runner,
                        record.status,
                        record.step_semantic_hash,
                        record.step_spec_hash,
                        json.dumps(list(record.input_artifact_refs)),
                        record.artifact_id,
                        record.artifact_kind,
                        record.started_at,
                        record.finished_at,
                        record.runtime_app_id,
                        record.reused_from_run_id,
                        record.reused_from_artifact_id,
                    ),
                )
            conn.commit()

    def list_workflow_steps(self, run_id: str) -> list[WorkflowStepRecord]:
        import psycopg

        with psycopg.connect(self.source.connection_url()) as conn:
            with conn.cursor() as cur:
                self._ensure_schema(cur)
                cur.execute(
                    """
                    SELECT
                        run_id,
                        workflow_hash,
                        workflow_step_key,
                        step_name,
                        step_index,
                        runner,
                        status,
                        step_semantic_hash,
                        step_spec_hash,
                        input_artifact_refs,
                        artifact_id,
                        artifact_kind,
                        started_at,
                        finished_at,
                        runtime_app_id,
                        reused_from_run_id,
                        reused_from_artifact_id
                    FROM pipelines_v2_workflow_steps
                    WHERE run_id = %s
                    ORDER BY step_index ASC, step_name ASC
                    """,
                    (run_id,),
                )
                rows = cur.fetchall()
        records: list[WorkflowStepRecord] = []
        for row in rows:
            records.append(
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
                        "started_at": row[12].isoformat() if row[12] is not None and hasattr(row[12], "isoformat") else row[12],
                        "finished_at": row[13].isoformat() if row[13] is not None and hasattr(row[13], "isoformat") else row[13],
                        "runtime_app_id": row[14],
                        "reused_from_run_id": row[15],
                        "reused_from_artifact_id": row[16],
                    }
                )
            )
        return records

    def find_latest_reusable_step(
        self,
        *,
        step_name: str,
        step_semantic_hash: str,
        input_artifact_refs: tuple[str, ...],
    ) -> WorkflowStepRecord | None:
        import json

        import psycopg

        with psycopg.connect(self.source.connection_url()) as conn:
            with conn.cursor() as cur:
                self._ensure_schema(cur)
                cur.execute(
                    """
                    SELECT
                        run_id,
                        workflow_hash,
                        workflow_step_key,
                        step_name,
                        step_index,
                        runner,
                        status,
                        step_semantic_hash,
                        step_spec_hash,
                        input_artifact_refs,
                        artifact_id,
                        artifact_kind,
                        started_at,
                        finished_at,
                        runtime_app_id,
                        reused_from_run_id,
                        reused_from_artifact_id
                    FROM pipelines_v2_workflow_steps
                    WHERE step_name = %s
                      AND step_semantic_hash = %s
                      AND input_artifact_refs = %s::jsonb
                      AND status IN ('completed', 'reused')
                      AND artifact_id IS NOT NULL
                    ORDER BY finished_at DESC NULLS LAST
                    LIMIT 1
                    """,
                    (step_name, step_semantic_hash, json.dumps(list(input_artifact_refs))),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return WorkflowStepRecord.from_dict(
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
                "started_at": row[12].isoformat() if row[12] is not None and hasattr(row[12], "isoformat") else row[12],
                "finished_at": row[13].isoformat() if row[13] is not None and hasattr(row[13], "isoformat") else row[13],
                "runtime_app_id": row[14],
                "reused_from_run_id": row[15],
                "reused_from_artifact_id": row[16],
            }
        )

    def _ensure_schema(self, cur: Any) -> None:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pipelines_v2_artifacts (
                artifact_id TEXT PRIMARY KEY,
                artifact_kind TEXT NOT NULL,
                operation_spec_hash TEXT NOT NULL,
                operation_semantic_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                manifest JSONB NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pipelines_v2_workflow_runs (
                run_id TEXT PRIMARY KEY,
                workflow_name TEXT NULL,
                workflow_hash TEXT NOT NULL,
                workflow_spec_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TIMESTAMPTZ NOT NULL,
                parent_run_id TEXT NULL,
                finished_at TIMESTAMPTZ NULL,
                error TEXT NULL,
                workflow_payload JSONB NOT NULL
            )
            """
        )
        cur.execute(
            """
            ALTER TABLE pipelines_v2_workflow_runs
            ADD COLUMN IF NOT EXISTS parent_run_id TEXT NULL
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pipelines_v2_workflow_steps (
                run_id TEXT NOT NULL,
                step_name TEXT NOT NULL,
                workflow_hash TEXT NOT NULL,
                workflow_step_key TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                runner TEXT NOT NULL,
                status TEXT NOT NULL,
                step_semantic_hash TEXT NOT NULL,
                step_spec_hash TEXT NOT NULL,
                input_artifact_refs JSONB NOT NULL,
                artifact_id TEXT NULL,
                artifact_kind TEXT NULL,
                started_at TIMESTAMPTZ NULL,
                finished_at TIMESTAMPTZ NULL,
                runtime_app_id TEXT NULL,
                reused_from_run_id TEXT NULL,
                reused_from_artifact_id TEXT NULL,
                PRIMARY KEY (run_id, step_name)
            )
            """
        )
        cur.execute(
            """
            ALTER TABLE pipelines_v2_workflow_steps
            ADD COLUMN IF NOT EXISTS runtime_app_id TEXT NULL
            """
        )
