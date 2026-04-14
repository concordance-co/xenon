"""Postgres-backed catalog descriptor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipelines_v2.data.sources import PostgresSource
from pipelines_v2.storage.artifacts import ArtifactManifest


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
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS pipelines_v2_artifacts (
                        artifact_id TEXT PRIMARY KEY,
                        artifact_kind TEXT NOT NULL,
                        operation_spec_hash TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL,
                        manifest JSONB NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    INSERT INTO pipelines_v2_artifacts (
                        artifact_id,
                        artifact_kind,
                        operation_spec_hash,
                        created_at,
                        manifest
                    ) VALUES (%s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (artifact_id) DO UPDATE SET
                        artifact_kind = EXCLUDED.artifact_kind,
                        operation_spec_hash = EXCLUDED.operation_spec_hash,
                        created_at = EXCLUDED.created_at,
                        manifest = EXCLUDED.manifest
                    """,
                    (
                        manifest.artifact_id,
                        manifest.artifact_kind,
                        manifest.operation_spec_hash,
                        manifest.created_at,
                        json.dumps(payload, sort_keys=True),
                    ),
                )
            conn.commit()
