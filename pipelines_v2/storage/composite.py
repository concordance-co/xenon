"""Composite catalog helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pipelines_v2.storage.artifacts import ArtifactManifest
from pipelines_v2.workflow.records import WorkflowRunRecord, WorkflowStepRecord


@dataclass(frozen=True, slots=True)
class CompositeCatalog:
    """Mirror catalog writes into multiple backends and read from the first hit."""

    catalogs: tuple[Any, ...] = field(default_factory=tuple)

    kind: str = "composite"

    def identity(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "catalogs": [
                catalog.identity() if hasattr(catalog, "identity") else {"kind": getattr(catalog, "kind", "unknown")}
                for catalog in self.catalogs
            ],
        }

    def record_artifact(self, manifest: ArtifactManifest) -> None:
        for catalog in self.catalogs:
            catalog.record_artifact(manifest)

    def load_artifact(self, artifact_id: str) -> ArtifactManifest | None:
        for catalog in self.catalogs:
            manifest = catalog.load_artifact(artifact_id)
            if manifest is not None:
                return manifest
        return None

    def find_artifact_for_workflow_step(
        self,
        *,
        run_id: str,
        workflow_step_key: str,
    ) -> ArtifactManifest | None:
        for catalog in self.catalogs:
            finder = getattr(catalog, "find_artifact_for_workflow_step", None)
            if not callable(finder):
                continue
            manifest = finder(run_id=run_id, workflow_step_key=workflow_step_key)
            if manifest is not None:
                return manifest
        return None

    def record_workflow_run(self, record: WorkflowRunRecord) -> None:
        for catalog in self.catalogs:
            catalog.record_workflow_run(record)

    def load_workflow_run(self, run_id: str) -> WorkflowRunRecord | None:
        for catalog in self.catalogs:
            record = catalog.load_workflow_run(run_id)
            if record is not None:
                return record
        return None

    def list_workflow_runs(
        self,
        *,
        workflow_name: str | None = None,
        workflow_hash: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[WorkflowRunRecord]:
        seen: set[str] = set()
        records: list[WorkflowRunRecord] = []
        for catalog in self.catalogs:
            lister = getattr(catalog, "list_workflow_runs", None)
            if not callable(lister):
                continue
            for record in lister(
                workflow_name=workflow_name,
                workflow_hash=workflow_hash,
                status=status,
                limit=None,
            ):
                if record.run_id in seen:
                    continue
                seen.add(record.run_id)
                records.append(record)
        records.sort(key=lambda item: (item.started_at, item.run_id), reverse=True)
        if limit is not None:
            return records[:limit]
        return records

    def record_workflow_step(self, record: WorkflowStepRecord) -> None:
        for catalog in self.catalogs:
            catalog.record_workflow_step(record)

    def list_workflow_steps(self, run_id: str) -> list[WorkflowStepRecord]:
        seen: set[tuple[str, str]] = set()
        records: list[WorkflowStepRecord] = []
        for catalog in self.catalogs:
            for record in catalog.list_workflow_steps(run_id):
                key = (record.run_id, record.step_name)
                if key in seen:
                    continue
                seen.add(key)
                records.append(record)
        records.sort(key=lambda item: (item.step_index, item.step_name))
        return records

    def find_latest_reusable_step(
        self,
        *,
        step_name: str,
        step_semantic_hash: str,
        input_artifact_refs: tuple[str, ...],
    ) -> WorkflowStepRecord | None:
        latest: WorkflowStepRecord | None = None
        for catalog in self.catalogs:
            record = catalog.find_latest_reusable_step(
                step_name=step_name,
                step_semantic_hash=step_semantic_hash,
                input_artifact_refs=input_artifact_refs,
            )
            if record is None:
                continue
            if latest is None or (record.finished_at or "") > (latest.finished_at or ""):
                latest = record
        return latest
