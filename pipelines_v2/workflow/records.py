"""Persisted workflow and step run records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class WorkflowRunRecord:
    """Persisted record for one workflow execution attempt."""

    run_id: str
    workflow_name: str | None
    workflow_hash: str
    workflow_spec_hash: str
    workflow_payload: Mapping[str, Any]
    status: str
    started_at: str
    finished_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workflow_name": self.workflow_name,
            "workflow_hash": self.workflow_hash,
            "workflow_spec_hash": self.workflow_spec_hash,
            "workflow_payload": dict(self.workflow_payload),
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkflowRunRecord":
        return cls(
            run_id=str(payload["run_id"]),
            workflow_name=str(payload["workflow_name"]) if payload.get("workflow_name") is not None else None,
            workflow_hash=str(payload["workflow_hash"]),
            workflow_spec_hash=str(payload["workflow_spec_hash"]),
            workflow_payload=dict(payload.get("workflow_payload", {})),
            status=str(payload["status"]),
            started_at=str(payload["started_at"]),
            finished_at=str(payload["finished_at"]) if payload.get("finished_at") is not None else None,
            error=str(payload["error"]) if payload.get("error") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class WorkflowStepRecord:
    """Persisted record for one workflow step within one workflow run."""

    run_id: str
    workflow_hash: str
    workflow_step_key: str
    step_name: str
    step_index: int
    runner: str
    status: str
    step_semantic_hash: str
    step_spec_hash: str
    input_artifact_refs: tuple[str, ...] = field(default_factory=tuple)
    artifact_id: str | None = None
    artifact_kind: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    runtime_app_id: str | None = None
    reused_from_run_id: str | None = None
    reused_from_artifact_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workflow_hash": self.workflow_hash,
            "workflow_step_key": self.workflow_step_key,
            "step_name": self.step_name,
            "step_index": self.step_index,
            "runner": self.runner,
            "status": self.status,
            "step_semantic_hash": self.step_semantic_hash,
            "step_spec_hash": self.step_spec_hash,
            "input_artifact_refs": list(self.input_artifact_refs),
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "runtime_app_id": self.runtime_app_id,
            "reused_from_run_id": self.reused_from_run_id,
            "reused_from_artifact_id": self.reused_from_artifact_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkflowStepRecord":
        return cls(
            run_id=str(payload["run_id"]),
            workflow_hash=str(payload["workflow_hash"]),
            workflow_step_key=str(payload["workflow_step_key"]),
            step_name=str(payload["step_name"]),
            step_index=int(payload["step_index"]),
            runner=str(payload["runner"]),
            status=str(payload["status"]),
            step_semantic_hash=str(payload["step_semantic_hash"]),
            step_spec_hash=str(payload["step_spec_hash"]),
            input_artifact_refs=tuple(str(item) for item in payload.get("input_artifact_refs", ())),
            artifact_id=str(payload["artifact_id"]) if payload.get("artifact_id") is not None else None,
            artifact_kind=str(payload["artifact_kind"]) if payload.get("artifact_kind") is not None else None,
            started_at=str(payload["started_at"]) if payload.get("started_at") is not None else None,
            finished_at=str(payload["finished_at"]) if payload.get("finished_at") is not None else None,
            runtime_app_id=str(payload["runtime_app_id"]) if payload.get("runtime_app_id") is not None else None,
            reused_from_run_id=(
                str(payload["reused_from_run_id"]) if payload.get("reused_from_run_id") is not None else None
            ),
            reused_from_artifact_id=(
                str(payload["reused_from_artifact_id"])
                if payload.get("reused_from_artifact_id") is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class WorkflowStepContext:
    """Workflow provenance attached to one runner execution."""

    run_id: str
    workflow_name: str | None
    workflow_hash: str
    workflow_spec_hash: str
    step_name: str
    step_index: int
    runner: str
    step_semantic_hash: str
    step_spec_hash: str

    @property
    def workflow_step_key(self) -> str:
        return f"{self.workflow_hash}.{self.step_name}"

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workflow_name": self.workflow_name,
            "workflow_hash": self.workflow_hash,
            "workflow_spec_hash": self.workflow_spec_hash,
            "step_name": self.step_name,
            "step_index": self.step_index,
            "runner": self.runner,
            "workflow_step_key": self.workflow_step_key,
            "step_semantic_hash": self.step_semantic_hash,
            "step_spec_hash": self.step_spec_hash,
        }
