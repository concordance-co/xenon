"""Structured workflow progress events and local persistence."""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from itertools import count
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from pipelines_v2.core.types import utc_now_iso
from pipelines_v2.storage.json import json_default

_PROGRESS_LOG = logging.getLogger("pipelines_v2.progress")
_PROGRESS_SEQUENCE = count(1)
_PROGRESS_SCHEMA_VERSION = "xenon.progress.v1"


@dataclass(frozen=True, slots=True)
class WorkflowProgressEvent:
    """One structured workflow or workflow-step progress update."""

    run_id: str
    workflow_name: str | None
    status: str
    stage: str
    schema_version: str = _PROGRESS_SCHEMA_VERSION
    event_id: str = field(default_factory=lambda: f"xpe_{uuid.uuid4().hex}")
    sequence: int = field(default_factory=lambda: next(_PROGRESS_SEQUENCE))
    created_at: str = field(default_factory=utc_now_iso)
    step_name: str | None = None
    step_index: int | None = None
    runner: str | None = None
    spec_kind: str | None = None
    message: str | None = None
    runtime_kind: str | None = None
    runtime_app_id: str | None = None
    artifact_id: str | None = None
    artifact_kind: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "sequence": self.sequence,
            "run_id": self.run_id,
            "workflow_name": self.workflow_name,
            "status": self.status,
            "stage": self.stage,
            "created_at": self.created_at,
            "step_name": self.step_name,
            "step_index": self.step_index,
            "runner": self.runner,
            "spec_kind": self.spec_kind,
            "message": self.message,
            "runtime_kind": self.runtime_kind,
            "runtime_app_id": self.runtime_app_id,
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkflowProgressEvent":
        return cls(
            run_id=str(payload["run_id"]),
            workflow_name=str(payload["workflow_name"]) if payload.get("workflow_name") is not None else None,
            status=str(payload["status"]),
            stage=str(payload["stage"]),
            schema_version=str(payload.get("schema_version") or _PROGRESS_SCHEMA_VERSION),
            event_id=str(payload.get("event_id") or f"xpe_{uuid.uuid4().hex}"),
            sequence=int(payload.get("sequence") or next(_PROGRESS_SEQUENCE)),
            created_at=str(payload["created_at"]),
            step_name=str(payload["step_name"]) if payload.get("step_name") is not None else None,
            step_index=int(payload["step_index"]) if payload.get("step_index") is not None else None,
            runner=str(payload["runner"]) if payload.get("runner") is not None else None,
            spec_kind=str(payload["spec_kind"]) if payload.get("spec_kind") is not None else None,
            message=str(payload["message"]) if payload.get("message") is not None else None,
            runtime_kind=str(payload["runtime_kind"]) if payload.get("runtime_kind") is not None else None,
            runtime_app_id=(
                str(payload["runtime_app_id"]) if payload.get("runtime_app_id") is not None else None
            ),
            artifact_id=str(payload["artifact_id"]) if payload.get("artifact_id") is not None else None,
            artifact_kind=str(payload["artifact_kind"]) if payload.get("artifact_kind") is not None else None,
            metrics=dict(payload.get("metrics", {})),
        )


@dataclass(slots=True)
class FileWorkflowProgressStore:
    """Persist workflow progress snapshots and event streams under the local registry."""

    root: Path | str
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def record_event(self, event: WorkflowProgressEvent) -> None:
        payload = event.to_dict()
        with self._lock:
            self._write_snapshot(event=event, payload=payload)
            self._append_event(event=event, payload=payload)

    def load_run_snapshot(self, run_id: str) -> dict[str, Any] | None:
        path = self._run_snapshot_path(run_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def load_step_snapshots(self, run_id: str) -> dict[str, dict[str, Any]]:
        root = self._step_snapshots_root(run_id)
        if not root.exists():
            return {}
        snapshots: dict[str, dict[str, Any]] = {}
        for path in sorted(root.glob("*.json")):
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            step_name = payload.get("step_name")
            if step_name:
                snapshots[str(step_name)] = dict(payload)
        return snapshots

    def _write_snapshot(self, *, event: WorkflowProgressEvent, payload: dict[str, Any]) -> None:
        if event.step_name is None:
            path = self._run_snapshot_path(event.run_id)
        else:
            path = self._step_snapshot_path(event.run_id, event.step_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    previous = json.load(f)
            except (OSError, json.JSONDecodeError):
                previous = {}
            for key in ("runtime_kind", "runtime_app_id"):
                if payload.get(key) is None and previous.get(key) is not None:
                    payload[key] = previous[key]
            payload["metrics"] = {
                **dict(previous.get("metrics") or {}),
                **dict(payload.get("metrics") or {}),
            }
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=json_default)
        os.replace(tmp, path)

    def _append_event(self, *, event: WorkflowProgressEvent, payload: dict[str, Any]) -> None:
        if event.step_name is None:
            path = self._run_events_path(event.run_id)
        else:
            path = self._step_events_path(event.run_id, event.step_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True, default=json_default))
            f.write("\n")

    def _run_snapshot_path(self, run_id: str) -> Path:
        return self.root / "workflow_progress" / "runs" / f"{run_id}.json"

    def _run_events_path(self, run_id: str) -> Path:
        return self.root / "workflow_progress" / "run_events" / f"{run_id}.jsonl"

    def _step_snapshots_root(self, run_id: str) -> Path:
        return self.root / "workflow_progress" / "steps" / run_id

    def _step_snapshot_path(self, run_id: str, step_name: str) -> Path:
        return self._step_snapshots_root(run_id) / f"{step_name}.json"

    def _step_events_path(self, run_id: str, step_name: str) -> Path:
        return self.root / "workflow_progress" / "step_events" / run_id / f"{step_name}.jsonl"


@dataclass(slots=True)
class WorkflowProgressSink:
    """Emit workflow progress to the local store and optional CLI logs."""

    store: FileWorkflowProgressStore
    log_level: int | None = None

    def emit(self, event: WorkflowProgressEvent) -> None:
        self.store.record_event(event)
        if self.log_level is None:
            return
        _PROGRESS_LOG.log(self.log_level, self._format_event(event))

    def _format_event(self, event: WorkflowProgressEvent) -> str:
        if event.step_name is None:
            parts = [
                f"run={event.run_id}",
                f"status={event.status}",
                f"stage={event.stage}",
            ]
            if event.message:
                parts.append(event.message)
            return "workflow progress " + " ".join(parts)
        parts = [
            f"run={event.run_id}",
            f"step={event.step_name}",
            f"status={event.status}",
            f"stage={event.stage}",
        ]
        if event.runner:
            parts.append(f"runner={event.runner}")
        if event.spec_kind:
            parts.append(f"spec={event.spec_kind}")
        if event.runtime_app_id:
            parts.append(f"app={event.runtime_app_id}")
        if event.artifact_id:
            parts.append(f"artifact={event.artifact_id}")
        if event.message:
            parts.append(event.message)
        return "workflow progress " + " ".join(parts)
