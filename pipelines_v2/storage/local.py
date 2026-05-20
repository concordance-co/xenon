"""Local filesystem artifact store and catalogs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors.numpy import load_file, save_file

from pipelines_v2.storage.artifacts import ArtifactManifest
from pipelines_v2.storage.json import json_default
from pipelines_v2.workflow.records import WorkflowRunRecord, WorkflowStepRecord


def _workflow_payload_summary(payload: Any) -> tuple[int, bool]:
    steps = payload.get("steps", ()) if isinstance(payload, dict) else ()
    if not isinstance(steps, (list, tuple)):
        return 0, False
    has_report = False
    for step in steps:
        if not isinstance(step, dict):
            continue
        spec = step.get("spec")
        if isinstance(spec, dict) and spec.get("kind") == "report":
            has_report = True
            break
    return len(steps), has_report


@dataclass(frozen=True, slots=True)
class LocalArtifactStore:
    """Artifact store backed by the local filesystem."""
    root: Path | str

    kind: str = "local"

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LocalArtifactStore":
        return cls(root=payload["root"])

    def identity(self) -> dict[str, Any]:
        return {"kind": self.kind, "root": str(self.root)}

    def make_artifact_dir(self, artifact_id: str) -> Path:
        path = Path(self.root) / artifact_id
        path.mkdir(parents=True, exist_ok=False)
        return path

    def ensure_artifact_dir(self, artifact_id: str) -> Path:
        path = Path(self.root) / artifact_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def has_local_artifact(self, artifact_id: str) -> bool:
        return (Path(self.root) / artifact_id).exists()

    def write_safetensors(self, artifact_id: str, relative_path: str, tensors: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_inside_artifact(artifact_id, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        save_file(tensors, str(tmp))
        os.replace(tmp, path)
        return {"store": self.kind, "path": str(path), "format": "safetensors", "bytes": path.stat().st_size}

    def write_json(self, artifact_id: str, relative_path: str, payload: Any) -> dict[str, Any]:
        path = self._resolve_inside_artifact(artifact_id, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True, indent=2, default=json_default)
        os.replace(tmp, path)
        return {
            "store": self.kind,
            "path": str(path),
            "format": "json",
            "bytes": path.stat().st_size,
        }

    def has_local_ref(self, ref: dict[str, Any]) -> bool:
        return Path(str(ref["path"])).exists()

    def read_json_ref(self, ref: dict[str, Any]) -> Any:
        path = Path(ref["path"])
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def read_safetensors_ref(self, ref: dict[str, Any]) -> dict[str, Any]:
        path = Path(str(ref["path"]))
        return load_file(str(path))

    def localize(self, artifact_id: str) -> Path:
        return Path(self.root) / artifact_id

    def estimate_download_bytes(self, ref: dict[str, Any]) -> int | None:
        if "bytes" in ref and ref["bytes"] is not None:
            return int(ref["bytes"])
        path = Path(ref["path"])
        if path.exists():
            return path.stat().st_size
        return None

    def validate_transfer(self, *, bytes: int | None, label: str) -> None:
        return None

    def _resolve_inside_artifact(self, artifact_id: str, relative_path: str) -> Path:
        root = (Path(self.root) / artifact_id).resolve()
        path = (root / relative_path).resolve()
        if root != path and root not in path.parents:
            raise ValueError(f"Refusing to write outside artifact root: {relative_path!r}")
        return path


@dataclass(frozen=True, slots=True)
class FileCatalog:
    """Catalog that mirrors manifests into a local directory as JSON files."""
    root: Path | str

    kind: str = "file"

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FileCatalog":
        return cls(root=payload["root"])

    def identity(self) -> dict[str, Any]:
        return {"kind": self.kind, "root": str(self.root)}

    def record_artifact(self, manifest: ArtifactManifest) -> None:
        Path(self.root).mkdir(parents=True, exist_ok=True)
        path = self._artifacts_root() / f"{manifest.artifact_id}.json"
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, sort_keys=True, indent=2, default=json_default)
        os.replace(tmp, path)

    def load_artifact(self, artifact_id: str) -> ArtifactManifest | None:
        path = self._artifacts_root() / f"{artifact_id}.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return ArtifactManifest.from_dict(json.load(f))

    def find_artifact_for_workflow_step(
        self,
        *,
        run_id: str,
        workflow_step_key: str,
    ) -> ArtifactManifest | None:
        latest: ArtifactManifest | None = None
        for path in self._artifacts_root().glob("*.json"):
            with path.open("r", encoding="utf-8") as f:
                manifest = ArtifactManifest.from_dict(json.load(f))
            context = manifest.workflow_context
            if context.get("run_id") != run_id:
                continue
            if context.get("workflow_step_key") != workflow_step_key:
                continue
            if latest is None or manifest.created_at > latest.created_at:
                latest = manifest
        return latest

    def record_workflow_run(self, record: WorkflowRunRecord) -> None:
        path = self._workflow_runs_root() / f"{record.run_id}.json"
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, sort_keys=True, indent=2, default=json_default)
        os.replace(tmp, path)
        existing_summary = self._load_workflow_run_summary(record.run_id)
        step_counts = (
            dict(existing_summary.get("step_counts", {}))
            if existing_summary is not None
            else self._compute_step_status_counts(record.run_id)
        )
        self._write_workflow_run_summary(
            self._build_workflow_run_summary(record, step_counts=step_counts)
        )

    def load_workflow_run(self, run_id: str) -> WorkflowRunRecord | None:
        path = self._workflow_runs_root() / f"{run_id}.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return WorkflowRunRecord.from_dict(json.load(f))

    def list_workflow_runs(
        self,
        *,
        workflow_name: str | None = None,
        workflow_hash: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[WorkflowRunRecord]:
        root = self._workflow_runs_root()
        records: list[WorkflowRunRecord] = []
        for path in sorted(root.glob("*.json"), reverse=True):
            with path.open("r", encoding="utf-8") as f:
                record = WorkflowRunRecord.from_dict(json.load(f))
            if workflow_name is not None and record.workflow_name != workflow_name:
                continue
            if workflow_hash is not None and record.workflow_hash != workflow_hash:
                continue
            if status is not None and record.status != status:
                continue
            records.append(record)
        records.sort(key=lambda item: (item.started_at, item.run_id), reverse=True)
        if limit is not None:
            return records[:limit]
        return records

    def list_workflow_runs_light(
        self,
        *,
        workflow_name: str | None = None,
        workflow_hash: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_workflow_run_summaries()
        root = self._workflow_run_summaries_root()
        rows: list[dict[str, Any]] = []
        for path in root.glob("*.json"):
            with path.open("r", encoding="utf-8") as f:
                row = dict(json.load(f))
            if workflow_name is not None and row.get("workflow_name") != workflow_name:
                continue
            if workflow_hash is not None and row.get("workflow_hash") != workflow_hash:
                continue
            if status is not None and row.get("status") != status:
                continue
            rows.append(row)
        rows.sort(key=lambda item: (str(item.get("started_at", "")), str(item.get("run_id", ""))), reverse=True)
        if limit is not None:
            return rows[:limit]
        return rows

    def record_workflow_step(self, record: WorkflowStepRecord) -> None:
        path = self._workflow_steps_root(record.run_id) / f"{record.step_name}.json"
        previous: WorkflowStepRecord | None = None
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                previous = WorkflowStepRecord.from_dict(json.load(f))
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, sort_keys=True, indent=2, default=json_default)
        os.replace(tmp, path)
        self._update_workflow_run_summary_for_step(
            run_id=record.run_id,
            previous=previous,
            current=record,
        )

    def list_workflow_steps(self, run_id: str) -> list[WorkflowStepRecord]:
        root = self._workflow_steps_root(run_id)
        if not root.exists():
            return []
        records: list[WorkflowStepRecord] = []
        for path in sorted(root.glob("*.json")):
            with path.open("r", encoding="utf-8") as f:
                records.append(WorkflowStepRecord.from_dict(json.load(f)))
        return records

    def find_latest_reusable_step(
        self,
        *,
        step_name: str,
        step_semantic_hash: str,
        input_artifact_refs: tuple[str, ...],
    ) -> WorkflowStepRecord | None:
        latest: WorkflowStepRecord | None = None
        runs_root = self._workflow_runs_root()
        if not runs_root.exists():
            return None
        for path in runs_root.glob("*.json"):
            run_id = path.stem
            for record in self.list_workflow_steps(run_id):
                if record.step_name != step_name:
                    continue
                if record.status not in {"completed", "reused"}:
                    continue
                if record.step_semantic_hash != step_semantic_hash:
                    continue
                if tuple(record.input_artifact_refs) != tuple(input_artifact_refs):
                    continue
                if latest is None or (record.finished_at or "") > (latest.finished_at or ""):
                    latest = record
        return latest

    def _artifacts_root(self) -> Path:
        root = Path(self.root)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _workflow_runs_root(self) -> Path:
        root = Path(self.root) / "workflow_runs"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _workflow_steps_root(self, run_id: str) -> Path:
        root = Path(self.root) / "workflow_steps" / run_id
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _workflow_run_summaries_root(self) -> Path:
        root = Path(self.root) / "workflow_run_summaries"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _workflow_run_summary_path(self, run_id: str) -> Path:
        return self._workflow_run_summaries_root() / f"{run_id}.json"

    def _load_workflow_run_summary(self, run_id: str) -> dict[str, Any] | None:
        path = self._workflow_run_summary_path(run_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return None
        return payload

    def _write_workflow_run_summary(self, payload: dict[str, Any]) -> None:
        path = self._workflow_run_summary_path(str(payload["run_id"]))
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True, indent=2, default=json_default)
        os.replace(tmp, path)

    def _ensure_workflow_run_summaries(self) -> None:
        runs_root = self._workflow_runs_root()
        summary_root = self._workflow_run_summaries_root()
        summary_ids = {path.stem for path in summary_root.glob("*.json")}
        missing = [path for path in runs_root.glob("*.json") if path.stem not in summary_ids]
        for path in missing:
            with path.open("r", encoding="utf-8") as f:
                record = WorkflowRunRecord.from_dict(json.load(f))
            self._write_workflow_run_summary(
                self._build_workflow_run_summary(
                    record,
                    step_counts=self._compute_step_status_counts(record.run_id),
                )
            )

    def _build_workflow_run_summary(
        self,
        record: WorkflowRunRecord,
        *,
        step_counts: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        step_total, has_report = _workflow_payload_summary(dict(record.workflow_payload))
        counts = {
            str(name): int(count)
            for name, count in dict(step_counts or {}).items()
            if int(count) > 0
        }
        step_total = max(step_total, sum(counts.values()))
        return {
            "run_id": record.run_id,
            "workflow_name": record.workflow_name,
            "workflow_hash": record.workflow_hash,
            "workflow_spec_hash": record.workflow_spec_hash,
            "status": record.status,
            "started_at": record.started_at,
            "finished_at": record.finished_at,
            "parent_run_id": record.parent_run_id,
            "error": record.error,
            "has_report": has_report,
            "step_total": step_total,
            "step_counts": counts,
        }

    def _compute_step_status_counts(self, run_id: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.list_workflow_steps(run_id):
            status = str(record.status or "").lower()
            if not status:
                continue
            counts[status] = counts.get(status, 0) + 1
        return counts

    def _update_workflow_run_summary_for_step(
        self,
        *,
        run_id: str,
        previous: WorkflowStepRecord | None,
        current: WorkflowStepRecord,
    ) -> None:
        summary = self._load_workflow_run_summary(run_id)
        if summary is None:
            run = self.load_workflow_run(run_id)
            if run is None:
                return
            summary = self._build_workflow_run_summary(
                run,
                step_counts=self._compute_step_status_counts(run_id),
            )
        counts = {
            str(name): int(count)
            for name, count in dict(summary.get("step_counts", {})).items()
            if int(count) > 0
        }
        if previous is not None:
            previous_status = str(previous.status or "").lower()
            if previous_status in counts:
                new_value = counts[previous_status] - 1
                if new_value > 0:
                    counts[previous_status] = new_value
                else:
                    counts.pop(previous_status, None)
        current_status = str(current.status or "").lower()
        if current_status:
            counts[current_status] = counts.get(current_status, 0) + 1
        summary["step_counts"] = counts
        summary["step_total"] = max(
            int(summary.get("step_total") or 0),
            current.step_index + 1,
            sum(counts.values()),
        )
        self._write_workflow_run_summary(summary)


@dataclass(frozen=True, slots=True)
class NullCatalog:
    """No-op catalog used when no secondary manifest index is needed."""
    kind: str = "none"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NullCatalog":
        return cls()

    def identity(self) -> dict[str, Any]:
        return {"kind": self.kind}

    def record_artifact(self, manifest: ArtifactManifest) -> None:
        return None

    def load_artifact(self, artifact_id: str) -> ArtifactManifest | None:
        return None

    def find_artifact_for_workflow_step(
        self,
        *,
        run_id: str,
        workflow_step_key: str,
    ) -> ArtifactManifest | None:
        return None

    def record_workflow_run(self, record: WorkflowRunRecord) -> None:
        return None

    def load_workflow_run(self, run_id: str) -> WorkflowRunRecord | None:
        return None

    def list_workflow_runs(
        self,
        *,
        workflow_name: str | None = None,
        workflow_hash: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[WorkflowRunRecord]:
        return []

    def record_workflow_step(self, record: WorkflowStepRecord) -> None:
        return None

    def list_workflow_steps(self, run_id: str) -> list[WorkflowStepRecord]:
        return []

    def find_latest_reusable_step(
        self,
        *,
        step_name: str,
        step_semantic_hash: str,
        input_artifact_refs: tuple[str, ...],
    ) -> WorkflowStepRecord | None:
        return None
