"""Storage contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from pipelines_v2.storage.artifacts import ArtifactManifest
from pipelines_v2.workflow.records import WorkflowRunRecord, WorkflowStepRecord


class ArtifactStore(Protocol):
    """Persistence backend for manifests, tensors, labels, and report outputs."""

    kind: str

    def make_artifact_dir(self, artifact_id: str) -> Path:
        """Create storage for one artifact and return its root path."""
        ...

    def has_local_artifact(self, artifact_id: str) -> bool:
        """Return whether the full artifact is already available locally."""
        ...

    def write_safetensors(self, artifact_id: str, relative_path: str, tensors: dict[str, Any]) -> dict[str, Any]:
        """Persist a safetensors bundle and return a storage ref."""
        ...

    def write_json(self, artifact_id: str, relative_path: str, payload: Any) -> dict[str, Any]:
        """Persist JSON and return a storage ref."""
        ...

    def has_local_ref(self, ref: dict[str, Any]) -> bool:
        """Return whether the specific storage ref is already local."""
        ...

    def read_safetensors_ref(self, ref: dict[str, Any]) -> dict[str, Any]:
        """Load a safetensors ref into memory."""
        ...

    def read_json_ref(self, ref: dict[str, Any]) -> Any:
        """Load a JSON ref into memory."""
        ...

    def localize(self, artifact_id: str) -> Path:
        """Ensure one artifact is locally available and return its root path."""
        ...

    def estimate_download_bytes(self, ref: dict[str, Any]) -> int | None:
        """Estimate bytes needed to materialize the given ref locally."""
        ...

    def validate_transfer(self, *, bytes: int | None, label: str) -> None:
        """Raise if materializing the ref violates store transfer policy."""
        ...


class Catalog(Protocol):
    """Secondary manifest index or registry."""

    kind: str

    def record_artifact(self, manifest: ArtifactManifest) -> None:
        """Record one manifest into the catalog backend."""
        ...

    def load_artifact(self, artifact_id: str) -> ArtifactManifest | None:
        """Load one recorded artifact manifest by id."""
        ...

    def record_workflow_run(self, record: WorkflowRunRecord) -> None:
        """Record or update one workflow run."""
        ...

    def load_workflow_run(self, run_id: str) -> WorkflowRunRecord | None:
        """Load one workflow run by id."""
        ...

    def record_workflow_step(self, record: WorkflowStepRecord) -> None:
        """Record or update one workflow step run."""
        ...

    def list_workflow_steps(self, run_id: str) -> list[WorkflowStepRecord]:
        """List workflow step records for one workflow run."""
        ...

    def find_latest_reusable_step(
        self,
        *,
        step_name: str,
        step_semantic_hash: str,
        input_artifact_refs: tuple[str, ...],
    ) -> WorkflowStepRecord | None:
        """Return the latest reusable completed step matching the semantic lineage."""
        ...
