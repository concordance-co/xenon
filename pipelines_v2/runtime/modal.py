"""Modal runner implementation."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

from pipelines_v2.core.types import OperationSpec
from pipelines_v2.mechinterp.assistant_axis import (
    AssistantAxisPrecomputedCoordinateSpec,
    AssistantAxisScoreSpec,
    AssistantAxisVectorSpec,
)
from pipelines_v2.mechinterp.emotions import (
    EmotionDirectionSpec,
    EmotionGeometrySpec,
    EmotionPrecomputedVectorSpaceSpec,
    EmotionScoreSpec,
    EmotionVectorSpaceSpec,
)
from pipelines_v2.operations.specs import (
    ActivationBankSpec,
    BasisSpec,
    CaptureSpec,
    CentroidSpec,
    CoordinateImportSpec,
    DirectionSpec,
    ExplicitPathMaskSpec,
    GenerationRunSpec,
    GeometrySpec,
    LabelFieldsSpec,
    LabelMapSpec,
    PatchComparisonSpec,
    PairDeltaSpec,
    PatchedGenerationSpec,
    ProbeSpec,
    ProjectionCalibrationSpec,
    ProjectionSpec,
    ReportSpec,
    ResidualizedProbeSpec,
    SubspaceSpec,
    TextBaselineSpec,
    TransferProbeSpec,
    TransformSpec,
)
from pipelines_v2.runtime.base import ExecutionPlan
from pipelines_v2.runtime.modal_worker import run_on_modal
from pipelines_v2.storage.artifacts import ArtifactManifest, CaptureArtifact, OperationArtifact
from pipelines_v2.storage.base import Catalog
from pipelines_v2.storage.local import NullCatalog
from pipelines_v2.storage.modal import ModalVolumeStore
from pipelines_v2.workflow.records import WorkflowStepContext
from pipelines_v2.operations.interventions.runtime import patched_generation_plan_errors


@dataclass(frozen=True, slots=True)
class ModalVolumeMount:
    """Extra Modal volume mount requested by a runner."""
    name: str
    mount_path: str
    create_if_missing: bool = False
    commit_on_success: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the mount for remote execution."""
        return {
            "name": self.name,
            "mount_path": self.mount_path,
            "create_if_missing": self.create_if_missing,
            "commit_on_success": self.commit_on_success,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModalVolumeMount":
        return cls(
            name=str(payload["name"]),
            mount_path=str(payload["mount_path"]),
            create_if_missing=bool(payload.get("create_if_missing", False)),
            commit_on_success=bool(payload.get("commit_on_success", False)),
        )


@dataclass(frozen=True, slots=True)
class ModalSecret:
    """Binding from a Modal secret to one or more runtime env vars."""
    name: str
    env_vars: tuple[str, ...]

    @classmethod
    def from_env_var(cls, env_var: str, *, secret_name: str | None = None) -> "ModalSecret":
        """Create a one-env-var Modal secret binding."""
        return cls(name=secret_name or env_var, env_vars=(env_var,))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the secret binding for remote execution."""
        return {
            "name": self.name,
            "env_vars": list(self.env_vars),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModalSecret":
        return cls(
            name=str(payload["name"]),
            env_vars=tuple(str(env_var) for env_var in payload.get("env_vars", ())),
        )


@dataclass(frozen=True, slots=True)
class ModalResources:
    """Resource profile for one remote Modal execution environment."""
    gpu: str | None = None
    cpu: int | float | None = None
    memory_mb: int | None = None
    timeout_seconds: int | None = None
    max_containers: int | None = None
    shard_count: int | None = None
    env: dict[str, str] = field(default_factory=dict)
    secrets: tuple[ModalSecret, ...] = ()
    volumes: tuple[ModalVolumeMount, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu": self.gpu,
            "cpu": self.cpu,
            "memory_mb": self.memory_mb,
            "timeout_seconds": self.timeout_seconds,
            "max_containers": self.max_containers,
            "shard_count": self.shard_count,
            "env": dict(self.env),
            "secrets": [secret.to_dict() for secret in self.secrets],
            "volumes": [volume.to_dict() for volume in self.volumes],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModalResources":
        return cls(
            gpu=payload.get("gpu"),
            cpu=payload.get("cpu"),
            memory_mb=int(payload["memory_mb"]) if payload.get("memory_mb") is not None else None,
            timeout_seconds=int(payload["timeout_seconds"]) if payload.get("timeout_seconds") is not None else None,
            max_containers=int(payload["max_containers"]) if payload.get("max_containers") is not None else None,
            shard_count=int(payload["shard_count"]) if payload.get("shard_count") is not None else None,
            env={str(key): str(value) for key, value in dict(payload.get("env", {})).items()},
            secrets=tuple(ModalSecret.from_dict(dict(secret)) for secret in payload.get("secrets", ())),
            volumes=tuple(ModalVolumeMount.from_dict(dict(volume)) for volume in payload.get("volumes", ())),
        )


@dataclass(slots=True)
class ModalRunner:
    """Execute capture and artifact-bound specs on Modal."""
    resources: ModalResources
    artifacts: ModalVolumeStore
    catalog: Catalog = field(default_factory=NullCatalog)

    kind: str = "modal"

    def identity(self) -> dict[str, Any]:
        """Return a serializable description of this runner."""
        return {
            "kind": self.kind,
            "resources": self.resources.to_dict(),
        }

    def plan(self, spec: OperationSpec) -> ExecutionPlan:
        """Preflight a spec against Modal secret bindings and capabilities."""
        engine = spec.bound_engine()
        if isinstance(spec, CaptureSpec):
            artifact_kinds = ("capture",)
        elif isinstance(spec, (GenerationRunSpec, PatchedGenerationSpec)):
            artifact_kinds = (spec.kind,)
        else:
            artifact_kinds = ((spec.kind,) if isinstance(spec, _ARTIFACT_BOUND_SPECS) else ())
        errors = list(_spec_plan_errors(spec))
        errors.extend(self._plan_errors(spec))
        return ExecutionPlan(
            spec_kind=spec.kind,
            required_capabilities=frozenset(spec.required_capabilities()),
            engine_capabilities=frozenset(engine.capabilities()) if engine is not None else frozenset(),
            artifact_kinds=artifact_kinds,
            checks=("capabilities", "runtime_spec", "artifact_store", "catalog"),
            errors=tuple(errors),
        )

    def run(
        self,
        spec: OperationSpec,
        *,
        workflow_context: WorkflowStepContext | None = None,
        progress_callback: Any | None = None,
    ) -> Any:
        """Execute one supported spec remotely and return its artifact."""
        self.plan(spec).validate()
        if isinstance(spec, (CaptureSpec, GenerationRunSpec, PatchedGenerationSpec, *_ARTIFACT_BOUND_SPECS)):
            return self._run_remote(spec, workflow_context=workflow_context, progress_callback=progress_callback)
        raise NotImplementedError(f"ModalRunner cannot run {spec.kind!r} specs yet")

    def _run_remote(
        self,
        spec: OperationSpec,
        *,
        workflow_context: WorkflowStepContext | None = None,
        progress_callback: Any | None = None,
    ) -> CaptureArtifact | OperationArtifact:
        run_kwargs = {
            "runner_config": self.identity(),
            "store_config": self.artifacts.identity(),
            "spec_payload": spec.to_dict(),
        }
        try:
            signature = inspect.signature(run_on_modal)
        except (TypeError, ValueError):
            signature = None
        if signature is None or "workflow_context" in signature.parameters:
            run_kwargs["workflow_context"] = (
                workflow_context.to_manifest_dict() if workflow_context is not None else None
            )
        if signature is None or "progress_callback" in signature.parameters:
            run_kwargs["progress_callback"] = progress_callback
        manifest_payload = run_on_modal(**run_kwargs)
        if manifest_payload is None:
            raise RuntimeError("Modal runner did not receive a manifest payload; the remote run was likely cancelled")
        manifest = ArtifactManifest.from_dict(manifest_payload)
        try:
            self.catalog.record_artifact(manifest)
        except NotImplementedError:
            pass
        if manifest.artifact_kind == "capture":
            return CaptureArtifact(_manifest=manifest, store=self.artifacts)
        return OperationArtifact(_manifest=manifest, store=self.artifacts)

    def _plan_errors(self, spec: OperationSpec) -> list[str]:
        errors: list[str] = []
        required = {secret.env_var for secret in spec.runtime_secrets()}
        runtime_spec = spec.runtime_spec()
        if runtime_spec is not None:
            required.update(secret.env_var for secret in runtime_spec.secrets)
        if not required:
            return errors
        provided = {env_var for secret in self.resources.secrets for env_var in secret.env_vars}
        missing = sorted(required - provided)
        if missing:
            errors.append(
                "ModalRunner is missing required runtime secret bindings for env vars: "
                f"{missing}"
            )
        return errors


_ARTIFACT_BOUND_SPECS = (
    ProbeSpec,
    TransferProbeSpec,
    TextBaselineSpec,
    ResidualizedProbeSpec,
    DirectionSpec,
    BasisSpec,
    CentroidSpec,
    GeometrySpec,
    SubspaceSpec,
    ActivationBankSpec,
    ExplicitPathMaskSpec,
    PairDeltaSpec,
    LabelMapSpec,
    LabelFieldsSpec,
    TransformSpec,
    PatchComparisonSpec,
    CoordinateImportSpec,
    ProjectionSpec,
    ProjectionCalibrationSpec,
    AssistantAxisPrecomputedCoordinateSpec,
    AssistantAxisVectorSpec,
    AssistantAxisScoreSpec,
    EmotionPrecomputedVectorSpaceSpec,
    EmotionVectorSpaceSpec,
    EmotionScoreSpec,
    EmotionDirectionSpec,
    EmotionGeometrySpec,
    ReportSpec,
)


def _spec_plan_errors(spec: OperationSpec) -> list[str]:
    errors: list[str] = []
    engine = spec.bound_engine()
    if engine is not None:
        planning_errors = getattr(engine, "planning_errors", None)
        if callable(planning_errors):
            errors.extend(str(error) for error in planning_errors(spec))
    if isinstance(spec, PatchedGenerationSpec):
        errors.extend(patched_generation_plan_errors(spec))
    return errors
