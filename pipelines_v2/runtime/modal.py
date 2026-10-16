"""Modal runner implementation."""

from __future__ import annotations

import inspect
import math
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from pipelines_v2.core.provenance import resolve_for_runner
from pipelines_v2.core.types import OperationSpec, stable_hash
from pipelines_v2.engine.base import PythonRuntimeSpec
from pipelines_v2.operations.specs import CaptureSpec, GenerationRunSpec, PatchedGenerationSpec
from pipelines_v2.operations.registry import is_artifact_operation, operation_artifact_kind
from pipelines_v2.runtime.base import ExecutionPlan
from pipelines_v2.runtime.modal_worker import _mounted_volumes, run_many_on_modal, run_on_modal
from pipelines_v2.runtime.planning import spec_plan_errors
from pipelines_v2.storage.artifacts import ArtifactManifest, CaptureArtifact, OperationArtifact
from pipelines_v2.storage.base import Catalog
from pipelines_v2.storage.local import NullCatalog
from pipelines_v2.storage.modal import ModalVolumeStore
from pipelines_v2.workflow.records import WorkflowStepContext


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
    enable_workflow_batching: bool = False
    enable_workflow_sessions: bool = field(default=False, metadata={"omit_if_default": True})
    readiness_timeout_seconds: int | None = field(default=None, metadata={"omit_if_default": True})
    client_timeout_seconds: int | None = field(default=None, metadata={"omit_if_default": True})
    merge_timeout_seconds: int | None = field(default=None, metadata={"omit_if_default": True})
    merge_memory_mb: int | None = field(default=None, metadata={"omit_if_default": True})
    env: dict[str, str] = field(default_factory=dict)
    secrets: tuple[ModalSecret, ...] = ()
    volumes: tuple[ModalVolumeMount, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            **{name: getattr(self, name) for name in ("client_timeout_seconds", "merge_timeout_seconds", "merge_memory_mb", "readiness_timeout_seconds") if getattr(self, name) is not None},
            **({"enable_workflow_sessions": True} if self.enable_workflow_sessions else {}),
            "gpu": self.gpu,
            "cpu": self.cpu,
            "memory_mb": self.memory_mb,
            "timeout_seconds": self.timeout_seconds,
            "max_containers": self.max_containers,
            "shard_count": self.shard_count,
            "enable_workflow_batching": self.enable_workflow_batching,
            "env": dict(self.env),
            "secrets": [secret.to_dict() for secret in self.secrets],
            "volumes": [volume.to_dict() for volume in self.volumes],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModalResources":
        unknown = set(payload) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError("Unknown ModalResources fields: " + ", ".join(sorted(unknown)))
        return cls(
            readiness_timeout_seconds=payload.get("readiness_timeout_seconds"),
            client_timeout_seconds=payload.get("client_timeout_seconds"),
            merge_timeout_seconds=payload.get("merge_timeout_seconds"),
            merge_memory_mb=payload.get("merge_memory_mb"),
            gpu=payload.get("gpu"),
            cpu=payload.get("cpu"),
            memory_mb=payload.get("memory_mb"),
            timeout_seconds=payload.get("timeout_seconds"),
            max_containers=payload.get("max_containers"),
            shard_count=payload.get("shard_count"),
            enable_workflow_batching=payload.get("enable_workflow_batching", False),
            enable_workflow_sessions=payload.get("enable_workflow_sessions", False),
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
    workspace_root: Path | None = field(default=None, repr=False, compare=False)

    admission_records: dict[str, Any] = field(default_factory=dict, repr=False)

    admission_readiness_records: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _refreshed_admissions: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _reuse_only_admissions: set[str] = field(default_factory=set, init=False, repr=False)
    _workflow_session: Any = field(default=None, init=False, repr=False)
    session_events: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    kind: str = "modal"

    def identity(self) -> dict[str, Any]:
        """Return a serializable description of this runner."""
        return {
            "kind": self.kind,
            "resources": self.resources.to_dict(),
        }

    @contextmanager
    def workflow_scope(self):
        if not self.resources.enable_workflow_sessions:
            yield
            return
        if self._workflow_session is not None:
            raise RuntimeError('A ModalRunner cannot serve concurrent workflow scopes')
        from pipelines_v2.runtime.sdxl_scope import SDXLWorkflowScope
        scope = SDXLWorkflowScope(self)
        previous = dict(self.admission_records)
        self._workflow_session = scope
        try:
            yield scope
        finally:
            import sys
            active_error = sys.exc_info()[0] is not None
            try:
                scope.close()
            except BaseException as close_error:
                scope.events.append({'event':'close_failed','error':type(close_error).__name__})
                if not active_error:
                    raise
            finally:
                self.session_events.extend(scope.events)
                self._workflow_session = None
                self.admission_records.clear()
                self.admission_records.update(previous)

    def prepare_execution_identity(self, spec):
        """Recover execution facts, then refresh CPU-visible target facts before reuse.

        The catalog is only a candidate index. A previous GPU observation is
        retained verbatim only after request/source/envelope and fresh target
        runtime/assets agree; CPU readiness never supplies GPU numerical facts.
        Actual payload integrity remains the orchestrator's C03 gate.
        """
        scope = self._workflow_session
        if scope is not None and scope.supports(spec):
            self.plan(spec).validate()
            self.admission_records[spec.semantic_hash()] = scope.observe(spec)
            return
        from pipelines_v2.runtime.readiness import AdmissionRecord, admitted_provenance_for_request, readiness_matches_execution
        key = spec.semantic_hash()
        self._refreshed_admissions.pop(key, None)
        self.admission_readiness_records.pop(key, None)
        existing = self.admission_records.pop(key, None)
        candidates = [(existing, key in self._reuse_only_admissions)] if existing is not None else []
        self._reuse_only_admissions.discard(key)
        # Bounded discovery supports a fresh CLI process and both resume and
        # cross-run reuse. Unsupported/legacy catalogs simply leave C01 unknown.
        lister = getattr(self.catalog, 'list_workflow_runs', None)
        if callable(lister):
            for run in lister(limit=100):
                for step in self.catalog.list_workflow_steps(run.run_id):
                    if step.status not in {'completed', 'reused'} or not step.artifact_id:
                        continue
                    manifest = self.catalog.load_artifact(step.artifact_id)
                    if manifest is None or manifest.operation_semantic_hash != key:
                        continue
                    payload = manifest.metadata.get('admission_v1') or (manifest.workflow_context or {}).get('admission_v1')
                    if payload is None:
                        continue
                    try:
                        record = AdmissionRecord.from_dict(payload)
                        provenance = manifest.provenance()
                        if provenance is None or provenance.scientific_identity != record.provenance.scientific_identity:
                            continue
                        # Check the descriptor before spending a readiness call.
                        from pipelines_v2.core.provenance_summaries import payload_identity
                        payload_identity(manifest)
                    except (ValueError, KeyError, TypeError):
                        continue
                    candidates.append((record, True))
        from pipelines_v2.core.provenance_observation import _OBSERVERS
        target_observer = spec.kind in _OBSERVERS
        readiness = None
        for candidate, reuse_only in candidates:
            try:
                record = candidate if isinstance(candidate, AdmissionRecord) else AdmissionRecord.from_dict(candidate)
                if target_observer:
                    from pipelines_v2.runtime.admission import workload_envelope, WorkloadEnvelope
                    from pipelines_v2.core.provenance import source_snapshot
                    from pipelines_v2.core.paths import find_workspace_root
                    if record.scope != 'execution' or not record.ready or not record.provenance.identity_complete:
                        continue
                    if workload_envelope(spec, self.identity()).identity != WorkloadEnvelope.from_dict(record.envelope).identity:
                        continue
                    mounts = tuple(getattr(spec.runtime_spec(), 'local_python_sources', ()) or ('pipelines_v2',))
                    current_source = source_snapshot(self.workspace_root or find_workspace_root(), mounts)
                    if current_source.get('unresolved') or current_source['mounted_source_digest'] != record.provenance.to_dict()['source']['mounted_source_digest']:
                        continue
                else:
                    admitted_provenance_for_request(spec, self.identity(), record, workspace_root=self.workspace_root)
            except (ValueError, KeyError, TypeError, OSError):
                continue
            if readiness is None:
                readiness = self.readiness(spec)
            if not readiness_matches_execution(readiness, record):
                continue
            if target_observer:
                try:
                    admitted_provenance_for_request(spec, self.identity(), record, workspace_root=self.workspace_root, target_readiness=readiness)
                except (ValueError, KeyError, TypeError, OSError):
                    continue
                self.admission_readiness_records[key] = readiness
            self.admission_records[key] = record
            if reuse_only:
                self._reuse_only_admissions.add(key)
            self._refreshed_admissions[key] = record.provenance.scientific_identity
            return

    def plan(self, spec: OperationSpec) -> ExecutionPlan:
        """Preflight a spec against Modal secret bindings and capabilities."""
        from pipelines_v2.runtime.admission import static_admission, effective_configuration, workload_envelope
        engine = spec.bound_engine()
        if isinstance(spec, CaptureSpec):
            artifact_kinds = ("capture",)
        elif isinstance(spec, (GenerationRunSpec, PatchedGenerationSpec)):
            artifact_kinds = (spec.kind,)
        else:
            artifact_kinds = ((operation_artifact_kind(spec),) if is_artifact_operation(spec) else ())
        errors = list(spec_plan_errors(spec, remote_assets=True))
        errors.extend(self._plan_errors(spec))
        return ExecutionPlan(
            spec_kind=spec.kind,
            required_capabilities=frozenset(spec.required_capabilities()),
            engine_capabilities=frozenset(engine.capabilities()) if engine is not None else frozenset(),
            artifact_kinds=artifact_kinds,
            checks=("capabilities", "runtime_spec", "artifact_store", "catalog"),
            errors=tuple(errors),
            admission_checks=static_admission(spec, self.identity(), workspace_root=self.workspace_root, artifact_store=self.artifacts, catalog=self.catalog),
            effective_config=effective_configuration(spec, self.identity()),
            workload_envelope=workload_envelope(spec, self.identity()),
        )

    def readiness(self, spec: OperationSpec):
        """Explicit online CPU checks in the target image; allocates no GPU."""
        from pipelines_v2.runtime.readiness import AdmissionRecord
        self.plan(spec).validate()
        payload = run_on_modal(runner_config=self.identity(), store_config=self.artifacts.identity(),
            spec_payload=spec.to_dict(), workspace_root=self.workspace_root, readiness_only=True)
        return AdmissionRecord.from_dict(payload)

    def run(
        self,
        spec: OperationSpec,
        *,
        workflow_context: WorkflowStepContext | None = None,
        progress_callback: Any | None = None,
    ) -> Any:
        """Execute one supported spec remotely and return its artifact."""
        self.plan(spec).validate()
        if (isinstance(spec, (CaptureSpec, GenerationRunSpec, PatchedGenerationSpec)) or is_artifact_operation(spec)):
            return self._run_remote(spec, workflow_context=workflow_context, progress_callback=progress_callback)
        raise NotImplementedError(f"ModalRunner cannot run {spec.kind!r} specs yet")

    def workflow_batch_key(self, spec: OperationSpec) -> str | None:
        """Return a conservative key for workflow steps that can share one Modal batch."""
        if not self.resources.enable_workflow_batching:
            return None
        if not (isinstance(spec, (CaptureSpec, GenerationRunSpec, PatchedGenerationSpec)) or is_artifact_operation(spec)):
            return None
        runtime_spec = spec.runtime_spec()
        if getattr(runtime_spec, "kind", None) == "external_tool":
            return None  # One isolated method/seed unit; never share a native engine session.
        if not isinstance(runtime_spec, PythonRuntimeSpec):
            return None
        engine = spec.bound_engine()
        engine_identity = engine.identity() if engine is not None else {}
        if dict(engine_identity).get("kind") == "vllm" and isinstance(
            spec,
            (CaptureSpec, GenerationRunSpec, PatchedGenerationSpec),
        ):
            return stable_hash(
                {
                    "runner": self.identity(),
                    "kind": "vllm_model_session",
                    "engine": engine_identity,
                    "runtime": {
                        "lock": runtime_spec.lock.identity if runtime_spec.lock else None,
                        "runtime_identity": runtime_spec.runtime_identity,
                        "python_version": runtime_spec.python_version,
                        "pip_packages": list(runtime_spec.pip_packages),
                        "local_python_sources": list(runtime_spec.local_python_sources),
                    },
                }
            )
        if int(self.resources.shard_count or 1) != 1:
            return None
        patch_family = None
        if isinstance(spec, PatchedGenerationSpec):
            operator = str(spec.patch.operator)
            if operator in {"project_out", "add_direction", "swap_mean", "swap_components", "random_control"}:
                patch_family = "subspace"
            elif spec.patch.requires_pairing():
                patch_family = "paired"
            else:
                patch_family = operator
        return stable_hash(
            {
                "runner": self.identity(),
                "spec_kind": spec.kind,
                "engine": engine_identity,
                "patch_family": patch_family,
                "runtime": {
                    "lock": runtime_spec.lock.identity if runtime_spec.lock else None,
                    "runtime_identity": runtime_spec.runtime_identity,
                    "python_version": runtime_spec.python_version,
                    "pip_packages": list(runtime_spec.pip_packages),
                    "env": dict(runtime_spec.env),
                    "local_python_sources": list(runtime_spec.local_python_sources),
                },
            }
        )

    def run_many(
        self,
        specs: Sequence[OperationSpec],
        *,
        workflow_contexts: Sequence[WorkflowStepContext | None] | None = None,
        progress_callback: Any | None = None,
    ) -> list[CaptureArtifact | OperationArtifact]:
        """Execute compatible specs in one Modal app invocation."""
        specs_tuple = tuple(specs)
        if not specs_tuple:
            return []
        contexts = (
            tuple(workflow_contexts)
            if workflow_contexts is not None
            else tuple(None for _ in specs_tuple)
        )
        if len(contexts) != len(specs_tuple):
            raise ValueError(
                "ModalRunner.run_many expected one workflow context per spec: "
                f"got {len(contexts)} contexts for {len(specs_tuple)} specs"
            )
        for spec in specs_tuple:
            self.plan(spec).validate()
            if not (isinstance(spec, (CaptureSpec, GenerationRunSpec, PatchedGenerationSpec)) or is_artifact_operation(spec)):
                raise NotImplementedError(f"ModalRunner cannot run {spec.kind!r} specs yet")
        if self._workflow_session is not None and all(self._workflow_session.supports(spec) for spec in specs_tuple):
            return [self._run_remote(spec, workflow_context=context, progress_callback=progress_callback)
                    for spec, context in zip(specs_tuple, contexts, strict=True)]
        manifest_payloads = run_many_on_modal(
            runner_config=self.identity(),
            store_config=self.artifacts.identity(),
            spec_payloads=[spec.to_dict() for spec in specs_tuple],
            workflow_contexts=[
                {**(context.to_manifest_dict() if context is not None else {}),
                 "provenance_v1": resolve_for_runner(spec, self).to_dict(),
                 **self._admission_context(spec)}
                for spec, context in zip(specs_tuple, contexts, strict=True)
            ],
            workspace_root=self.workspace_root,
            progress_callback=progress_callback,
        )
        if len(manifest_payloads) != len(specs_tuple):
            raise RuntimeError(
                "Modal runner received a different number of batch results than specs: "
                f"got {len(manifest_payloads)}, expected {len(specs_tuple)}"
            )
        artifacts: list[CaptureArtifact | OperationArtifact] = []
        for manifest_payload in manifest_payloads:
            if manifest_payload is None:
                raise RuntimeError("Modal runner did not receive a manifest payload for one batched spec")
            manifest = ArtifactManifest.from_dict(manifest_payload)
            try:
                self.catalog.record_artifact(manifest)
            except NotImplementedError:
                pass
            if manifest.artifact_kind == "capture":
                artifacts.append(CaptureArtifact(_manifest=manifest, store=self.artifacts))
            else:
                artifacts.append(OperationArtifact(_manifest=manifest, store=self.artifacts))
        return artifacts

    def refresh_admission_for_reuse(self, spec, record) -> bool:
        """Mutable remote checkpoint aliases require fresh CPU observations."""
        from pipelines_v2.runtime.readiness import readiness_matches_execution
        if self._refreshed_admissions.get(spec.semantic_hash()) == record.provenance.scientific_identity:
            return True
        return readiness_matches_execution(self.readiness(spec), record)

    def _admission_context(self, spec):
        # Historical catalog observations authorize reuse, never constrain a new
        # GPU execution after a cache miss. Explicit admissions remain strict.
        if spec.semantic_hash() in self._reuse_only_admissions and self._workflow_session is None:
            return {}
        record = self.admission_records.get(spec.semantic_hash())
        if record is None:
            return {}
        from pipelines_v2.runtime.readiness import AdmissionRecord, admitted_provenance_for_request
        record = record if isinstance(record, AdmissionRecord) else AdmissionRecord.from_dict(record)
        admitted_provenance_for_request(spec, self.identity(), record, workspace_root=self.workspace_root,
            target_readiness=self.admission_readiness_records.get(spec.semantic_hash()))
        return {"admission_v1": record.to_dict()}

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
                {**(workflow_context.to_manifest_dict() if workflow_context is not None else {}),
                 "provenance_v1": resolve_for_runner(spec, self).to_dict(),
                 **self._admission_context(spec)}
            )
        if signature is None or "progress_callback" in signature.parameters:
            run_kwargs["progress_callback"] = progress_callback
        if signature is None or "workspace_root" in signature.parameters:
            run_kwargs["workspace_root"] = self.workspace_root
        if self._workflow_session is not None and self._workflow_session.supports(spec):
            if spec.semantic_hash() not in self.admission_records:
                self.prepare_execution_identity(spec)
                run_kwargs['workflow_context'] = {**run_kwargs.get('workflow_context', {}),
                    'provenance_v1': resolve_for_runner(spec, self).to_dict(), **self._admission_context(spec)}
            manifest_payload = self._workflow_session.execute(spec, run_kwargs.get('workflow_context'), progress_callback)
        else:
            manifest_payload = run_on_modal(**run_kwargs)
        if manifest_payload is None:
            raise RuntimeError("Modal runner did not receive a manifest payload; the remote run was likely cancelled")
        manifest = ArtifactManifest.from_dict(manifest_payload)
        qualification = manifest.metadata.get('qualification_v1')
        if qualification is not None:
            self.catalog.record_artifact(ArtifactManifest.from_dict(qualification))
        try:
            self.catalog.record_artifact(manifest)
        except NotImplementedError:
            pass
        if manifest.artifact_kind == "capture":
            return CaptureArtifact(_manifest=manifest, store=self.artifacts)
        return OperationArtifact(_manifest=manifest, store=self.artifacts)

    def _plan_errors(self, spec: OperationSpec) -> list[str]:
        errors: list[str] = []
        for name in ("memory_mb", "timeout_seconds", "max_containers", "shard_count", "client_timeout_seconds", "merge_timeout_seconds", "merge_memory_mb", "readiness_timeout_seconds"):
            value = getattr(self.resources, name)
            if value is not None and (type(value) is not int or value <= 0):
                errors.append(f"ModalResources.{name} must be a positive integer")
        cpu = self.resources.cpu
        if cpu is not None and (type(cpu) not in (int, float) or not math.isfinite(cpu) or cpu <= 0):
            errors.append("ModalResources.cpu must be a positive number")
        if type(self.resources.enable_workflow_sessions) is not bool:
            errors.append("ModalResources.enable_workflow_sessions must be a boolean")
        if self.resources.enable_workflow_sessions and spec.bound_engine() is not None and spec.bound_engine().identity().get('kind') == 'sdxl':
            if (self.resources.shard_count or 1) != 1 or (self.resources.max_containers or 1) != 1:
                errors.append('SDXL workflow sessions require one worker and one shard')
            if not self.resources.gpu or not self.resources.memory_mb or not self.resources.timeout_seconds:
                errors.append('SDXL workflow sessions require explicit GPU, host memory and timeout bounds')
        if type(self.resources.enable_workflow_batching) is not bool:
            errors.append("ModalResources.enable_workflow_batching must be a boolean")
        mounts = ()
        try:
            mounts = _mounted_volumes(store_config=self.artifacts.identity(), resources=self.resources.to_dict())
        except ValueError as exc:
            errors.append(str(exc))
        engine = spec.bound_engine()
        if engine is not None and engine.identity().get('kind') in {'torch','torch_native'}:
            if engine.device=='cuda' and not self.resources.gpu:
                errors.append('CUDA Torch execution requires an explicit Modal GPU')
            if engine.device=='cpu' and self.resources.gpu:
                errors.append('CPU Torch execution must not allocate a GPU')
            if not self.resources.memory_mb or not self.resources.timeout_seconds:
                errors.append('Torch Modal execution requires explicit host memory and timeout')
            model_path=PurePosixPath(engine.resolved_model_path())
            if not model_path.is_absolute() or not any(model_path.is_relative_to(PurePosixPath(m.mount_path)) for m in mounts):
                errors.append('Prepared Torch model path must be covered by a Modal volume mount')
        if engine is not None and engine.identity().get("kind") == "sdxl":
            model_path = PurePosixPath(engine.resolved_model_path())
            if not any(model_path.is_relative_to(PurePosixPath(mount.mount_path)) for mount in mounts):
                errors.append(f"SDXL model path {str(model_path)!r} is not covered by a Modal volume mount")
        if engine is not None and engine.identity().get("kind") == "vllm":
            gpu = self.resources.gpu
            gpu_count = None
            if not isinstance(gpu, str) or not gpu.strip():
                errors.append("Modal vLLM execution requires ModalResources.gpu")
            else:
                name, separator, count = gpu.partition(":")
                if not name.strip() or (separator and (not count.isdigit() or int(count) <= 0)):
                    errors.append("ModalResources.gpu must be a GPU type optionally followed by :count")
                else:
                    gpu_count = int(count) if separator else 1
            tp = getattr(engine, "tensor_parallel_size", 1)
            pp = getattr(engine, "pipeline_parallel_size", 1)
            if gpu_count is not None and type(tp) is int and type(pp) is int and tp * pp > gpu_count:
                errors.append(
                    f"vLLM tensor_parallel_size * pipeline_parallel_size requires {tp * pp} GPUs "
                    f"per container, but ModalResources.gpu={gpu!r} provides {gpu_count}; "
                    "shard_count does not add GPUs to a container"
                )
            model_path = PurePosixPath(engine.resolved_model_path())
            if model_path.is_absolute() and not any(
                model_path.is_relative_to(PurePosixPath(mount.mount_path)) for mount in mounts
            ):
                errors.append(f"Model path {str(model_path)!r} is not covered by a Modal volume mount")
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
