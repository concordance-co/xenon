"""Generic remote execution entrypoint for runner backends."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import replace
from itertools import count
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pipelines_v2.core.types import OperationSpec, stable_hash, utc_now_iso
from pipelines_v2.data.datasets import Dataset
from pipelines_v2.operations import operation_spec_from_dict
from pipelines_v2.operations.specs import (
    CaptureSpec,
    GenerationRunSpec,
    PatchedGenerationSpec,
    ReportSpec,
    MoERoutingSite,
    ResidualSite,
)
from pipelines_v2.runtime.artifacts import ARTIFACT_BOUND_SPECS as _ARTIFACT_BOUND_SPECS
from pipelines_v2.storage import ArtifactManifest, artifact_store_from_dict
from pipelines_v2.storage.artifacts import ArtifactLabelRef, CaptureArtifact, FeatureLayerRef, FeatureRef, OperationArtifact
from pipelines_v2.storage.features import load_feature_payload, write_capture_features
from pipelines_v2.operations.interventions.runtime import rows_example_coverage

_PROGRESS_LOG = logging.getLogger("pipelines_v2.remote_progress")
_REMOTE_PROGRESS_SEQUENCE = count(1)
RemoteProgressCallback = Callable[[Mapping[str, Any]], None]


def execute_remote(
    *,
    runner_config: dict[str, Any],
    store_config: dict[str, Any],
    spec_payload: dict[str, Any],
    workflow_context: dict[str, Any] | None = None,
    progress_callback: RemoteProgressCallback | None = None,
) -> dict[str, Any]:
    """Execute a serialized operation in a remote worker."""
    spec = operation_spec_from_dict(spec_payload)
    return _execute_remote_spec(
        runner_config=runner_config,
        store_config=store_config,
        spec=spec,
        workflow_context=workflow_context,
        execution_session=None,
        progress_callback=progress_callback,
    )


def execute_remote_many(
    *,
    runner_config: dict[str, Any],
    store_config: dict[str, Any],
    spec_payloads: list[dict[str, Any]],
    workflow_contexts: list[dict[str, Any] | None] | None = None,
    progress_callback: RemoteProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Execute multiple serialized operations in one remote worker process."""
    contexts = list(workflow_contexts) if workflow_contexts is not None else [None] * len(spec_payloads)
    if len(contexts) != len(spec_payloads):
        raise ValueError(
            "execute_remote_many expected one workflow context per spec payload: "
            f"got {len(contexts)} contexts for {len(spec_payloads)} specs"
        )
    specs = [operation_spec_from_dict(spec_payload) for spec_payload in spec_payloads]
    execution_session = _RemoteExecutionSession(specs=specs)
    results: list[dict[str, Any]] = []
    for spec, workflow_context in zip(specs, contexts, strict=True):
        results.append(
            _execute_remote_spec(
                runner_config=runner_config,
                store_config=store_config,
                spec=spec,
                workflow_context=workflow_context,
                execution_session=execution_session,
                progress_callback=progress_callback,
            )
        )
    return results


def _execute_remote_spec(
    *,
    runner_config: dict[str, Any],
    store_config: dict[str, Any],
    spec: OperationSpec,
    workflow_context: dict[str, Any] | None,
    execution_session: "_RemoteExecutionSession | None",
    progress_callback: RemoteProgressCallback | None,
) -> dict[str, Any]:
    _emit_remote_progress(
        workflow_context=workflow_context,
        status="running",
        stage="remote_started",
        spec_kind=spec.kind,
        message=f"remote execution started for {spec.kind}",
        progress_callback=progress_callback,
    )
    store = artifact_store_from_dict(store_config)

    if isinstance(spec, CaptureSpec):
        return _execute_capture(
            runner_config=runner_config,
            store=store,
            spec=spec,
            workflow_context=workflow_context,
            execution_session=execution_session,
            remote_progress_callback=progress_callback,
        )
    if isinstance(spec, GenerationRunSpec):
        return _execute_generation(
            runner_config=runner_config,
            store=store,
            spec=spec,
            workflow_context=workflow_context,
            execution_session=execution_session,
            remote_progress_callback=progress_callback,
        )
    if isinstance(spec, PatchedGenerationSpec):
        return _execute_intervention(
            runner_config=runner_config,
            store=store,
            spec=spec,
            workflow_context=workflow_context,
            execution_session=execution_session,
            remote_progress_callback=progress_callback,
        )
    if isinstance(spec, _ARTIFACT_BOUND_SPECS):
        return _execute_artifact_operation(
            runner_config=runner_config,
            store=store,
            spec=spec,
            workflow_context=workflow_context,
        )
    raise NotImplementedError(f"Remote executor cannot run {spec.kind!r} specs yet")


class _RemoteExecutionSession:
    """Remote process-local caches shared by one execute_remote_many call."""

    def __init__(self, *, specs: Sequence[OperationSpec] = ()) -> None:
        self._specs = tuple(specs)
        self._vllm_sessions: dict[str, Any] = {}
        self._vllm_intervention_runtimes: dict[str, Any] = {}

    def capture(
        self,
        *,
        engine: Any,
        spec: CaptureSpec,
        progress_callback: Any | None = None,
    ) -> Any:
        session = self._vllm_session(
            engine=engine,
            spec=spec,
            progress_callback=progress_callback,
        )
        if session is None:
            incremental_capture = getattr(engine, "capture_incremental", None)
            if progress_callback is not None and callable(incremental_capture):
                return incremental_capture(spec, progress_callback=progress_callback)
            return engine.capture(spec)
        if progress_callback is None:
            return session.capture(spec)
        return session.capture(spec, progress_callback=progress_callback)

    def generate(
        self,
        *,
        engine: Any,
        spec: GenerationRunSpec,
        batch_callback: Any | None = None,
        progress_callback: Any | None = None,
    ) -> Any:
        session = self._vllm_session(
            engine=engine,
            spec=spec,
            progress_callback=progress_callback,
        )
        if session is None:
            incremental_generate = getattr(engine, "generate_incremental", None)
            if batch_callback is not None and callable(incremental_generate):
                kwargs = {"batch_callback": batch_callback}
                if progress_callback is not None:
                    kwargs["progress_callback"] = progress_callback
                return incremental_generate(spec, **kwargs)
            result = engine.generate(spec)
            if batch_callback is not None:
                batch_callback(list(result.rows), dict(result.metadata))
            return result
        kwargs = {"batch_callback": batch_callback}
        if progress_callback is not None:
            kwargs["progress_callback"] = progress_callback
        return session.generate(spec, **kwargs)

    def intervene(
        self,
        *,
        engine: Any,
        spec: PatchedGenerationSpec,
        progress_callback: Any | None = None,
    ) -> Any:
        session = self._vllm_session(
            engine=engine,
            spec=spec,
            progress_callback=progress_callback,
        )
        if session is not None:
            return session.intervene(spec)
        identity = engine.identity() if callable(getattr(engine, "identity", None)) else {}
        if dict(identity).get("kind") != "vllm":
            return engine.intervene(spec)

        from pipelines_v2.engine.vllm.intervene import (
            build_vllm_intervention_runtime,
            run_vllm_intervention_with_runtime,
            vllm_intervention_session_key,
        )

        session_key = vllm_intervention_session_key(engine=engine, spec=spec)
        runtime = self._vllm_intervention_runtimes.get(session_key)
        if runtime is None:
            runtime = build_vllm_intervention_runtime(engine=engine, spec=spec)
            self._vllm_intervention_runtimes[session_key] = runtime
        return run_vllm_intervention_with_runtime(runtime=runtime, spec=spec)

    def _vllm_session(
        self,
        *,
        engine: Any,
        spec: OperationSpec,
        progress_callback: Any | None = None,
    ) -> Any | None:
        identity = engine.identity() if callable(getattr(engine, "identity", None)) else {}
        if dict(identity).get("kind") != "vllm":
            return None

        from pipelines_v2.engine.vllm.session import (
            build_vllm_session_runtime,
            vllm_session_key,
        )

        session_specs = self._vllm_session_specs(engine=engine, spec=spec)
        session_key = vllm_session_key(engine=engine, specs=session_specs)
        runtime = self._vllm_sessions.get(session_key)
        if runtime is None:
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "model_loading",
                        "status": "running",
                        "message": "Loading model runtime",
                    }
                )
            try:
                runtime = build_vllm_session_runtime(
                    engine=engine,
                    specs=session_specs,
                    progress_callback=progress_callback,
                )
            except Exception:
                if progress_callback is not None:
                    progress_callback(
                        {
                            "stage": "model_loading",
                            "status": "error",
                            "message": "Model runtime failed to load",
                        }
                    )
                raise
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "model_loading",
                        "status": "complete",
                        "current": 1,
                        "total": 1,
                        "message": "Model runtime ready",
                    }
                )
            self._vllm_sessions[session_key] = runtime
        return runtime

    def _vllm_session_specs(self, *, engine: Any, spec: OperationSpec) -> tuple[OperationSpec, ...]:
        identity = engine.identity() if callable(getattr(engine, "identity", None)) else {}

        from pipelines_v2.engine.vllm.session import vllm_modal_batch_family

        candidates = tuple(
            candidate
            for candidate in self._specs
            if isinstance(candidate, (CaptureSpec, GenerationRunSpec, PatchedGenerationSpec))
            and _same_engine_identity(candidate.bound_engine(), identity)
        )
        if not candidates:
            return (spec,)
        patch_families = {
            str(family)
            for candidate in candidates
            if isinstance(candidate, PatchedGenerationSpec)
            for family in (vllm_modal_batch_family(candidate),)
            if family is not None
        }
        if patch_families and patch_families <= {"subspace", "paired"}:
            return candidates
        spec_family = vllm_modal_batch_family(spec)
        if isinstance(spec, PatchedGenerationSpec):
            return tuple(
                candidate
                for candidate in candidates
                if not isinstance(candidate, PatchedGenerationSpec)
                or vllm_modal_batch_family(candidate) == spec_family
            )
        if len(patch_families) == 1:
            family = next(iter(patch_families))
            return tuple(
                candidate
                for candidate in candidates
                if not isinstance(candidate, PatchedGenerationSpec)
                or vllm_modal_batch_family(candidate) == family
            )
        if len(patch_families) > 1:
            return tuple(candidate for candidate in candidates if not isinstance(candidate, PatchedGenerationSpec))
        return candidates


def _same_engine_identity(engine: Any | None, identity: dict[str, Any]) -> bool:
    if engine is None:
        return False
    engine_identity = engine.identity() if callable(getattr(engine, "identity", None)) else {}
    return dict(engine_identity) == dict(identity)


def merge_remote_shards(
    *,
    runner_config: dict[str, Any],
    store_config: dict[str, Any],
    spec_payload: dict[str, Any],
    shard_manifests: list[dict[str, Any]],
    workflow_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge per-shard remote artifacts into the step's final artifact."""
    spec = operation_spec_from_dict(spec_payload)
    store = artifact_store_from_dict(store_config)
    manifests = [ArtifactManifest.from_dict(payload) for payload in shard_manifests]
    if isinstance(spec, GenerationRunSpec):
        return _merge_generation_shards(
            runner_config=runner_config,
            store=store,
            spec=spec,
            shard_manifests=manifests,
            workflow_context=workflow_context,
        )
    if isinstance(spec, CaptureSpec):
        return _merge_capture_shards(
            runner_config=runner_config,
            store=store,
            spec=spec,
            shard_manifests=manifests,
            workflow_context=workflow_context,
        )
    if isinstance(spec, PatchedGenerationSpec):
        return _merge_intervention_shards(
            runner_config=runner_config,
            store=store,
            spec=spec,
            shard_manifests=manifests,
            workflow_context=workflow_context,
        )
    raise NotImplementedError(f"Remote executor cannot merge sharded {spec.kind!r} specs yet")


def _emit_remote_progress(
    *,
    workflow_context: dict[str, Any] | None,
    status: str,
    stage: str,
    spec_kind: str,
    message: str,
    metrics: dict[str, Any] | None = None,
    progress_callback: RemoteProgressCallback | None = None,
) -> None:
    context = dict(workflow_context or {})
    event_metrics = dict(metrics or {})
    shard = context.get("execution_shard")
    if isinstance(shard, dict):
        container_index = int(shard.get("index") or 0)
        event_metrics.setdefault("container_index", container_index)
        event_metrics.setdefault("container_count", int(shard.get("count") or 1))
        event_metrics.setdefault("container_id", f"container-{container_index + 1}")
        event_metrics.setdefault("container_label", f"Container {container_index + 1}")
    else:
        event_metrics.setdefault("container_index", int(context.get("container_index") or 0))
        event_metrics.setdefault("container_count", int(context.get("container_count") or 1))
        event_metrics.setdefault(
            "container_id",
            str(context.get("container_id") or "container-1"),
        )
        event_metrics.setdefault(
            "container_label",
            str(context.get("container_label") or "Container 1"),
        )
    if context.get("runtime_app_name"):
        event_metrics.setdefault("app_name", str(context["runtime_app_name"]))
    payload = {
        "schema_version": "xenon.progress.v1",
        "event_id": f"xpe_{uuid.uuid4().hex}",
        "sequence": next(_REMOTE_PROGRESS_SEQUENCE),
        "created_at": utc_now_iso(),
        "run_id": context.get("run_id"),
        "workflow_name": context.get("workflow_name"),
        "step_name": context.get("step_name"),
        "step_index": context.get("step_index"),
        "runner": context.get("runner"),
        "status": status,
        "stage": stage,
        "spec_kind": spec_kind,
        "message": message,
        "runtime_kind": context.get("runtime_kind"),
        "runtime_app_id": context.get("runtime_app_id"),
        "metrics": event_metrics,
    }
    _PROGRESS_LOG.info("XENON_PROGRESS %s", json.dumps(payload, sort_keys=True))
    if progress_callback is not None:
        try:
            progress_callback(payload)
        except Exception:
            _PROGRESS_LOG.warning(
                "remote progress callback failed",
                exc_info=True,
            )


def _engine_progress_callback(
    *,
    workflow_context: dict[str, Any] | None,
    spec_kind: str,
    remote_progress_callback: RemoteProgressCallback | None = None,
) -> Any:
    def _callback(payload: dict[str, Any]) -> None:
        metrics = {
            key: payload[key]
            for key in ("current", "total", "unit")
            if payload.get(key) is not None
        }
        _emit_remote_progress(
            workflow_context=workflow_context,
            status=str(payload.get("status") or "running"),
            stage=str(payload.get("stage") or "running"),
            spec_kind=spec_kind,
            message=str(payload.get("message") or ""),
            metrics=metrics,
            progress_callback=remote_progress_callback,
        )

    return _callback


def _execute_capture(
    *,
    runner_config: dict[str, Any],
    store: Any,
    spec: CaptureSpec,
    workflow_context: dict[str, Any] | None = None,
    execution_session: _RemoteExecutionSession | None = None,
    remote_progress_callback: RemoteProgressCallback | None = None,
) -> dict[str, Any]:
    engine = spec.bound_engine()
    if engine is None:
        raise RuntimeError("CaptureSpec is missing a bound engine")
    artifact_id = _artifact_id_for(spec=spec, workflow_context=workflow_context)
    existing = _load_existing_manifest(store=store, artifact_id=artifact_id, spec=spec)
    if existing is not None:
        return existing.to_dict()
    _ensure_artifact_dir(store, artifact_id)
    engine_progress_callback = (
        _engine_progress_callback(
            workflow_context=workflow_context,
            spec_kind=spec.kind,
            remote_progress_callback=remote_progress_callback,
        )
        if remote_progress_callback is not None
        else None
    )

    streamed_result = _execute_streaming_capture_if_available(
        engine=engine,
        spec=spec,
        workflow_context=workflow_context,
        execution_session=execution_session,
        progress_callback=engine_progress_callback,
    )
    if streamed_result is not None:
        result_features, result_generations, result_metadata, example_coverage = streamed_result
    else:
        resolved_spec = _resolve_dataset_for_execution(spec=spec, workflow_context=workflow_context)
        resolved_spec = _apply_execution_shard(spec=resolved_spec, workflow_context=workflow_context)
        if not resolved_spec.dataset.examples:
            result_features = _empty_capture_features(resolved_spec)
            result_generations = []
            result_metadata = {"empty_shard": True}
        else:
            if execution_session is None:
                incremental_capture = getattr(engine, "capture_incremental", None)
                if engine_progress_callback is not None and callable(incremental_capture):
                    result = incremental_capture(
                        resolved_spec,
                        progress_callback=engine_progress_callback,
                    )
                else:
                    result = engine.capture(resolved_spec)
            else:
                result = execution_session.capture(
                    engine=engine,
                    spec=resolved_spec,
                    progress_callback=engine_progress_callback,
                )
            result_features = result.features
            result_generations = result.generations
            result_metadata = dict(result.metadata)
        example_coverage = resolved_spec.dataset.coverage()

    storage_refs: dict[str, Any] = {"features": write_capture_features(store, artifact_id, result_features)}

    if result_generations:
        storage_refs["generations"] = store.write_json(
            artifact_id,
            "generations.json",
            result_generations,
        )

    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        artifact_kind=spec.kind,
        schema_version=1,
        operation_spec_hash=spec.spec_hash(),
        operation_semantic_hash=spec.semantic_hash(),
        created_at=utc_now_iso(),
        engine=engine.identity(),
        runner=runner_config,
        input_artifact_refs=(),
        example_coverage=example_coverage,
        storage_refs=storage_refs,
        metadata=_with_execution_shard_metadata(result_metadata, workflow_context=workflow_context),
        workflow_context=dict(workflow_context or {}),
    )
    storage_refs["manifest"] = store.write_json(
        artifact_id,
        "manifest.json",
        manifest.to_dict(),
    )
    return manifest.to_dict()


def _execute_streaming_capture_if_available(
    *,
    engine: Any,
    spec: CaptureSpec,
    workflow_context: dict[str, Any] | None,
    execution_session: _RemoteExecutionSession | None,
    progress_callback: Any | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]] | None:
    """Stream deferred Postgres capture inputs through a reusable execution session."""

    if execution_session is None:
        return None
    if not _is_streamable_deferred_postgres_dataset(spec.dataset):
        return None

    batch_size = max(1, int(getattr(engine, "max_num_seqs", 0) or 1))
    features = _empty_capture_features(spec)
    generations: list[dict[str, Any]] = []
    example_metadata: list[dict[str, Any]] = []
    prompt_hashes: dict[str, str | None] = {}
    example_keys: list[str] = []
    result_metadata: dict[str, Any] = {}
    batch_count = 0

    for batch_dataset in _iter_streamed_dataset_batches(
        dataset=spec.dataset,
        workflow_context=workflow_context,
        batch_size=batch_size,
    ):
        batch_spec = replace(spec, dataset=batch_dataset)
        batch_spec = _apply_execution_shard(spec=batch_spec, workflow_context=workflow_context)
        if not batch_spec.dataset.examples:
            continue
        batch_count += 1
        result = execution_session.capture(
            engine=engine,
            spec=batch_spec,
            progress_callback=progress_callback,
        )
        if not result_metadata:
            result_metadata = dict(result.metadata)
        for name, payload in result.features.items():
            _merge_feature_payload(features, str(name), payload)
        generations.extend(dict(row) for row in result.generations)
        for example in batch_spec.dataset.examples:
            example_keys.append(str(example.key))
            prompt_hashes[str(example.key)] = str(example.prompt_hash) if example.prompt_hash is not None else None
        batch_example_metadata = dict(result.metadata).get("example_metadata")
        if isinstance(batch_example_metadata, list):
            example_metadata.extend(dict(item) for item in batch_example_metadata if isinstance(item, dict))

    if batch_count == 0:
        result_metadata = {
            "empty_shard": True,
            "streamed_deferred_dataset": True,
            "streamed_dataset_batches": 0,
            "dataset_fetch_batch_size": batch_size,
        }
    else:
        result_metadata = {
            **result_metadata,
            "example_metadata": example_metadata,
            "example_count": len(example_keys),
            "spec_hash": spec.spec_hash(),
            "streamed_deferred_dataset": True,
            "streamed_dataset_batches": batch_count,
            "dataset_fetch_batch_size": batch_size,
        }

    coverage = {
        "dataset_id": spec.dataset.id,
        "dataset_name": spec.dataset.name,
        "materialized": True,
        "streamed": True,
        "example_count": len(example_keys),
        "example_keys": example_keys,
        "prompt_hashes": prompt_hashes,
    }
    return features, generations, result_metadata, coverage


def _is_streamable_deferred_postgres_dataset(dataset: Dataset) -> bool:
    if not getattr(dataset, "is_deferred", False):
        return False
    source = dict(getattr(dataset, "source", {}) or {})
    return source.get("kind") == "postgres"


def _iter_streamed_dataset_batches(
    *,
    dataset: Dataset,
    workflow_context: dict[str, Any] | None,
    batch_size: int,
) -> Any:
    from pipelines_v2.data.sources import source_from_dict

    source = source_from_dict(dict(dataset.source or {}))
    batch_iter = getattr(source, "iter_dataset_batches", None)
    if not callable(batch_iter):
        raise RuntimeError("Deferred Postgres dataset source does not support streaming batches")
    fetch = dict(dataset.fetch or {})
    shard = _execution_shard(workflow_context)
    if shard is not None and fetch.get("prompt_hash_column"):
        fetch["execution_shard"] = shard

    selection_keys = dataset.selection.get("keys")
    allowed_keys = {str(key) for key in selection_keys} if isinstance(selection_keys, list) else None
    selection_limit = dataset.selection.get("limit")
    remaining = int(selection_limit) if selection_limit is not None else None
    batch_index = 0
    for source_batch in batch_iter(batch_size=int(batch_size), **fetch):
        examples = []
        for example in source_batch.examples:
            if allowed_keys is not None and str(example.key) not in allowed_keys:
                continue
            if remaining is not None and remaining <= 0:
                break
            examples.append(example)
            if remaining is not None:
                remaining -= 1
        if examples:
            yield Dataset.from_examples(
                examples,
                id=dataset.id,
                name=f"{dataset.name or source_batch.name or 'dataset'}_batch_{batch_index}",
            )
            batch_index += 1
        if remaining is not None and remaining <= 0:
            break


def _execute_artifact_operation(
    *,
    runner_config: dict[str, Any],
    store: Any,
    spec: OperationSpec,
    workflow_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from pipelines_v2.operations.execute import execute_artifact_operation

    artifact_id = f"{spec.kind}_{spec.schema_version}_{uuid.uuid4().hex[:8]}"
    store.make_artifact_dir(artifact_id)
    result = execute_artifact_operation(spec)

    storage_refs: dict[str, Any] = {}
    if result.payload:
        storage_refs["result"] = store.write_json(
            artifact_id,
            "result.json",
            result.payload,
        )
    if result.features:
        storage_refs["features"] = write_capture_features(store, artifact_id, result.features)
    if result.labels:
        storage_refs["labels"] = {
            name: store.write_json(artifact_id, f"labels/{name}.json", payload)
            for name, payload in result.labels.items()
        }
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        artifact_kind=spec.kind,
        schema_version=1,
        operation_spec_hash=spec.spec_hash(),
        operation_semantic_hash=spec.semantic_hash(),
        created_at=utc_now_iso(),
        engine={},
        runner=runner_config,
        input_artifact_refs=tuple(_input_artifact_ids(spec)),
        example_coverage=result.example_coverage,
        storage_refs=storage_refs,
        metadata=result.metadata,
        workflow_context=dict(workflow_context or {}),
    )
    storage_refs["manifest"] = store.write_json(
        artifact_id,
        "manifest.json",
        manifest.to_dict(),
    )
    return manifest.to_dict()


def _execute_intervention(
    *,
    runner_config: dict[str, Any],
    store: Any,
    spec: PatchedGenerationSpec,
    workflow_context: dict[str, Any] | None = None,
    execution_session: _RemoteExecutionSession | None = None,
    remote_progress_callback: RemoteProgressCallback | None = None,
) -> dict[str, Any]:
    engine = spec.bound_engine()
    if engine is None:
        raise RuntimeError("PatchedGenerationSpec is missing a bound engine")
    resolved_spec = _resolve_dataset_for_execution(spec=spec, workflow_context=workflow_context)
    resolved_spec = _apply_execution_shard(spec=resolved_spec, workflow_context=workflow_context)
    shard = _execution_shard(workflow_context)
    artifact_id = (
        _artifact_id_for(spec=spec, workflow_context=workflow_context)
        if shard is not None
        else f"{spec.kind}_{spec.schema_version}_{uuid.uuid4().hex[:8]}"
    )
    _ensure_artifact_dir(store, artifact_id)
    if not resolved_spec.dataset.examples:
        result_summary = {"example_count": 0, "patched_count": 0, "skipped_count": 0}
        result_rows: list[dict[str, Any]] = []
        result_metadata: dict[str, Any] = {"empty_shard": True}
    else:
        total_examples = len(resolved_spec.dataset.examples)
        _emit_remote_progress(
            workflow_context=workflow_context,
            status="running",
            stage="generation",
            spec_kind=spec.kind,
            message="Running patched generation",
            metrics={"current": 0, "total": total_examples, "unit": "prompts"},
            progress_callback=remote_progress_callback,
        )
        engine_progress_callback = (
            _engine_progress_callback(
                workflow_context=workflow_context,
                spec_kind=spec.kind,
                remote_progress_callback=remote_progress_callback,
            )
            if remote_progress_callback is not None
            else None
        )
        if execution_session is None:
            result = engine.intervene(resolved_spec)
        else:
            result = execution_session.intervene(
                engine=engine,
                spec=resolved_spec,
                progress_callback=engine_progress_callback,
            )
        _emit_remote_progress(
            workflow_context=workflow_context,
            status="complete",
            stage="generation",
            spec_kind=spec.kind,
            message="Patched generation complete",
            metrics={
                "current": total_examples,
                "total": total_examples,
                "unit": "prompts",
            },
            progress_callback=remote_progress_callback,
        )
        result_summary = dict(result.summary)
        result_rows = list(result.rows)
        result_metadata = dict(result.metadata)

    payload = {
        "kind": "patched_generation_result",
        "summary": result_summary,
        "rows": result_rows,
    }
    storage_refs: dict[str, Any] = {
        "result": store.write_json(artifact_id, "result.json", payload),
    }
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        artifact_kind=spec.kind,
        schema_version=1,
        operation_spec_hash=spec.spec_hash(),
        operation_semantic_hash=spec.semantic_hash(),
        created_at=utc_now_iso(),
        engine=engine.identity(),
        runner=runner_config,
        input_artifact_refs=tuple(_input_artifact_ids(spec)),
        example_coverage=rows_example_coverage(dataset=resolved_spec.dataset, rows=result_rows),
        storage_refs=storage_refs,
        metadata=_with_execution_shard_metadata(result_metadata, workflow_context=workflow_context),
        workflow_context=dict(workflow_context or {}),
    )
    storage_refs["manifest"] = store.write_json(
        artifact_id,
        "manifest.json",
        manifest.to_dict(),
    )
    return manifest.to_dict()


def _execute_generation(
    *,
    runner_config: dict[str, Any],
    store: Any,
    spec: GenerationRunSpec,
    workflow_context: dict[str, Any] | None = None,
    execution_session: _RemoteExecutionSession | None = None,
    remote_progress_callback: RemoteProgressCallback | None = None,
) -> dict[str, Any]:
    engine = spec.bound_engine()
    if engine is None:
        raise RuntimeError("GenerationRunSpec is missing a bound engine")
    resolved_spec = _resolve_dataset_for_execution(spec=spec, workflow_context=workflow_context)
    resolved_spec = _apply_execution_shard(spec=resolved_spec, workflow_context=workflow_context)
    artifact_id = _artifact_id_for(spec=spec, workflow_context=workflow_context)
    existing = _load_existing_manifest(store=store, artifact_id=artifact_id, spec=spec)
    if existing is not None:
        return existing.to_dict()
    _ensure_artifact_dir(store, artifact_id)
    engine_progress_callback = (
        _engine_progress_callback(
            workflow_context=workflow_context,
            spec_kind=spec.kind,
            remote_progress_callback=remote_progress_callback,
        )
        if remote_progress_callback is not None
        else None
    )
    partial_rows = _load_partial_generation_rows(store=store, artifact_id=artifact_id)
    result_rows = _merge_generation_rows(dataset=resolved_spec.dataset, rows=partial_rows)
    completed_keys = _generation_resume_keys(result_rows)
    remaining_examples = [
        example
        for example in resolved_spec.dataset.examples
        if str(example.prompt_hash or "") not in completed_keys and str(example.key) not in completed_keys
    ]
    if not resolved_spec.dataset.examples:
        result_rows: list[dict[str, Any]] = []
        result_metadata: dict[str, Any] = {"empty_shard": True}
    elif not remaining_examples:
        result_metadata = {
            "backend": "resume",
            "resumed_from_partial": bool(partial_rows),
            "partial_row_count": len(partial_rows),
            "completed_from_partial": True,
        }
    else:
        generation_spec = replace(
            resolved_spec,
            dataset=Dataset.from_examples(
                remaining_examples,
                id=resolved_spec.dataset.id,
                name=f"{resolved_spec.dataset.name or 'dataset'}_remaining",
            ),
        )

        def _record_generation_checkpoint(batch_rows: list[dict[str, Any]], batch_metadata: dict[str, Any]) -> None:
            nonlocal result_rows
            if not batch_rows:
                return
            result_rows = _merge_generation_rows(
                dataset=resolved_spec.dataset,
                rows=[*result_rows, *batch_rows],
            )
            _write_generation_result_payload(
                store=store,
                artifact_id=artifact_id,
                rows=result_rows,
                total_example_count=len(resolved_spec.dataset.examples),
                partial=True,
                metadata={
                    "checkpoint_metadata": dict(batch_metadata),
                    "resumed_from_partial": bool(partial_rows),
                    "partial_row_count": len(partial_rows),
                },
            )
            _emit_remote_progress(
                workflow_context=workflow_context,
                status="running",
                stage="generation",
                spec_kind=spec.kind,
                message="generation checkpoint written",
                metrics={
                    "current": len(result_rows),
                    "total": len(resolved_spec.dataset.examples),
                    "unit": "prompts",
                    "completed_examples": len(result_rows),
                    "remaining_examples": max(0, len(resolved_spec.dataset.examples) - len(result_rows)),
                    "total_examples": len(resolved_spec.dataset.examples),
                },
                progress_callback=remote_progress_callback,
            )

        incremental_generate = getattr(engine, "generate_incremental", None)
        if execution_session is not None:
            result = execution_session.generate(
                engine=engine,
                spec=generation_spec,
                batch_callback=_record_generation_checkpoint,
                progress_callback=engine_progress_callback,
            )
        elif callable(incremental_generate):
            kwargs = {"batch_callback": _record_generation_checkpoint}
            if engine_progress_callback is not None:
                kwargs["progress_callback"] = engine_progress_callback
            result = incremental_generate(generation_spec, **kwargs)
        else:
            result = engine.generate(generation_spec)
            _record_generation_checkpoint(list(result.rows), dict(result.metadata))
        result_rows = _merge_generation_rows(
            dataset=resolved_spec.dataset,
            rows=[*result_rows, *list(result.rows)],
        )
        result_metadata = dict(result.metadata)
        if partial_rows:
            result_metadata["resumed_from_partial"] = True
            result_metadata["partial_row_count"] = len(partial_rows)

    storage_refs: dict[str, Any] = {
        "result": _write_generation_result_payload(
            store=store,
            artifact_id=artifact_id,
            rows=result_rows,
            total_example_count=len(resolved_spec.dataset.examples),
            partial=False,
            metadata=result_metadata,
        ),
    }
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        artifact_kind=spec.kind,
        schema_version=1,
        operation_spec_hash=spec.spec_hash(),
        operation_semantic_hash=spec.semantic_hash(),
        created_at=utc_now_iso(),
        engine=engine.identity(),
        runner=runner_config,
        input_artifact_refs=tuple(_input_artifact_ids(spec)),
        example_coverage=rows_example_coverage(dataset=resolved_spec.dataset, rows=result_rows),
        storage_refs=storage_refs,
        metadata=_with_execution_shard_metadata(result_metadata, workflow_context=workflow_context),
        workflow_context=dict(workflow_context or {}),
    )
    storage_refs["manifest"] = store.write_json(
        artifact_id,
        "manifest.json",
        manifest.to_dict(),
    )
    return manifest.to_dict()


def _merge_generation_shards(
    *,
    runner_config: dict[str, Any],
    store: Any,
    spec: GenerationRunSpec,
    shard_manifests: list[ArtifactManifest],
    workflow_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_spec = spec.resolve_dataset()
    artifact_id = _artifact_id_for(spec=spec, workflow_context=workflow_context)
    existing = _load_existing_manifest(store=store, artifact_id=artifact_id, spec=spec)
    if existing is not None:
        return existing.to_dict()
    _ensure_artifact_dir(store, artifact_id)

    rows: list[dict[str, Any]] = []
    seen_prompt_hashes: set[str] = set()
    for shard in _ordered_shard_manifests(shard_manifests):
        ref = shard.storage_refs.get("result")
        if not isinstance(ref, dict):
            continue
        payload = store.read_json_ref(ref)
        for row in payload.get("rows", []) if isinstance(payload, dict) else []:
            if not isinstance(row, dict):
                continue
            row_key = _row_prompt_hash(row) or str(row.get("example_key") or "")
            if row_key in seen_prompt_hashes:
                continue
            seen_prompt_hashes.add(row_key)
            rows.append(dict(row))

    order = {example.key: index for index, example in enumerate(resolved_spec.dataset.examples)}
    rows.sort(key=lambda row: order.get(str(row.get("example_key") or ""), len(order)))
    payload = {
        "kind": "generation_run_result",
        "summary": {
            "example_count": len(rows),
            "sharded": True,
            "shard_count": len(shard_manifests),
        },
        "rows": rows,
    }
    storage_refs: dict[str, Any] = {
        "result": store.write_json(artifact_id, "result.json", payload),
    }
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        artifact_kind=spec.kind,
        schema_version=1,
        operation_spec_hash=spec.spec_hash(),
        operation_semantic_hash=spec.semantic_hash(),
        created_at=utc_now_iso(),
        engine=spec.engine.identity() if spec.engine is not None else {},
        runner=runner_config,
        input_artifact_refs=tuple(_input_artifact_ids(spec)),
        example_coverage=rows_example_coverage(dataset=resolved_spec.dataset, rows=rows),
        storage_refs=storage_refs,
        metadata={
            "sharded": True,
            "shards": [_shard_manifest_summary(shard) for shard in _ordered_shard_manifests(shard_manifests)],
        },
        workflow_context=dict(workflow_context or {}),
    )
    storage_refs["manifest"] = store.write_json(artifact_id, "manifest.json", manifest.to_dict())
    return manifest.to_dict()


def _merge_capture_shards(
    *,
    runner_config: dict[str, Any],
    store: Any,
    spec: CaptureSpec,
    shard_manifests: list[ArtifactManifest],
    workflow_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_id = _artifact_id_for(spec=spec, workflow_context=workflow_context)
    existing = _load_existing_manifest(store=store, artifact_id=artifact_id, spec=spec)
    if existing is not None:
        return existing.to_dict()
    _ensure_artifact_dir(store, artifact_id)

    features = _empty_capture_features(spec)
    generations: list[dict[str, Any]] = []
    example_metadata: list[dict[str, Any]] = []
    for shard in _ordered_shard_manifests(shard_manifests):
        for name, ref in dict(shard.storage_refs.get("features", {})).items():
            if not isinstance(ref, dict):
                continue
            payload = load_feature_payload(store, ref)
            _merge_feature_payload(features, str(name), payload)
        ref = shard.storage_refs.get("generations")
        if isinstance(ref, dict):
            payload = store.read_json_ref(ref)
            if isinstance(payload, list):
                generations.extend(dict(row) for row in payload if isinstance(row, dict))
        metadata = dict(shard.metadata)
        shard_example_metadata = metadata.get("example_metadata")
        if isinstance(shard_example_metadata, list):
            example_metadata.extend(dict(item) for item in shard_example_metadata if isinstance(item, dict))

    storage_refs: dict[str, Any] = {
        "features": write_capture_features(store, artifact_id, features),
    }
    if generations:
        storage_refs["generations"] = store.write_json(artifact_id, "generations.json", generations)
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        artifact_kind=spec.kind,
        schema_version=1,
        operation_spec_hash=spec.spec_hash(),
        operation_semantic_hash=spec.semantic_hash(),
        created_at=utc_now_iso(),
        engine=spec.engine.identity() if spec.engine is not None else {},
        runner=runner_config,
        input_artifact_refs=(),
        example_coverage=spec.dataset.coverage(),
        storage_refs=storage_refs,
        metadata={
            "sharded": True,
            "example_metadata": sorted(example_metadata, key=lambda item: str(item.get("example_key") or "")),
            "shards": [_shard_manifest_summary(shard) for shard in _ordered_shard_manifests(shard_manifests)],
        },
        workflow_context=dict(workflow_context or {}),
    )
    storage_refs["manifest"] = store.write_json(artifact_id, "manifest.json", manifest.to_dict())
    return manifest.to_dict()


def _merge_intervention_shards(
    *,
    runner_config: dict[str, Any],
    store: Any,
    spec: PatchedGenerationSpec,
    shard_manifests: list[ArtifactManifest],
    workflow_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_spec = spec.resolve_dataset()
    artifact_id = _artifact_id_for(spec=spec, workflow_context=workflow_context)
    existing = _load_existing_manifest(store=store, artifact_id=artifact_id, spec=spec)
    if existing is not None:
        return existing.to_dict()
    _ensure_artifact_dir(store, artifact_id)

    rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for shard in _ordered_shard_manifests(shard_manifests):
        ref = shard.storage_refs.get("result")
        if not isinstance(ref, dict):
            continue
        payload = store.read_json_ref(ref)
        for row in payload.get("rows", []) if isinstance(payload, dict) else []:
            if not isinstance(row, dict):
                continue
            row_key = _patched_generation_row_key(row)
            if row_key in seen_keys:
                continue
            seen_keys.add(row_key)
            rows.append(dict(row))

    order = {example.key: index for index, example in enumerate(resolved_spec.dataset.examples)}
    rows.sort(
        key=lambda row: (
            order.get(str(row.get("example_key") or ""), len(order)),
            str(row.get("case_key") or ""),
            str(row.get("donor_example_key") or ""),
            str(row.get("request_id") or ""),
        )
    )
    ok_rows = [row for row in rows if str(row.get("status") or "ok") == "ok"]
    payload = {
        "kind": "patched_generation_result",
        "summary": {
            "example_count": len(rows),
            "patched_count": len(ok_rows),
            "skipped_count": len(rows) - len(ok_rows),
            "sharded": True,
            "shard_count": len(shard_manifests),
        },
        "rows": rows,
    }
    storage_refs: dict[str, Any] = {
        "result": store.write_json(artifact_id, "result.json", payload),
    }
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        artifact_kind=spec.kind,
        schema_version=1,
        operation_spec_hash=spec.spec_hash(),
        operation_semantic_hash=spec.semantic_hash(),
        created_at=utc_now_iso(),
        engine=spec.engine.identity() if spec.engine is not None else {},
        runner=runner_config,
        input_artifact_refs=tuple(_input_artifact_ids(spec)),
        example_coverage=rows_example_coverage(dataset=resolved_spec.dataset, rows=rows),
        storage_refs=storage_refs,
        metadata={
            "sharded": True,
            "shards": [_shard_manifest_summary(shard) for shard in _ordered_shard_manifests(shard_manifests)],
        },
        workflow_context=dict(workflow_context or {}),
    )
    storage_refs["manifest"] = store.write_json(artifact_id, "manifest.json", manifest.to_dict())
    return manifest.to_dict()


def _artifact_id_for(*, spec: OperationSpec, workflow_context: dict[str, Any] | None) -> str:
    context = dict(workflow_context or {})
    step_key = str(context.get("workflow_step_key") or "").strip()
    run_id = str(context.get("run_id") or "").strip()
    if not step_key or not run_id:
        return f"{spec.kind}_{spec.schema_version}_{uuid.uuid4().hex[:8]}"
    shard = _execution_shard(context)
    suffix_payload: list[Any] = [run_id, step_key, spec.semantic_hash()]
    suffix = stable_hash(suffix_payload)[:12]
    artifact_id = f"{spec.kind}_{spec.schema_version}_{suffix}"
    if shard is None:
        return artifact_id
    return f"{artifact_id}_shard_{shard['index']:05d}_of_{shard['count']:05d}"


def _resolve_dataset_for_execution(*, spec: Any, workflow_context: dict[str, Any] | None) -> Any:
    dataset = getattr(spec, "dataset", None)
    shard = _execution_shard(workflow_context)
    if shard is not None and getattr(dataset, "is_deferred", False):
        source = dict(getattr(dataset, "source", {}) or {})
        fetch = dict(getattr(dataset, "fetch", {}) or {})
        if source.get("kind") == "postgres" and fetch.get("prompt_hash_column"):
            fetch["execution_shard"] = shard
            dataset = replace(dataset, fetch=fetch)
            spec = replace(spec, dataset=dataset)
    return spec.resolve_dataset()


def _apply_execution_shard(*, spec: Any, workflow_context: dict[str, Any] | None) -> Any:
    shard = _execution_shard(workflow_context)
    if shard is None:
        return spec
    dataset = spec.dataset
    selected = _shard_examples_for_spec(spec=spec, shard=shard)
    return replace(
        spec,
        dataset=Dataset.from_examples(
            selected,
            id=dataset.id,
            name=f"{dataset.name}_shard_{int(shard['index'])}_of_{int(shard['count'])}",
        ),
    )


def _shard_examples_for_spec(*, spec: Any, shard: dict[str, int]) -> list[Any]:
    dataset = spec.dataset
    examples = list(dataset.examples)
    count = int(shard["count"])
    index = int(shard["index"])
    if isinstance(spec, GenerationRunSpec) and spec.select_when is not None:
        allowed = _resolved_example_key_set(spec.select_when)
        examples = [example for example in examples if str(example.key) in allowed]
    if isinstance(spec, PatchedGenerationSpec):
        patch = spec.patch
        if patch is not None and patch.requires_pairing():
            case_values = _resolved_values_map(spec.pair_by)
            case_by_key = {
                str(key): str(value)
                for key, value in case_values.items()
                if value is not None
            }
            selected_cases = {
                case_key
                for case_key in set(case_by_key.values())
                if _stable_text_shard(case_key, count) == index
            }
            return [
                example
                for example in examples
                if case_by_key.get(str(example.key)) in selected_cases
            ]
        if spec.select_when is not None:
            allowed = _resolved_example_key_set(spec.select_when)
            examples = [example for example in examples if str(example.key) in allowed]
    return [
        example
        for example in examples
        if _prompt_hash_shard(example.prompt_hash, count) == index
    ]


def _resolved_example_key_set(value: Any) -> set[str]:
    if not hasattr(value, "resolve_example_keys"):
        return set()
    return {str(key) for key in value.resolve_example_keys()}


def _resolved_values_map(value: Any) -> dict[str, Any]:
    if not hasattr(value, "resolve_values"):
        return {}
    return {str(key): item for key, item in value.resolve_values().items()}


def _execution_shard(workflow_context: dict[str, Any] | None) -> dict[str, int] | None:
    if not isinstance(workflow_context, dict):
        return None
    raw = workflow_context.get("execution_shard")
    if not isinstance(raw, dict):
        return None
    count = int(raw.get("count") or 1)
    index = int(raw.get("index") or 0)
    if count <= 1:
        return None
    if index < 0 or index >= count:
        raise ValueError(f"Invalid execution shard index {index} for shard count {count}")
    return {"index": index, "count": count}


def _prompt_hash_shard(prompt_hash: str, count: int) -> int:
    prompt_hash_text = str(prompt_hash)
    try:
        value = int(prompt_hash_text[:16], 16)
    except ValueError:
        value = int(stable_hash(prompt_hash_text)[:16], 16)
    return value % int(count)


def _stable_text_shard(text: str, count: int) -> int:
    return int(stable_hash(str(text))[:16], 16) % int(count)


def _patched_generation_row_key(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("case_key") or ""),
        str(row.get("example_key") or ""),
        str(row.get("donor_example_key") or ""),
        str(row.get("request_id") or ""),
        str(row.get("skip_reason") or ""),
    ]
    return stable_hash(parts)


def _with_execution_shard_metadata(
    metadata: dict[str, Any],
    *,
    workflow_context: dict[str, Any] | None,
) -> dict[str, Any]:
    shard = _execution_shard(workflow_context)
    if shard is None:
        return metadata
    return {**metadata, "execution_shard": shard}


def _load_existing_manifest(*, store: Any, artifact_id: str, spec: OperationSpec) -> ArtifactManifest | None:
    root = getattr(store, "root", None)
    if root is None:
        return None
    ref = {
        "store": getattr(store, "kind", ""),
        "name": getattr(store, "name", None),
        "path": str(Path(str(root)) / artifact_id / "manifest.json"),
        "format": "json",
    }
    try:
        payload = store.read_json_ref(ref)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    manifest = ArtifactManifest.from_dict(payload)
    if manifest.artifact_kind != spec.kind:
        return None
    if manifest.operation_semantic_hash != spec.semantic_hash():
        return None
    return manifest


def _load_partial_generation_rows(*, store: Any, artifact_id: str) -> list[dict[str, Any]]:
    root = getattr(store, "root", None)
    if root is None:
        return []
    ref = {
        "store": getattr(store, "kind", ""),
        "name": getattr(store, "name", None),
        "path": str(Path(str(root)) / artifact_id / "result.json"),
        "format": "json",
    }
    try:
        payload = store.read_json_ref(ref)
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _write_generation_result_payload(
    *,
    store: Any,
    artifact_id: str,
    rows: list[dict[str, Any]],
    total_example_count: int,
    partial: bool,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "kind": "generation_run_result",
        "summary": {
            "example_count": len(rows),
            "completed_example_count": len(rows),
            "total_example_count": int(total_example_count),
            "partial": bool(partial),
        },
        "rows": rows,
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    return store.write_json(artifact_id, "result.json", payload)


def _merge_generation_rows(*, dataset: Dataset, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row_by_key: dict[str, dict[str, Any]] = {}
    fallback_order: dict[str, int] = {}
    for row in rows:
        key = _generation_row_key(row)
        if not key:
            key = f"row_{len(fallback_order):08d}"
        if key not in fallback_order:
            fallback_order[key] = len(fallback_order)
        row_by_key[key] = dict(row)

    dataset_order: dict[str, int] = {}
    for index, example in enumerate(dataset.examples):
        if example.prompt_hash:
            dataset_order[str(example.prompt_hash)] = index
        dataset_order[str(example.key)] = index

    def _sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
        key = _generation_row_key(row)
        return (
            dataset_order.get(key, len(dataset_order)),
            fallback_order.get(key, len(fallback_order)),
            key,
        )

    return sorted(row_by_key.values(), key=_sort_key)


def _generation_row_key(row: dict[str, Any]) -> str:
    return _row_prompt_hash(row) or str(row.get("example_key") or "")


def _generation_resume_keys(rows: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for row in rows:
        prompt_hash = _row_prompt_hash(row)
        if prompt_hash:
            keys.add(prompt_hash)
        example_key = str(row.get("example_key") or "")
        if example_key:
            keys.add(example_key)
    return keys


def _ensure_artifact_dir(store: Any, artifact_id: str) -> None:
    ensure = getattr(store, "ensure_artifact_dir", None)
    if callable(ensure):
        ensure(artifact_id)
    else:
        store.make_artifact_dir(artifact_id)


def _empty_capture_features(spec: CaptureSpec) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for site in spec.sites:
        if isinstance(site, ResidualSite):
            payload = {
                "kind": "residual",
                "site": site.site,
                "storage": {"dtype": site.storage.dtype, "format": site.storage.format},
                "layers": {str(layer): {} for layer in site.layers},
            }
            if site.pooling is not None:
                payload["pooling"] = {"kind": site.pooling.kind}
            features[site.name] = payload
        elif isinstance(site, MoERoutingSite):
            features[site.name] = {
                "kind": "moe_routing",
                "routing_policy": {
                    "source": "vllm_gate_logits",
                    "observed_routing_decisions": True,
                },
                "layers": {str(layer): {} for layer in site.layers},
            }
    return features


def _merge_feature_payload(features: dict[str, dict[str, Any]], name: str, payload: dict[str, Any]) -> None:
    if name not in features:
        features[name] = dict(payload)
        return
    target = features[name]
    for layer, records in dict(payload.get("layers", {})).items():
        target.setdefault("layers", {}).setdefault(str(layer), {}).update(dict(records))


def _ordered_shard_manifests(shard_manifests: list[ArtifactManifest]) -> list[ArtifactManifest]:
    return sorted(
        shard_manifests,
        key=lambda manifest: int(dict(manifest.metadata).get("execution_shard", {}).get("index", 0)),
    )


def _shard_manifest_summary(manifest: ArtifactManifest) -> dict[str, Any]:
    return {
        "artifact_id": manifest.artifact_id,
        "example_coverage": dict(manifest.example_coverage),
        "execution_shard": dict(manifest.metadata).get("execution_shard"),
    }


def _row_prompt_hash(row: dict[str, Any]) -> str:
    example = row.get("example")
    if isinstance(example, dict):
        return str(example.get("prompt_hash") or "")
    return ""


def _input_artifact_ids(spec: OperationSpec) -> list[str]:
    artifact_ids: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, FeatureLayerRef):
            artifact_ids.append(value.feature.artifact.id)
        elif isinstance(value, FeatureRef):
            artifact_ids.append(value.artifact.id)
        elif isinstance(value, ArtifactLabelRef):
            artifact_ids.append(value.artifact.id)
        elif isinstance(value, (CaptureArtifact, OperationArtifact)):
            artifact_ids.append(value.id)
        elif isinstance(value, tuple | list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif hasattr(value, "__dataclass_fields__"):
            for field_name in value.__dataclass_fields__:
                visit(getattr(value, field_name))

    visit(spec)
    return sorted(set(artifact_ids))
