"""Generic remote execution entrypoint for runner backends."""

from __future__ import annotations

import uuid
from typing import Any

from pipelines_v2.core.types import OperationSpec, utc_now_iso
from pipelines_v2.operations import operation_spec_from_dict
from pipelines_v2.operations.specs import (
    ActivationBankSpec,
    BasisSpec,
    CaptureSpec,
    CentroidSpec,
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
    ReportSpec,
    ResidualizedProbeSpec,
    SubspaceSpec,
    TextBaselineSpec,
    TransferProbeSpec,
    TransformSpec,
)
from pipelines_v2.storage import ArtifactManifest, artifact_store_from_dict
from pipelines_v2.storage.artifacts import ArtifactLabelRef, CaptureArtifact, FeatureLayerRef, FeatureRef, OperationArtifact
from pipelines_v2.storage.features import write_capture_features
from pipelines_v2.operations.interventions.runtime import rows_example_coverage


def execute_remote(
    *,
    runner_config: dict[str, Any],
    store_config: dict[str, Any],
    spec_payload: dict[str, Any],
    workflow_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a serialized operation in a remote worker."""
    spec = operation_spec_from_dict(spec_payload)
    store = artifact_store_from_dict(store_config)

    if isinstance(spec, CaptureSpec):
        return _execute_capture(
            runner_config=runner_config,
            store=store,
            spec=spec,
            workflow_context=workflow_context,
        )
    if isinstance(spec, GenerationRunSpec):
        return _execute_generation(
            runner_config=runner_config,
            store=store,
            spec=spec,
            workflow_context=workflow_context,
        )
    if isinstance(spec, PatchedGenerationSpec):
        return _execute_intervention(
            runner_config=runner_config,
            store=store,
            spec=spec,
            workflow_context=workflow_context,
        )
    if isinstance(spec, _ARTIFACT_BOUND_SPECS):
        return _execute_artifact_operation(
            runner_config=runner_config,
            store=store,
            spec=spec,
            workflow_context=workflow_context,
        )
    raise NotImplementedError(f"Remote executor cannot run {spec.kind!r} specs yet")


def _execute_capture(
    *,
    runner_config: dict[str, Any],
    store: Any,
    spec: CaptureSpec,
    workflow_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    engine = spec.bound_engine()
    if engine is None:
        raise RuntimeError("CaptureSpec is missing a bound engine")
    resolved_spec = spec.resolve_dataset()
    artifact_id = f"{spec.kind}_{spec.schema_version}_{uuid.uuid4().hex[:8]}"
    store.make_artifact_dir(artifact_id)
    result = engine.capture(resolved_spec)

    storage_refs: dict[str, Any] = {"features": write_capture_features(store, artifact_id, result.features)}

    if result.generations:
        storage_refs["generations"] = store.write_json(
            artifact_id,
            "generations.json",
            result.generations,
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
        example_coverage=resolved_spec.dataset.coverage(),
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
) -> dict[str, Any]:
    engine = spec.bound_engine()
    if engine is None:
        raise RuntimeError("PatchedGenerationSpec is missing a bound engine")
    resolved_spec = spec.resolve_dataset()
    artifact_id = f"{spec.kind}_{spec.schema_version}_{uuid.uuid4().hex[:8]}"
    store.make_artifact_dir(artifact_id)
    result = engine.intervene(resolved_spec)

    payload = {
        "kind": "patched_generation_result",
        "summary": dict(result.summary),
        "rows": list(result.rows),
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
        example_coverage=rows_example_coverage(dataset=resolved_spec.dataset, rows=result.rows),
        storage_refs=storage_refs,
        metadata=dict(result.metadata),
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
) -> dict[str, Any]:
    engine = spec.bound_engine()
    if engine is None:
        raise RuntimeError("GenerationRunSpec is missing a bound engine")
    resolved_spec = spec.resolve_dataset()
    artifact_id = f"{spec.kind}_{spec.schema_version}_{uuid.uuid4().hex[:8]}"
    store.make_artifact_dir(artifact_id)
    result = engine.generate(resolved_spec)

    payload = {
        "kind": "generation_run_result",
        "summary": {"example_count": len(result.rows)},
        "rows": list(result.rows),
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
        example_coverage=rows_example_coverage(dataset=resolved_spec.dataset, rows=result.rows),
        storage_refs=storage_refs,
        metadata=dict(result.metadata),
        workflow_context=dict(workflow_context or {}),
    )
    storage_refs["manifest"] = store.write_json(
        artifact_id,
        "manifest.json",
        manifest.to_dict(),
    )
    return manifest.to_dict()


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
    ReportSpec,
)
