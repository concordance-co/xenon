"""Local runner implementation."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipelines_v2.core.types import OperationSpec, utc_now_iso
from pipelines_v2.operations.specs import BasisSpec, CaptureSpec, DirectionSpec, LabelFieldsSpec, LabelMapSpec, PairDeltaSpec, ProbeSpec, ReportSpec, TransformSpec
from pipelines_v2.runtime.base import ExecutionPlan
from pipelines_v2.storage.artifacts import (
    ArtifactLabelRef,
    ArtifactManifest,
    CaptureArtifact,
    FeatureLayerRef,
    FeatureRef,
    OperationArtifact,
)
from pipelines_v2.storage.base import Catalog
from pipelines_v2.storage.features import write_capture_features
from pipelines_v2.storage.local import LocalArtifactStore, NullCatalog


@dataclass(frozen=True, slots=True)
class LocalResources:
    """Optional local resource hints for local execution."""
    device: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"device": self.device}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LocalResources":
        return cls(device=payload.get("device"))


@dataclass(slots=True)
class LocalRunner:
    """Execute captures and artifact-bound specs in the local process."""
    resources: LocalResources | None = None
    artifacts: LocalArtifactStore = field(default_factory=lambda: LocalArtifactStore(Path("artifacts")))
    catalog: Catalog = field(default_factory=NullCatalog)

    kind: str = "local"

    def identity(self) -> dict[str, Any]:
        """Return a serializable description of this runner."""
        return {
            "kind": self.kind,
            "resources": {"device": self.resources.device if self.resources else None},
        }

    def plan(self, spec: OperationSpec) -> ExecutionPlan:
        """Preflight a spec against local env vars, stores, and capabilities."""
        engine = spec.bound_engine()
        engine_capabilities = frozenset(engine.capabilities()) if engine is not None else frozenset()
        artifact_kinds = ("capture",) if isinstance(spec, CaptureSpec) else ((spec.kind,) if isinstance(spec, _ARTIFACT_BOUND_SPECS) else ())
        errors = list(_spec_plan_errors(spec))
        errors.extend(_local_plan_errors(spec))
        return ExecutionPlan(
            spec_kind=spec.kind,
            required_capabilities=frozenset(spec.required_capabilities()),
            engine_capabilities=engine_capabilities,
            artifact_kinds=artifact_kinds,
            checks=("capabilities", "runtime_env", "artifact_store", "catalog"),
            errors=tuple(errors),
        )

    def run(self, spec: OperationSpec) -> Any:
        """Execute one supported spec locally and return its artifact."""
        self.plan(spec).validate()
        if isinstance(spec, CaptureSpec):
            return self._run_capture(spec)
        if isinstance(spec, _ARTIFACT_BOUND_SPECS):
            return self._run_artifact_operation(spec)
        raise NotImplementedError(f"LocalRunner cannot run {spec.kind!r} specs yet")

    def _run_capture(self, spec: CaptureSpec) -> CaptureArtifact:
        engine = spec.bound_engine()
        if engine is None:
            raise RuntimeError("CaptureSpec is missing a bound engine")
        resolved_spec = spec.resolve_dataset()
        result = engine.capture(resolved_spec)
        artifact_id = f"capture_{spec.spec_hash()[:12]}_{uuid.uuid4().hex[:8]}"
        self.artifacts.make_artifact_dir(artifact_id)

        storage_refs: dict[str, Any] = {"features": write_capture_features(self.artifacts, artifact_id, result.features)}

        if result.generations:
            storage_refs["generations"] = self.artifacts.write_json(
                artifact_id,
                "generations.json",
                result.generations,
            )

        manifest = ArtifactManifest(
            artifact_id=artifact_id,
            artifact_kind="capture",
            schema_version=1,
            operation_spec_hash=spec.spec_hash(),
            created_at=utc_now_iso(),
            engine=engine.identity(),
            runner=self.identity(),
            input_artifact_refs=(),
            example_coverage=resolved_spec.dataset.coverage(),
            storage_refs=storage_refs,
            metadata=result.metadata,
        )
        storage_refs["manifest"] = self.artifacts.write_json(
            artifact_id,
            "manifest.json",
            manifest.to_dict(),
        )
        self.catalog.record_artifact(manifest)
        return CaptureArtifact(_manifest=manifest, store=self.artifacts)

    def _run_artifact_operation(self, spec: OperationSpec) -> OperationArtifact:
        from pipelines_v2.operations.execute import execute_artifact_operation

        artifact_id = f"{spec.kind}_{spec.spec_hash()[:12]}_{uuid.uuid4().hex[:8]}"
        self.artifacts.make_artifact_dir(artifact_id)
        result = execute_artifact_operation(spec)

        storage_refs: dict[str, Any] = {}
        if result.payload:
            storage_refs["result"] = self.artifacts.write_json(artifact_id, "result.json", result.payload)
        if result.features:
            storage_refs["features"] = write_capture_features(self.artifacts, artifact_id, result.features)
        if result.labels:
            storage_refs["labels"] = {
                name: self.artifacts.write_json(artifact_id, f"labels/{name}.json", payload)
                for name, payload in result.labels.items()
            }
        metadata = dict(result.metadata)
        if isinstance(spec, ReportSpec):
            report_refs, published = _materialize_local_report_outputs(
                artifact_id=artifact_id,
                spec=spec,
                payload=result.payload,
            )
            storage_refs.update(report_refs)
            if published:
                metadata["published_report"] = published
        manifest = ArtifactManifest(
            artifact_id=artifact_id,
            artifact_kind=spec.kind,
            schema_version=1,
            operation_spec_hash=spec.spec_hash(),
            created_at=utc_now_iso(),
            engine={},
            runner=self.identity(),
            input_artifact_refs=tuple(_input_artifact_ids(spec)),
            example_coverage=result.example_coverage,
            storage_refs=storage_refs,
            metadata=metadata,
        )
        storage_refs["manifest"] = self.artifacts.write_json(artifact_id, "manifest.json", manifest.to_dict())
        self.catalog.record_artifact(manifest)
        return OperationArtifact(_manifest=manifest, store=self.artifacts)


_ARTIFACT_BOUND_SPECS = (ProbeSpec, DirectionSpec, BasisSpec, PairDeltaSpec, LabelMapSpec, LabelFieldsSpec, TransformSpec, ReportSpec)


def _spec_plan_errors(spec: OperationSpec) -> list[str]:
    errors: list[str] = []
    engine = spec.bound_engine()
    if engine is not None:
        planning_errors = getattr(engine, "planning_errors", None)
        if callable(planning_errors):
            errors.extend(str(error) for error in planning_errors(spec))
    return errors


def _local_plan_errors(spec: OperationSpec) -> list[str]:
    errors: list[str] = []
    for secret in spec.runtime_secrets():
        if not os.environ.get(secret.env_var):
            errors.append(f"Missing required environment variable for local runtime: {secret.env_var}")
    for ref in _iter_transfer_refs(spec):
        store = _ref_store(ref)
        transfer_bytes = _ref_transfer_bytes(ref)
        try:
            if isinstance(ref, FeatureLayerRef):
                if not store.has_local_ref(ref.feature.artifact.manifest().storage_refs["features"][ref.feature.name]):
                    store.validate_transfer(bytes=transfer_bytes, label=f"feature {ref.feature.name!r}")
            elif isinstance(ref, FeatureRef):
                if not store.has_local_ref(ref.artifact.manifest().storage_refs["features"][ref.name]):
                    store.validate_transfer(bytes=transfer_bytes, label=f"feature {ref.name!r}")
            elif isinstance(ref, ArtifactLabelRef):
                if not store.has_local_ref(ref.artifact.manifest().storage_refs["labels"][ref.name]):
                    store.validate_transfer(bytes=transfer_bytes, label=f"label {ref.name!r}")
        except Exception as exc:
            errors.append(str(exc))
    return errors


def _iter_transfer_refs(value: Any) -> list[FeatureRef | FeatureLayerRef | ArtifactLabelRef]:
    refs: list[FeatureRef | FeatureLayerRef | ArtifactLabelRef] = []
    if isinstance(value, FeatureLayerRef):
        refs.append(value)
    elif isinstance(value, FeatureRef):
        refs.append(value)
    elif isinstance(value, ArtifactLabelRef):
        refs.append(value)
    elif isinstance(value, tuple | list):
        for item in value:
            refs.extend(_iter_transfer_refs(item))
    elif isinstance(value, dict):
        for item in value.values():
            refs.extend(_iter_transfer_refs(item))
    elif hasattr(value, "__dataclass_fields__"):
        for field_name in value.__dataclass_fields__:
            refs.extend(_iter_transfer_refs(getattr(value, field_name)))
    return refs


def _ref_store(ref: FeatureRef | FeatureLayerRef | ArtifactLabelRef) -> Any:
    if isinstance(ref, FeatureLayerRef):
        return ref.feature.artifact.store
    return ref.artifact.store


def _ref_transfer_bytes(ref: FeatureRef | FeatureLayerRef | ArtifactLabelRef) -> int | None:
    if isinstance(ref, FeatureLayerRef):
        return ref.feature.estimated_transfer_bytes()
    return ref.estimated_transfer_bytes()


def _input_artifact_ids(spec: OperationSpec) -> list[str]:
    artifact_ids: list[str] = []
    for ref in _iter_transfer_refs(spec):
        artifact = ref.feature.artifact if isinstance(ref, FeatureLayerRef) else ref.artifact
        artifact_ids.append(artifact.id)
    for artifact in _iter_direct_artifacts(spec):
        artifact_ids.append(artifact.id)
    return sorted(set(artifact_ids))


def _iter_direct_artifacts(value: Any) -> list[CaptureArtifact | OperationArtifact]:
    artifacts: list[CaptureArtifact | OperationArtifact] = []
    if isinstance(value, (CaptureArtifact, OperationArtifact)):
        artifacts.append(value)
    elif isinstance(value, tuple | list):
        for item in value:
            artifacts.extend(_iter_direct_artifacts(item))
    elif isinstance(value, dict):
        for item in value.values():
            artifacts.extend(_iter_direct_artifacts(item))
    elif hasattr(value, "__dataclass_fields__"):
        for field_name in value.__dataclass_fields__:
            artifacts.extend(_iter_direct_artifacts(getattr(value, field_name)))
    return artifacts


def _materialize_local_report_outputs(
    *,
    artifact_id: str,
    spec: ReportSpec,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not spec.output_dir:
        return {}, None
    output_dir = Path(spec.output_dir) / artifact_id
    output_dir.mkdir(parents=True, exist_ok=True)

    report_json_path = output_dir / "report.json"
    summary_json_path = output_dir / "summary.json"
    report_md_path = output_dir / "report.md"

    _write_json(report_json_path, payload)
    _write_json(summary_json_path, payload.get("summary", payload))
    _write_text(report_md_path, _render_report_markdown(spec=spec, payload=payload))

    return (
        {
            "report": {
                "store": "local_path",
                "path": str(report_md_path),
                "format": "markdown",
                "bytes": report_md_path.stat().st_size,
            },
            "summary": {
                "store": "local_path",
                "path": str(summary_json_path),
                "format": "json",
                "bytes": summary_json_path.stat().st_size,
            },
        },
        {
            "output_dir": str(output_dir),
            "report_path": str(report_md_path),
            "summary_path": str(summary_json_path),
            "report_json_path": str(report_json_path),
        },
    )


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, indent=2)
    os.replace(tmp, path)


def _write_text(path: Path, payload: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(payload)
    os.replace(tmp, path)


def _render_report_markdown(*, spec: ReportSpec, payload: dict[str, Any]) -> str:
    lines = [
        f"# {spec.template}",
        "",
        f"- template: `{spec.template}`",
        f"- input_count: {payload.get('summary', {}).get('input_count', len(payload.get('inputs', [])))}",
        "",
        "## Inputs",
        "",
    ]
    for item in payload.get("inputs", []):
        if isinstance(item, dict):
            artifact_id = item.get("artifact_id")
            artifact_kind = item.get("artifact_kind")
            if artifact_id and artifact_kind:
                lines.append(f"- `{artifact_id}` ({artifact_kind})")
            else:
                lines.append(f"- `{json.dumps(item, sort_keys=True)}`")
        else:
            lines.append(f"- `{item}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(payload.get("summary", {}), sort_keys=True, indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)
