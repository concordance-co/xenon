"""Artifact manifests and typed artifact accessors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping

from pipelines_v2.storage.features import load_feature_payload


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Canonical manifest for one produced artifact."""
    artifact_id: str
    artifact_kind: str
    schema_version: int
    operation_spec_hash: str
    operation_semantic_hash: str
    created_at: str
    engine: Mapping[str, Any]
    runner: Mapping[str, Any]
    input_artifact_refs: tuple[str, ...]
    example_coverage: Mapping[str, Any]
    storage_refs: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    workflow_context: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactManifest":
        return cls(
            artifact_id=str(payload["artifact_id"]),
            artifact_kind=str(payload["artifact_kind"]),
            schema_version=int(payload["schema_version"]),
            operation_spec_hash=str(payload["operation_spec_hash"]),
            operation_semantic_hash=str(payload.get("operation_semantic_hash", payload["operation_spec_hash"])),
            created_at=str(payload["created_at"]),
            engine=dict(payload["engine"]),
            runner=dict(payload["runner"]),
            input_artifact_refs=tuple(payload.get("input_artifact_refs", ())),
            example_coverage=dict(payload["example_coverage"]),
            storage_refs=dict(payload["storage_refs"]),
            metadata=dict(payload.get("metadata", {})),
            workflow_context=dict(payload.get("workflow_context", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "schema_version": self.schema_version,
            "operation_spec_hash": self.operation_spec_hash,
            "operation_semantic_hash": self.operation_semantic_hash,
            "created_at": self.created_at,
            "engine": dict(self.engine),
            "runner": dict(self.runner),
            "input_artifact_refs": list(self.input_artifact_refs),
            "example_coverage": dict(self.example_coverage),
            "storage_refs": dict(self.storage_refs),
            "metadata": dict(self.metadata),
            "workflow_context": dict(self.workflow_context),
        }


@dataclass(frozen=True, slots=True)
class FeatureRef:
    """Reference to one named feature stored inside an artifact."""
    artifact: Any
    name: str

    kind: ClassVar[str] = "feature_ref"

    def load(self) -> dict[str, Any]:
        """Load the decoded feature payload from storage."""
        return self.artifact.load_feature(self.name)

    def estimated_transfer_bytes(self) -> int | None:
        """Estimate bytes needed to materialize this feature locally."""
        features = self.artifact.manifest().storage_refs.get("features", {})
        if self.name not in features:
            raise KeyError(f"Capture artifact {self.artifact.id!r} has no feature {self.name!r}")
        return self.artifact.store.estimate_download_bytes(features[self.name])

    def layer(self, layer: int) -> "FeatureLayerRef":
        """Reference one specific layer inside this feature."""
        return FeatureLayerRef(feature=self, layer=layer)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureRef":
        return cls(
            artifact=artifact_from_dict(payload["artifact"]),
            name=str(payload["name"]),
        )


@dataclass(frozen=True, slots=True)
class ArtifactLabelRef:
    """Reference to one named label payload stored inside an artifact."""
    artifact: Any
    name: str

    kind: ClassVar[str] = "artifact_label_ref"

    def equals(self, value: Any) -> Any:
        """Build a predicate over this artifact-backed label set."""
        from pipelines_v2.data.datasets import LabelPredicate

        return LabelPredicate(label_set=self, op="equals", value=value)

    def load(self) -> dict[str, Any]:
        """Load the raw label payload from storage."""
        return self.artifact.load_label(self.name)

    def resolve_values(self) -> Mapping[str, Any]:
        """Return the materialized example_key -> value mapping."""
        payload = self.load()
        values = payload.get("values")
        if not isinstance(values, Mapping):
            raise TypeError("Artifact label payload must contain a 'values' mapping")
        return {str(key): value for key, value in values.items()}

    def resolve_example_keys(self) -> list[str]:
        """Return the example keys covered by this artifact label."""
        return sorted(str(key) for key in self.resolve_values())

    def runtime_secrets(self) -> tuple[Any, ...]:
        return ()

    def estimated_transfer_bytes(self) -> int | None:
        """Estimate bytes needed to materialize this label locally."""
        labels = self.artifact.manifest().storage_refs.get("labels", {})
        if self.name not in labels:
            raise KeyError(f"Artifact {self.artifact.id!r} has no label {self.name!r}")
        return self.artifact.store.estimate_download_bytes(labels[self.name])

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactLabelRef":
        return cls(
            artifact=artifact_from_dict(payload["artifact"]),
            name=str(payload["name"]),
        )


@dataclass(frozen=True, slots=True)
class FeatureLayerRef:
    """Reference to one layer inside a feature payload."""
    feature: FeatureRef
    layer: int

    kind: ClassVar[str] = "feature_layer_ref"

    def load(self) -> dict[str, Any]:
        """Load only the layer payload addressed by this ref."""
        payload = self.feature.load()
        return payload["layers"][str(self.layer)]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureLayerRef":
        return cls(
            feature=FeatureRef.from_dict(payload["feature"]),
            layer=int(payload["layer"]),
        )


@dataclass(frozen=True, slots=True)
class CaptureArtifact:
    """Typed handle over a capture manifest plus its backing store."""
    _manifest: ArtifactManifest
    store: Any

    kind: ClassVar[str] = "capture_artifact"

    @property
    def id(self) -> str:
        """Artifact id."""
        return self._manifest.artifact_id

    def manifest(self) -> ArtifactManifest:
        """Return the parsed artifact manifest."""
        return self._manifest

    def manifest_dict(self) -> dict[str, Any]:
        """Return the manifest as a plain dict."""
        return self._manifest.to_dict()

    def localize(self) -> Path:
        """Ensure the artifact is locally available and return its root path."""
        if not self.store.has_local_artifact(self.id):
            self.store.validate_transfer(
                bytes=self.estimated_local_transfer_bytes(),
                label=f"capture artifact {self.id}",
            )
        return self.store.localize(self.id)

    def estimated_local_transfer_bytes(self) -> int | None:
        """Estimate bytes needed to localize this artifact."""
        resources = _unique_storage_resources(self._manifest.storage_refs)
        estimates = [resource["bytes"] for resource in resources]
        if any(value is None for value in estimates):
            return None
        return sum(int(value) for value in estimates if value is not None)

    def feature(self, name: str) -> FeatureRef:
        """Return a typed feature ref by name."""
        features = self._manifest.storage_refs.get("features", {})
        if name not in features:
            raise KeyError(f"Capture artifact {self.id!r} has no feature {name!r}")
        return FeatureRef(artifact=self, name=name)

    def load_feature(self, name: str) -> dict[str, Any]:
        """Load and decode a named feature payload."""
        features = self._manifest.storage_refs.get("features", {})
        if name not in features:
            raise KeyError(f"Capture artifact {self.id!r} has no feature {name!r}")
        ref = features[name]
        if not self.store.has_local_ref(ref):
            self.store.validate_transfer(
                bytes=self.store.estimate_download_bytes(ref),
                label=f"feature {name!r} from capture artifact {self.id}",
            )
        return load_feature_payload(self.store, ref)

    def label(self, name: str) -> ArtifactLabelRef:
        """Return a typed label ref by name."""
        labels = self._manifest.storage_refs.get("labels", {})
        if name not in labels:
            raise KeyError(f"Capture artifact {self.id!r} has no label {name!r}")
        return ArtifactLabelRef(artifact=self, name=name)

    def load_label(self, name: str) -> dict[str, Any]:
        """Load a named label payload."""
        labels = self._manifest.storage_refs.get("labels", {})
        if name not in labels:
            raise KeyError(f"Capture artifact {self.id!r} has no label {name!r}")
        ref = labels[name]
        if not self.store.has_local_ref(ref):
            self.store.validate_transfer(
                bytes=self.store.estimate_download_bytes(ref),
                label=f"label {name!r} from capture artifact {self.id}",
            )
        payload = self.store.read_json_ref(ref)
        if not isinstance(payload, dict):
            raise TypeError("Label payload must be a mapping")
        return payload

    def generations(self) -> list[dict[str, Any]]:
        """Load captured generations if this artifact includes them."""
        ref = self._manifest.storage_refs.get("generations")
        if ref is None:
            return []
        if not self.store.has_local_ref(ref):
            self.store.validate_transfer(
                bytes=self.store.estimate_download_bytes(ref),
                label=f"generations from capture artifact {self.id}",
            )
        payload = self.store.read_json_ref(ref)
        if not isinstance(payload, list):
            raise TypeError("Generation payload must be a list")
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CaptureArtifact":
        from pipelines_v2.storage import artifact_store_from_dict

        return cls(
            _manifest=ArtifactManifest.from_dict(payload["_manifest"]),
            store=artifact_store_from_dict(dict(payload["store"])),
        )


@dataclass(frozen=True, slots=True)
class OperationArtifact:
    """Typed handle over a non-capture artifact plus its backing store."""
    _manifest: ArtifactManifest
    store: Any

    kind: ClassVar[str] = "operation_artifact"

    @property
    def id(self) -> str:
        """Artifact id."""
        return self._manifest.artifact_id

    def manifest(self) -> ArtifactManifest:
        """Return the parsed artifact manifest."""
        return self._manifest

    def manifest_dict(self) -> dict[str, Any]:
        """Return the manifest as a plain dict."""
        return self._manifest.to_dict()

    def localize(self) -> Path:
        """Ensure the artifact is locally available and return its root path."""
        if not self.store.has_local_artifact(self.id):
            self.store.validate_transfer(
                bytes=self.estimated_local_transfer_bytes(),
                label=f"artifact {self.id}",
            )
        return self.store.localize(self.id)

    def estimated_local_transfer_bytes(self) -> int | None:
        """Estimate bytes needed to localize this artifact."""
        resources = _unique_storage_resources(self._manifest.storage_refs)
        estimates = [resource["bytes"] for resource in resources]
        if any(value is None for value in estimates):
            return None
        return sum(int(value) for value in estimates if value is not None)

    def result(self) -> Any:
        """Load the main result payload."""
        ref = self._manifest.storage_refs.get("result")
        if ref is None:
            return {}
        if not self.store.has_local_ref(ref):
            self.store.validate_transfer(
                bytes=self.store.estimate_download_bytes(ref),
                label=f"result payload for artifact {self.id}",
            )
        return self.store.read_json_ref(ref)

    def feature(self, name: str) -> FeatureRef:
        """Return a typed feature ref by name."""
        features = self._manifest.storage_refs.get("features", {})
        if name not in features:
            raise KeyError(f"Operation artifact {self.id!r} has no feature {name!r}")
        return FeatureRef(artifact=self, name=name)

    def load_feature(self, name: str) -> dict[str, Any]:
        """Load and decode a named derived feature payload."""
        features = self._manifest.storage_refs.get("features", {})
        if name not in features:
            raise KeyError(f"Operation artifact {self.id!r} has no feature {name!r}")
        ref = features[name]
        if not self.store.has_local_ref(ref):
            self.store.validate_transfer(
                bytes=self.store.estimate_download_bytes(ref),
                label=f"feature {name!r} from artifact {self.id}",
            )
        return load_feature_payload(self.store, ref)

    def label(self, name: str) -> ArtifactLabelRef:
        """Return a typed label ref by name."""
        labels = self._manifest.storage_refs.get("labels", {})
        if name not in labels:
            raise KeyError(f"Operation artifact {self.id!r} has no label {name!r}")
        return ArtifactLabelRef(artifact=self, name=name)

    def load_label(self, name: str) -> dict[str, Any]:
        """Load a named label payload."""
        labels = self._manifest.storage_refs.get("labels", {})
        if name not in labels:
            raise KeyError(f"Operation artifact {self.id!r} has no label {name!r}")
        ref = labels[name]
        if not self.store.has_local_ref(ref):
            self.store.validate_transfer(
                bytes=self.store.estimate_download_bytes(ref),
                label=f"label {name!r} from artifact {self.id}",
            )
        payload = self.store.read_json_ref(ref)
        if not isinstance(payload, dict):
            raise TypeError("Label payload must be a mapping")
        return payload

    def summary(self) -> Any:
        """Return ``result()['summary']`` when present, otherwise the raw result."""
        payload = self.result()
        if isinstance(payload, Mapping) and "summary" in payload:
            return payload["summary"]
        return payload

    @property
    def uri(self) -> str:
        """Best-effort URI/path for the artifact's human-facing output."""
        report_ref = self._manifest.storage_refs.get("report")
        if isinstance(report_ref, Mapping) and "path" in report_ref:
            return str(report_ref["path"])
        return str(self.localize())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperationArtifact":
        from pipelines_v2.storage import artifact_store_from_dict

        return cls(
            _manifest=ArtifactManifest.from_dict(payload["_manifest"]),
            store=artifact_store_from_dict(dict(payload["store"])),
        )


def artifact_from_manifest(manifest: ArtifactManifest, *, store: Any) -> CaptureArtifact | OperationArtifact:
    """Construct the correct typed artifact handle for a manifest and store."""
    if manifest.artifact_kind == "capture":
        return CaptureArtifact(_manifest=manifest, store=store)
    return OperationArtifact(_manifest=manifest, store=store)


def artifact_from_dict(payload: Mapping[str, Any]) -> CaptureArtifact | OperationArtifact:
    """Reconstruct the correct typed artifact handle from a serialized payload."""
    kind = str(payload.get("kind") or "")
    if kind == CaptureArtifact.kind:
        return CaptureArtifact.from_dict(payload)
    if kind == OperationArtifact.kind:
        return OperationArtifact.from_dict(payload)
    manifest = ArtifactManifest.from_dict(payload["_manifest"])
    if manifest.artifact_kind == "capture":
        return CaptureArtifact.from_dict(payload)
    return OperationArtifact.from_dict(payload)


def _iter_storage_refs(value: Any) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        if "path" in value and "store" in value:
            refs.append(dict(value))
        else:
            for child in value.values():
                refs.extend(_iter_storage_refs(child))
    return refs


def _unique_storage_resources(value: Any) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str | None, str], dict[str, Any]] = {}
    for ref in _iter_storage_refs(value):
        if "metadata_path" in ref:
            metadata_key = (str(ref["store"]), ref.get("name"), str(ref["metadata_path"]))
            unique.setdefault(metadata_key, {"bytes": ref.get("metadata_bytes")})
        if "tensor_path" in ref:
            tensor_key = (str(ref["store"]), ref.get("name"), str(ref["tensor_path"]))
            unique.setdefault(tensor_key, {"bytes": ref.get("tensor_bytes")})
        else:
            key = (str(ref["store"]), ref.get("name"), str(ref["path"]))
            unique.setdefault(key, {"bytes": ref.get("bytes")})
    return list(unique.values())
