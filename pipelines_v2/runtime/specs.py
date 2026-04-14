"""Serializable runner specs that materialize concrete runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pipelines_v2.runtime.local import LocalResources, LocalRunner
from pipelines_v2.runtime.modal import ModalResources, ModalRunner
from pipelines_v2.storage import (
    ArtifactStore,
    Catalog,
    LocalArtifactStore,
    ModalVolumeStore,
    NullCatalog,
    artifact_store_from_dict,
    catalog_from_dict,
)


class RunnerSpec(Protocol):
    """Serializable execution-profile description that can materialize a runner."""

    kind: str

    def to_runner(self) -> Any:
        """Materialize the concrete runner instance."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize the runner spec into a JSON-safe payload."""
        ...


@dataclass(frozen=True, slots=True)
class LocalRunnerSpec:
    """Serializable spec for constructing a ``LocalRunner``."""

    resources: LocalResources | None = None
    artifacts: ArtifactStore = field(default_factory=lambda: LocalArtifactStore(Path("artifacts")))
    catalog: Catalog = field(default_factory=NullCatalog)

    kind: str = "local"

    def to_runner(self) -> LocalRunner:
        return LocalRunner(
            resources=self.resources,
            artifacts=self.artifacts,
            catalog=self.catalog,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "resources": self.resources.to_dict() if self.resources is not None else None,
            "artifacts": self.artifacts.identity(),
            "catalog": self.catalog.identity(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LocalRunnerSpec":
        resources_payload = payload.get("resources")
        return cls(
            resources=LocalResources.from_dict(dict(resources_payload)) if resources_payload is not None else None,
            artifacts=artifact_store_from_dict(dict(payload["artifacts"])),
            catalog=catalog_from_dict(dict(payload.get("catalog", {"kind": "none"}))),
        )


@dataclass(frozen=True, slots=True)
class ModalRunnerSpec:
    """Serializable spec for constructing a ``ModalRunner``."""

    resources: ModalResources
    artifacts: ModalVolumeStore
    catalog: Catalog = field(default_factory=NullCatalog)

    kind: str = "modal"

    def to_runner(self) -> ModalRunner:
        return ModalRunner(
            resources=self.resources,
            artifacts=self.artifacts,
            catalog=self.catalog,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "resources": self.resources.to_dict(),
            "artifacts": self.artifacts.identity(),
            "catalog": self.catalog.identity(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModalRunnerSpec":
        return cls(
            resources=ModalResources.from_dict(dict(payload["resources"])),
            artifacts=artifact_store_from_dict(dict(payload["artifacts"])),
            catalog=catalog_from_dict(dict(payload.get("catalog", {"kind": "none"}))),
        )


def runner_spec_from_dict(payload: dict[str, Any]) -> RunnerSpec:
    kind = str(payload.get("kind") or "").strip()
    if kind == "local":
        return LocalRunnerSpec.from_dict(payload)
    if kind == "modal":
        return ModalRunnerSpec.from_dict(payload)
    raise ValueError(f"Unknown runner spec kind: {kind!r}")
