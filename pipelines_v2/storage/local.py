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
        path = Path(self.root) / f"{manifest.artifact_id}.json"
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, sort_keys=True, indent=2, default=json_default)
        os.replace(tmp, path)


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
