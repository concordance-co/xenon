"""Infer artifact stores from persisted artifact manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pipelines_v2.storage.local import LocalArtifactStore
from pipelines_v2.storage.modal import ModalVolumeStore


def artifact_store_from_manifest(
    manifest: Any,
    *,
    local_cache_root: Path | None = None,
    purpose: str = "artifact access",
) -> Any:
    """Infer a concrete artifact store from the first storage ref in a manifest."""

    ref = first_storage_ref(getattr(manifest, "storage_refs", {}))
    artifact_id = str(getattr(manifest, "artifact_id", ""))
    if ref is None:
        raise RuntimeError(f"Artifact {artifact_id!r} has no storage refs to infer a store from")
    store_kind = str(ref.get("store") or "").strip()
    path = ref.get("path")
    if not path:
        raise RuntimeError(f"Artifact {artifact_id!r} storage ref is missing a path")
    root = infer_artifact_root(path, artifact_id)
    if store_kind == "modal_volume":
        name = ref.get("name")
        if not name:
            raise RuntimeError(f"Artifact {artifact_id!r} Modal ref is missing a volume name")
        return ModalVolumeStore(name=str(name), root=str(root), local_cache_root=local_cache_root)
    if store_kind in {"local", "local_path"}:
        return LocalArtifactStore(root=root)
    raise RuntimeError(f"Unsupported artifact store kind for {purpose}: {store_kind!r}")


def first_storage_ref(value: Any) -> Mapping[str, Any] | None:
    """Return the first nested storage ref containing store/path fields."""

    if isinstance(value, Mapping):
        if "store" in value and "path" in value:
            return value
        for child in value.values():
            found = first_storage_ref(child)
            if found is not None:
                return found
    return None


def infer_artifact_root(path: str | Path, artifact_id: str) -> Path:
    """Infer the store root by stripping the artifact id and following path."""

    resolved = Path(path)
    parts = resolved.parts
    try:
        index = parts.index(artifact_id)
    except ValueError:
        return resolved.parent
    if index == 0:
        return resolved.parent
    root = Path(parts[0])
    for part in parts[1:index]:
        root /= part
    return root


__all__ = ["artifact_store_from_manifest", "first_storage_ref", "infer_artifact_root"]
