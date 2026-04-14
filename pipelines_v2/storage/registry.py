"""Artifact-store and catalog registries and deserialization helpers."""

from __future__ import annotations

from typing import Any, Callable

from pipelines_v2.storage.base import ArtifactStore
from pipelines_v2.storage.base import Catalog
from pipelines_v2.storage.local import FileCatalog, LocalArtifactStore, NullCatalog
from pipelines_v2.storage.modal import ModalVolumeStore
from pipelines_v2.storage.postgres import PostgresCatalog

ArtifactStoreLoader = Callable[[dict[str, Any]], ArtifactStore]
CatalogLoader = Callable[[dict[str, Any]], Catalog]

_ARTIFACT_STORE_LOADERS: dict[str, ArtifactStoreLoader] = {
    "local": LocalArtifactStore.from_dict,
    "modal_volume": ModalVolumeStore.from_dict,
}

_CATALOG_LOADERS: dict[str, CatalogLoader] = {
    "file": FileCatalog.from_dict,
    "none": NullCatalog.from_dict,
    "postgres": PostgresCatalog.from_dict,
}


def artifact_store_from_dict(payload: dict[str, Any]) -> ArtifactStore:
    kind = str(payload.get("kind") or "").strip()
    if not kind:
        raise ValueError("Artifact store payload is missing 'kind'")
    try:
        loader = _ARTIFACT_STORE_LOADERS[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown artifact store kind: {kind!r}") from exc
    return loader(payload)


def catalog_from_dict(payload: dict[str, Any]) -> Catalog:
    kind = str(payload.get("kind") or "").strip()
    if not kind:
        raise ValueError("Catalog payload is missing 'kind'")
    try:
        loader = _CATALOG_LOADERS[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown catalog kind: {kind!r}") from exc
    return loader(payload)
