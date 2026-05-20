"""Artifact-store and catalog registries and deserialization helpers."""

from __future__ import annotations

from typing import Any, Callable

from pipelines_v2.core.registry import load_from_kind_registry
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
    return load_from_kind_registry(payload, _ARTIFACT_STORE_LOADERS, missing_message="Artifact store payload is missing 'kind'", unknown_message="Unknown artifact store kind: {kind!r}")


def catalog_from_dict(payload: dict[str, Any]) -> Catalog:
    return load_from_kind_registry(payload, _CATALOG_LOADERS, missing_message="Catalog payload is missing 'kind'", unknown_message="Unknown catalog kind: {kind!r}")
