"""Source loading helpers for intervention execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pipelines_v2.core.types import SpecValidationError

from .recipes import (
    ActivationPatchSpec,
    AddDirectionPatch,
    ResidualPathPatch,
    SwapComponentsPatch,
    SwapMeanPatch,
)


def load_activation_bank_source(patch: ActivationPatchSpec) -> dict[str, Any]:
    source = getattr(patch, "activation_bank", None)
    if source is None or not hasattr(source, "result"):
        raise SpecValidationError(f"{type(patch).__name__} activation_bank source must be an operation artifact ref")
    payload = source.result()
    if not isinstance(payload, Mapping):
        raise TypeError("Activation bank payload must be a mapping")
    if str(payload.get("kind") or "") != "activation_bank_result":
        raise SpecValidationError("Activation patch currently requires an activation_bank_result source")
    layers = payload.get("layers")
    if not isinstance(layers, Mapping):
        raise TypeError("Activation bank payload must contain a 'layers' mapping")
    required_layers = required_source_layers_for_patch(patch)
    missing = [int(layer) for layer in required_layers if str(int(layer)) not in layers]
    if missing:
        raise SpecValidationError(
            f"{type(patch).__name__} activation_bank source is missing required layers: {sorted(missing)}"
        )
    site = str(payload.get("site") or "")
    if site and str(patch.write_site.site) and site != str(patch.write_site.site):
        raise SpecValidationError(
            f"{type(patch).__name__} activation_bank site {site!r} does not match write_site {patch.write_site.site!r}"
        )
    return dict(payload)


def load_subspace_source(patch: ActivationPatchSpec) -> dict[str, Any]:
    source = getattr(patch, "subspace", None)
    if source is None or not hasattr(source, "result"):
        raise SpecValidationError(f"{type(patch).__name__} subspace source must be an operation artifact ref")
    payload = source.result()
    if not isinstance(payload, Mapping):
        raise TypeError("Subspace payload must be a mapping")
    if str(payload.get("kind") or "") != "subspace_result":
        raise SpecValidationError("Activation patch currently requires a subspace_result source")
    layers = payload.get("layers")
    if not isinstance(layers, Mapping):
        raise TypeError("Subspace source payload must contain a 'layers' mapping")
    missing = [int(layer) for layer in required_source_layers_for_patch(patch) if str(int(layer)) not in layers]
    if missing:
        raise SpecValidationError(
            f"{type(patch).__name__} subspace source is missing required layers: {sorted(missing)}"
        )
    return dict(payload)


def load_direction_source(patch: AddDirectionPatch) -> dict[str, Any]:
    source = patch.direction
    if source is None or not hasattr(source, "result"):
        raise SpecValidationError("AddDirectionPatch direction source must be an operation artifact ref")
    payload = source.result()
    if not isinstance(payload, Mapping):
        raise TypeError("Direction payload must be a mapping")
    if str(payload.get("kind") or "") != "direction_result":
        raise SpecValidationError("AddDirectionPatch currently requires a direction_result source")
    layers = payload.get("layers")
    if not isinstance(layers, Mapping):
        raise TypeError("Direction payload must contain a 'layers' mapping")
    missing = [int(layer) for layer in required_source_layers_for_patch(patch) if str(int(layer)) not in layers]
    if missing:
        raise SpecValidationError(
            f"AddDirectionPatch direction source is missing required layers: {sorted(missing)}"
        )
    return dict(payload)


def load_centroid_source(patch: SwapMeanPatch | SwapComponentsPatch) -> dict[str, Any]:
    source = patch.centroids
    if source is None or not hasattr(source, "result"):
        raise SpecValidationError(f"{type(patch).__name__} centroid source must be an operation artifact ref")
    payload = source.result()
    if not isinstance(payload, Mapping):
        raise TypeError("Centroid payload must be a mapping")
    if str(payload.get("kind") or "") != "centroid_result":
        raise SpecValidationError("Activation patch currently requires a centroid_result source")
    layers = payload.get("layers")
    if not isinstance(layers, Mapping):
        raise TypeError("Centroid payload must contain a 'layers' mapping")
    missing_layers: list[int] = []
    missing_centroids: list[int] = []
    for source_layer in required_source_layers_for_patch(patch):
        layer_payload = layers.get(str(int(source_layer)))
        if not isinstance(layer_payload, Mapping):
            missing_layers.append(int(source_layer))
            continue
        centroids = layer_payload.get("centroids")
        if not isinstance(centroids, Mapping) or patch.centroid_name not in centroids:
            missing_centroids.append(int(source_layer))
    if missing_layers:
        raise SpecValidationError(
            f"{type(patch).__name__} centroid source is missing required layers: {sorted(missing_layers)}"
        )
    if missing_centroids:
        raise SpecValidationError(
            f"{type(patch).__name__} centroid source is missing centroid {patch.centroid_name!r} "
            f"for layers {sorted(missing_centroids)}"
        )
    return dict(payload)


def load_path_mask_source(patch: ResidualPathPatch) -> dict[str, Any]:
    source = patch.path_mask
    if source is None or not hasattr(source, "result"):
        raise SpecValidationError("ResidualPathPatch path_mask source must be an operation artifact ref")
    payload = source.result()
    if not isinstance(payload, Mapping):
        raise TypeError("Path mask payload must be a mapping")
    if str(payload.get("kind") or "") != "explicit_path_mask_result":
        raise SpecValidationError("ResidualPathPatch currently requires an explicit_path_mask_result source")
    edges = payload.get("edges")
    if not isinstance(edges, list):
        raise TypeError("Path mask payload must contain an 'edges' list")
    return dict(payload)


def path_mask_edges(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    edges = payload.get("edges")
    if not isinstance(edges, list):
        raise SpecValidationError("Path mask payload must contain an 'edges' list")
    normalized: list[dict[str, Any]] = []
    for edge in edges:
        if not isinstance(edge, Mapping):
            raise SpecValidationError("Path mask edges must be mappings")
        normalized.append(
            {
                "source_layer": int(edge["source_layer"]),
                "write_layer": int(edge["write_layer"]),
                "weight": float(edge.get("weight", 1.0)),
            }
        )
    return normalized


def required_source_layers_for_patch(patch: ActivationPatchSpec) -> tuple[int, ...]:
    if isinstance(patch, ResidualPathPatch):
        return tuple(sorted({int(edge["source_layer"]) for edge in path_mask_edges(load_path_mask_source(patch))}))
    return tuple(int(patch.source_layer_for(int(write_layer))) for write_layer in patch.write_site.layers)


__all__ = [
    "load_activation_bank_source",
    "load_centroid_source",
    "load_direction_source",
    "load_path_mask_source",
    "load_subspace_source",
    "path_mask_edges",
    "required_source_layers_for_patch",
]
