"""Planning-time validation for intervention workflows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from ._runtime_resolution import (
    activation_bank_case_errors,
    resolve_patched_generation_cases,
    resolve_patched_generation_targets,
)
from ._runtime_sources import (
    load_activation_bank_source,
    load_centroid_source,
    load_direction_source,
    load_path_mask_source,
    load_subspace_source,
    path_mask_edges,
)
from .generation import PatchedGenerationSpec
from .recipes import ProjectOutPatch, RandomControlPatch, ResidualPathPatch, SwapComponentsPatch


def patched_generation_plan_errors(spec: PatchedGenerationSpec) -> list[str]:
    try:
        resolved_spec = spec.resolve_dataset()
    except Exception as exc:
        return [f"PatchedGenerationSpec dataset resolution failed during plan: {exc}"]

    patch = resolved_spec.patch

    if patch.requires_pairing():
        try:
            resolved_cases, _ = resolve_patched_generation_cases(resolved_spec)
        except Exception as exc:
            return [str(exc)]
        activation_bank_source = getattr(patch, "activation_bank", None)
        path_mask_source = getattr(patch, "path_mask", None)
        if patch.uses_activation_bank() and (_is_step_ref(activation_bank_source) or _is_remote_artifact_ref(activation_bank_source)):
            if patch.is_residual_path() and not _is_step_ref(path_mask_source):
                try:
                    path_mask = load_path_mask_source(patch)  # type: ignore[arg-type]
                except Exception as exc:
                    return [str(exc)]
                return _path_mask_errors(patch=patch, payload=path_mask)
            return []
        if patch.uses_path_mask() and _is_step_ref(path_mask_source):
            return []
        try:
            activation_bank = load_activation_bank_source(patch)
        except Exception as exc:
            return [str(exc)]
        errors = activation_bank_case_errors(
            spec=resolved_spec,
            activation_bank=activation_bank,
            resolved_cases=resolved_cases,
        )
        if patch.is_residual_path():
            try:
                path_mask = load_path_mask_source(patch)  # type: ignore[arg-type]
            except Exception as exc:
                return [str(exc)]
            errors.extend(_path_mask_errors(patch=patch, payload=path_mask))
        return errors

    try:
        resolve_patched_generation_targets(resolved_spec)
    except Exception as exc:
        return [str(exc)]

    loaders: list[tuple[Any, Any]] = []
    if patch.uses_subspace() and getattr(patch, "subspace", None) is not None:
        if not _is_step_ref(getattr(patch, "subspace", None)):
            loaders.append((load_subspace_source, patch))
    if patch.uses_direction():
        if not _is_step_ref(getattr(patch, "direction", None)):
            loaders.append((load_direction_source, patch))
    if patch.uses_centroid():
        if not _is_step_ref(getattr(patch, "centroids", None)):
            loaders.append((load_centroid_source, patch))
    for loader, value in loaders:
        try:
            loader(value)
        except Exception as exc:
            return [str(exc)]
    return _patch_source_errors(patch)


def _patch_source_errors(patch: Any) -> list[str]:
    errors: list[str] = []
    if isinstance(patch, (ProjectOutPatch, RandomControlPatch, SwapComponentsPatch)):
        component_map = getattr(patch, "component_indices_by_layer", {})
        for write_layer, indices in dict(component_map).items():
            invalid = [int(index) for index in indices if int(index) < 0]
            if invalid:
                errors.append(
                    f"{type(patch).__name__} component indices must be non-negative for write layer {int(write_layer)}: {invalid}"
                )
    return errors


def _path_mask_errors(*, patch: ResidualPathPatch, payload: Mapping[str, Any]) -> list[str]:
    edges = path_mask_edges(payload)
    by_write_layer = defaultdict(list)
    for edge in edges:
        by_write_layer[int(edge["write_layer"])].append(edge)
    errors: list[str] = []
    for write_layer in patch.write_site.layers:
        layer_edges = by_write_layer.get(int(write_layer), [])
        if not layer_edges:
            errors.append(
                f"ResidualPathPatch path_mask is missing edges for write layer {int(write_layer)}"
            )
    return errors


def _is_step_ref(value: Any) -> bool:
    return bool(getattr(value, "kind", "") == "step_ref" and hasattr(value, "step"))


def _is_remote_artifact_ref(value: Any) -> bool:
    store = getattr(value, "store", None)
    manifest = getattr(value, "manifest", None)
    if not callable(manifest):
        return False
    if getattr(store, "kind", "") != "modal_volume":
        return False
    return bool(getattr(value, "kind", "") == "operation_artifact")


__all__ = ["patched_generation_plan_errors"]
