"""Coordinate artifacts and vector loading helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pipelines_v2.core.types import SpecValidationError


@dataclass(frozen=True, slots=True)
class ResolvedCoordinate:
    """One named coordinate family resolved to concrete vectors."""

    name: str
    layers: dict[int, np.ndarray]
    source_kind: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_coordinate_import_payload(
    *,
    path: str,
    format: str,
    select_layer: int | None,
    normalize: str,
    name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize an external vector artifact into a canonical coordinate result."""

    resolved_path = Path(path).expanduser()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Coordinate import path does not exist: {resolved_path}")

    tensor = _load_coordinate_tensor(path=resolved_path, format=format)
    matrix = np.asarray(tensor, dtype=np.float32)
    if matrix.ndim == 1:
        layer_indices = (int(select_layer) if select_layer is not None else 0,)
        vectors_by_layer = {int(layer_indices[0]): matrix}
    elif matrix.ndim == 2:
        if select_layer is not None:
            vectors_by_layer = {int(select_layer): matrix[int(select_layer)]}
        else:
            vectors_by_layer = {int(layer): matrix[int(layer)] for layer in range(int(matrix.shape[0]))}
    else:
        raise SpecValidationError(
            f"Coordinate imports must resolve to rank-1 or rank-2 tensors, got shape {tuple(matrix.shape)}"
        )

    payload_layers: dict[str, Any] = {}
    for layer, raw_vector in vectors_by_layer.items():
        unit, norm = _normalize_vector(raw_vector, normalize=normalize)
        payload_layers[str(layer)] = {
            "vector": unit.astype(np.float32).tolist(),
            "raw_vector": np.asarray(raw_vector, dtype=np.float32).tolist(),
            "norm": float(norm),
        }

    coordinate_name = str(name or resolved_path.stem)
    return {
        "kind": "coordinate_result",
        "coordinate_kind": "direction",
        "name": coordinate_name,
        "format": str(format),
        "path": str(resolved_path),
        "layers": payload_layers,
        "metadata": dict(metadata or {}),
        "summary": {
            "layer_count": len(payload_layers),
            "selected_layer": int(select_layer) if select_layer is not None else None,
            "normalized": str(normalize),
        },
    }


def resolve_coordinate(
    source: Any,
    *,
    fallback_name: str | None = None,
) -> ResolvedCoordinate:
    """Load a coordinate artifact or direction artifact into in-memory vectors."""

    payload = source.result() if hasattr(source, "result") else source
    if not isinstance(payload, Mapping):
        raise TypeError(f"Coordinate source must resolve to a mapping, got {type(payload).__name__}")
    payload_kind = str(payload.get("kind") or "")
    if payload_kind not in {"coordinate_result", "direction_result"}:
        raise SpecValidationError(
            "Projection coordinates must resolve to coordinate_result or direction_result payloads, "
            f"got {payload_kind!r}"
        )
    raw_layers = payload.get("layers")
    if not isinstance(raw_layers, Mapping) or not raw_layers:
        raise SpecValidationError("Coordinate payload must contain a non-empty 'layers' mapping")
    layers: dict[int, np.ndarray] = {}
    for layer_name, layer_payload in raw_layers.items():
        if not isinstance(layer_payload, Mapping):
            raise TypeError("Coordinate layer payloads must be mappings")
        vector_payload = layer_payload.get("vector")
        if vector_payload is None:
            raise SpecValidationError(f"Coordinate layer {layer_name!r} is missing a 'vector'")
        vector = np.asarray(vector_payload, dtype=np.float32)
        if vector.ndim != 1:
            raise SpecValidationError(f"Coordinate layer {layer_name!r} vector must be rank-1")
        layers[int(layer_name)] = vector

    name = str(payload.get("name") or fallback_name or getattr(source, "id", "") or "coordinate")
    metadata = payload.get("metadata")
    return ResolvedCoordinate(
        name=name,
        layers=layers,
        source_kind=payload_kind,
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
    )


def coordinate_name_key(name: str) -> str:
    """Return a label-safe coordinate key."""

    return re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip()).strip("_").lower() or "coordinate"


def _load_coordinate_tensor(*, path: Path, format: str) -> Any:
    normalized = str(format).strip().lower()
    if normalized == "torch_tensor_or_axis_dict":
        import torch

        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, Mapping) and payload.get("axis") is not None:
            payload = payload["axis"]
        if hasattr(payload, "detach"):
            return payload.detach().cpu().numpy()
        return np.asarray(payload)
    if normalized == "npy":
        return np.load(path)
    if normalized == "json_vector":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, Mapping) and payload.get("vector") is not None:
            payload = payload["vector"]
        return np.asarray(payload)
    raise SpecValidationError(f"Unsupported coordinate import format: {format!r}")


def _normalize_vector(vector: np.ndarray, *, normalize: str) -> tuple[np.ndarray, float]:
    raw = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(raw))
    mode = str(normalize).strip().lower()
    if mode in {"none", ""}:
        return raw.astype(np.float32), norm
    if mode == "l2":
        if norm <= 0:
            return raw.astype(np.float32), norm
        return (raw / norm).astype(np.float32), norm
    raise SpecValidationError(f"Unsupported coordinate normalization mode: {normalize!r}")
