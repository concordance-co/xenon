"""Small vector helpers shared by projection-style operations."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from pipelines_v2.core.types import SpecValidationError


def coordinate_name_key(name: str, *, fallback: str = "coordinate") -> str:
    """Return a stable label-safe key for a named projection coordinate."""

    return re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip()).strip("_").lower() or str(fallback)


def normalize_vector(
    vector: Any,
    *,
    normalize: str,
    error_label: str = "vector",
) -> tuple[np.ndarray, float]:
    """Normalize a rank-1 vector and return the normalized vector plus raw norm."""

    raw = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(raw))
    mode = str(normalize).strip().lower()
    if mode in {"none", ""}:
        return raw.astype(np.float32), norm
    if mode == "l2":
        if norm <= 0:
            return raw.astype(np.float32), norm
        return (raw / norm).astype(np.float32), norm
    raise SpecValidationError(f"Unsupported {error_label} normalization mode: {normalize!r}")


__all__ = ["coordinate_name_key", "normalize_vector"]
