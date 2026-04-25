"""Vector pooling and projection kernels."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from pipelines_v2.core.types import SpecValidationError
from pipelines_v2.operations.common.tokens import TokenPooling


def pool_positions(
    values: NDArray[np.float32],
    *,
    positions: Sequence[int],
    pooling: TokenPooling,
) -> NDArray[np.float32]:
    """Pool one token-matrix slice into a single readout vector.

    ``values`` is the feature-local token matrix for one example/layer. The
    positions are already rebased into that feature-local coordinate system by
    capture, so this helper only validates and pools.
    """

    selected_positions = [int(position) for position in positions]
    if not selected_positions:
        raise SpecValidationError("Projection slices must contain at least one token position")
    selected = np.asarray(values[selected_positions], dtype=np.float32)
    if selected.ndim != 2:
        raise TypeError("Projection pooling expects a rank-2 token matrix")
    if pooling.kind == "mean":
        return selected.mean(axis=0).astype(np.float32)
    if pooling.kind == "first":
        return selected[0].astype(np.float32)
    if pooling.kind == "last":
        return selected[-1].astype(np.float32)
    raise SpecValidationError(f"Unsupported projection pooling mode: {pooling.kind}")


def project_vector(
    pooled: NDArray[np.float32],
    *,
    direction: NDArray[np.float32],
    metric: str,
) -> float:
    """Project one pooled vector onto one coordinate.

    ``signed_dot`` assumes callers have normalized coordinates when they want a
    pure signed projection. ``cosine`` normalizes both operands at score time.
    """

    value = np.asarray(pooled, dtype=np.float32)
    axis = np.asarray(direction, dtype=np.float32)
    if value.ndim != 1 or axis.ndim != 1:
        raise TypeError("Projection metrics expect rank-1 pooled vectors and directions")
    if value.shape[0] != axis.shape[0]:
        raise SpecValidationError(
            f"Projection dimension mismatch: pooled vector width={value.shape[0]} direction width={axis.shape[0]}"
        )
    normalized_metric = str(metric).strip().lower()
    if normalized_metric == "signed_dot":
        return float(np.dot(value, axis))
    if normalized_metric == "cosine":
        denom = float(np.linalg.norm(value) * np.linalg.norm(axis))
        if denom <= 0:
            return 0.0
        return float(np.dot(value, axis) / denom)
    raise SpecValidationError(f"Unsupported projection metric: {metric!r}")
