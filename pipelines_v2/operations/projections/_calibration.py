"""Calibration helpers for structured projection outputs."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def fit_quantile_bands(
    values: Sequence[float],
    *,
    bands: Sequence[str],
) -> dict[str, object]:
    """Fit simple quantile thresholds for ordered score bands."""

    normalized_bands = [str(band) for band in bands if str(band).strip()]
    if len(normalized_bands) < 2:
        raise ValueError("Quantile band calibration requires at least two named bands")
    scores = np.asarray([float(value) for value in values], dtype=np.float32)
    if scores.size == 0:
        raise ValueError("Quantile band calibration requires at least one score")
    quantiles = [index / len(normalized_bands) for index in range(1, len(normalized_bands))]
    thresholds = np.quantile(scores, quantiles).astype(np.float32)
    return {
        "bands": normalized_bands,
        "thresholds": thresholds.tolist(),
        "quantiles": quantiles,
        "min": float(scores.min()),
        "max": float(scores.max()),
        "count": int(scores.size),
    }
