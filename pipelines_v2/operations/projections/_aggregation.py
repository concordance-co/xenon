"""Aggregation helpers for structured projection scores."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def summarize_scores(
    scores: Sequence[float],
    *,
    summary_names: Sequence[str],
    order_values: Sequence[float] | None = None,
) -> dict[str, float]:
    """Compute reusable summaries over ordered slice scores.

    ``order_values`` lets callers summarize repeated semantic slices in their
    original order, for example chat-turn index. The ``trend`` summary is the
    least-squares slope of score versus that order value.
    """

    values = np.asarray([float(score) for score in scores], dtype=np.float32)
    if values.size == 0:
        return {}
    ordered_axis = np.asarray(list(order_values) if order_values is not None else list(range(len(values))), dtype=np.float32)
    summaries: dict[str, float] = {}
    for name in summary_names:
        normalized = str(name).strip().lower()
        if normalized == "mean":
            summaries["mean"] = float(values.mean())
        elif normalized == "min":
            summaries["min"] = float(values.min())
        elif normalized == "max":
            summaries["max"] = float(values.max())
        elif normalized == "std":
            summaries["std"] = float(values.std())
        elif normalized == "trend":
            summaries["trend"] = _linear_trend(values, ordered_axis)
        elif normalized == "first_last_delta":
            summaries["first_last_delta"] = float(values[-1] - values[0])
    return summaries


def _linear_trend(values: np.ndarray, axis: np.ndarray) -> float:
    if values.size < 2 or axis.size != values.size:
        return 0.0
    centered_axis = axis - axis.mean()
    denom = float(np.dot(centered_axis, centered_axis))
    if denom <= 0:
        return 0.0
    centered_values = values - values.mean()
    return float(np.dot(centered_axis, centered_values) / denom)
