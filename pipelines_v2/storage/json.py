"""JSON serialization helpers for artifact payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def json_default(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a core dependency for v2.
        np = None

    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
