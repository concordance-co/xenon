"""Small serializable schema fragments shared by operation specs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class TensorStorage:
    """Requested storage policy for captured tensor payloads."""

    dtype: str = "float16"
    format: str = "safetensors"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TensorStorage":
        return cls(
            dtype=str(payload.get("dtype", "float16")),
            format=str(payload.get("format", "safetensors")),
        )
