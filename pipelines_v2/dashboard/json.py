"""JSON helpers shared by dashboard readers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json_object_optional(path: Path) -> dict[str, Any] | None:
    """Read a JSON object from a file if it exists and contains an object."""

    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return None
    return payload


__all__ = ["read_json_object_optional"]
