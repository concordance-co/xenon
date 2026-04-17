"""Row shaping helpers for intervention artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pipelines_v2.data.datasets import Dataset


def row_example_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("example")
    if isinstance(payload, Mapping):
        return dict(payload)
    example_key = str(row.get("example_key") or "")
    return {"key": example_key}


def rows_example_coverage(
    *,
    dataset: Dataset,
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    coverage = dataset.coverage()
    example_keys = [str(row.get("example_key")) for row in rows if str(row.get("example_key") or "").strip()]
    ordered_keys = sorted(dict.fromkeys(example_keys))
    coverage["example_keys"] = ordered_keys
    coverage["example_count"] = len(ordered_keys)
    prompt_hashes = dict(coverage.get("prompt_hashes") or {})
    if prompt_hashes:
        coverage["prompt_hashes"] = {key: prompt_hashes[key] for key in ordered_keys if key in prompt_hashes}
    return coverage


__all__ = ["row_example_payload", "rows_example_coverage"]
