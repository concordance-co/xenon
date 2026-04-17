"""Comparison-time evaluation helpers for intervention workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pipelines_v2.core.types import SpecValidationError
from pipelines_v2.operations.common.builders import TransformResult

from .comparison import PatchComparisonSpec


def evaluate_patch_comparison_row(
    *,
    spec: PatchComparisonSpec,
    example: Mapping[str, Any],
    baseline: Mapping[str, Any],
    variants: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    raw = spec.row_evaluator.build(
        {
            "example": dict(example),
            "baseline": dict(baseline),
            "variants": {str(name): dict(payload) for name, payload in variants.items()},
        }
    )
    if isinstance(raw, TransformResult):
        payload = dict(raw.payload)
    elif isinstance(raw, Mapping):
        payload = dict(raw)
    else:
        raise SpecValidationError(
            "PatchComparisonSpec row_evaluator must return a mapping or TransformResult"
        )
    payload.setdefault("metrics", {})
    payload.setdefault("evaluation", {})
    return payload


def aggregate_patch_comparison_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    usable_rows = [row for row in rows if str(row.get("status") or "ok") == "ok"]
    metrics: dict[str, dict[str, Any]] = {}
    for row in usable_rows:
        evaluation = row.get("evaluation")
        if not isinstance(evaluation, Mapping):
            continue
        raw_metrics = evaluation.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            continue
        for name, value in raw_metrics.items():
            normalized = _metric_value(value)
            if normalized is None:
                continue
            entry = metrics.setdefault(
                str(name),
                {
                    "count": 0,
                    "sum": 0.0,
                    "kind": "boolean_rate" if isinstance(value, bool) else "mean",
                },
            )
            entry["count"] += 1
            entry["sum"] += normalized
            if isinstance(value, bool):
                entry["kind"] = "boolean_rate"
    for entry in metrics.values():
        count = max(1, int(entry["count"]))
        entry["value"] = float(entry["sum"] / count)
        del entry["sum"]
    return {
        "example_count": len(rows),
        "compared_count": len(usable_rows),
        "skipped_count": len(rows) - len(usable_rows),
        "metrics": metrics,
    }


def _metric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    return None


__all__ = ["aggregate_patch_comparison_rows", "evaluate_patch_comparison_row"]
