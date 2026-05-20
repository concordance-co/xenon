"""Artifact-bound execution for intervention comparison specs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pipelines_v2.core.types import SpecValidationError
from pipelines_v2.operations.execution.common import OperationExecutionResult
from pipelines_v2.operations.interventions import PatchComparisonSpec
from pipelines_v2.operations.interventions.runtime import (
    aggregate_patch_comparison_rows,
    evaluate_patch_comparison_row,
    row_example_payload,
)


def run_patch_comparison(spec: PatchComparisonSpec) -> OperationExecutionResult:
    baseline_payload = _operation_result_payload(spec.baseline, label="baseline")
    baseline_rows = _rows_by_example_key(baseline_payload, label="baseline")
    variant_rows = {
        str(name): _rows_by_example_key(
            _operation_result_payload(artifact, label=f"variants[{name}]"),
            label=f"variants[{name}]",
        )
        for name, artifact in spec.variants.items()
    }
    baseline_keys = set(baseline_rows)
    for name, mapping in variant_rows.items():
        variant_keys = set(mapping)
        missing = sorted(baseline_keys - variant_keys)
        extras = sorted(variant_keys - baseline_keys)
        if missing or extras:
            problems: list[str] = []
            if missing:
                problems.append(f"missing={missing[:5]}{' ...' if len(missing) > 5 else ''}")
            if extras:
                problems.append(f"extra={extras[:5]}{' ...' if len(extras) > 5 else ''}")
            raise SpecValidationError(
                "PatchComparisonSpec baseline and variant row sets must match exactly "
                f"for variants[{name}]: {'; '.join(problems)}"
            )

    rows: list[dict[str, Any]] = []
    for example_key in sorted(baseline_rows):
        baseline_row = baseline_rows[example_key]
        missing_variants = [name for name, mapping in variant_rows.items() if example_key not in mapping]
        if missing_variants:
            rows.append(
                {
                    "example_key": example_key,
                    "status": "skipped",
                    "skip_reason": f"missing_variants:{','.join(sorted(missing_variants))}",
                }
            )
            continue
        aligned_variants = {
            name: mapping[example_key]
            for name, mapping in variant_rows.items()
        }
        evaluation = evaluate_patch_comparison_row(
            spec=spec,
            example=row_example_payload(baseline_row),
            baseline=baseline_row,
            variants=aligned_variants,
        )
        rows.append(
            {
                "example_key": example_key,
                "status": "ok",
                "skip_reason": "",
                "example": row_example_payload(baseline_row),
                "baseline": baseline_row,
                "variants": aligned_variants,
                "evaluation": evaluation,
            }
        )

    summary = aggregate_patch_comparison_rows(rows)
    summary["variant_names"] = sorted(str(name) for name in spec.variants)
    return OperationExecutionResult(
        payload={
            "kind": "patch_comparison_result",
            "summary": summary,
            "rows": rows,
        },
        metadata={"variant_names": summary["variant_names"]},
        example_coverage={"example_keys": sorted(baseline_rows)},
    )


def _operation_result_payload(value: Any, *, label: str) -> Mapping[str, Any]:
    if value is None or not hasattr(value, "result"):
        raise SpecValidationError(f"PatchComparisonSpec {label} must be an operation artifact ref")
    payload = value.result()
    if not isinstance(payload, Mapping):
        raise SpecValidationError(f"PatchComparisonSpec {label} result payload must be a mapping")
    return payload


def _rows_by_example_key(payload: Mapping[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise SpecValidationError(f"PatchComparisonSpec {label} payload must contain a 'rows' list")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise SpecValidationError(f"PatchComparisonSpec {label} rows must be mappings")
        example_key = str(row.get("example_key") or "")
        if not example_key:
            raise SpecValidationError(f"PatchComparisonSpec {label} rows must include example_key")
        indexed[example_key] = dict(row)
    return indexed


__all__ = ["run_patch_comparison"]
