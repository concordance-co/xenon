"""Execution helpers for report specs."""

from __future__ import annotations

from pipelines_v2.operations.reports import ReportSpec

from .common import OperationExecutionResult, report_example_keys, summarize_report_input


def run_report(spec: ReportSpec) -> OperationExecutionResult:
    inputs = [summarize_report_input(item) for item in spec.inputs]
    example_keys = report_example_keys(spec.inputs)
    payload = {
        "kind": "report_result",
        "template": spec.template,
        "output_dir": spec.output_dir,
        "inputs": inputs,
        "summary": {
            "template": spec.template,
            "input_count": len(inputs),
            "example_count": len(example_keys) if example_keys is not None else None,
        },
    }
    example_coverage = {
        "materialized": example_keys is not None,
        "example_count": len(example_keys) if example_keys is not None else None,
    }
    if example_keys is not None:
        example_coverage["example_keys"] = example_keys
    return OperationExecutionResult(payload=payload, example_coverage=example_coverage)
