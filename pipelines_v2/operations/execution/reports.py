"""Execution helpers for report specs."""

from __future__ import annotations

import logging
import time

from pipelines_v2.operations.reports import ReportSpec

from .common import OperationExecutionResult, report_example_keys, summarize_report_input


logger = logging.getLogger(__name__)


def run_report(spec: ReportSpec) -> OperationExecutionResult:
    started = time.perf_counter()
    logger.info(
        "Running report template=%s inputs=%d output_dir=%s",
        spec.template,
        len(spec.inputs),
        spec.output_dir,
    )
    inputs = [summarize_report_input(item) for item in spec.inputs]
    logger.info(
        "Summarized report inputs template=%s inputs=%d elapsed=%.2fs",
        spec.template,
        len(inputs),
        time.perf_counter() - started,
    )
    coverage_started = time.perf_counter()
    example_keys = report_example_keys(spec.inputs)
    logger.info(
        "Computed report example coverage template=%s materialized=%s example_count=%s elapsed=%.2fs total_elapsed=%.2fs",
        spec.template,
        example_keys is not None,
        len(example_keys) if example_keys is not None else None,
        time.perf_counter() - coverage_started,
        time.perf_counter() - started,
    )
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
