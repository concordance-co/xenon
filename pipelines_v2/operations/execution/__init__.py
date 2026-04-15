"""Artifact-bound execution dispatch split by operation family."""

from __future__ import annotations

from typing import Any

from pipelines_v2.operations.derive import LabelFieldsSpec, LabelMapSpec, PairDeltaSpec, TransformSpec
from pipelines_v2.operations.readouts import ProbeSpec, ResidualizedProbeSpec, TextBaselineSpec, TransferProbeSpec
from pipelines_v2.operations.representation import BasisSpec, DirectionSpec, GeometrySpec
from pipelines_v2.operations.reports import ReportSpec

from .common import OperationExecutionResult, feature_matrices
from .derive import run_label_fields, run_label_map, run_pair_delta, run_transform
from .readouts import run_probe, run_residualized_probe, run_text_baseline, run_transfer_probe
from .reports import run_report
from .representation import run_basis, run_direction, run_geometry


def execute_artifact_operation(spec: Any) -> OperationExecutionResult:
    if isinstance(spec, ProbeSpec):
        return run_probe(spec)
    if isinstance(spec, TransferProbeSpec):
        return run_transfer_probe(spec)
    if isinstance(spec, TextBaselineSpec):
        return run_text_baseline(spec)
    if isinstance(spec, ResidualizedProbeSpec):
        return run_residualized_probe(spec)
    if isinstance(spec, DirectionSpec):
        return run_direction(spec)
    if isinstance(spec, BasisSpec):
        return run_basis(spec)
    if isinstance(spec, GeometrySpec):
        return run_geometry(spec)
    if isinstance(spec, PairDeltaSpec):
        return run_pair_delta(spec)
    if isinstance(spec, LabelMapSpec):
        return run_label_map(spec)
    if isinstance(spec, LabelFieldsSpec):
        return run_label_fields(spec)
    if isinstance(spec, TransformSpec):
        return run_transform(spec)
    if isinstance(spec, ReportSpec):
        return run_report(spec)
    raise NotImplementedError(f"Artifact-bound execution is not implemented for {type(spec).__name__}")


__all__ = [
    "OperationExecutionResult",
    "execute_artifact_operation",
    "feature_matrices",
]
