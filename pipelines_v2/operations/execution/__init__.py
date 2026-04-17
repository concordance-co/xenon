"""Artifact-bound execution dispatch split by operation family."""

from __future__ import annotations

from typing import Any

from .common import OperationExecutionResult, feature_matrices


def execute_artifact_operation(spec: Any) -> OperationExecutionResult:
    from pipelines_v2.operations.derive import LabelFieldsSpec, LabelMapSpec, PairDeltaSpec, TransformSpec
    from pipelines_v2.operations.interventions import ActivationBankSpec, ExplicitPathMaskSpec, PatchComparisonSpec
    from pipelines_v2.operations.representation import BasisSpec, CentroidSpec, DirectionSpec, GeometrySpec, SubspaceSpec
    from pipelines_v2.operations.reports import ReportSpec

    if type(spec).__name__ == "ProbeSpec":
        from pipelines_v2.operations.readouts import ProbeSpec
        from .readouts import run_probe
        if isinstance(spec, ProbeSpec):
            return run_probe(spec)
    if type(spec).__name__ == "TransferProbeSpec":
        from pipelines_v2.operations.readouts import TransferProbeSpec
        from .readouts import run_transfer_probe
        if isinstance(spec, TransferProbeSpec):
            return run_transfer_probe(spec)
    if type(spec).__name__ == "TextBaselineSpec":
        from pipelines_v2.operations.readouts import TextBaselineSpec
        from .readouts import run_text_baseline
        if isinstance(spec, TextBaselineSpec):
            return run_text_baseline(spec)
    if type(spec).__name__ == "ResidualizedProbeSpec":
        from pipelines_v2.operations.readouts import ResidualizedProbeSpec
        from .readouts import run_residualized_probe
        if isinstance(spec, ResidualizedProbeSpec):
            return run_residualized_probe(spec)
    if isinstance(spec, DirectionSpec):
        from .representation import run_direction
        return run_direction(spec)
    if isinstance(spec, BasisSpec):
        from .representation import run_basis
        return run_basis(spec)
    if isinstance(spec, CentroidSpec):
        from .representation import run_centroid
        return run_centroid(spec)
    if isinstance(spec, GeometrySpec):
        from .representation import run_geometry
        return run_geometry(spec)
    if isinstance(spec, SubspaceSpec):
        from .representation import run_subspace
        return run_subspace(spec)
    if isinstance(spec, ActivationBankSpec):
        from .intervention_sources import run_activation_bank
        return run_activation_bank(spec)
    if isinstance(spec, ExplicitPathMaskSpec):
        from .intervention_sources import run_explicit_path_mask
        return run_explicit_path_mask(spec)
    if isinstance(spec, PairDeltaSpec):
        from .derive import run_pair_delta
        return run_pair_delta(spec)
    if isinstance(spec, LabelMapSpec):
        from .derive import run_label_map
        return run_label_map(spec)
    if isinstance(spec, LabelFieldsSpec):
        from .derive import run_label_fields
        return run_label_fields(spec)
    if isinstance(spec, TransformSpec):
        from .derive import run_transform
        return run_transform(spec)
    if isinstance(spec, PatchComparisonSpec):
        from .interventions import run_patch_comparison
        return run_patch_comparison(spec)
    if isinstance(spec, ReportSpec):
        from .reports import run_report
        return run_report(spec)
    raise NotImplementedError(f"Artifact-bound execution is not implemented for {type(spec).__name__}")


__all__ = [
    "OperationExecutionResult",
    "execute_artifact_operation",
    "feature_matrices",
]
