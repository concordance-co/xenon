"""Runtime registry of operation specs that produce operation artifacts."""

from __future__ import annotations

from pipelines_v2.mechinterp.assistant_axis import (
    AssistantAxisPrecomputedCoordinateSpec,
    AssistantAxisScoreSpec,
    AssistantAxisVectorSpec,
)
from pipelines_v2.mechinterp.emotions import (
    EmotionDirectionSpec,
    EmotionGeometrySpec,
    EmotionPrecomputedVectorSpaceSpec,
    EmotionScoreSpec,
    EmotionVectorSpaceSpec,
)
from pipelines_v2.operations.specs import (
    ActivationBankSpec,
    BasisSpec,
    CentroidSpec,
    CoordinateImportSpec,
    DirectionSpec,
    ExplicitPathMaskSpec,
    GeometrySpec,
    LabelFieldsSpec,
    LabelMapSpec,
    PatchComparisonSpec,
    PairDeltaSpec,
    ProbeSpec,
    ProjectionCalibrationSpec,
    ProjectionSpec,
    ReportSpec,
    ResidualizedProbeSpec,
    SubspaceSpec,
    TextBaselineSpec,
    TransferProbeSpec,
    TransformSpec,
)


ARTIFACT_BOUND_SPECS = (
    ProbeSpec,
    TransferProbeSpec,
    TextBaselineSpec,
    ResidualizedProbeSpec,
    DirectionSpec,
    BasisSpec,
    CentroidSpec,
    GeometrySpec,
    SubspaceSpec,
    ActivationBankSpec,
    ExplicitPathMaskSpec,
    PairDeltaSpec,
    LabelMapSpec,
    LabelFieldsSpec,
    TransformSpec,
    PatchComparisonSpec,
    CoordinateImportSpec,
    ProjectionSpec,
    ProjectionCalibrationSpec,
    AssistantAxisPrecomputedCoordinateSpec,
    AssistantAxisVectorSpec,
    AssistantAxisScoreSpec,
    EmotionPrecomputedVectorSpaceSpec,
    EmotionVectorSpaceSpec,
    EmotionScoreSpec,
    EmotionDirectionSpec,
    EmotionGeometrySpec,
    ReportSpec,
)


__all__ = ["ARTIFACT_BOUND_SPECS"]
