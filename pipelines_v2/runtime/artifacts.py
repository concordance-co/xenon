"""Runtime registry of operation specs that produce operation artifacts."""

from __future__ import annotations

from pipelines_v2.mechinterp.assistant_axis import (
    AssistantAxisPrecomputedCoordinateSpec,
    AssistantAxisScoreSpec,
    AssistantAxisTraitCoordinateSpec,
    AssistantAxisVectorSpec,
)
from pipelines_v2.mechinterp.emotions import (
    EmotionDirectionSpec,
    EmotionGeometrySpec,
    EmotionPrecomputedVectorSpaceSpec,
    EmotionScoreSpec,
    EmotionVectorSpaceSpec,
)
from pipelines_v2.mechinterp.refusal import (
    RefusalAblationSubspaceSpec,
    RefusalDirectionSelectionSpec,
    RefusalDirectionSpec,
    RefusalScoreSpec,
)
from pipelines_v2.mechinterp.truthfulness import (
    TruthfulnessAblationSubspaceSpec,
    TruthfulnessDirectionSelectionSpec,
    TruthfulnessDirectionSpec,
    TruthfulnessScoreSpec,
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
    AssistantAxisTraitCoordinateSpec,
    EmotionPrecomputedVectorSpaceSpec,
    EmotionVectorSpaceSpec,
    EmotionScoreSpec,
    EmotionDirectionSpec,
    EmotionGeometrySpec,
    RefusalDirectionSpec,
    RefusalScoreSpec,
    RefusalDirectionSelectionSpec,
    RefusalAblationSubspaceSpec,
    TruthfulnessDirectionSpec,
    TruthfulnessScoreSpec,
    TruthfulnessDirectionSelectionSpec,
    TruthfulnessAblationSubspaceSpec,
    ReportSpec,
)


__all__ = ["ARTIFACT_BOUND_SPECS"]
