"""Legacy built-in type snapshot; runtime dispatch uses the live operation registry.

ARTIFACT_BOUND_SPECS remains for old importers. New code must use
is_artifact_operation so registrations added after runner import are visible.
"""

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
    PersistedProbeImportSpec,
    PersistedProbeInferenceSpec,
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
    PersistedProbeImportSpec,
    PersistedProbeInferenceSpec,
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


from pipelines_v2.operations.registry import is_artifact_operation, operation_artifact_kind

__all__ = ["ARTIFACT_BOUND_SPECS", "is_artifact_operation", "operation_artifact_kind"]
