"""Operation spec registry and deserialization helpers."""

from __future__ import annotations

from typing import Any, Callable

from pipelines_v2.core.registry import load_from_kind_registry
from pipelines_v2.core.types import OperationSpec
from pipelines_v2.operations.capture import CaptureSpec
from pipelines_v2.operations.derive import LabelFieldsSpec, LabelMapSpec, PairDeltaSpec, TransformSpec
from pipelines_v2.operations.interventions import (
    ActivationBankSpec,
    ExplicitPathMaskSpec,
    GenerationRunSpec,
    PatchComparisonSpec,
    PatchedGenerationSpec,
)
from pipelines_v2.operations.projections import CoordinateImportSpec, ProjectionCalibrationSpec, ProjectionSpec
from pipelines_v2.operations.readouts import ProbeSpec, ResidualizedProbeSpec, TextBaselineSpec, TransferProbeSpec
from pipelines_v2.operations.reports import ReportSpec
from pipelines_v2.operations.representation import BasisSpec, CentroidSpec, DirectionSpec, GeometrySpec, SubspaceSpec

OperationLoader = Callable[[dict[str, Any]], OperationSpec]


def _assistant_axis_precomputed_coordinate_from_dict(payload: dict[str, Any]) -> OperationSpec:
    from pipelines_v2.mechinterp.assistant_axis import AssistantAxisPrecomputedCoordinateSpec

    return AssistantAxisPrecomputedCoordinateSpec.from_dict(payload)


def _assistant_axis_vector_from_dict(payload: dict[str, Any]) -> OperationSpec:
    from pipelines_v2.mechinterp.assistant_axis import AssistantAxisVectorSpec

    return AssistantAxisVectorSpec.from_dict(payload)


def _assistant_axis_score_from_dict(payload: dict[str, Any]) -> OperationSpec:
    from pipelines_v2.mechinterp.assistant_axis import AssistantAxisScoreSpec

    return AssistantAxisScoreSpec.from_dict(payload)


def _emotion_precomputed_vector_space_from_dict(payload: dict[str, Any]) -> OperationSpec:
    from pipelines_v2.mechinterp.emotions import EmotionPrecomputedVectorSpaceSpec

    return EmotionPrecomputedVectorSpaceSpec.from_dict(payload)


def _emotion_vector_space_from_dict(payload: dict[str, Any]) -> OperationSpec:
    from pipelines_v2.mechinterp.emotions import EmotionVectorSpaceSpec

    return EmotionVectorSpaceSpec.from_dict(payload)


def _emotion_score_from_dict(payload: dict[str, Any]) -> OperationSpec:
    from pipelines_v2.mechinterp.emotions import EmotionScoreSpec

    return EmotionScoreSpec.from_dict(payload)


def _emotion_direction_from_dict(payload: dict[str, Any]) -> OperationSpec:
    from pipelines_v2.mechinterp.emotions import EmotionDirectionSpec

    return EmotionDirectionSpec.from_dict(payload)


def _emotion_geometry_from_dict(payload: dict[str, Any]) -> OperationSpec:
    from pipelines_v2.mechinterp.emotions import EmotionGeometrySpec

    return EmotionGeometrySpec.from_dict(payload)


_OPERATION_LOADERS: dict[str, OperationLoader] = {
    CaptureSpec.kind: CaptureSpec.from_dict,
    ProbeSpec.kind: ProbeSpec.from_dict,
    TransferProbeSpec.kind: TransferProbeSpec.from_dict,
    TextBaselineSpec.kind: TextBaselineSpec.from_dict,
    ResidualizedProbeSpec.kind: ResidualizedProbeSpec.from_dict,
    DirectionSpec.kind: DirectionSpec.from_dict,
    BasisSpec.kind: BasisSpec.from_dict,
    CentroidSpec.kind: CentroidSpec.from_dict,
    SubspaceSpec.kind: SubspaceSpec.from_dict,
    GeometrySpec.kind: GeometrySpec.from_dict,
    ActivationBankSpec.kind: ActivationBankSpec.from_dict,
    ExplicitPathMaskSpec.kind: ExplicitPathMaskSpec.from_dict,
    PairDeltaSpec.kind: PairDeltaSpec.from_dict,
    LabelMapSpec.kind: LabelMapSpec.from_dict,
    LabelFieldsSpec.kind: LabelFieldsSpec.from_dict,
    TransformSpec.kind: TransformSpec.from_dict,
    GenerationRunSpec.kind: GenerationRunSpec.from_dict,
    PatchedGenerationSpec.kind: PatchedGenerationSpec.from_dict,
    PatchComparisonSpec.kind: PatchComparisonSpec.from_dict,
    CoordinateImportSpec.kind: CoordinateImportSpec.from_dict,
    ProjectionSpec.kind: ProjectionSpec.from_dict,
    ProjectionCalibrationSpec.kind: ProjectionCalibrationSpec.from_dict,
    "assistant_axis_precomputed_coordinate": _assistant_axis_precomputed_coordinate_from_dict,
    "assistant_axis_vector": _assistant_axis_vector_from_dict,
    "assistant_axis_score": _assistant_axis_score_from_dict,
    "emotion_precomputed_vector_space": _emotion_precomputed_vector_space_from_dict,
    "emotion_vector_space": _emotion_vector_space_from_dict,
    "emotion_score": _emotion_score_from_dict,
    "emotion_direction": _emotion_direction_from_dict,
    "emotion_geometry": _emotion_geometry_from_dict,
    ReportSpec.kind: ReportSpec.from_dict,
}


def operation_spec_from_dict(payload: dict[str, Any]) -> OperationSpec:
    return load_from_kind_registry(payload, _OPERATION_LOADERS, missing_message="Operation spec payload is missing 'kind'", unknown_message="Unknown operation spec kind: {kind!r}")
