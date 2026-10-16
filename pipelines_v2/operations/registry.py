"""Operation spec registry and deserialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from pipelines_v2.core.registry import load_from_kind_registry, register_kind
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
from pipelines_v2.operations.readouts import (
    PersistedProbeImportSpec,
    PersistedProbeInferenceSpec,
    ProbeSpec,
    ResidualizedProbeSpec,
    TextBaselineSpec,
    TransferProbeSpec,
)
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


def _assistant_axis_trait_coordinate_from_dict(payload: dict[str, Any]) -> OperationSpec:
    from pipelines_v2.mechinterp.assistant_axis import AssistantAxisTraitCoordinateSpec

    return AssistantAxisTraitCoordinateSpec.from_dict(payload)


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


def _refusal_direction_from_dict(payload: dict[str, Any]) -> OperationSpec:
    from pipelines_v2.mechinterp.refusal import RefusalDirectionSpec

    return RefusalDirectionSpec.from_dict(payload)


def _refusal_score_from_dict(payload: dict[str, Any]) -> OperationSpec:
    from pipelines_v2.mechinterp.refusal import RefusalScoreSpec

    return RefusalScoreSpec.from_dict(payload)


def _refusal_direction_selection_from_dict(payload: dict[str, Any]) -> OperationSpec:
    from pipelines_v2.mechinterp.refusal import RefusalDirectionSelectionSpec

    return RefusalDirectionSelectionSpec.from_dict(payload)


def _refusal_ablation_subspace_from_dict(payload: dict[str, Any]) -> OperationSpec:
    from pipelines_v2.mechinterp.refusal import RefusalAblationSubspaceSpec

    return RefusalAblationSubspaceSpec.from_dict(payload)


def _truthfulness_direction_from_dict(payload: dict[str, Any]) -> OperationSpec:
    from pipelines_v2.mechinterp.truthfulness import TruthfulnessDirectionSpec

    return TruthfulnessDirectionSpec.from_dict(payload)


def _truthfulness_score_from_dict(payload: dict[str, Any]) -> OperationSpec:
    from pipelines_v2.mechinterp.truthfulness import TruthfulnessScoreSpec

    return TruthfulnessScoreSpec.from_dict(payload)


def _truthfulness_direction_selection_from_dict(payload: dict[str, Any]) -> OperationSpec:
    from pipelines_v2.mechinterp.truthfulness import TruthfulnessDirectionSelectionSpec

    return TruthfulnessDirectionSelectionSpec.from_dict(payload)


def _truthfulness_ablation_subspace_from_dict(payload: dict[str, Any]) -> OperationSpec:
    from pipelines_v2.mechinterp.truthfulness import TruthfulnessAblationSubspaceSpec

    return TruthfulnessAblationSubspaceSpec.from_dict(payload)


_OPERATION_LOADERS: dict[str, OperationLoader] = {
    CaptureSpec.kind: CaptureSpec.from_dict,
    ProbeSpec.kind: ProbeSpec.from_dict,
    PersistedProbeImportSpec.kind: PersistedProbeImportSpec.from_dict,
    PersistedProbeInferenceSpec.kind: PersistedProbeInferenceSpec.from_dict,
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
    "assistant_axis_trait_coordinate": _assistant_axis_trait_coordinate_from_dict,
    "emotion_precomputed_vector_space": _emotion_precomputed_vector_space_from_dict,
    "emotion_vector_space": _emotion_vector_space_from_dict,
    "emotion_score": _emotion_score_from_dict,
    "emotion_direction": _emotion_direction_from_dict,
    "emotion_geometry": _emotion_geometry_from_dict,
    "refusal_direction": _refusal_direction_from_dict,
    "refusal_score": _refusal_score_from_dict,
    "refusal_direction_selection": _refusal_direction_selection_from_dict,
    "refusal_ablation_subspace": _refusal_ablation_subspace_from_dict,
    "truthfulness_direction": _truthfulness_direction_from_dict,
    "truthfulness_score": _truthfulness_score_from_dict,
    "truthfulness_direction_selection": _truthfulness_direction_selection_from_dict,
    "truthfulness_ablation_subspace": _truthfulness_ablation_subspace_from_dict,
    ReportSpec.kind: ReportSpec.from_dict,
}


@dataclass(frozen=True, slots=True)
class OperationRegistration:
    """One source of truth for loading, execution placement, and output identity."""

    loader: OperationLoader
    placement: Literal["capture", "generation", "intervention", "artifact"]
    artifact_kind: str
    executor: Callable[[Any], Any] | None = None


def _execute_builtin_artifact(spec: OperationSpec) -> Any:
    from pipelines_v2.operations.execution import _execute_builtin_artifact_operation

    return _execute_builtin_artifact_operation(spec)


# Keep built-in loaders explicit and lazy. No import path comes from a payload.
_OPERATION_REGISTRATIONS: dict[str, OperationRegistration] = {
    kind: OperationRegistration(
        loader=loader,
        placement={"capture": "capture", "generation_run": "generation", "patched_generation": "intervention"}.get(kind, "artifact"),
        artifact_kind=kind,
        executor=None if kind in {"capture", "generation_run", "patched_generation"} else _execute_builtin_artifact,
    )
    for kind, loader in _OPERATION_LOADERS.items()
}
BUILTIN_OPERATION_KINDS = frozenset(_OPERATION_REGISTRATIONS)


def register_operation(
    kind: str,
    *,
    loader: OperationLoader,
    executor: Callable[[Any], Any],
    artifact_kind: str | None = None,
    placement: Literal["artifact"] = "artifact",
) -> None:
    """Register an artifact operation atomically before loading its specs.

    Model-bound capture/generation/intervention use the existing engine protocol.
    New artifact operations need no edits to runner dispatch. Registrations are
    process-local: trusted local and worker bootstrap must install the same ones.
    """
    if placement != "artifact":
        raise ValueError("New operation registrations require artifact placement")
    if not callable(loader) or not callable(executor):
        raise TypeError("Operation loader and executor must be callable")
    output = kind if artifact_kind is None else artifact_kind
    if not isinstance(output, str) or not output.strip() or output != output.strip() or output == "capture":
        raise ValueError("Operation artifact_kind must be non-empty and cannot reserve 'capture'")
    registration = OperationRegistration(loader, placement, output, executor)
    register_kind(_OPERATION_REGISTRATIONS, kind, registration, label="Operation")
    _OPERATION_LOADERS[kind] = loader


def _load_external_mattr(payload):
    from pipelines_v2.operations.attribution.specs import MAttrSpec
    return MAttrSpec.from_dict(payload)


def _run_external_mattr(spec):
    from pipelines_v2.operations.attribution.execution import run_mattr
    return run_mattr(spec)


register_operation('external_mattr', loader=_load_external_mattr, executor=_run_external_mattr,
                   artifact_kind='external_attribution')


def operation_registration(spec_or_kind: OperationSpec | str) -> OperationRegistration | None:
    kind = spec_or_kind if isinstance(spec_or_kind, str) else spec_or_kind.kind
    return _OPERATION_REGISTRATIONS.get(kind)


def is_artifact_operation(spec: OperationSpec) -> bool:
    registration = operation_registration(spec)
    return registration is not None and registration.placement == "artifact"


def operation_artifact_kind(spec: OperationSpec) -> str:
    registration = operation_registration(spec)
    if registration is None:
        raise ValueError(f"Unknown operation spec kind: {spec.kind!r}")
    return registration.artifact_kind


def _load_diffusion_operation(payload):
    from pipelines_v2.engine.diffusers.specs import load_diffusion_spec
    return load_diffusion_spec(payload)


def _diffusion_requires_context(spec):
    from pipelines_v2.engine.diffusers.specs import requires_context
    return requires_context(spec)


for _diffusion_kind in ("diffusion_generate", "diffusion_capture", "diffusion_intervene"):
    register_operation(_diffusion_kind, loader=_load_diffusion_operation, executor=_diffusion_requires_context)


def _load_native_route(payload):
    from pipelines_v2.operations.graphs import NativeRouteSpec
    return NativeRouteSpec.from_dict(payload)


def _run_native_route(spec):
    from pipelines_v2.operations.graphs.execution import run_native_route
    return run_native_route(spec)


register_operation('native_route', loader=_load_native_route, executor=_run_native_route, artifact_kind='attribution_graph')


def _load_circuit_trace(payload):
    from pipelines_v2.operations.circuit_tracer import CircuitTraceSpec
    return CircuitTraceSpec.from_dict(payload)


def _run_circuit_trace(spec):
    from pipelines_v2.operations.circuit_tracer.execution import run_circuit_trace
    return run_circuit_trace(spec)


register_operation('circuit_trace', loader=_load_circuit_trace, executor=_run_circuit_trace, artifact_kind='attribution_graph')


def operation_spec_from_dict(payload: dict[str, Any]) -> OperationSpec:
    spec = load_from_kind_registry(payload, _OPERATION_LOADERS, missing_message="Operation spec payload is missing 'kind'", unknown_message="Unknown operation spec kind: {kind!r}")
    if not isinstance(spec, OperationSpec) or spec.kind != str(payload["kind"]).strip():
        raise ValueError("Operation loader returned an incompatible spec kind")
    return spec


__all__ = [
    "OperationRegistration", "register_operation", "operation_registration",
    "operation_spec_from_dict", "is_artifact_operation", "operation_artifact_kind",
]


# Keep training libraries and artifact executors lazy at registry bootstrap.
def _load_sae_operation(payload):
    from pipelines_v2.operations.representation import sae
    classes = {'sae_dictionary_import':sae.SAEDictionaryImportSpec,'sae_encode':sae.SAEEncodeSpec,
               'sae_decode':sae.SAEDecodeSpec,'sae_feature_splice':sae.SAEFeatureSpliceSpec}
    return classes[payload['kind']].from_dict(payload)


def _execute_sae_operation(spec):
    from pipelines_v2.representation.sae import execution
    executors = {'sae_dictionary_import':execution.run_import,'sae_encode':execution.run_encode,
                 'sae_decode':execution.run_decode,'sae_feature_splice':execution.run_splice}
    return executors[spec.kind](spec)


for _sae_kind in ('sae_dictionary_import','sae_encode','sae_decode','sae_feature_splice'):
    register_operation(_sae_kind,loader=_load_sae_operation,executor=_execute_sae_operation)


def _load_recognition(payload):
    from pipelines_v2.engine.recognition import RecognitionSpec
    return RecognitionSpec.from_dict(payload)


def _run_recognition(spec):
    from pipelines_v2.engine.recognition.execution import run_recognition
    return run_recognition(spec)


register_operation('recognition_microscope', loader=_load_recognition, executor=_run_recognition)


# Crosscoder fitting dependencies remain lazy and isolated.
def _load_crosscoder_operation(payload):
    from pipelines_v2.operations.representation import crosscoder
    return getattr(crosscoder, _CROSSCODER_TYPES[payload['kind']]).from_dict(payload)


def _execute_crosscoder_operation(spec):
    from pipelines_v2.representation.crosscoder import execution
    return getattr(execution, _CROSSCODER_EXECUTORS[spec.kind])(spec)


_CROSSCODER_TYPES=dict(zip(('crosscoder_pair_capture','crosscoder_fit','crosscoder_import','crosscoder_encode','crosscoder_decode','crosscoder_feature_splice'),
    ('CrosscoderPairCaptureSpec','CrosscoderFitSpec','CrosscoderImportSpec','CrosscoderEncodeSpec','CrosscoderDecodeSpec','CrosscoderFeatureSpliceSpec')))
_CROSSCODER_EXECUTORS=dict(zip(_CROSSCODER_TYPES,('run_capture','run_fit','run_import','run_encode','run_decode','run_splice')))
for _crosscoder_kind in _CROSSCODER_TYPES:
    register_operation(_crosscoder_kind,loader=_load_crosscoder_operation,executor=_execute_crosscoder_operation)
