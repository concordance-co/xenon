"""Public import surface for the v2 pipelines API."""

from __future__ import annotations

from .engine.torch import TorchEngine, TorchSession
from .engine.native import NativeTorchEngine, NativeSession
from .operations.common.native import NativeSite, NativeSelection, ExpertLocalCoordinates
from .operations.interventions.native import NativeInterventionSite, NativeCoordinatePatch
from .operations.interventions import ScalePatch

from .storage.finalization import CheckpointJournal, IntegrityError, finalize_manifest, validate_completion
from .runtime.recovery import register_capture_coverage_validator
from .core.provenance import ProvenanceV1, ReuseDecision, compare_provenance
from .core.types import (
    CapabilityError,
    EngineCapability,
    OperationSpec,
    PipelinesV2Error,
    RuntimeSecret,
    SpecValidationError,
    TransferPolicy,
    TransferPolicyError,
)
from .data.datasets import CaseSet, Dataset, Example, LabelPredicate, LabelSet
from .data.images import ImageInput, SpatialSelector, SpatialMapping, DiffusionCoordinates, prepare_image
from .storage.image_features import SpatialTensor, tensor_content_digest, spatial_summary
from .data.sources import ArtifactDatasetSource, HuggingFaceListContrastSource, HuggingFaceSource, InMemorySource, PostgresSource, Source, UrlJsonSource
from .engine import (
    Engine,
    EngineCaptureResult,
    EngineGenerationResult,
    EngineInterventionResult,
    PythonRuntimeSpec,
    RuntimeSpec,
    ToyEngine,
    VLLMEngine,
)
from .mechinterp.assistant_axis import (
    ASSISTANT_AXIS_PROMPT_DATASET_REPO,
    ASSISTANT_AXIS_VECTOR_REPO,
    AssistantAxisPrecomputedCoordinateSpec,
    AssistantAxisScoreSpec,
    AssistantAxisTraitCoordinateSpec,
    AssistantAxisVectorSpec,
    KNOWN_ASSISTANT_AXIS_MODELS,
    assistant_axis_model_config,
    assistant_axis_trait_filename,
    assistant_axis_prompt_dataset,
    discover_assistant_axis_layer_and_capping,
)
from .mechinterp.emotions import (
    EMOTION_PROBES_DATASET_REPO,
    EMOTION_VECTOR_SPACE_KIND,
    EmotionDirectionSpec,
    EmotionGeometrySpec,
    EmotionPrecomputedVectorSpaceSpec,
    EmotionScoreSpec,
    EmotionVectorSpaceSpec,
    emotion_contrast_dataset,
    emotion_probe_story_dataset,
)
from .mechinterp.refusal import (
    REFUSAL_DIRECTION_DATASETS,
    REFUSAL_DIRECTION_REPO_RAW_BASE,
    RefusalAblationSubspaceSpec,
    RefusalDirectionSelectionSpec,
    RefusalDirectionSpec,
    RefusalScoreSpec,
    refusal_direction_processed_dataset,
    refusal_direction_split_dataset,
)
from .mechinterp.truthfulness import (
    TRUTHFULQA_DATASET_REPO,
    TRUTHFULQA_GENERATION_CONFIG,
    TruthfulnessAblationSubspaceSpec,
    TruthfulnessDirectionSelectionSpec,
    TruthfulnessDirectionSpec,
    TruthfulnessScoreSpec,
    truthfulqa_answer_contrast_dataset,
    truthfulqa_generation_dataset,
)
from .operations.capture import CaptureSpec, GenerationSpec, MoERoutingSite, ResidualSite, RoutingRecord
from .operations.common.builders import PromptMetadataBuilder, TransformBuilder, TransformResult
from .operations.common.schemas import TensorStorage
from .operations.common.tokens import TokenPooling, TokenSelector
from .operations.derive import LabelFieldsSpec, LabelMapSpec, PairDeltaSpec, TransformSpec
from .operations.interventions import (
    ActivationPatchSpec,
    ActivationBankSpec,
    AddDirectionPatch,
    ExplicitPathEdge,
    ExplicitPathMaskSpec,
    GenerationRunSpec,
    InterchangePatch,
    InterventionSite,
    PatchApplication,
    PatchComparisonSpec,
    PatchedGenerationSpec,
    ProjectOutPatch,
    RandomControlPatch,
    ResidualInterventionSite,
    ResidualPathPatch,
    SwapComponentsPatch,
    SwapMeanPatch,
)
from .operations.projections import CoordinateImportSpec, ProjectionCalibrationSpec, ProjectionSpec, SectionSelector
from .operations.readouts import (
    PersistedProbeImportSpec,
    PersistedProbeInferenceSpec,
    ProbeSpec,
    ResidualizedProbeSpec,
    TextBaselineSpec,
    TransferProbeSpec,
)
from .operations.reports import ReportSpec
from .operations.representation import BasisSpec, CentroidSpec, DirectionSpec, GeometrySpec, SubspaceSpec
from .runtime import (
    DeploymentConfig,
    DeploymentContext,
    DeploymentController,
    DeploymentHandle,
    DeploymentPlan,
    DeploymentSpec,
    DeploymentTargetSpec,
    EndpointSpec,
    ExecutionPlan,
    HTTPEndpoint,
    LocalResources,
    LocalRunner,
    LocalRunnerSpec,
    ModalDeploymentTargetSpec,
    ModalResources,
    ModalRunner,
    ModalRunnerSpec,
    ModalSecret,
    ModalVolumeMount,
    Runner,
    RunnerSpec,
    ServiceFactory,
)
from .storage.artifacts import ArtifactManifest, CaptureArtifact, FeatureLayerRef, FeatureRef, OperationArtifact
from .storage.artifacts import ArtifactLabelRef
from .storage import (
    ArtifactStore,
    Catalog,
    CompositeCatalog,
    FileCatalog,
    LocalArtifactStore,
    ModalVolumeStore,
    NullCatalog,
    PostgresCatalog,
)
from .workflow import (
    StepFeatureRef,
    StepLabelRef,
    StepRef,
    WorkflowOrchestrator,
    WorkflowPlan,
    WorkflowResult,
    WorkflowRunRecord,
    WorkflowSpec,
    WorkflowStep,
    WorkflowStepContext,
    WorkflowStepRecord,
)

# Explicit adapter registration; these imports do not load optional runtimes.
from .engine.registry import register_engine
from .operations.registry import register_operation
from .operations.capture.specs import register_capture_site
from .operations.interventions.sites import register_intervention_site
from .storage.features import register_feature_codec

__all__ = [
    "TorchEngine", "TorchSession", "ScalePatch",
    "CheckpointJournal", "IntegrityError", "finalize_manifest", "validate_completion",
    "register_capture_coverage_validator",
    "ImageInput", "SpatialSelector", "SpatialMapping", "DiffusionCoordinates", "prepare_image",
    "SpatialTensor", "tensor_content_digest", "spatial_summary",
    "ProvenanceV1",
    "ReuseDecision",
    "compare_provenance",
    "register_engine",
    "register_operation",
    "register_capture_site",
    "register_intervention_site",
    "register_feature_codec",
    "ActivationPatchSpec",
    "ActivationBankSpec",
    "AddDirectionPatch",
    "ASSISTANT_AXIS_PROMPT_DATASET_REPO",
    "ASSISTANT_AXIS_VECTOR_REPO",
    "ArtifactManifest",
    "ArtifactLabelRef",
    "ArtifactDatasetSource",
    "ArtifactStore",
    "AssistantAxisPrecomputedCoordinateSpec",
    "AssistantAxisScoreSpec",
    "AssistantAxisTraitCoordinateSpec",
    "AssistantAxisVectorSpec",
    "BasisSpec",
    "CapabilityError",
    "CaptureArtifact",
    "CaptureSpec",
    "CaseSet",
    "Catalog",
    "CentroidSpec",
    "CompositeCatalog",
    "CoordinateImportSpec",
    "DeploymentConfig",
    "DeploymentContext",
    "DeploymentController",
    "DeploymentHandle",
    "DeploymentPlan",
    "DeploymentSpec",
    "DeploymentTargetSpec",
    "Dataset",
    "DirectionSpec",
    "Engine",
    "EngineCapability",
    "EngineCaptureResult",
    "EngineGenerationResult",
    "EngineInterventionResult",
    "EndpointSpec",
    "EMOTION_PROBES_DATASET_REPO",
    "EMOTION_VECTOR_SPACE_KIND",
    "Example",
    "ExecutionPlan",
    "EmotionDirectionSpec",
    "EmotionGeometrySpec",
    "EmotionPrecomputedVectorSpaceSpec",
    "EmotionScoreSpec",
    "EmotionVectorSpaceSpec",
    "FeatureLayerRef",
    "FeatureRef",
    "FileCatalog",
    "ExplicitPathEdge",
    "ExplicitPathMaskSpec",
    "GenerationSpec",
    "GenerationRunSpec",
    "GeometrySpec",
    "HTTPEndpoint",
    "HuggingFaceSource",
    "HuggingFaceListContrastSource",
    "InMemorySource",
    "KNOWN_ASSISTANT_AXIS_MODELS",
    "LabelFieldsSpec",
    "LabelPredicate",
    "LabelMapSpec",
    "LabelSet",
    "LocalArtifactStore",
    "LocalResources",
    "LocalRunner",
    "LocalRunnerSpec",
    "MoERoutingSite",
    "ModalDeploymentTargetSpec",
    "ModalResources",
    "ModalRunner",
    "ModalRunnerSpec",
    "ModalSecret",
    "ModalVolumeMount",
    "ModalVolumeStore",
    "NullCatalog",
    "OperationSpec",
    "OperationArtifact",
    "PatchApplication",
    "PatchComparisonSpec",
    "PatchedGenerationSpec",
    "PipelinesV2Error",
    "PostgresCatalog",
    "PostgresSource",
    "PromptMetadataBuilder",
    "PersistedProbeImportSpec",
    "PersistedProbeInferenceSpec",
    "ProbeSpec",
    "ProjectionCalibrationSpec",
    "ProjectionSpec",
    "PythonRuntimeSpec",
    "REFUSAL_DIRECTION_DATASETS",
    "REFUSAL_DIRECTION_REPO_RAW_BASE",
    "RefusalAblationSubspaceSpec",
    "RefusalDirectionSelectionSpec",
    "RefusalDirectionSpec",
    "RefusalScoreSpec",
    "ResidualizedProbeSpec",
    "ReportSpec",
    "InterchangePatch",
    "InterventionSite",
    "ProjectOutPatch",
    "RandomControlPatch",
    "ResidualInterventionSite",
    "ResidualPathPatch",
    "ResidualSite",
    "RuntimeSecret",
    "RuntimeSpec",
    "RoutingRecord",
    "Runner",
    "RunnerSpec",
    "SectionSelector",
    "ServiceFactory",
    "Source",
    "SpecValidationError",
    "StepFeatureRef",
    "StepLabelRef",
    "StepRef",
    "TensorStorage",
    "TextBaselineSpec",
    "SubspaceSpec",
    "SwapComponentsPatch",
    "SwapMeanPatch",
    "TransformBuilder",
    "TransformResult",
    "TransformSpec",
    "TransferProbeSpec",
    "TRUTHFULQA_DATASET_REPO",
    "TRUTHFULQA_GENERATION_CONFIG",
    "TruthfulnessAblationSubspaceSpec",
    "TruthfulnessDirectionSelectionSpec",
    "TruthfulnessDirectionSpec",
    "TruthfulnessScoreSpec",
    "TokenPooling",
    "TokenSelector",
    "ToyEngine",
    "TransferPolicy",
    "TransferPolicyError",
    "UrlJsonSource",
    "VLLMEngine",
    "WorkflowOrchestrator",
    "WorkflowPlan",
    "WorkflowResult",
    "WorkflowRunRecord",
    "WorkflowSpec",
    "WorkflowStep",
    "WorkflowStepContext",
    "WorkflowStepRecord",
    "PairDeltaSpec",
    "assistant_axis_model_config",
    "assistant_axis_prompt_dataset",
    "assistant_axis_trait_filename",
    "discover_assistant_axis_layer_and_capping",
    "emotion_contrast_dataset",
    "emotion_probe_story_dataset",
    "refusal_direction_processed_dataset",
    "refusal_direction_split_dataset",
    "truthfulqa_answer_contrast_dataset",
    "truthfulqa_generation_dataset",
]

# Runtime admission is metadata-only; optional model libraries remain lazy.
from pipelines_v2.runtime.admission import AdmissionCheck, WorkloadEnvelope, read_capability_matrix
from pipelines_v2.runtime.locks import RuntimeLock, load_runtime_lock
from pipelines_v2.runtime.readiness import AdmissionRecord

__all__ += ["AdmissionCheck", "AdmissionRecord", "WorkloadEnvelope", "RuntimeLock", "load_runtime_lock", "read_capability_matrix"]

# Optional SDXL adapter: model libraries are imported only inside execution.
from pipelines_v2.engine.diffusers.hooks import DenoisingHook
from pipelines_v2.engine.diffusers.session import ImageRequest
from pipelines_v2.engine.diffusers.specs import SDXLEngine, DiffusionGenerationSpec, DiffusionCaptureSpec, DiffusionInterventionSpec
from pipelines_v2.engine.diffusers.execution import diffusion_image
__all__ += ["DenoisingHook", "ImageRequest", "SDXLEngine", "DiffusionGenerationSpec", "DiffusionCaptureSpec", "DiffusionInterventionSpec", "diffusion_image"]

from pipelines_v2.operations.attribution import MAttrSpec, official_tool_lock
from pipelines_v2.runtime.external_tool import ExternalToolRuntimeSpec
__all__ += ["MAttrSpec", "official_tool_lock", "ExternalToolRuntimeSpec"]
__all__ += ['NativeTorchEngine','NativeSession','NativeSite','NativeSelection','ExpertLocalCoordinates','NativeInterventionSite','NativeCoordinatePatch']

from .operations.representation.sae import SAEDictionaryImportSpec, SAEEncodeSpec, SAEDecodeSpec, SAEFeatureSpliceSpec
from .representation.sae import SAEDictionary, splice as sae_feature_splice
__all__ += ['SAEDictionaryImportSpec', 'SAEEncodeSpec', 'SAEDecodeSpec', 'SAEFeatureSpliceSpec', 'SAEDictionary', 'sae_feature_splice']

from .operations.graphs import AttributionGraph, GraphNode, GraphEdge, RouteEvidence, NativeRouteSpec, RoutePair
__all__ += ['AttributionGraph', 'GraphNode', 'GraphEdge', 'RouteEvidence', 'NativeRouteSpec', 'RoutePair']

from pipelines_v2.engine.recognition import ViTRecognitionEngine, ViTSite, RecognitionSpec
__all__ += ['ViTRecognitionEngine', 'ViTSite', 'RecognitionSpec']

from pipelines_v2.operations.circuit_tracer import CircuitTraceSpec
from pipelines_v2.runtime.circuit_tracer import CircuitTracerRuntimeSpec
__all__ += ["CircuitTraceSpec", "CircuitTracerRuntimeSpec"]

from pipelines_v2.storage.publication import PublicationTarget, PublicationPlan, plan_publication, publish_artifact
__all__ += ['PublicationTarget', 'PublicationPlan', 'plan_publication', 'publish_artifact']

from .operations.representation.crosscoder import CrosscoderPairCaptureSpec, CrosscoderFitSpec, CrosscoderImportSpec, CrosscoderEncodeSpec, CrosscoderDecodeSpec, CrosscoderFeatureSpliceSpec
__all__ += ['CrosscoderPairCaptureSpec','CrosscoderFitSpec','CrosscoderImportSpec','CrosscoderEncodeSpec','CrosscoderDecodeSpec','CrosscoderFeatureSpliceSpec']
