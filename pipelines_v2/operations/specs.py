"""Compatibility import surface for operation specs and shared primitives."""

from __future__ import annotations

from pipelines_v2.operations.capture import CaptureSpec, GenerationSpec, MoERoutingSite, ResidualSite, RoutingRecord
from pipelines_v2.operations.capture.specs import capture_site_from_dict
from pipelines_v2.operations.capture.sites import CaptureSite
from pipelines_v2.operations.common._shared import (
    analysis_runtime_spec as _analysis_runtime_spec,
    callable_import_ref as _callable_import_ref,
    contains_section_token_selector as _contains_section_token_selector,
    load_importable_function as _load_importable_function,
    merge_string_tuples as _merge_string_tuples,
    runtime_secrets_from_refs as _runtime_secrets_from_refs,
    spec_uses_section_token_selector,
    spec_value_from_dict as _spec_value_from_dict,
)
from pipelines_v2.operations.common.builders import PromptMetadataBuilder, TransformBuilder, TransformResult
from pipelines_v2.operations.common.schemas import TensorStorage
from pipelines_v2.operations.common.tokens import TokenPooling, TokenSelector
from pipelines_v2.operations.derive import LabelFieldsSpec, LabelMapSpec, PairDeltaSpec, TransformSpec
from pipelines_v2.operations.interventions import (
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
from pipelines_v2.operations.projections import CoordinateImportSpec, ProjectionCalibrationSpec, ProjectionSpec, SectionSelector
from pipelines_v2.operations.readouts import ProbeSpec, ResidualizedProbeSpec, TextBaselineSpec, TransferProbeSpec
from pipelines_v2.operations.reports import ReportSpec
from pipelines_v2.operations.representation import BasisSpec, CentroidSpec, DirectionSpec, GeometrySpec, SubspaceSpec

_capture_site_from_dict = capture_site_from_dict

__all__ = [
    "ActivationPatchSpec",
    "ActivationBankSpec",
    "AddDirectionPatch",
    "BasisSpec",
    "CaptureSite",
    "CaptureSpec",
    "CentroidSpec",
    "CoordinateImportSpec",
    "DirectionSpec",
    "ExplicitPathEdge",
    "ExplicitPathMaskSpec",
    "GenerationRunSpec",
    "GeometrySpec",
    "GenerationSpec",
    "InterchangePatch",
    "InterventionSite",
    "PatchApplication",
    "LabelFieldsSpec",
    "LabelMapSpec",
    "MoERoutingSite",
    "PatchComparisonSpec",
    "PairDeltaSpec",
    "PatchedGenerationSpec",
    "PromptMetadataBuilder",
    "ProbeSpec",
    "ProjectionCalibrationSpec",
    "ProjectionSpec",
    "ResidualizedProbeSpec",
    "ReportSpec",
    "ProjectOutPatch",
    "RandomControlPatch",
    "ResidualInterventionSite",
    "ResidualPathPatch",
    "ResidualSite",
    "RoutingRecord",
    "SectionSelector",
    "SwapComponentsPatch",
    "SwapMeanPatch",
    "TensorStorage",
    "TextBaselineSpec",
    "SubspaceSpec",
    "TokenPooling",
    "TokenSelector",
    "TransferProbeSpec",
    "TransformBuilder",
    "TransformResult",
    "TransformSpec",
    "spec_uses_section_token_selector",
]
