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
from pipelines_v2.operations.interventions import ActivationPatchControl, ActivationPatchSpec, ResidualInterventionSite
from pipelines_v2.operations.readouts import ProbeSpec, ResidualizedProbeSpec, TextBaselineSpec, TransferProbeSpec
from pipelines_v2.operations.reports import ReportSpec
from pipelines_v2.operations.representation import BasisSpec, DirectionSpec, GeometrySpec

_capture_site_from_dict = capture_site_from_dict

__all__ = [
    "ActivationPatchSpec",
    "ActivationPatchControl",
    "BasisSpec",
    "CaptureSite",
    "CaptureSpec",
    "DirectionSpec",
    "GeometrySpec",
    "GenerationSpec",
    "LabelFieldsSpec",
    "LabelMapSpec",
    "MoERoutingSite",
    "PairDeltaSpec",
    "PromptMetadataBuilder",
    "ProbeSpec",
    "ResidualizedProbeSpec",
    "ReportSpec",
    "ResidualInterventionSite",
    "ResidualSite",
    "RoutingRecord",
    "TensorStorage",
    "TextBaselineSpec",
    "TokenPooling",
    "TokenSelector",
    "TransferProbeSpec",
    "TransformBuilder",
    "TransformResult",
    "TransformSpec",
    "spec_uses_section_token_selector",
]
