"""Intervention specs."""

from .evaluation import PatchComparisonSpec
from .patching import (
    ActivationPatchSpec,
    ScalePatch,
    AddDirectionPatch,
    GenerationRunSpec,
    InterchangePatch,
    InterventionSite,
    PatchApplication,
    PatchedGenerationSpec,
    ProjectOutPatch,
    RandomControlPatch,
    ResidualInterventionSite,
    ResidualPathPatch,
    SwapComponentsPatch,
    SwapMeanPatch,
)
from .sources import ActivationBankSpec, ExplicitPathEdge, ExplicitPathMaskSpec

__all__ = [
    "ActivationPatchSpec",
    "ScalePatch",
    "ActivationBankSpec",
    "AddDirectionPatch",
    "ExplicitPathEdge",
    "ExplicitPathMaskSpec",
    "GenerationRunSpec",
    "InterchangePatch",
    "InterventionSite",
    "PatchApplication",
    "PatchComparisonSpec",
    "PatchedGenerationSpec",
    "ProjectOutPatch",
    "RandomControlPatch",
    "ResidualInterventionSite",
    "ResidualPathPatch",
    "SwapComponentsPatch",
    "SwapMeanPatch",
]

from .native import NativeInterventionSite, NativeCoordinatePatch
from .sites import register_intervention_site
from .recipes import PATCH_LOADERS
from pipelines_v2.core.registry import register_kind
register_intervention_site(NativeInterventionSite.kind, NativeInterventionSite.from_dict)
register_kind(PATCH_LOADERS, NativeCoordinatePatch.kind, NativeCoordinatePatch.from_dict, label='Activation patch')
__all__ += ['NativeInterventionSite','NativeCoordinatePatch']
