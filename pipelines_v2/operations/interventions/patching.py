"""Compatibility façade for intervention patching specs."""

from .comparison import PatchComparisonSpec
from .generation import GenerationRunSpec, PatchedGenerationSpec
from .recipes import (
    ActivationPatchSpec,
    ScalePatch,
    AddDirectionPatch,
    InterchangePatch,
    PatchApplication,
    ProjectOutPatch,
    RandomControlPatch,
    ResidualPathPatch,
    SwapComponentsPatch,
    SwapMeanPatch,
)
from .sites import InterventionSite, ResidualInterventionSite

__all__ = [
    "ActivationPatchSpec",
    "ScalePatch",
    "AddDirectionPatch",
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
