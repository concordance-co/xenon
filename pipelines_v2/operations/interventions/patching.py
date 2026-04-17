"""Compatibility façade for intervention patching specs."""

from .comparison import PatchComparisonSpec
from .generation import GenerationRunSpec, PatchedGenerationSpec
from .recipes import (
    ActivationPatchSpec,
    AddDirectionPatch,
    InterchangePatch,
    ProjectOutPatch,
    RandomControlPatch,
    ResidualPathPatch,
    SwapComponentsPatch,
    SwapMeanPatch,
)
from .sites import InterventionSite, ResidualInterventionSite

__all__ = [
    "ActivationPatchSpec",
    "AddDirectionPatch",
    "GenerationRunSpec",
    "InterchangePatch",
    "InterventionSite",
    "PatchComparisonSpec",
    "PatchedGenerationSpec",
    "ProjectOutPatch",
    "RandomControlPatch",
    "ResidualInterventionSite",
    "ResidualPathPatch",
    "SwapComponentsPatch",
    "SwapMeanPatch",
]
