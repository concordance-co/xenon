"""Out-of-the-box mech-interp helpers and presets."""

from .assistant_axis import (
    ASSISTANT_AXIS_PROMPT_DATASET_REPO,
    ASSISTANT_AXIS_VECTOR_REPO,
    AssistantAxisPrecomputedCoordinateSpec,
    AssistantAxisScoreSpec,
    AssistantAxisVectorSpec,
    KNOWN_ASSISTANT_AXIS_MODELS,
    assistant_axis_model_config,
    assistant_axis_prompt_dataset,
    discover_assistant_axis_layer_and_capping,
)
from .emotions import (
    EMOTION_VECTOR_SPACE_KIND,
    EmotionDirectionSpec,
    EmotionGeometrySpec,
    EmotionPrecomputedVectorSpaceSpec,
    EmotionScoreSpec,
    EmotionVectorSpaceSpec,
)

__all__ = [
    "ASSISTANT_AXIS_PROMPT_DATASET_REPO",
    "ASSISTANT_AXIS_VECTOR_REPO",
    "AssistantAxisPrecomputedCoordinateSpec",
    "AssistantAxisScoreSpec",
    "AssistantAxisVectorSpec",
    "EMOTION_VECTOR_SPACE_KIND",
    "EmotionDirectionSpec",
    "EmotionGeometrySpec",
    "EmotionPrecomputedVectorSpaceSpec",
    "EmotionScoreSpec",
    "EmotionVectorSpaceSpec",
    "KNOWN_ASSISTANT_AXIS_MODELS",
    "assistant_axis_model_config",
    "assistant_axis_prompt_dataset",
    "discover_assistant_axis_layer_and_capping",
]
