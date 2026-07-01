"""Editable data surface for the Assistant Axis method.

The method is a default-vs-role contrast. The paper's examples are persona
roles, but BYOD roles can be support modes, policy modes, analyst styles, or
any other grouped non-default condition.
"""

from __future__ import annotations

from pathlib import Path


METHOD_NAME = "assistant_axis_default_vs_role"
SCHEMA_PATH = Path(__file__).parents[1] / "schemas" / "assistant_axis_method.schema.json"

REQUIRED_BYOD_FIELDS = (
    "example_id",
    "text",
    "assistant_response",
)

DERIVATION_FIELDS = (
    "axis_kind",
    "role",
)

DEFAULT_SYNTH_RECIPE = {
    "default_condition": "ordinary helpful assistant response",
    "role_condition": "role- or style-conditioned assistant response",
    "response_section": "assistant_response",
    "axis_formula": "mean(default_response_activations) - mean(per_role_role_playing_vectors)",
    "paper_default_roles": ["pirate", "villain"],
    "edit_here": {
        "roles": ["TODO: replace with your grouped conditions"],
        "source_prompts": ["TODO: replace with prompts/tasks to answer"],
        "examples_per_role": "TODO: choose count",
        "adherence_filter": "TODO: optional score threshold for role rows",
    },
}

PRECOMPUTED_SCORE_TARGETS = {
    "gemma_2_27b": {"model_id": "google/gemma-2-27b-it", "layer": 22},
    "qwen_3_32b": {"model_id": "Qwen/Qwen3-32B", "layer": 32},
    "llama_3_3_70b": {"model_id": "meta-llama/Llama-3.3-70B-Instruct", "layer": 40},
}
