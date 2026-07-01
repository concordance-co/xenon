"""Editable data surface for the concept-vector-space method.

The paper instantiates this with emotion-labeled stories plus neutral
dialogues. BYOD concepts do not need to be emotions: any labeled concept set
can use the same mean-vector and optional neutral-PC projection recipe.
"""

from __future__ import annotations

from pathlib import Path


METHOD_NAME = "concept_vector_space_with_neutral_projection"
SCHEMA_PATH = Path(__file__).parents[1] / "schemas" / "concept_vector_space_method.schema.json"

REQUIRED_BYOD_FIELDS = (
    "example_id",
    "text",
    "row_role",
)

CONCEPT_FIELDS = (
    "concept",
    "split",
)

DEFAULT_SYNTH_RECIPE = {
    "concept_examples": {
        "row_role": "concept",
        "split": "train",
        "concept": "TODO: label such as happy, recession, churn_risk, refund_intent",
    },
    "heldout_examples": {
        "row_role": "heldout",
        "split": "heldout",
        "concept": "TODO: same label space as train",
    },
    "neutral_examples": {
        "row_role": "neutral",
        "split": "neutral",
        "purpose": "background PCs to project out",
    },
    "vector_formula": "mean(concept examples) - mean(concept means)",
    "neutral_projection": "project out top PCs explaining the configured variance threshold",
    "paper_defaults": {
        "concept_domain": "emotions",
        "neutral_domain": "neutral Human/Assistant dialogues",
    },
    "edit_here": {
        "concepts": ["TODO: replace with your concept labels"],
        "topics": ["TODO: replace with your seed topics/items"],
        "examples_per_concept_topic": "TODO: choose count",
        "neutral_topics": ["TODO: replace with background topics/items"],
    },
}
