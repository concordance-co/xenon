"""Editable data surface for the binary truth/correctness readout method.

The paper default is truthfulness. BYOD rows can be any binary correctness or
veracity substrate, including domain-specific claims or task answers.
"""

from __future__ import annotations

from pathlib import Path


METHOD_NAME = "binary_truth_or_correctness_readout"
SCHEMA_PATH = Path(__file__).parents[1] / "schemas" / "binary_truth_readout_method.schema.json"

REQUIRED_BYOD_FIELDS = (
    "example_id",
    "prompt",
    "answer",
    "label",
)

DEFAULT_SYNTH_RECIPE = {
    "positive_label": "true",
    "negative_label": "false",
    "rendered_text": "prompt + answer",
    "capture_section": "answer",
    "direction_formula": "mean(true/correct answer activations) - mean(false/incorrect answer activations)",
    "paper_defaults": {
        "dataset_shape": "TruthfulQA-style prompt plus answer variants",
        "labels": ["true", "false"],
    },
    "edit_here": {
        "label_mapping": {"TODO_positive": "true", "TODO_negative": "false"},
        "claim_or_question_seeds": ["TODO: replace with your domain items"],
        "paired_variants": "TODO: whether each claim_id has both label variants",
        "heldout_design": "TODO: fresh topics, fresh aliases, or same-item variants",
    },
}
