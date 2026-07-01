"""Editable data surface for the contrast-direction method.

The paper default is harmful-vs-harmless/refusal behavior. BYOD labels can be
any two-pole behavioral contrast as long as the positive and negative poles are
defined before capture.
"""

from __future__ import annotations

from pathlib import Path


METHOD_NAME = "two_pole_contrast_direction"
SCHEMA_PATH = Path(__file__).parents[1] / "schemas" / "contrast_direction_method.schema.json"

REQUIRED_BYOD_FIELDS = (
    "example_id",
    "text",
    "label",
)

DEFAULT_SYNTH_RECIPE = {
    "positive_label": "TODO: label for add direction pole",
    "negative_label": "TODO: label for subtract direction pole",
    "capture_section": "TODO: instruction, response, or full trace",
    "direction_formula": "mean(positive_label activations) - mean(negative_label activations)",
    "paper_defaults": {
        "positive_label": "refusal",
        "negative_label": "non_refusal",
        "capture_section": "instruction",
    },
    "edit_here": {
        "contrast_name": "TODO: e.g. cautious_vs_direct, escalation_vs_resolution",
        "seed_items": ["TODO: replace with data-generation seeds"],
        "label_balance": "TODO: target rows per pole",
        "heldout_design": "TODO: same-topic hard negatives or fresh topics",
    },
}
