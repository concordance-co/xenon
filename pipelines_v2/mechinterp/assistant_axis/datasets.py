"""Canonical dataset helpers for Assistant Axis vector prompts.

The library does not build this dataset from local Yora files at runtime. The
source prompts are uploaded once to Hugging Face and then referenced through
normal ``Dataset`` infrastructure so remote runs stay reproducible.
"""

from __future__ import annotations

from pipelines_v2.data.datasets import Dataset
from pipelines_v2.data.sources import HuggingFaceSource

ASSISTANT_AXIS_PROMPT_DATASET_REPO = "belmore/assistant-axis-vector-prompts"


def assistant_axis_prompt_dataset(
    *,
    repo_id: str = ASSISTANT_AXIS_PROMPT_DATASET_REPO,
    revision: str | None = None,
    split: str = "train",
    limit: int | None = None,
) -> Dataset:
    """Return the canonical deferred HF dataset for Assistant Axis prompt source rows.

    The dataset is intentionally compact: one row per role/trait source with
    nested ``instructions`` and ``questions`` columns. It is not pre-expanded
    into every instruction/question prompt pair. Consumers that want concrete
    prompt invocations should expand those nested columns in their own vector
    generation spec.
    """

    return Dataset.from_huggingface(
        source=HuggingFaceSource(path=repo_id, revision=revision),
        split=split,
        prompt_column="name",
        prompt_template="{source_type}:{name}",
        example_key_column="example_id",
        label_columns=(
            "source_type",
            "name",
            "is_default",
            "instruction_count",
            "question_count",
            "polarity_count",
        ),
        case_columns=("name", "source_type"),
        metadata_columns=("instructions", "questions", "eval_prompt", "question_source", "schema_version"),
        limit=limit,
        name="assistant_axis_prompt_sources",
    )


__all__ = [
    "ASSISTANT_AXIS_PROMPT_DATASET_REPO",
    "assistant_axis_prompt_dataset",
]
