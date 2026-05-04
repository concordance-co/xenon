"""Dataset helpers for TruthfulQA / ITI-style workflows."""

from __future__ import annotations

from pipelines_v2.data.datasets import Dataset
from pipelines_v2.data.sources import HuggingFaceListContrastSource, HuggingFaceSource

TRUTHFULQA_DATASET_REPO = "truthful_qa"
TRUTHFULQA_GENERATION_CONFIG = "generation"


def truthfulqa_generation_dataset(
    *,
    revision: str | None = None,
    split: str = "validation",
    limit: int | None = None,
    name: str = "truthfulqa_generation",
) -> Dataset:
    """Return TruthfulQA generation rows as a deferred question dataset."""

    return Dataset.from_huggingface(
        source=HuggingFaceSource(
            path=TRUTHFULQA_DATASET_REPO,
            name=TRUTHFULQA_GENERATION_CONFIG,
            revision=revision,
        ),
        split=split,
        prompt_column="question",
        example_key_column="example_id",
        index_column="example_id",
        index_prefix="truthfulqa",
        label_columns=("category",),
        case_columns=("category",),
        metadata_columns=("type", "best_answer", "correct_answers", "incorrect_answers", "source"),
        limit=limit,
        name=name,
    )


def truthfulqa_answer_contrast_dataset(
    *,
    revision: str | None = None,
    split: str = "validation",
    prompt_template: object | None = "Question: {question}\nAnswer: {answer}",
    limit: int | None = None,
    name: str = "truthfulqa_answer_contrast",
) -> Dataset:
    """Return answer-level truthful/untruthful contrasts from TruthfulQA.

    Each correct answer becomes a ``truthful`` row and each incorrect answer
    becomes an ``untruthful`` row. The prompt template can be replaced to match
    a product's own chat or completion format while preserving the same labels.
    """

    return Dataset.from_source(
        source=HuggingFaceListContrastSource(
            path=TRUTHFULQA_DATASET_REPO,
            name=TRUTHFULQA_GENERATION_CONFIG,
            revision=revision,
        ),
        defer=True,
        split=split,
        prompt_column="question",
        prompt_template=prompt_template,
        example_key_column="example_id",
        label_name="truthfulness",
        positive_column="correct_answers",
        negative_column="incorrect_answers",
        positive_label="truthful",
        negative_label="untruthful",
        answer_column="answer",
        metadata_columns=("category", "type", "best_answer", "source"),
        limit=limit,
        name=name,
    )


__all__ = [
    "TRUTHFULQA_DATASET_REPO",
    "TRUTHFULQA_GENERATION_CONFIG",
    "truthfulqa_answer_contrast_dataset",
    "truthfulqa_generation_dataset",
]
