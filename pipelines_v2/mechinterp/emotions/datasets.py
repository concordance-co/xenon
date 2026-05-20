"""Dataset helpers for emotion-vector workflows."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from pipelines_v2.data.datasets import Dataset
from pipelines_v2.data.sources import HuggingFaceSource

EMOTION_PROBES_DATASET_REPO = "ryancodrai/emotion-probes"


def emotion_contrast_dataset(
    records: Iterable[Mapping[str, Any]],
    *,
    prompt_column: str = "text",
    emotion_column: str = "emotion",
    example_key_column: str = "example_id",
    topic_column: str | None = "topic",
    prompt_template: object | None = None,
    metadata_columns: Sequence[str] = (),
    name: str = "emotion_contrast",
) -> Dataset:
    """Build a materialized labeled emotion dataset from user records.

    This is the preferred helper when a user has their own agent prompts,
    stories, transcripts, or sections and wants to recompute emotion spaces
    with the same Xenon workflow shape.
    """

    case_columns = (topic_column,) if topic_column is not None else ()
    return Dataset.from_records(
        records,
        prompt_column=prompt_column,
        prompt_template=prompt_template,
        example_key_column=example_key_column,
        label_columns=(emotion_column,),
        case_columns=case_columns,
        metadata_columns=tuple(dict.fromkeys([*(case_columns or ()), *metadata_columns])),
        name=name,
    )


def emotion_probe_story_dataset(
    *,
    revision: str | None = None,
    split: str = "train",
    prompt_column: str = "text",
    emotion_column: str = "real_emotion",
    limit: int | None = None,
    name: str = "emotion_probe_stories",
) -> Dataset:
    """Return a deferred HF mirror of the paper-style generated emotion probes.

    The upstream paper documents the generation recipe; this helper points at a
    lightweight public mirror with story/dialogue rows and labels suitable for
    smoke recomputation and product experiments.
    """

    return Dataset.from_huggingface(
        source=HuggingFaceSource(
            path=EMOTION_PROBES_DATASET_REPO,
            revision=revision,
        ),
        split=split,
        prompt_column=prompt_column,
        example_key_column="example_id",
        index_column="example_id",
        index_prefix="emotion_probe",
        label_columns=(emotion_column,),
        case_columns=("topic",),
        case_key_column="topic",
        metadata_columns=("displayed_emotion", "topic", "name_a", "name_b"),
        limit=limit,
        name=name,
    )


__all__ = [
    "EMOTION_PROBES_DATASET_REPO",
    "emotion_contrast_dataset",
    "emotion_probe_story_dataset",
]
