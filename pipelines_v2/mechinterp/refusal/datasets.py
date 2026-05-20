"""Dataset helpers for refusal-direction workflows."""

from __future__ import annotations

from pipelines_v2.data.datasets import Dataset
from pipelines_v2.data.sources import UrlJsonSource

REFUSAL_DIRECTION_REPO_RAW_BASE = "https://raw.githubusercontent.com/andyrdt/refusal_direction"
REFUSAL_DIRECTION_DATASETS = (
    "advbench",
    "tdc2023",
    "maliciousinstruct",
    "harmbench_val",
    "harmbench_test",
    "jailbreakbench",
    "strongreject",
    "alpaca",
)


def refusal_direction_split_dataset(
    *,
    revision: str = "main",
    split: str = "train",
    include_harmful: bool = True,
    include_harmless: bool = True,
    limit: int | None = None,
    prompt_template: object | None = None,
) -> Dataset:
    """Return the paper repo's harmful/harmless split JSON as a deferred dataset."""

    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of {'train', 'val', 'test'}")
    harmtypes = [
        harmtype
        for harmtype, enabled in (("harmful", include_harmful), ("harmless", include_harmless))
        if enabled
    ]
    if not harmtypes:
        raise ValueError("At least one of include_harmful/include_harmless must be true")
    return _refusal_url_dataset(
        [
            {
                "source_name": f"{harmtype}_{split}",
                "url": _repo_url(revision, f"dataset/splits/{harmtype}_{split}.json"),
                "labels": {
                    "harmtype": harmtype,
                    "split": split,
                    "source_dataset": "paper_split",
                },
            }
            for harmtype in harmtypes
        ],
        name=f"refusal_direction_{split}",
        limit=limit,
        prompt_template=prompt_template,
    )


def refusal_direction_processed_dataset(
    dataset_name: str,
    *,
    revision: str = "main",
    harmtype: str | None = None,
    limit: int | None = None,
    prompt_template: object | None = None,
) -> Dataset:
    """Return one processed source dataset from the refusal-direction repo."""

    normalized = str(dataset_name).strip().lower()
    if normalized not in REFUSAL_DIRECTION_DATASETS:
        raise ValueError(f"dataset_name must be one of {REFUSAL_DIRECTION_DATASETS}")
    labels = {"source_dataset": normalized}
    if harmtype is not None:
        labels["harmtype"] = str(harmtype)
    return _refusal_url_dataset(
        [
            {
                "source_name": normalized,
                "url": _repo_url(revision, f"dataset/processed/{normalized}.json"),
                "labels": labels,
            }
        ],
        name=f"refusal_direction_{normalized}",
        limit=limit,
        prompt_template=prompt_template,
    )


def _refusal_url_dataset(
    files: list[dict[str, object]],
    *,
    name: str,
    limit: int | None,
    prompt_template: object | None,
) -> Dataset:
    return Dataset.from_source(
        source=UrlJsonSource(files=tuple(files)),
        defer=True,
        prompt_column="instruction",
        prompt_template=prompt_template,
        example_key_column="example_id",
        label_columns=("harmtype", "split", "source_dataset", "category"),
        case_columns=("source_dataset",),
        metadata_columns=("source_name", "source_url"),
        limit_per_file=limit,
        name=name,
    )


def _repo_url(revision: str, path: str) -> str:
    return f"{REFUSAL_DIRECTION_REPO_RAW_BASE}/{revision}/{path}"


__all__ = [
    "REFUSAL_DIRECTION_DATASETS",
    "REFUSAL_DIRECTION_REPO_RAW_BASE",
    "refusal_direction_processed_dataset",
    "refusal_direction_split_dataset",
]
