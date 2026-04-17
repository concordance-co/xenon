"""Shared helpers for intervention specs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pipelines_v2.core.types import SpecValidationError
from pipelines_v2.data.datasets import Dataset
from pipelines_v2.operations.common._shared import example_has_explicit_token_sections
from pipelines_v2.operations.common.tokens import TokenSelector


def requires_token_sections(tokens: TokenSelector | None) -> bool:
    return isinstance(tokens, TokenSelector) and tokens.kind == "section"


def dataset_provides_token_sections(dataset: Dataset) -> bool:
    if dataset.is_deferred:
        return False
    return all(example_has_explicit_token_sections(example) for example in dataset.examples)


def normalize_layer_map(value: Mapping[int, int] | None) -> dict[int, int]:
    return {
        int(write_layer): int(source_layer)
        for write_layer, source_layer in dict(value or {}).items()
    }


def normalize_component_map(value: Mapping[int, tuple[int, ...]] | None) -> dict[int, tuple[int, ...]]:
    return {
        int(layer): tuple(int(index) for index in indices)
        for layer, indices in dict(value or {}).items()
    }


def token_selector_from_payload(
    payload: Any,
    *,
    default: TokenSelector | None = None,
) -> TokenSelector | None:
    if payload is None:
        return default
    if isinstance(payload, TokenSelector):
        return payload
    if isinstance(payload, Mapping):
        return TokenSelector.from_dict(payload)
    raise SpecValidationError(f"Expected token selector mapping, got {type(payload).__name__}")
