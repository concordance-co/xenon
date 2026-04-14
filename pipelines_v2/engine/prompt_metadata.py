"""Helpers for explicit prompt-derived metadata used during capture."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pipelines_v2.operations.specs import PromptMetadataBuilder


def resolve_prompt_metadata(
    *,
    metadata: Any,
    rendered_prompt: str,
    builder: PromptMetadataBuilder | None,
) -> dict[str, Any]:
    """Merge example metadata with builder-derived prompt metadata."""
    resolved: dict[str, Any] = dict(metadata) if isinstance(metadata, Mapping) else {}
    if builder is None:
        return resolved
    built = builder.build(rendered_prompt)
    for key, value in built.items():
        resolved.setdefault(str(key), value)
    return resolved


def token_sections_from_metadata(
    *,
    metadata: Mapping[str, Any] | None,
    offsets: list[tuple[int, int]] | None,
    require_sections: bool,
    allow_char_spans: bool,
) -> dict[str, list[int]]:
    """Resolve explicit ``token_sections`` metadata into token positions."""
    raw_sections = metadata.get("token_sections") if isinstance(metadata, Mapping) else None
    if raw_sections is None:
        if require_sections:
            raise RuntimeError(
                "TokenSelector.section(...) requires explicit token_sections metadata. "
                "Provide prompt_metadata_builder=... on the capture spec or metadata['token_sections'] on each example."
            )
        return {}
    if isinstance(raw_sections, str):
        raw_sections = json.loads(raw_sections)
    if not isinstance(raw_sections, Mapping):
        raise TypeError("token_sections metadata must be a mapping or JSON object string")

    resolved: dict[str, list[int]] = {}
    for name, raw_value in raw_sections.items():
        positions = _token_positions_for_section(
            section_name=str(name),
            raw_value=raw_value,
            offsets=offsets,
            allow_char_spans=allow_char_spans,
        )
        if positions:
            resolved[str(name)] = positions
    if require_sections and not resolved:
        raise RuntimeError("TokenSelector.section(...) was requested, but token_sections resolved to no token positions")
    return resolved


def rebase_token_sections(
    *,
    token_sections: Mapping[str, Sequence[int]] | None,
    selected_positions: Sequence[int],
) -> dict[str, list[int]]:
    """Translate prompt-level token sections into feature-local coordinates."""

    if not isinstance(token_sections, Mapping):
        return {}
    local_index_by_position = {
        int(position): local_index
        for local_index, position in enumerate(selected_positions)
    }
    rebased: dict[str, list[int]] = {}
    for name, positions in token_sections.items():
        if not isinstance(positions, Sequence) or isinstance(positions, str | bytes | bytearray):
            continue
        local_positions = [
            local_index_by_position[int(position)]
            for position in positions
            if int(position) in local_index_by_position
        ]
        if local_positions:
            rebased[str(name)] = local_positions
    return rebased


def _token_positions_for_section(
    *,
    section_name: str,
    raw_value: Any,
    offsets: list[tuple[int, int]] | None,
    allow_char_spans: bool,
) -> list[int]:
    if raw_value is None:
        return []
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, str | bytes | bytearray):
        if not all(isinstance(position, int) for position in raw_value):
            raise TypeError(
                f"token_sections[{section_name!r}] sequences must contain only integer token positions"
            )
        return [int(position) for position in raw_value]
    if not isinstance(raw_value, Mapping):
        raise TypeError(
            f"token_sections[{section_name!r}] must be a list of token positions or a span mapping"
        )

    if "token_positions" in raw_value:
        payload = raw_value["token_positions"]
        if not isinstance(payload, Sequence) or isinstance(payload, str | bytes | bytearray):
            raise TypeError(f"token_sections[{section_name!r}]['token_positions'] must be a sequence of integers")
        if not all(isinstance(position, int) for position in payload):
            raise TypeError(f"token_sections[{section_name!r}]['token_positions'] must contain only integers")
        return [int(position) for position in payload]

    start = raw_value.get("char_start", raw_value.get("start"))
    end = raw_value.get("char_end", raw_value.get("end"))
    if start is None or end is None:
        raise TypeError(
            f"token_sections[{section_name!r}] span mappings must define char_start/char_end or start/end"
        )
    if not allow_char_spans:
        raise RuntimeError(
            f"token_sections[{section_name!r}] uses character spans, but this engine requires explicit token positions"
        )
    if offsets is None:
        raise RuntimeError(f"token_sections[{section_name!r}] uses character spans, but token offsets are unavailable")
    char_start = int(start)
    char_end = int(end)
    positions = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_end > char_start and token_start < char_end
    ]
    if not positions:
        raise RuntimeError(f"token_sections[{section_name!r}] span did not map to any token positions")
    return positions
