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
    prompt: Any = None,
) -> dict[str, Any]:
    """Merge example metadata with builder-derived prompt metadata.

    Builder output fills missing keys only. This lets datasets provide explicit
    ``token_sections`` or ``section_records`` while still allowing a capture
    spec to add best-effort defaults for examples that do not already define
    them.
    """
    resolved: dict[str, Any] = dict(metadata) if isinstance(metadata, Mapping) else {}
    if builder is None:
        return resolved
    built = builder.build(
        rendered_prompt,
        prompt=prompt,
        metadata=resolved,
    )
    for key, value in built.items():
        resolved.setdefault(str(key), value)
    return resolved


def build_chat_turn_metadata(
    rendered_prompt: str,
    *,
    prompt: Any = None,
    name_template: str = "{role}_turn_{index:03d}",
    include_section_records: bool = True,
    include_assistant_colon: bool = False,
    assistant_colon_name: str = "assistant_colon",
) -> dict[str, object]:
    """Build best-effort turn spans over rendered chat prompts.

    ``prompt`` is expected to be a chat-message sequence. The function searches
    each message's textual content inside ``rendered_prompt`` and emits
    character spans that tokenization later resolves into ``token_sections`` and
    optional structured ``section_records``. It is intentionally best-effort:
    if a message cannot be found exactly, that turn is omitted rather than
    inventing a span.
    """

    if not isinstance(prompt, Sequence) or isinstance(prompt, str | bytes | bytearray):
        payload: dict[str, object] = {}
        return _with_assistant_colon_metadata(
            payload,
            rendered_prompt=rendered_prompt,
            include_assistant_colon=include_assistant_colon,
            assistant_colon_name=assistant_colon_name,
        )

    token_sections: dict[str, dict[str, int]] = {}
    section_records: list[dict[str, object]] = []
    search_start = 0

    for index, message in enumerate(prompt):
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "unknown")
        content_text = _message_content_text(message.get("content"))
        if not content_text:
            continue
        start = rendered_prompt.find(content_text, search_start)
        if start < 0:
            start = rendered_prompt.find(content_text)
        if start < 0:
            continue
        end = start + len(content_text)
        search_start = end
        name = name_template.format(role=role, index=index)
        token_sections[name] = {"char_start": int(start), "char_end": int(end)}
        if include_section_records:
            section_records.append(
                {
                    "name": name,
                    "role": role,
                    "unit": "turn",
                    "index": int(index),
                    "char_start": int(start),
                    "char_end": int(end),
                }
            )

    payload: dict[str, object] = {"token_sections": token_sections}
    if include_section_records:
        payload["section_records"] = section_records
    return _with_assistant_colon_metadata(
        payload,
        rendered_prompt=rendered_prompt,
        include_assistant_colon=include_assistant_colon,
        assistant_colon_name=assistant_colon_name,
    )


def _with_assistant_colon_metadata(
    payload: dict[str, object],
    *,
    rendered_prompt: str,
    include_assistant_colon: bool,
    assistant_colon_name: str,
) -> dict[str, object]:
    if not include_assistant_colon:
        return payload
    colon_index = _assistant_colon_index(rendered_prompt)
    if colon_index < 0:
        return payload

    token_sections = dict(payload.get("token_sections", {})) if isinstance(payload.get("token_sections"), Mapping) else {}
    token_sections[str(assistant_colon_name)] = {"char_start": colon_index, "char_end": colon_index + 1}
    payload["token_sections"] = token_sections

    raw_records = payload.get("section_records")
    if isinstance(raw_records, Sequence) and not isinstance(raw_records, str | bytes | bytearray):
        records = list(raw_records)
        records.append(
            {
                "name": str(assistant_colon_name),
                "role": "assistant",
                "unit": "marker",
                "index": len(records),
                "char_start": colon_index,
                "char_end": colon_index + 1,
                "tags": {"marker": "assistant_colon"},
            }
        )
        payload["section_records"] = records
    return payload


def _assistant_colon_index(rendered_prompt: str) -> int:
    lowered = rendered_prompt.lower()
    start = max(lowered.rfind("assistant:"), lowered.rfind("assistant\n"))
    if start < 0:
        return -1
    colon = rendered_prompt.find(":", start)
    if colon >= 0:
        return colon
    newline = rendered_prompt.find("\n", start)
    return newline if newline >= 0 else -1


def token_sections_from_metadata(
    *,
    metadata: Mapping[str, Any] | None,
    offsets: list[tuple[int, int]] | None,
    require_sections: bool,
    allow_char_spans: bool,
) -> dict[str, list[int]]:
    """Resolve explicit ``token_sections`` metadata into token positions.

    Sections may be provided as token positions or character spans. Character
    spans require tokenizer offsets from the capture engine; engines that cannot
    provide offsets should pass ``allow_char_spans=False`` to fail early.
    """
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


def section_records_from_metadata(
    *,
    metadata: Mapping[str, Any] | None,
    offsets: list[tuple[int, int]] | None,
    token_sections: Mapping[str, Sequence[int]] | None,
    allow_char_spans: bool,
) -> list[dict[str, Any]]:
    """Resolve structured section records into prompt-level token positions.

    Section records are richer than ``token_sections`` because they preserve
    fields such as ``role``, ``unit``, ``index``, and ``tags``. Projection
    scoring uses these records to select and label repeated spans while still
    relying on token positions for the actual activation pooling.
    """

    raw_records = metadata.get("section_records") if isinstance(metadata, Mapping) else None
    if raw_records is None:
        return []
    if isinstance(raw_records, str):
        raw_records = json.loads(raw_records)
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, str | bytes | bytearray):
        raise TypeError("section_records metadata must be a sequence or JSON array string")

    resolved: list[dict[str, Any]] = []
    for item in raw_records:
        if not isinstance(item, Mapping):
            raise TypeError("section_records entries must be mappings")
        name = str(item.get("name") or item.get("section_name") or "").strip()
        if not name:
            raise TypeError("section_records entries must define 'name'")
        positions = _token_positions_for_section_record(
            section_name=name,
            raw_record=item,
            offsets=offsets,
            allow_char_spans=allow_char_spans,
            token_sections=token_sections,
        )
        if not positions:
            continue
        record = {
            str(key): value
            for key, value in dict(item).items()
            if key not in {"section_name", "positions"}
        }
        record["name"] = name
        record["token_positions"] = positions
        if record.get("index") is not None:
            record["index"] = int(record["index"])
        tags = record.get("tags")
        if tags is not None:
            if not isinstance(tags, Mapping):
                raise TypeError(f"section_records[{name!r}]['tags'] must be a mapping")
            record["tags"] = {str(key): value for key, value in tags.items()}
        resolved.append(record)
    return resolved


def rebase_section_records(
    *,
    section_records: Sequence[Mapping[str, Any]] | None,
    selected_positions: Sequence[int],
) -> list[dict[str, Any]]:
    """Translate prompt-level section records into feature-local coordinates."""

    if not isinstance(section_records, Sequence) or isinstance(section_records, str | bytes | bytearray):
        return []
    local_index_by_position = {
        int(position): local_index
        for local_index, position in enumerate(selected_positions)
    }
    rebased: list[dict[str, Any]] = []
    for raw_record in section_records:
        if not isinstance(raw_record, Mapping):
            continue
        positions = raw_record.get("token_positions")
        if not isinstance(positions, Sequence) or isinstance(positions, str | bytes | bytearray):
            continue
        local_positions = [
            local_index_by_position[int(position)]
            for position in positions
            if int(position) in local_index_by_position
        ]
        if not local_positions:
            continue
        record = dict(raw_record)
        record["token_positions"] = local_positions
        rebased.append(record)
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


def _token_positions_for_section_record(
    *,
    section_name: str,
    raw_record: Mapping[str, Any],
    offsets: list[tuple[int, int]] | None,
    allow_char_spans: bool,
    token_sections: Mapping[str, Sequence[int]] | None,
) -> list[int]:
    positions = raw_record.get("token_positions", raw_record.get("positions"))
    if positions is not None:
        if not isinstance(positions, Sequence) or isinstance(positions, str | bytes | bytearray):
            raise TypeError(f"section_records[{section_name!r}] token positions must be a sequence of integers")
        if not all(isinstance(position, int) for position in positions):
            raise TypeError(f"section_records[{section_name!r}] token positions must contain only integers")
        return [int(position) for position in positions]
    if raw_record.get("char_start", raw_record.get("start")) is not None:
        return _token_positions_for_section(
            section_name=section_name,
            raw_value={
                "char_start": raw_record.get("char_start", raw_record.get("start")),
                "char_end": raw_record.get("char_end", raw_record.get("end")),
            },
            offsets=offsets,
            allow_char_spans=allow_char_spans,
        )
    if isinstance(token_sections, Mapping) and section_name in token_sections:
        return [int(position) for position in token_sections[section_name]]
    return []


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, str | bytes | bytearray):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, Mapping):
                continue
            item_type = str(item.get("type") or "")
            if item_type == "text" and item.get("text") is not None:
                parts.append(str(item["text"]))
        return "".join(parts)
    return ""
