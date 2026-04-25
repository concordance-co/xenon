"""Section-record helpers for structured activation projections."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from pipelines_v2.core.types import SpecValidationError


@dataclass(frozen=True, slots=True)
class SectionSelector:
    """Select repeated semantic slices from one captured example."""

    names: tuple[str, ...] = ()
    where_fields: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def all(cls) -> "SectionSelector":
        return cls()

    @classmethod
    def named(cls, *names: str) -> "SectionSelector":
        return cls(names=tuple(str(name) for name in names if str(name).strip()))

    @classmethod
    def where(cls, **criteria: Any) -> "SectionSelector":
        return cls(where_fields={str(key): value for key, value in criteria.items()})

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "where_fields": dict(self.where_fields),
        }

    def semantic_dict(self) -> dict[str, Any]:
        return self.to_dict()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "SectionSelector":
        if payload is None:
            return cls.all()
        return cls(
            names=tuple(str(name) for name in payload.get("names", ())),
            where_fields={str(key): value for key, value in dict(payload.get("where_fields", {})).items()},
        )


def coerce_section_records(
    raw: Any,
    *,
    token_sections: Mapping[str, Sequence[int]] | None = None,
) -> list[dict[str, Any]]:
    """Normalize stored section records or synthesize them from token sections."""

    if raw is None:
        return _records_from_token_sections(token_sections)
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes | bytearray):
        raise TypeError("section_records must be a sequence of mappings or a JSON array string")
    records: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise TypeError("section_records items must be mappings")
        normalized = _normalize_section_record(item, token_sections=token_sections)
        if normalized is not None:
            records.append(normalized)
    records.sort(key=_section_record_sort_key)
    return records


def select_section_records(
    records: Sequence[Mapping[str, Any]],
    selector: SectionSelector | None,
) -> list[dict[str, Any]]:
    """Return the ordered section records matching one selector."""

    resolved_selector = selector or SectionSelector.all()
    matched: list[dict[str, Any]] = []
    for raw_record in records:
        record = dict(raw_record)
        if _record_matches_selector(record, resolved_selector):
            matched.append(record)
    matched.sort(key=_section_record_sort_key)
    return matched


def _record_matches_selector(record: Mapping[str, Any], selector: SectionSelector) -> bool:
    names = tuple(selector.names)
    if names and str(record.get("name") or "") not in set(names):
        return False
    tags = record.get("tags")
    tag_mapping = dict(tags) if isinstance(tags, Mapping) else {}
    for key, expected in selector.where_fields.items():
        if key in record:
            actual = record.get(key)
        else:
            actual = tag_mapping.get(key)
        if actual != expected:
            return False
    return True


def _records_from_token_sections(
    token_sections: Mapping[str, Sequence[int]] | None,
) -> list[dict[str, Any]]:
    if not isinstance(token_sections, Mapping):
        return []
    records: list[dict[str, Any]] = []
    for name, positions in token_sections.items():
        if not isinstance(positions, Sequence) or isinstance(positions, str | bytes | bytearray):
            continue
        normalized_positions = [int(position) for position in positions]
        if not normalized_positions:
            continue
        records.append(
            {
                "name": str(name),
                "token_positions": normalized_positions,
            }
        )
    records.sort(key=_section_record_sort_key)
    return records


def _normalize_section_record(
    raw: Mapping[str, Any],
    *,
    token_sections: Mapping[str, Sequence[int]] | None,
) -> dict[str, Any] | None:
    name = str(raw.get("name") or raw.get("section_name") or "").strip()
    if not name:
        raise SpecValidationError("section_records entries must define a non-empty 'name'")
    if "token_positions" in raw:
        positions_raw = raw.get("token_positions")
    elif "positions" in raw:
        positions_raw = raw.get("positions")
    elif isinstance(token_sections, Mapping) and name in token_sections:
        positions_raw = token_sections[name]
    else:
        positions_raw = ()
    if not isinstance(positions_raw, Sequence) or isinstance(positions_raw, str | bytes | bytearray):
        raise TypeError(f"section_records[{name!r}] token positions must be a sequence of integers")
    token_positions = [int(position) for position in positions_raw]
    if not token_positions:
        return None

    normalized = {
        str(key): value
        for key, value in dict(raw).items()
        if key not in {"positions", "section_name"}
    }
    normalized["name"] = name
    normalized["token_positions"] = token_positions
    if normalized.get("index") is not None:
        normalized["index"] = int(normalized["index"])
    if "tags" in normalized and normalized["tags"] is not None:
        tags = normalized["tags"]
        if not isinstance(tags, Mapping):
            raise TypeError(f"section_records[{name!r}]['tags'] must be a mapping when present")
        normalized["tags"] = {str(key): value for key, value in tags.items()}
    return normalized


def _section_record_sort_key(record: Mapping[str, Any]) -> tuple[int, int, str]:
    index = record.get("index")
    if index is not None:
        return (0, int(index), str(record.get("name") or ""))
    positions = record.get("token_positions")
    if isinstance(positions, Sequence) and not isinstance(positions, str | bytes | bytearray) and positions:
        return (1, min(int(position) for position in positions), str(record.get("name") or ""))
    return (2, 0, str(record.get("name") or ""))
