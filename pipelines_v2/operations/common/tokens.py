"""Token selection and pooling primitives shared across operation families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pipelines_v2.core.types import SpecValidationError


@dataclass(frozen=True, slots=True)
class TokenSelector:
    """Select token positions from a captured token axis."""

    kind: str
    value: Any = None

    @classmethod
    def last(cls) -> "TokenSelector":
        return cls(kind="last")

    @classmethod
    def full_sequence(cls) -> "TokenSelector":
        return cls(kind="full_sequence")

    @classmethod
    def slice(cls, start: int, stop: int | None = None) -> "TokenSelector":
        return cls(kind="slice", value={"start": start, "stop": stop})

    @classmethod
    def section(cls, name: str) -> "TokenSelector":
        return cls(kind="section", value=name)

    def resolve(
        self,
        token_count: int,
        *,
        token_sections: Mapping[str, Sequence[int]] | None = None,
    ) -> list[int]:
        if token_count <= 0:
            return []
        if self.kind == "last":
            return [token_count - 1]
        if self.kind == "full_sequence":
            return list(range(token_count))
        if self.kind == "slice":
            start = int(self.value["start"])
            stop = self.value.get("stop")
            return list(range(token_count))[start:stop]
        if self.kind == "section":
            section_name = str(self.value)
            if token_sections is None or section_name not in token_sections:
                raise SpecValidationError(f"Token selector section {section_name!r} is not available for this example")
            return [int(position) for position in token_sections[section_name]]
        raise SpecValidationError(f"Unsupported token selector: {self.kind}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TokenSelector":
        return cls(kind=str(payload["kind"]), value=payload.get("value"))


@dataclass(frozen=True, slots=True)
class TokenPooling:
    """Reduce selected token vectors to one vector per example."""

    kind: str

    @classmethod
    def mean(cls) -> "TokenPooling":
        return cls(kind="mean")

    @classmethod
    def last(cls) -> "TokenPooling":
        return cls(kind="last")

    @classmethod
    def first(cls) -> "TokenPooling":
        return cls(kind="first")

    def from_count(self, token_count: int) -> list[int]:
        if token_count <= 0:
            return []
        if self.kind == "mean":
            return list(range(token_count))
        if self.kind == "last":
            return [token_count - 1]
        if self.kind == "first":
            return [0]
        raise SpecValidationError(f"Unsupported token pooling mode: {self.kind}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TokenPooling":
        return cls(kind=str(payload["kind"]))
