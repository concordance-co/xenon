"""Token selection and pooling primitives shared across operation families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pipelines_v2.core.types import SpecValidationError


@dataclass(frozen=True, slots=True)
class TokenSlice:
    """One contiguous token span with half-open [start, stop) semantics."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        start = int(self.start)
        stop = int(self.stop)
        if start < 0 or stop <= start:
            raise SpecValidationError(f"Invalid TokenSlice(start={start}, stop={stop})")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)

    def positions(self) -> tuple[int, ...]:
        return tuple(range(int(self.start), int(self.stop)))


@dataclass(frozen=True, slots=True)
class TokenSliceSet:
    """Ordered set of contiguous token spans."""

    slices: tuple[TokenSlice, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.slices, key=lambda item: (int(item.start), int(item.stop))))
        for prev, current in zip(ordered, ordered[1:]):
            if int(current.start) <= int(prev.stop):
                raise SpecValidationError(
                    "TokenSliceSet slices must be strictly ordered and non-overlapping"
                )
        object.__setattr__(self, "slices", ordered)

    def positions(self) -> list[int]:
        positions: list[int] = []
        for token_slice in self.slices:
            positions.extend(token_slice.positions())
        return positions

    def token_count(self) -> int:
        return sum(int(token_slice.stop) - int(token_slice.start) for token_slice in self.slices)

    def is_single_contiguous(self) -> bool:
        return len(self.slices) == 1

    def single_span(self) -> tuple[int, int] | None:
        if len(self.slices) != 1:
            return None
        token_slice = self.slices[0]
        return (int(token_slice.start), int(token_slice.stop))

    @classmethod
    def from_positions(cls, positions: Sequence[int]) -> "TokenSliceSet":
        normalized = sorted({int(position) for position in positions})
        if not normalized:
            return cls(slices=())
        slices: list[TokenSlice] = []
        start = normalized[0]
        prev = start
        for position in normalized[1:]:
            if int(position) == int(prev) + 1:
                prev = int(position)
                continue
            slices.append(TokenSlice(start=int(start), stop=int(prev) + 1))
            start = int(position)
            prev = int(position)
        slices.append(TokenSlice(start=int(start), stop=int(prev) + 1))
        return cls(slices=tuple(slices))


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

    def resolve_slices(
        self,
        token_count: int,
        *,
        token_sections: Mapping[str, Sequence[int]] | None = None,
    ) -> TokenSliceSet:
        if token_count <= 0:
            return TokenSliceSet(slices=())
        if self.kind == "last":
            return TokenSliceSet(slices=(TokenSlice(start=token_count - 1, stop=token_count),))
        if self.kind == "full_sequence":
            return TokenSliceSet(slices=(TokenSlice(start=0, stop=token_count),))
        if self.kind == "slice":
            start = int(self.value["start"])
            stop = self.value.get("stop")
            selected = list(range(token_count))[start:stop]
            return TokenSliceSet.from_positions(selected)
        if self.kind == "section":
            section_name = str(self.value)
            if token_sections is None or section_name not in token_sections:
                raise SpecValidationError(f"Token selector section {section_name!r} is not available for this example")
            positions: list[int] = []
            seen: set[int] = set()
            for raw_position in token_sections[section_name]:
                position = int(raw_position)
                if position < 0 or position >= token_count:
                    raise SpecValidationError(
                        f"Token selector section {section_name!r} includes out-of-bounds position "
                        f"{position} for token_count={token_count}"
                    )
                if position in seen:
                    continue
                seen.add(position)
                positions.append(position)
            return TokenSliceSet.from_positions(positions)
        raise SpecValidationError(f"Unsupported token selector: {self.kind}")

    def resolve(
        self,
        token_count: int,
        *,
        token_sections: Mapping[str, Sequence[int]] | None = None,
    ) -> list[int]:
        return self.resolve_slices(
            token_count,
            token_sections=token_sections,
        ).positions()

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


__all__ = ["TokenPooling", "TokenSelector", "TokenSlice", "TokenSliceSet"]
