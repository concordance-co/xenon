"""Small helpers for kind-keyed registries."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

T = TypeVar("T")


def load_from_kind_registry(
    payload: dict[str, Any],
    loaders: Mapping[str, Callable[[dict[str, Any]], T]],
    *,
    missing_message: str,
    unknown_message: str,
) -> T:
    """Load a payload through a registry keyed by its ``kind`` field."""

    kind = str(payload.get("kind") or "").strip()
    if not kind:
        raise ValueError(missing_message)
    try:
        loader = loaders[kind]
    except KeyError as exc:
        raise ValueError(unknown_message.format(kind=kind)) from exc
    return loader(payload)


def register_kind(registry: dict[str, T], kind: str, value: T, *, label: str) -> None:
    """Register trusted code explicitly; serialized payloads never supply loaders."""
    if not isinstance(kind, str) or not kind.strip() or kind != kind.strip():
        raise ValueError(f"{label} kind must be a non-empty, whitespace-free name")
    if kind in registry:
        raise ValueError(f"{label} kind already registered: {kind!r}")
    registry[kind] = value


__all__ = ["load_from_kind_registry", "register_kind"]
