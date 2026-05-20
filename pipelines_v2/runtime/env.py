"""Shared runtime-environment helpers for runner implementations."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager


_MISSING = object()


def merged_runtime_env(*layers: Mapping[str, str] | None) -> dict[str, str]:
    """Merge runtime env layers, with later layers overriding earlier ones."""

    merged: dict[str, str] = {}
    for layer in layers:
        if not layer:
            continue
        for key, value in layer.items():
            env_key = str(key).strip()
            if not env_key:
                continue
            merged[env_key] = str(value)
    return merged


@contextmanager
def applied_runtime_env(env: Mapping[str, str] | None) -> Iterator[None]:
    """Temporarily overlay environment variables for a local execution."""

    if not env:
        yield
        return

    previous: dict[str, object] = {}
    for key, value in env.items():
        previous[key] = os.environ.get(key, _MISSING)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, prior_value in previous.items():
            if prior_value is _MISSING:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(prior_value)
