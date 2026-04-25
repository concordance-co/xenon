"""Shared helpers for built-in mech-interp specs."""

from __future__ import annotations

import os
from collections.abc import Sequence

from pipelines_v2.core.types import RuntimeSecret


def hf_token_from_env(token_env_var: str | None) -> str | None:
    """Resolve an optional Hugging Face token env var."""

    if token_env_var is None:
        return None
    token = os.environ.get(token_env_var)
    if not token:
        raise RuntimeError(f"Missing required environment variable: {token_env_var}")
    return token


def merge_runtime_secrets(*values: Sequence[RuntimeSecret]) -> tuple[RuntimeSecret, ...]:
    """Merge runtime-secret sequences by env var while preserving first value."""

    merged: dict[str, RuntimeSecret] = {}
    for value in values:
        for secret in value:
            merged.setdefault(secret.env_var, secret)
    return tuple(merged.values())


__all__ = ["hf_token_from_env", "merge_runtime_secrets"]
