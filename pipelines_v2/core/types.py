"""Shared primitives for the v2 pipelines API."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar


class EngineCapability(str, Enum):
    """Backend features exposed to operation planning."""

    GENERATION = "generation"
    LOGPROBS = "logprobs"
    RESIDUAL_CAPTURE = "residual_capture"
    ATTENTION_CAPTURE = "attention_capture"
    KV_CACHE_CAPTURE = "kv_cache_capture"
    MOE_ROUTING_CAPTURE = "moe_routing_capture"
    ACTIVATION_PATCHING = "activation_patching"
    REQUEST_SCOPED_INTERVENTIONS = "request_scoped_interventions"
    STRUCTURED_OUTPUT = "structured_output"


class PipelinesV2Error(Exception):
    """Base exception for the v2 API."""


class CapabilityError(PipelinesV2Error):
    """Raised when an engine cannot satisfy an operation spec."""


class SpecValidationError(PipelinesV2Error, ValueError):
    """Raised when a user-facing spec is internally inconsistent."""


class TransferPolicyError(PipelinesV2Error):
    """Raised when a remote transfer violates the configured transfer policy."""


@dataclass(frozen=True, slots=True)
class RuntimeSecret:
    env_var: str

    def __post_init__(self) -> None:
        if not str(self.env_var).strip():
            raise ValueError("RuntimeSecret env_var cannot be empty")
        object.__setattr__(self, "env_var", str(self.env_var))

    def to_dict(self) -> dict[str, Any]:
        return {"env_var": self.env_var}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeSecret":
        return cls(env_var=str(payload["env_var"]))


@dataclass(frozen=True, slots=True)
class TransferPolicy:
    allow_large_transfer: bool = False
    max_download_bytes: int = 64 * 1024 * 1024

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_large_transfer": self.allow_large_transfer,
            "max_download_bytes": self.max_download_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "TransferPolicy":
        if payload is None:
            return cls()
        return cls(
            allow_large_transfer=bool(payload.get("allow_large_transfer", False)),
            max_download_bytes=int(payload.get("max_download_bytes", 64 * 1024 * 1024)),
        )


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(data: Any) -> str:
    return hashlib.sha256(stable_json(data).encode("utf-8")).hexdigest()


def to_primitive(value: Any) -> Any:
    """Convert public API objects to stable JSON-like values."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): to_primitive(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [to_primitive(v) for v in value]
    if dataclasses.is_dataclass(value):
        data: dict[str, Any] = {}
        for field in dataclasses.fields(value):
            data[field.name] = to_primitive(getattr(value, field.name))
        kind = getattr(value, "kind", None)
        if isinstance(kind, str):
            data.setdefault("kind", kind)
        return data
    return value


def to_semantic_primitive(value: Any) -> Any:
    """Convert public API objects into stable semantic JSON-like values."""

    semantic_dict = getattr(value, "semantic_dict", None)
    if callable(semantic_dict):
        return to_semantic_primitive(semantic_dict())
    semantic_identity = getattr(value, "semantic_identity", None)
    if callable(semantic_identity):
        return to_semantic_primitive(semantic_identity())
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): to_semantic_primitive(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [to_semantic_primitive(v) for v in value]
    if dataclasses.is_dataclass(value):
        data: dict[str, Any] = {}
        for field in dataclasses.fields(value):
            data[field.name] = to_semantic_primitive(getattr(value, field.name))
        kind = getattr(value, "kind", None)
        if isinstance(kind, str):
            data.setdefault("kind", kind)
        return data
    return value

@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Serializable description of a unit of work."""

    schema_version: int = 1

    kind: ClassVar[str] = "operation"

    def to_dict(self) -> dict[str, Any]:
        data = to_primitive(self)
        if not isinstance(data, dict):
            raise TypeError("OperationSpec serializer returned non-dict data")
        data["kind"] = self.kind
        return data

    def spec_hash(self) -> str:
        return stable_hash(self.to_dict())

    def semantic_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for field in dataclasses.fields(self):
            data[field.name] = to_semantic_primitive(getattr(self, field.name))
        data["kind"] = self.kind
        return data

    def semantic_hash(self) -> str:
        return stable_hash(self.semantic_dict())

    def required_capabilities(self) -> set[EngineCapability]:
        return set()

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return ()

    def bound_engine(self) -> Any | None:
        return None

    def runtime_spec(self) -> Any | None:
        return None
