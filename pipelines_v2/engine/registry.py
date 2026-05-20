"""Engine registry and deserialization helpers."""

from __future__ import annotations

from typing import Any, Callable

from pipelines_v2.core.registry import load_from_kind_registry
from pipelines_v2.engine.base import Engine
from pipelines_v2.engine.toy import ToyEngine
from pipelines_v2.engine.vllm.engine import VLLMEngine

EngineLoader = Callable[[dict[str, Any]], Engine]

_ENGINE_LOADERS: dict[str, EngineLoader] = {
    "toy": ToyEngine.from_dict,
    "vllm": VLLMEngine.from_dict,
}


def engine_from_dict(payload: dict[str, Any]) -> Engine:
    return load_from_kind_registry(payload, _ENGINE_LOADERS, missing_message="Engine payload is missing 'kind'", unknown_message="Unknown engine kind: {kind!r}")
