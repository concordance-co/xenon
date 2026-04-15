"""Engine registry and deserialization helpers."""

from __future__ import annotations

from typing import Any, Callable

from pipelines_v2.engine.base import Engine
from pipelines_v2.engine.toy import ToyEngine
from pipelines_v2.engine.vllm.engine import VLLMEngine

EngineLoader = Callable[[dict[str, Any]], Engine]

_ENGINE_LOADERS: dict[str, EngineLoader] = {
    "toy": ToyEngine.from_dict,
    "vllm": VLLMEngine.from_dict,
}


def engine_from_dict(payload: dict[str, Any]) -> Engine:
    kind = str(payload.get("kind") or "").strip()
    if not kind:
        raise ValueError("Engine payload is missing 'kind'")
    try:
        loader = _ENGINE_LOADERS[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown engine kind: {kind!r}") from exc
    return loader(payload)
