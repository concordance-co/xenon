"""Engine registry and deserialization helpers."""

from __future__ import annotations

from typing import Any, Callable

from pipelines_v2.core.registry import load_from_kind_registry, register_kind
from pipelines_v2.engine.base import Engine
from pipelines_v2.engine.toy import ToyEngine
from pipelines_v2.engine.torch import TorchEngine
from pipelines_v2.engine.native import NativeTorchEngine
from pipelines_v2.engine.vllm.engine import VLLMEngine

EngineLoader = Callable[[dict[str, Any]], Engine]

_ENGINE_LOADERS: dict[str, EngineLoader] = {
    "toy": ToyEngine.from_dict,
    "torch": TorchEngine.from_dict,
    "torch_native": NativeTorchEngine.from_dict,
    "vllm": VLLMEngine.from_dict,
}


def engine_from_dict(payload: dict[str, Any]) -> Engine:
    if payload.get("kind") == "vit_recognition":
        from pipelines_v2.engine.recognition import ViTRecognitionEngine
        return ViTRecognitionEngine.from_dict(payload)
    if payload.get("kind") == "sdxl":
        from pipelines_v2.engine.diffusers.specs import SDXLEngine
        return SDXLEngine.from_dict(payload)
    engine = load_from_kind_registry(payload, _ENGINE_LOADERS, missing_message="Engine payload is missing 'kind'", unknown_message="Unknown engine kind: {kind!r}")
    if engine.identity().get("kind") != str(payload["kind"]).strip():
        raise ValueError("Engine loader returned an incompatible kind")
    return engine


def register_engine(kind: str, loader: EngineLoader) -> None:
    """Register an engine loader from trusted bootstrap code (no plugin discovery)."""
    if not callable(loader):
        raise TypeError("Engine loader must be callable")
    register_kind(_ENGINE_LOADERS, kind, loader, label="Engine")


__all__ = ["engine_from_dict", "register_engine"]
