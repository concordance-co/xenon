"""Compatibility helpers for vLLM hidden-state extraction.

vLLM 0.25.1's upstream ``ExampleHiddenStatesConnector`` supports both
prompt-only and prompt-plus-generated-token capture. It also writes
asynchronously, so callers must use its synchronized loader before cleaning up
the tensor and companion lock files.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

_CONNECTOR_MODULE = (
    "vllm.distributed.kv_transfer.kv_connector.v1."
    "example_hidden_states_connector"
)


def load_and_cleanup_hidden_states(path: str) -> dict[str, Any]:
    """Load one connector result and remove all connector-owned files.

    On vLLM 0.25.1 this delegates to ``load_hidden_states`` so an asynchronous
    writer is allowed to finish under its companion filesystem lock. The
    direct safetensors fallback keeps CPU-only tests usable and preserves
    compatibility with vLLM releases predating the synchronized helpers.
    """

    connector_module: Any | None
    try:
        connector_module = import_module(_CONNECTOR_MODULE)
    except ModuleNotFoundError as exc:
        if not exc.name or not (
            exc.name == "vllm" or exc.name.startswith("vllm.")
        ):
            raise
        connector_module = None

    upstream_load = (
        getattr(connector_module, "load_hidden_states", None)
        if connector_module is not None
        else None
    )
    upstream_cleanup = (
        getattr(connector_module, "cleanup_hidden_states", None)
        if connector_module is not None
        else None
    )
    if callable(upstream_load):
        tensors = upstream_load(str(path))
        try:
            if not isinstance(tensors, dict):
                raise TypeError(
                    "vLLM hidden-state loader returned "
                    f"{type(tensors).__name__}, expected a tensor mapping"
                )
        finally:
            if callable(upstream_cleanup):
                upstream_cleanup(str(path))
        return tensors

    from safetensors.torch import load_file

    connector_file = Path(path)
    tensors = load_file(str(connector_file))
    connector_file.unlink(missing_ok=True)
    Path(f"{connector_file}.lock").unlink(missing_ok=True)
    return tensors


def __getattr__(name: str) -> Any:
    """Keep old serialized connector configs importable during migration."""

    if name == "PipelinesV2HiddenStatesConnector":
        connector_module = import_module(_CONNECTOR_MODULE)
        return connector_module.ExampleHiddenStatesConnector
    raise AttributeError(name)


__all__ = [
    "PipelinesV2HiddenStatesConnector",
    "load_and_cleanup_hidden_states",
]
