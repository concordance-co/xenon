"""Translate Xenon block-output coordinates to reviewed vLLM EAGLE boundaries.

EAGLE index 0 is the embedding boundary; index L+1 is the unnormalized
combined residual after decoder block L in the reviewed implementations.
Never apply this translation to legacy artifacts or unknown architectures.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import warnings
from typing import Any, Sequence

from pipelines_v2.operations.capture.sites import ResidualSite


COORDINATE_SCHEMA = "xenon.residual_coordinates.v1"
_REVIEWED_MODELS = {
    "0.19.0": {("vllm.model_executor.models.llama", "LlamaModel")},
    "0.25.1": {
        ("vllm.model_executor.models.llama", "LlamaModel"),
        ("vllm.model_executor.models.qwen2", "Qwen2Model"),
        ("vllm.model_executor.models.qwen3", "Qwen3Model"),
        ("vllm.model_executor.models.qwen3_moe", "Qwen3MoeModel"),
    },
}


def auxiliary_index(site: ResidualSite, layer: int) -> int:
    return int(layer) + (1 if site.layer_indexing == "decoder_block" else 0)


def auxiliary_indices(sites: Sequence[ResidualSite]) -> list[int]:
    return sorted({auxiliary_index(site, layer) for site in sites for layer in site.layers})


def coordinate_metadata(site: ResidualSite) -> dict[str, Any]:
    return {
        "schema": COORDINATE_SCHEMA,
        "layer_indexing": site.layer_indexing,
        "backend": "vllm",
        "site": site.site,
        "capture_aux_indices": {str(layer): auxiliary_index(site, layer) for layer in site.layers},
        # Raw auxiliary coordinates do not assert physical semantics on an
        # unreviewed model. In particular, legacy index 0 is not block -1.
        "decoder_block_indices": (
            {str(layer): int(layer) for layer in site.layers}
            if site.layer_indexing == "decoder_block" else None
        ),
    }


def validate_capture_coordinates(engine: Any, sites: Sequence[ResidualSite]) -> None:
    if any(site.layer_indexing == "decoder_block" and site.site != "resid_post" for site in sites):
        raise ValueError("vLLM decoder-block residual capture currently supports only resid_post")
    if any(site.layer_indexing == "decoder_block" for site in sites):
        if int(engine.pipeline_parallel_size) != 1:
            raise ValueError("Decoder-block capture coordinates currently require pipeline_parallel_size=1")
    if any(site.layer_indexing == "vllm_aux" for site in sites):
        warnings.warn(
            "Legacy vllm_aux capture: layer labels are raw auxiliary indices, not decoder-block "
            "outputs. Use an audited source_layer_map when patching; see platform/residual_layer_indexing.md.",
            UserWarning, stacklevel=2,
        )


def audit_model_coordinates(model: Any, *, requested_blocks: tuple[int, ...]) -> dict[str, Any]:
    """Run on every model worker before generation; reject unreviewed mappings."""
    from .patching.base import unwrap_model

    model = unwrap_model(model)
    decoder = getattr(model, "model", None)
    if decoder is None:
        raise ValueError("Cannot verify decoder-block capture coordinates: model has no decoder")
    version = importlib.metadata.version("vllm")
    identity = (type(decoder).__module__, type(decoder).__name__)
    if identity not in _REVIEWED_MODELS.get(version, set()):
        raise ValueError(
            f"Unreviewed residual capture coordinates: vLLM {version}, {identity}. "
            "Audit this backend before using decoder_block; vllm_aux is explicit raw-index compatibility mode."
        )
    count = len(decoder.layers)
    if getattr(decoder, "start_layer", 0) != 0 or getattr(decoder, "end_layer", count) != count:
        raise ValueError("Decoder-block capture requires the complete non-pipeline-parallel decoder")
    if any(layer < 0 or layer >= count for layer in requested_blocks):
        raise ValueError(f"Requested decoder blocks {requested_blocks} outside [0, {count})")
    source = inspect.getsource(type(decoder).forward)
    compact = " ".join(source.split())
    if not any(
        f"aux_hidden_states, {index} + 1, hidden_states, residual" in compact
        for index in ("idx", "layer_idx")
    ):
        raise ValueError("Reviewed vLLM forward no longer exposes the expected post-block auxiliary boundary")
    return {
        "vllm_version": version,
        "decoder_class": ".".join(identity),
        "decoder_block_count": count,
        "forward_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "schema": COORDINATE_SCHEMA,
    }
