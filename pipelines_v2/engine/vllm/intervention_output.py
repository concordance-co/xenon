"""Output normalization helpers for patched vLLM generation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pipelines_v2.operations.interventions import PatchedGenerationSpec

from .capture import _generation_result_from_output


def normalize_generation_output(
    request_output: Any,
    *,
    capture_reasoning: bool,
    reasoning_parser: Any | None,
) -> dict[str, Any]:
    result = _generation_result_from_output(
        request_output,
        capture_reasoning=capture_reasoning,
        reasoning_parser=reasoning_parser,
    )
    payload = {
        "generated_text": str(result.get("text") or ""),
        "generated_token_ids": list(result.get("generated_token_ids") or ()),
        "finish_reason": str(result.get("finish_reason") or ""),
        "request_id": str(result.get("request_id") or ""),
    }
    if "reasoning_text" in result:
        payload["reasoning_text"] = str(result.get("reasoning_text") or "")
    if "structured_output" in result:
        payload["structured_output"] = result.get("structured_output")
    return payload


def stats_for_request(batch_stats: Mapping[str, Any], request_id: str) -> dict[str, Any]:
    if request_id in batch_stats and isinstance(batch_stats[request_id], Mapping):
        return {str(key): dict(value) for key, value in dict(batch_stats[request_id]).items()}
    short_id = request_id.split("-", 1)[0]
    for candidate_id, payload in batch_stats.items():
        candidate_text = str(candidate_id)
        if candidate_text == short_id or candidate_text.startswith(f"{short_id}-"):
            if isinstance(payload, Mapping):
                return {str(key): dict(value) for key, value in dict(payload).items()}
    return {}


def missing_patch_stats(
    *,
    spec: PatchedGenerationSpec,
    target_positions: list[int],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        str(int(write_layer)): {
            "layer": int(write_layer),
            "source_layer": int(spec.patch.source_layer_for(int(write_layer))),
            "status": "missing_runtime_stats",
            "operator": spec.patch.operator,
            "token_count": int(len(target_positions)),
            "query_positions": [int(pos) for pos in target_positions],
        }
        for write_layer in spec.patch.write_site.layers
    }
    if extra:
        for layer_payload in payload.values():
            layer_payload.update(dict(extra))
            if hasattr(spec.patch, "centroid_name"):
                layer_payload["centroid_name"] = str(spec.patch.centroid_name)
            if hasattr(spec.patch, "component_indices_by_layer"):
                layer_payload["selected_component_count"] = int(
                    len(getattr(spec.patch, "component_indices_by_layer", {}).get(int(layer_payload["layer"]), ()))
                )
    return payload


def iter_batches(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


__all__ = [
    "iter_batches",
    "missing_patch_stats",
    "normalize_generation_output",
    "stats_for_request",
]
