"""Shared helpers for domain-specific direction wrappers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from pipelines_v2.core.types import SpecValidationError
from pipelines_v2.operations.common.vectors import normalize_vector
from pipelines_v2.operations.execution.common import OperationExecutionResult


def annotate_direction_payload(
    payload: Mapping[str, Any],
    *,
    name: str,
    method: str,
    metadata: Mapping[str, Any] | None = None,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a direction payload with a stable name and method metadata."""

    result = dict(payload)
    existing_metadata = dict(result.get("metadata", {})) if isinstance(result.get("metadata"), Mapping) else {}
    existing_summary = dict(result.get("summary", {})) if isinstance(result.get("summary"), Mapping) else {}
    result["name"] = str(name)
    result["metadata"] = {
        **existing_metadata,
        "source": method,
        **dict(metadata or {}),
    }
    result["summary"] = {
        **existing_summary,
        **dict(summary or {}),
    }
    return result


def direction_payload_to_subspace(
    direction: Any,
    *,
    layers: Sequence[int] = (),
    name: str,
    method: str,
    metadata: Mapping[str, Any] | None = None,
) -> OperationExecutionResult:
    """Convert a direction_result/coordinate_result into a 1-component subspace."""

    payload = _payload(direction, label="direction")
    raw_layers = payload.get("layers")
    if not isinstance(raw_layers, Mapping) or not raw_layers:
        raise SpecValidationError("Direction ablation requires a non-empty direction layers payload")
    selected = _selected_layers(raw_layers, layers=layers)
    subspace_layers: dict[str, Any] = {}
    for layer in selected:
        layer_payload = raw_layers[str(layer)]
        if not isinstance(layer_payload, Mapping):
            raise TypeError("Direction layer payloads must be mappings")
        raw = np.asarray(layer_payload.get("vector"), dtype=np.float32)
        if raw.ndim != 1:
            raise SpecValidationError(f"Direction layer {layer} vector must be rank-1")
        unit, _norm = normalize_vector(raw, normalize="l2", error_label="direction ablation")
        width = int(unit.shape[0])
        subspace_layers[str(layer)] = {
            "method": "direction_component",
            "mean": np.zeros(width, dtype=np.float32).tolist(),
            "scale": np.ones(width, dtype=np.float32).tolist(),
            "safe_scale": np.ones(width, dtype=np.float32).tolist(),
            "components": [unit.astype(np.float32).tolist()],
            "explained_variance_ratio": [1.0],
            "example_count": 0,
            "component_count": 1,
            "named_components": {str(name): 0},
        }
    return OperationExecutionResult(
        payload={
            "kind": "subspace_result",
            "feature": str(payload.get("feature") or ""),
            "layers": subspace_layers,
            "metadata": {
                "source": method,
                "direction_name": str(payload.get("name") or ""),
                **dict(metadata or {}),
            },
            "summary": {
                "layer_count": len(subspace_layers),
                "component_count": 1,
                "method": "direction_component",
            },
        },
        example_coverage={"materialized": True, "example_count": 0, "example_keys": []},
    )


def select_direction_by_projection_gap(
    *,
    direction: Any,
    scores: Any,
    positive_when: Any,
    negative_when: Any,
    layers: Sequence[int] = (),
    summary_metric: str = "mean",
    name: str,
    method: str,
    metadata: Mapping[str, Any] | None = None,
) -> OperationExecutionResult:
    """Select the direction layer with the largest projection separation."""

    direction_payload = _payload(direction, label="direction")
    score_payload = _payload(scores, label="scores")
    raw_layers = direction_payload.get("layers")
    if not isinstance(raw_layers, Mapping) or not raw_layers:
        raise SpecValidationError("Direction selection requires a non-empty direction layers payload")

    positive_keys = _resolved_key_set(positive_when, label="positive_when")
    negative_keys = _resolved_key_set(negative_when, label="negative_when")
    selected_layers = _selected_layers(raw_layers, layers=layers)
    values_by_layer: dict[int, dict[str, list[float]]] = {
        int(layer): {"positive": [], "negative": []} for layer in selected_layers
    }

    for row in score_payload.get("example_summaries", ()):
        if not isinstance(row, Mapping):
            continue
        example_key = str(row.get("example_key") or "")
        layer = int(row.get("layer") or 0)
        if layer not in values_by_layer:
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping) or str(summary_metric) not in metrics:
            continue
        bucket = "positive" if example_key in positive_keys else "negative" if example_key in negative_keys else ""
        if bucket:
            values_by_layer[layer][bucket].append(float(metrics[str(summary_metric)]))

    layer_stats: dict[str, Any] = {}
    best_layer: int | None = None
    best_abs_gap = -1.0
    for layer in selected_layers:
        positive = values_by_layer[int(layer)]["positive"]
        negative = values_by_layer[int(layer)]["negative"]
        if not positive or not negative:
            continue
        pos_mean = float(np.mean(positive))
        neg_mean = float(np.mean(negative))
        gap = pos_mean - neg_mean
        layer_stats[str(layer)] = {
            "positive_count": len(positive),
            "negative_count": len(negative),
            "positive_mean": pos_mean,
            "negative_mean": neg_mean,
            "gap": float(gap),
            "abs_gap": float(abs(gap)),
        }
        if abs(gap) > best_abs_gap:
            best_layer = int(layer)
            best_abs_gap = float(abs(gap))

    if best_layer is None:
        raise SpecValidationError("Direction selection could not score any layer with both positive and negative rows")
    best_payload = raw_layers[str(best_layer)]
    result = annotate_direction_payload(
        {
            "kind": "direction_result",
            "feature": direction_payload.get("feature"),
            "layers": {str(best_layer): best_payload},
        },
        name=name,
        method=method,
        metadata={
            "selected_from_direction": str(direction_payload.get("name") or ""),
            "selection_metric": str(summary_metric),
            **dict(metadata or {}),
        },
        summary={
            "layer_count": 1,
            "selected_layer": int(best_layer),
            "selection": layer_stats[str(best_layer)],
            "candidate_layers": layer_stats,
        },
    )
    return OperationExecutionResult(
        payload=result,
        example_coverage={"materialized": True, "example_count": len(positive_keys | negative_keys), "example_keys": sorted(positive_keys | negative_keys)},
    )


def _payload(source: Any, *, label: str) -> Mapping[str, Any]:
    payload = source.result() if hasattr(source, "result") else source
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} must resolve to a mapping, got {type(payload).__name__}")
    return payload


def _resolved_key_set(source: Any, *, label: str) -> set[str]:
    if source is None or not hasattr(source, "resolve_example_keys"):
        raise SpecValidationError(f"{label} must support resolve_example_keys()")
    return {str(key) for key in source.resolve_example_keys()}


def _selected_layers(raw_layers: Mapping[str, Any], *, layers: Sequence[int]) -> tuple[int, ...]:
    available = sorted(int(layer) for layer in raw_layers)
    if not layers:
        return tuple(available)
    requested = tuple(int(layer) for layer in layers)
    missing = [layer for layer in requested if str(layer) not in raw_layers]
    if missing:
        raise SpecValidationError(f"Requested direction layers are missing: {missing}")
    return requested


__all__ = [
    "annotate_direction_payload",
    "direction_payload_to_subspace",
    "select_direction_by_projection_gap",
]
