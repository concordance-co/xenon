"""Artifact-bound execution for intervention source artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pipelines_v2.core.types import SpecValidationError
from pipelines_v2.operations.execution.common import (
    OperationExecutionResult,
    align_example_keys_to_rows,
    feature_name,
)
from pipelines_v2.operations.interventions.sources import ActivationBankSpec, ExplicitPathMaskSpec


def run_activation_bank(spec: ActivationBankSpec) -> OperationExecutionResult:
    feature = spec.feature
    if feature is None or not hasattr(feature, "load"):
        raise SpecValidationError("ActivationBankSpec feature must be a feature ref")
    payload = feature.load()
    if not isinstance(payload, Mapping):
        raise SpecValidationError("ActivationBankSpec feature payload must be a mapping")
    if str(payload.get("kind") or "") != "residual":
        raise SpecValidationError("ActivationBankSpec currently requires a residual feature")
    layers_payload = payload.get("layers")
    if not isinstance(layers_payload, Mapping):
        raise SpecValidationError("ActivationBankSpec feature payload is missing a 'layers' mapping")
    available_layers = sorted(int(layer) for layer in layers_payload)
    selected_layers = [layer for layer in available_layers if not spec.layers or int(layer) in set(spec.layers)]
    if not selected_layers:
        raise SpecValidationError("ActivationBankSpec requested layers were not present in the feature")
    first_layer = layers_payload[str(selected_layers[0])]
    if not isinstance(first_layer, Mapping):
        raise SpecValidationError("ActivationBankSpec feature layer payload must be a mapping")
    selected_keys = align_example_keys_to_rows(
        list(first_layer.keys()),
        spec.rows,
        label="ActivationBankSpec",
    )
    result_layers: dict[str, Any] = {}
    for layer in selected_layers:
        layer_payload = layers_payload[str(layer)]
        if not isinstance(layer_payload, Mapping):
            raise SpecValidationError(f"ActivationBankSpec layer {layer} payload must be a mapping")
        result_layers[str(layer)] = {
            key: {
                "values": layer_payload[key]["values"],
                "token_count": int(layer_payload[key].get("token_count", len(layer_payload[key]["values"]))),
                "token_sections": dict(layer_payload[key].get("token_sections", {})),
            }
            for key in selected_keys
        }
    payload_out = {
        "kind": "activation_bank_result",
        "feature": feature_name(spec.feature),
        "site": str(payload.get("site") or ""),
        "layers": result_layers,
        "summary": {
            "layer_count": len(result_layers),
            "example_count": len(selected_keys),
            "layer_indices": [int(layer) for layer in selected_layers],
        },
    }
    return OperationExecutionResult(
        payload=payload_out,
        example_coverage={
            "materialized": True,
            "example_count": len(selected_keys),
            "example_keys": list(selected_keys),
        },
    )


def run_explicit_path_mask(spec: ExplicitPathMaskSpec) -> OperationExecutionResult:
    edges = [
        {
            "source_layer": int(edge.source_layer),
            "write_layer": int(edge.write_layer),
            "weight": float(edge.weight),
        }
        for edge in spec.edges
    ]
    by_write_layer: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        by_write_layer.setdefault(str(edge["write_layer"]), []).append(edge)
    return OperationExecutionResult(
        payload={
            "kind": "explicit_path_mask_result",
            "edges": edges,
            "by_write_layer": by_write_layer,
            "summary": {
                "edge_count": len(edges),
                "write_layer_count": len(by_write_layer),
                "source_layers": sorted({int(edge["source_layer"]) for edge in edges}),
                "write_layers": sorted({int(edge["write_layer"]) for edge in edges}),
            },
        },
        example_coverage={},
    )


__all__ = ["run_activation_bank", "run_explicit_path_mask"]
