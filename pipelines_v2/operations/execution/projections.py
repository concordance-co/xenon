"""Execution helpers for structured projection specs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines_v2.core.types import SpecValidationError
from pipelines_v2.operations.execution.common import (
    OperationExecutionResult,
    align_example_keys_to_rows,
    coerce_token_sections,
    feature_name,
    make_label_payload,
    routing_vector_from_record,
)
from pipelines_v2.operations.projections import CoordinateImportSpec, ProjectionCalibrationSpec, ProjectionSpec
from pipelines_v2.operations.projections._aggregation import summarize_scores
from pipelines_v2.operations.projections._calibration import fit_quantile_bands
from pipelines_v2.operations.projections._coordinates import (
    coordinate_name_key,
    load_coordinate_import_payload,
    resolve_coordinate,
)
from pipelines_v2.operations.projections._kernels import pool_positions, project_vector
from pipelines_v2.operations.projections.slices import coerce_section_records, select_section_records
from pipelines_v2.storage.artifacts import FeatureLayerRef, FeatureRef


def run_coordinate_import(spec: CoordinateImportSpec) -> OperationExecutionResult:
    payload = load_coordinate_import_payload(
        path=spec.path,
        format=spec.format,
        select_layer=spec.select_layer,
        normalize=spec.normalize,
        name=spec.name,
        metadata=spec.metadata,
    )
    return OperationExecutionResult(
        payload=payload,
        example_coverage={
            "materialized": True,
            "example_count": 0,
            "example_keys": [],
        },
    )


def run_projection(spec: ProjectionSpec) -> OperationExecutionResult:
    feature_kind, layer_payloads, routing_policy = _load_projection_feature(
        spec.feature,
        layers=tuple(spec.layers) if spec.layers else None,
    )
    if not spec.coordinates:
        raise SpecValidationError("ProjectionSpec requires at least one coordinate source")

    selected_layers = sorted(layer_payloads)
    resolved_coordinates = [
        resolve_coordinate(source, fallback_name=f"coordinate_{index}")
        for index, source in enumerate(spec.coordinates)
    ]
    for coordinate in resolved_coordinates:
        if not set(coordinate.layers).intersection(selected_layers):
            raise SpecValidationError(
                f"Projection coordinate {coordinate.name!r} does not define any of the requested feature layers {selected_layers}"
            )

    example_keys = sorted(layer_payloads[selected_layers[0]])
    selected_keys = align_example_keys_to_rows(example_keys, spec.rows, label="projection")

    rows: list[dict[str, Any]] = []
    example_summaries: list[dict[str, Any]] = []
    label_values: dict[str, dict[str, Any]] = defaultdict(dict)

    for layer in selected_layers:
        layer_payload = layer_payloads[layer]
        for example_key in selected_keys:
            record = layer_payload[example_key]
            values = _projection_token_matrix(
                feature_kind=feature_kind,
                record=record,
                routing_policy=routing_policy,
            )
            token_sections = coerce_token_sections(record.get("token_sections"))
            section_records = coerce_section_records(
                record.get("section_records"),
                token_sections=token_sections,
            )
            selected_sections = select_section_records(section_records, spec.slices)
            if not selected_sections:
                continue

            scores_by_coordinate: dict[str, list[float]] = defaultdict(list)
            order_by_coordinate: dict[str, list[float]] = defaultdict(list)

            for ordinal, section in enumerate(selected_sections):
                positions = section.get("token_positions")
                if not isinstance(positions, Sequence) or isinstance(positions, str | bytes | bytearray):
                    continue
                pooled = pool_positions(
                    values,
                    positions=[int(position) for position in positions],
                    pooling=spec.pooling,
                )
                order_value = float(section.get("index")) if section.get("index") is not None else float(ordinal)

                for coordinate in resolved_coordinates:
                    direction = coordinate.layers.get(int(layer))
                    if direction is None:
                        continue
                    score = project_vector(
                        pooled,
                        direction=direction,
                        metric=spec.metric,
                    )
                    scores_by_coordinate[coordinate.name].append(float(score))
                    order_by_coordinate[coordinate.name].append(order_value)
                    rows.append(
                        {
                            "example_key": str(example_key),
                            "layer": int(layer),
                            "coordinate": coordinate.name,
                            "score": float(score),
                            "slice_name": str(section.get("name") or ""),
                            "slice_index": int(section["index"]) if section.get("index") is not None else int(ordinal),
                            "slice_token_count": len([int(position) for position in positions]),
                            "role": section.get("role"),
                            "unit": section.get("unit"),
                            "tags": dict(section.get("tags", {})) if isinstance(section.get("tags"), Mapping) else {},
                        }
                    )

            for coordinate in resolved_coordinates:
                coordinate_scores = scores_by_coordinate.get(coordinate.name)
                if not coordinate_scores:
                    continue
                metrics = summarize_scores(
                    coordinate_scores,
                    summary_names=tuple(spec.summaries),
                    order_values=order_by_coordinate.get(coordinate.name),
                )
                summary_row = {
                    "example_key": str(example_key),
                    "layer": int(layer),
                    "coordinate": coordinate.name,
                    "slice_count": len(coordinate_scores),
                    "metrics": metrics,
                }
                example_summaries.append(summary_row)
                if spec.emit_labels:
                    _projection_labels_for_summary(
                        label_values=label_values,
                        example_key=str(example_key),
                        layer=int(layer),
                        coordinate_name=coordinate.name,
                        slice_count=len(coordinate_scores),
                        metrics=metrics,
                    )

    payload = {
        "kind": "projection_result",
        "feature": feature_name(spec.feature),
        "metric": spec.metric,
        "pooling": spec.pooling.kind,
        "coordinates": [
            {
                "name": coordinate.name,
                "layers": sorted(int(layer) for layer in coordinate.layers),
                "source_kind": coordinate.source_kind,
                "metadata": dict(coordinate.metadata),
            }
            for coordinate in resolved_coordinates
        ],
        "rows": rows,
        "example_summaries": example_summaries,
        "summary": {
            "coordinate_count": len(resolved_coordinates),
            "layer_count": len(selected_layers),
            "slice_row_count": len(rows),
            "example_summary_count": len(example_summaries),
        },
    }
    labels = {
        name: make_label_payload(name, values)
        for name, values in dict(label_values).items()
    }
    return OperationExecutionResult(
        payload=payload,
        labels=labels,
        example_coverage={
            "materialized": True,
            "example_count": len(selected_keys),
            "example_keys": list(selected_keys),
        },
    )


def run_projection_calibration(spec: ProjectionCalibrationSpec) -> OperationExecutionResult:
    payload = spec.projections.result() if hasattr(spec.projections, "result") else spec.projections
    if not isinstance(payload, Mapping) or str(payload.get("kind") or "") != "projection_result":
        raise SpecValidationError("ProjectionCalibrationSpec requires a projection_result source")

    fit_on_keys = None
    if spec.fit_on is not None:
        if not hasattr(spec.fit_on, "resolve_example_keys"):
            raise SpecValidationError(
                f"ProjectionCalibrationSpec fit_on must support resolve_example_keys(), got {type(spec.fit_on).__name__}"
            )
        fit_on_keys = {str(key) for key in spec.fit_on.resolve_example_keys()}

    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    if spec.summary_name is not None:
        summary_name = str(spec.summary_name)
        for row in payload.get("example_summaries", ()):
            if not isinstance(row, Mapping):
                continue
            example_key = str(row.get("example_key") or "")
            if fit_on_keys is not None and example_key not in fit_on_keys:
                continue
            metrics = row.get("metrics")
            if not isinstance(metrics, Mapping) or summary_name not in metrics:
                continue
            grouped[(str(row.get("coordinate") or ""), int(row.get("layer") or 0))].append(float(metrics[summary_name]))
    else:
        for row in payload.get("rows", ()):
            if not isinstance(row, Mapping):
                continue
            example_key = str(row.get("example_key") or "")
            if fit_on_keys is not None and example_key not in fit_on_keys:
                continue
            grouped[(str(row.get("coordinate") or ""), int(row.get("layer") or 0))].append(float(row.get("score") or 0.0))

    if spec.strategy != "quantile_bands":
        raise SpecValidationError(f"Unsupported projection calibration strategy: {spec.strategy!r}")

    definitions: list[dict[str, Any]] = []
    for (coordinate_name, layer), values in sorted(grouped.items()):
        if not values:
            continue
        definition = fit_quantile_bands(values, bands=tuple(spec.bands) if spec.bands else ("low", "mid", "high"))
        definition.update(
            {
                "coordinate": coordinate_name,
                "layer": int(layer),
                "orientation": _coordinate_orientation(spec.orientation, coordinate_name),
            }
        )
        if spec.summary_name is not None:
            definition["summary_name"] = str(spec.summary_name)
        definitions.append(definition)

    payload_out = {
        "kind": "projection_calibration_result",
        "strategy": spec.strategy,
        "summary_name": spec.summary_name,
        "definitions": definitions,
        "summary": {
            "definition_count": len(definitions),
            "fit_example_count": len(fit_on_keys) if fit_on_keys is not None else None,
        },
    }
    return OperationExecutionResult(
        payload=payload_out,
        example_coverage={
            "materialized": True,
            "example_count": len(fit_on_keys) if fit_on_keys is not None else 0,
            "example_keys": sorted(fit_on_keys) if fit_on_keys is not None else [],
        },
    )


def _load_projection_feature(
    feature: Any,
    *,
    layers: Sequence[int] | None,
) -> tuple[str, dict[int, Mapping[str, Any]], Mapping[str, Any] | None]:
    if isinstance(feature, FeatureLayerRef):
        payload = feature.feature.load()
        feature_kind = str(payload.get("kind") or "")
        return (
            feature_kind,
            {int(feature.layer): payload["layers"][str(feature.layer)]},
            payload.get("routing_policy") if isinstance(payload, Mapping) else None,
        )
    if isinstance(feature, FeatureRef):
        payload = feature.load()
        feature_kind = str(payload.get("kind") or "")
        raw_layers = payload.get("layers")
        if not isinstance(raw_layers, Mapping):
            raise TypeError("Projection features must contain a 'layers' mapping")
        selected_layers = sorted(int(layer) for layer in raw_layers if layers is None or int(layer) in set(layers))
        if not selected_layers:
            raise SpecValidationError("Projection feature did not contain any requested layers")
        return (
            feature_kind,
            {int(layer): raw_layers[str(layer)] for layer in selected_layers},
            payload.get("routing_policy") if isinstance(payload, Mapping) else None,
        )
    raise TypeError(f"Unsupported projection feature reference type: {type(feature).__name__}")


def _projection_token_matrix(
    *,
    feature_kind: str,
    record: Mapping[str, Any],
    routing_policy: Mapping[str, Any] | None,
) -> np.ndarray:
    if feature_kind == "residual":
        values = np.asarray(record["values"], dtype=np.float32)
        if values.ndim != 2:
            raise TypeError("Projection residual feature rows must be rank-2 token matrices")
        return values
    if feature_kind == "moe_routing":
        token_positions = [int(position) for position in record.get("tokens", ())]
        raw_records = record.get("records")
        if not isinstance(raw_records, Mapping):
            raise TypeError("Projection routing feature rows must contain a 'records' mapping")
        vectors = [
            routing_vector_from_record(
                raw_records[str(token_position)],
                routing_policy=routing_policy,
            )
            for token_position in token_positions
        ]
        return np.stack(vectors, axis=0).astype(np.float32)
    raise SpecValidationError(f"ProjectionSpec does not support feature kind {feature_kind!r}")


def _projection_labels_for_summary(
    *,
    label_values: dict[str, dict[str, Any]],
    example_key: str,
    layer: int,
    coordinate_name: str,
    slice_count: int,
    metrics: Mapping[str, Any],
) -> None:
    key = coordinate_name_key(coordinate_name)
    label_values[f"projection__{key}__layer_{layer}__slice_count"][example_key] = int(slice_count)
    for metric_name, metric_value in metrics.items():
        label_values[f"projection__{key}__layer_{layer}__{metric_name}"][example_key] = metric_value


def _coordinate_orientation(orientation: Mapping[str, str], coordinate_name: str) -> str | None:
    if coordinate_name in orientation:
        return str(orientation[coordinate_name])
    key = coordinate_name_key(coordinate_name)
    if key in orientation:
        return str(orientation[key])
    return None


__all__ = [
    "run_coordinate_import",
    "run_projection",
    "run_projection_calibration",
]
