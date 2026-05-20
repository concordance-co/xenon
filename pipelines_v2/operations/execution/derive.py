"""Execution helpers for derived-label and contrast operations."""

from __future__ import annotations

from typing import Any

import numpy as np

from pipelines_v2.core.types import SpecValidationError, stable_hash
from pipelines_v2.operations.derive import LabelFieldsSpec, LabelMapSpec, PairDeltaSpec, TransformSpec

from .common import (
    OperationExecutionResult,
    coerce_mapping_value,
    coerce_transform_label_payload,
    coerce_transform_result,
    feature_matrices,
    infer_transform_example_keys,
    label_payload_from_grouped_source,
    make_label_payload,
    ordered_values,
    resolve_values_map,
)


def run_pair_delta(spec: PairDeltaSpec) -> OperationExecutionResult:
    from pipelines_v2.data.datasets import LabelPredicate

    if not isinstance(spec.positive, LabelPredicate) or not isinstance(spec.negative, LabelPredicate):
        raise SpecValidationError("PairDeltaSpec requires positive and negative LabelPredicate refs")

    matrices, example_keys = feature_matrices(
        spec.feature,
        layers=tuple(spec.layers) if spec.layers else None,
        token_selector=spec.tokens,
        token_pooling=spec.pooling,
    )
    case_values = ordered_values(spec.case, example_keys, label="case")
    positive_keys = {key for key in spec.positive.resolve_example_keys() if key in set(example_keys)}
    negative_keys = {key for key in spec.negative.resolve_example_keys() if key in set(example_keys)}
    if not positive_keys or not negative_keys:
        raise SpecValidationError("PairDeltaSpec requires at least one positive and one negative example")

    groups: dict[str, list[str]] = {}
    for example_key, case_value in zip(example_keys, case_values, strict=False):
        groups.setdefault(str(case_value), []).append(example_key)

    case_keys: list[str] = []
    side_keys: dict[str, list[str]] = {}
    for case_key, members in groups.items():
        pos_members = [key for key in members if key in positive_keys]
        neg_members = [key for key in members if key in negative_keys]
        if not pos_members or not neg_members:
            continue
        case_keys.append(case_key)
        side_keys[f"positive::{case_key}"] = pos_members
        side_keys[f"negative::{case_key}"] = neg_members

    if not case_keys:
        raise SpecValidationError("PairDeltaSpec could not find any case with both positive and negative members")

    feature_layers: dict[str, Any] = {}
    index_by_example = {key: index for index, key in enumerate(example_keys)}
    for layer, X in matrices.items():
        layer_payload: dict[str, Any] = {}
        for case_key in case_keys:
            pos_members = side_keys[f"positive::{case_key}"]
            neg_members = side_keys[f"negative::{case_key}"]
            pos_vec = X[[index_by_example[key] for key in pos_members]].mean(axis=0)
            neg_vec = X[[index_by_example[key] for key in neg_members]].mean(axis=0)
            delta = (pos_vec - neg_vec).astype(np.float32)
            layer_payload[case_key] = {
                "tokens": [0],
                "values": np.expand_dims(delta, axis=0),
                "prompt_hash": stable_hash([spec.output_feature_name, case_key])[:24],
            }
        feature_layers[str(layer)] = layer_payload

    propagated_example_keys = {
        case_key: side_keys[f"{spec.propagate_from}::{case_key}"]
        for case_key in case_keys
    }
    label_payloads = {
        name: label_payload_from_grouped_source(
            name=name,
            source=source,
            grouped_example_keys=propagated_example_keys,
        )
        for name, source in dict(spec.labels).items()
    }

    feature_payload = {
        "kind": "residual",
        "site": "pair_delta",
        "storage": {"dtype": "float32", "format": "safetensors"},
        "layers": feature_layers,
    }
    payload = {
        "kind": "pair_delta_result",
        "feature": spec.output_feature_name,
        "pair_count": len(case_keys),
        "layer_count": len(feature_layers),
        "propagate_from": spec.propagate_from,
        "labels": sorted(label_payloads),
    }
    return OperationExecutionResult(
        payload=payload,
        features={spec.output_feature_name: feature_payload},
        labels=label_payloads,
        example_coverage={
            "materialized": True,
            "example_count": len(case_keys),
            "example_keys": case_keys,
        },
    )


def run_transform(spec: TransformSpec) -> OperationExecutionResult:
    raw_result = spec.builder.build(spec.inputs)
    result = coerce_transform_result(raw_result)
    labels = {
        str(name): coerce_transform_label_payload(str(name), values)
        for name, values in result.labels.items()
    }
    example_keys = (
        [str(key) for key in result.example_keys]
        if result.example_keys is not None
        else infer_transform_example_keys(labels)
    )
    payload = dict(result.payload)
    payload.setdefault("kind", "transform_result")
    return OperationExecutionResult(
        payload=payload,
        labels=labels,
        metadata=dict(result.metadata),
        example_coverage={
            "materialized": example_keys is not None,
            "example_count": len(example_keys) if example_keys is not None else None,
            **({"example_keys": list(example_keys)} if example_keys is not None else {}),
        },
    )


def run_label_map(spec: LabelMapSpec) -> OperationExecutionResult:
    values = resolve_values_map(spec.source, label="source")
    mapped: dict[str, Any] = {}
    missing: set[str] = set()
    for key, value in values.items():
        mapping_key = str(value)
        if mapping_key in spec.mapping:
            mapped[str(key)] = spec.mapping[mapping_key]
        elif spec.strict:
            missing.add(mapping_key)
        else:
            mapped[str(key)] = spec.default_value
    if missing:
        raise SpecValidationError(
            f"LabelMapSpec output {spec.output_name!r} is missing mappings for source values: {sorted(missing)}"
        )
    payload = {
        "kind": "label_map_result",
        "source_name": getattr(spec.source, "name", None),
        "output_name": spec.output_name,
        "mapped_count": len(mapped),
        "strict": spec.strict,
    }
    labels = {spec.output_name: make_label_payload(spec.output_name, mapped)}
    return OperationExecutionResult(
        payload=payload,
        labels=labels,
        example_coverage={
            "materialized": True,
            "example_count": len(mapped),
            "example_keys": sorted(mapped),
        },
    )


def run_label_fields(spec: LabelFieldsSpec) -> OperationExecutionResult:
    values = resolve_values_map(spec.source, label="source")
    extracted: dict[str, dict[str, Any]] = {output_name: {} for output_name in spec.fields}
    missing_fields: set[str] = set()
    for key, value in values.items():
        raw = coerce_mapping_value(value, label="LabelFieldsSpec source value")
        for output_name, field_name in spec.fields.items():
            if field_name in raw:
                extracted[output_name][str(key)] = raw[field_name]
            elif spec.strict:
                missing_fields.add(field_name)
    if missing_fields:
        raise SpecValidationError(
            f"LabelFieldsSpec is missing requested fields in source payloads: {sorted(missing_fields)}"
        )
    labels = {
        output_name: make_label_payload(output_name, output_values)
        for output_name, output_values in extracted.items()
    }
    payload = {
        "kind": "label_fields_result",
        "source_name": getattr(spec.source, "name", None),
        "label_names": sorted(labels),
        "strict": spec.strict,
    }
    example_keys = sorted(next(iter(extracted.values()))) if extracted else []
    return OperationExecutionResult(
        payload=payload,
        labels=labels,
        example_coverage={
            "materialized": True,
            "example_count": len(example_keys),
            "example_keys": example_keys,
        },
    )
