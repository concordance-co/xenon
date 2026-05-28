"""Shared execution helpers for artifact-bound operations."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from pipelines_v2.core.types import SpecValidationError, stable_hash
from pipelines_v2.data.datasets import CaseSet, LabelSet
from pipelines_v2.operations.common.builders import TransformResult
from pipelines_v2.operations.common.tokens import TokenPooling, TokenSelector
from pipelines_v2.storage.artifacts import ArtifactLabelRef, CaptureArtifact, FeatureLayerRef, FeatureRef, OperationArtifact

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OperationExecutionResult:
    payload: dict[str, Any] = field(default_factory=dict)
    features: dict[str, dict[str, Any]] = field(default_factory=dict)
    labels: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    example_coverage: dict[str, Any] = field(default_factory=dict)


def feature_matrices(
    feature: Any,
    *,
    layers: Sequence[int] | None = None,
    token_selector: TokenSelector | None = None,
    token_pooling: TokenPooling | None = None,
) -> tuple[dict[int, NDArray[np.float32]], list[str]]:
    if isinstance(feature, FeatureLayerRef):
        logger.info(
            "Loading feature matrix from layer ref feature=%s layer=%s",
            feature.feature.name,
            feature.layer,
        )
        layer_payload = feature.load()
        example_keys = sorted(layer_payload)
        return {
            int(feature.layer): matrix_from_layer_payload(
                layer_payload,
                example_keys,
                token_selector=token_selector,
                token_pooling=token_pooling,
            )
        }, example_keys
    if isinstance(feature, FeatureRef):
        logger.info("Loading feature matrices feature=%s", feature.name)
        payload = feature.load()
        payload_kind = payload.get("kind")
        available_layers = sorted(int(layer) for layer in payload["layers"])
        selected_layers = [layer for layer in available_layers if layers is None or layer in set(layers)]
        if not selected_layers:
            raise SpecValidationError("No requested layers were present in the feature payload")
        logger.info(
            "Building feature matrices feature=%s kind=%s selected_layers=%s available_layer_count=%s",
            feature.name,
            payload_kind,
            selected_layers,
            len(available_layers),
        )
        example_keys = sorted(payload["layers"][str(selected_layers[0])])
        if payload_kind == "residual":
            return {
                int(layer): matrix_from_layer_payload(
                    payload["layers"][str(layer)],
                    example_keys,
                    token_selector=token_selector,
                    token_pooling=token_pooling,
                )
                for layer in selected_layers
            }, example_keys
        if payload_kind == "moe_routing":
            return {
                int(layer): router_matrix_from_layer_payload(
                    payload["layers"][str(layer)],
                    example_keys,
                    token_selector=token_selector,
                    token_pooling=token_pooling,
                    routing_policy=payload.get("routing_policy"),
                )
                for layer in selected_layers
            }, example_keys
        raise NotImplementedError(
            f"Artifact-bound ops do not support feature payload kind {payload_kind!r} yet"
        )
    raise TypeError(f"Unsupported feature reference type: {type(feature).__name__}")


def matrix_from_layer_payload(
    layer_payload: Mapping[str, Any],
    example_keys: Sequence[str],
    *,
    token_selector: TokenSelector | None,
    token_pooling: TokenPooling | None,
) -> NDArray[np.float32]:
    missing = [str(key) for key in example_keys if str(key) not in layer_payload]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise SpecValidationError(
            "Feature layer payload is missing selected examples: "
            f"{preview}{suffix} ({len(missing)} missing)"
        )
    logger.info(
        "Building residual matrix examples=%s token_selector=%s token_pooling=%s",
        len(example_keys),
        getattr(token_selector, "kind", None),
        getattr(token_pooling, "kind", None),
    )
    rows: list[NDArray[np.float32]] = []
    for index, key in enumerate(example_keys, start=1):
        record = dict(layer_payload[key])
        values = np.asarray(record["values"], dtype=np.float32)
        if values.ndim != 2:
            raise TypeError("Residual layer payload values must be rank-2")
        if token_selector is not None:
            selected = token_selector.resolve(
                values.shape[0],
                token_sections=coerce_token_sections(record.get("token_sections")),
            )
            if not selected:
                raise SpecValidationError("Token selector did not match any captured positions")
            values = values[selected]
        rows.append(pool_token_values(values, token_pooling=token_pooling))
        if index % 25000 == 0:
            logger.info("Built residual matrix rows=%s/%s", index, len(example_keys))
    matrix = np.stack(rows, axis=0).astype(np.float32)
    logger.info("Built residual matrix shape=%s dtype=%s", matrix.shape, matrix.dtype)
    return matrix


def router_matrix_from_layer_payload(
    layer_payload: Mapping[str, Any],
    example_keys: Sequence[str],
    *,
    token_selector: TokenSelector | None,
    token_pooling: TokenPooling | None,
    routing_policy: Mapping[str, Any] | None,
) -> NDArray[np.float32]:
    rows: list[NDArray[np.float32]] = []
    logger.info(
        "Building router matrix examples=%s token_selector=%s token_pooling=%s",
        len(example_keys),
        getattr(token_selector, "kind", None),
        getattr(token_pooling, "kind", None),
    )
    for index, key in enumerate(example_keys, start=1):
        record = dict(layer_payload[key])
        token_positions = [int(position) for position in record.get("tokens", ())]
        token_records = record.get("records")
        if not isinstance(token_records, Mapping):
            raise TypeError("MoE routing layer payload must contain a 'records' mapping")
        token_count = len(token_positions)
        if token_count <= 0:
            raise SpecValidationError("MoE routing payload did not contain any captured token positions")
        if token_selector is not None:
            selected = token_selector.resolve(
                token_count,
                token_sections=coerce_token_sections(record.get("token_sections")),
            )
        else:
            selected = list(range(token_count))
        if not selected:
            raise SpecValidationError("Token selector did not match any captured router positions")
        vectors: list[NDArray[np.float32]] = []
        for local_index in selected:
            token_position = token_positions[int(local_index)]
            token_record = token_records.get(str(token_position))
            if not isinstance(token_record, Mapping):
                raise SpecValidationError(
                    f"MoE routing payload is missing records for captured token position {token_position}"
                )
            vectors.append(routing_vector_from_record(token_record, routing_policy=routing_policy))
        values = np.stack(vectors, axis=0).astype(np.float32)
        rows.append(pool_token_values(values, token_pooling=token_pooling))
        if index % 25000 == 0:
            logger.info("Built router matrix rows=%s/%s", index, len(example_keys))
    matrix = np.stack(rows, axis=0).astype(np.float32)
    logger.info("Built router matrix shape=%s dtype=%s", matrix.shape, matrix.dtype)
    return matrix


def routing_vector_from_record(
    record: Mapping[str, Any],
    *,
    routing_policy: Mapping[str, Any] | None,
) -> NDArray[np.float32]:
    if "gate_logits" in record:
        return np.asarray(record["gate_logits"], dtype=np.float32)
    if "gate_probs" in record:
        return np.asarray(record["gate_probs"], dtype=np.float32)
    if "topk_from_gate" in record and isinstance(record["topk_from_gate"], Mapping):
        topk = record["topk_from_gate"]
        expert_ids = [int(expert_id) for expert_id in topk.get("expert_ids", ())]
        expert_weights = topk.get("weights", topk.get("expert_weights"))
        if expert_weights is None:
            weights = np.ones(len(expert_ids), dtype=np.float32)
        else:
            weights = np.asarray(expert_weights, dtype=np.float32)
        num_experts = None
        if isinstance(routing_policy, Mapping) and routing_policy.get("num_experts") is not None:
            num_experts = int(routing_policy["num_experts"])
        elif expert_ids:
            num_experts = max(expert_ids) + 1
        if num_experts is None:
            raise SpecValidationError("Could not infer MoE routing vector width from top-k router payload")
        dense = np.zeros(num_experts, dtype=np.float32)
        dense[np.asarray(expert_ids, dtype=np.int64)] = weights
        return dense
    if "expert_load" in record and isinstance(record["expert_load"], Mapping):
        expert_load = record["expert_load"]
        raw_counts = expert_load.get("counts", ())
        if isinstance(raw_counts, Mapping):
            pairs = [(int(expert_id), value) for expert_id, value in raw_counts.items()]
            expert_ids = [expert_id for expert_id, _value in pairs]
            counts = np.asarray([value for _expert_id, value in pairs], dtype=np.float32)
        else:
            expert_ids = [int(expert_id) for expert_id in expert_load.get("expert_ids", ())]
            counts = np.asarray(raw_counts, dtype=np.float32)
        if expert_ids and counts.size == len(expert_ids):
            num_experts = None
            if isinstance(routing_policy, Mapping) and routing_policy.get("num_experts") is not None:
                num_experts = int(routing_policy["num_experts"])
            else:
                num_experts = max(expert_ids) + 1
            dense = np.zeros(num_experts, dtype=np.float32)
            dense[np.asarray(expert_ids, dtype=np.int64)] = counts
            return dense
    raise SpecValidationError("MoE routing record does not contain a usable dense readout vector")


def pool_token_values(
    values: NDArray[np.float32],
    *,
    token_pooling: TokenPooling | None,
) -> NDArray[np.float32]:
    pooling = token_pooling or TokenPooling.mean()
    indices = pooling.from_count(int(values.shape[0]))
    if not indices:
        raise SpecValidationError("Token pooling did not match any token positions")
    selected = values[np.asarray(indices, dtype=np.int64)]
    if pooling.kind == "mean":
        return selected.mean(axis=0).astype(np.float32)
    if pooling.kind == "last":
        return selected[-1].astype(np.float32)
    if pooling.kind == "first":
        return selected[0].astype(np.float32)
    raise SpecValidationError(f"Unsupported token pooling mode: {pooling.kind}")


def ordered_values(source: Any, example_keys: Sequence[str], *, label: str) -> list[Any]:
    values = resolve_values_map(source, label=label)
    return [values[key] for key in example_keys]


def ordered_groups(groups: Any, example_keys: Sequence[str]) -> NDArray[np.object_] | None:
    if groups is None:
        return None
    values = resolve_values_map(groups, label="groups")
    return np.asarray([values[key] for key in example_keys], dtype=object)


def subset_example_keys(example_keys: Sequence[str], subset: Any | None) -> list[str]:
    if subset is None:
        return [str(key) for key in example_keys]
    if not hasattr(subset, "resolve_example_keys"):
        raise SpecValidationError(f"Subset selector must support resolve_example_keys(), got {type(subset).__name__}")
    allowed = {str(key) for key in subset.resolve_example_keys()}
    return [str(key) for key in example_keys if str(key) in allowed]


def requested_example_keys(rows: Any, *, label: str) -> list[str]:
    if rows is None:
        return []
    if not hasattr(rows, "resolve_example_keys"):
        raise SpecValidationError(
            f"{label} rows must support resolve_example_keys(), got {type(rows).__name__}"
        )
    return [str(key) for key in rows.resolve_example_keys()]


def align_example_keys_to_rows(
    example_keys: Sequence[str],
    rows: Any | None,
    *,
    label: str,
) -> list[str]:
    base_keys = [str(key) for key in example_keys]
    if rows is None:
        return base_keys
    requested = requested_example_keys(rows, label=label)
    requested_set = set(requested)
    base_set = set(base_keys)
    missing = [key for key in requested if key not in base_set]
    if missing:
        raise SpecValidationError(
            f"{label} rows requested {len(missing)} example keys not present in the referenced feature/text rows; "
            f"sample missing keys: {missing[:5]}"
        )
    selected = [key for key in base_keys if key in requested_set]
    if not selected:
        raise SpecValidationError(f"{label} rows did not match any referenced feature/text example keys")
    return selected


def filter_matrix_by_keys(
    matrix: NDArray[np.float32],
    example_keys: Sequence[str],
    selected_keys: Sequence[str],
) -> NDArray[np.float32]:
    index_by_key = {str(key): index for index, key in enumerate(example_keys)}
    indices = [index_by_key[str(key)] for key in selected_keys]
    return matrix[np.asarray(indices, dtype=np.int64)]


def encode_labels(values: Sequence[Any]) -> tuple[NDArray[np.int64], list[str]]:
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(np.asarray(values, dtype=object))
    classes = [str(item) for item in encoder.classes_]
    return encoded.astype(np.int64), classes


def feature_name(feature: Any) -> str:
    if isinstance(feature, FeatureLayerRef):
        return f"{feature.feature.name}:layer:{feature.layer}"
    if isinstance(feature, FeatureRef):
        return feature.name
    return type(feature).__name__


def summarize_report_input(value: Any) -> dict[str, Any]:
    if isinstance(value, (CaptureArtifact, OperationArtifact)):
        manifest = value.manifest()
        summary: Any | None = None
        if isinstance(value, OperationArtifact):
            try:
                summary = value.summary()
            except Exception:
                summary = None
        workflow_context = dict(manifest.workflow_context)
        runner = dict(manifest.runner)
        engine = dict(manifest.engine)
        return {
            "name": str(workflow_context.get("step_name") or manifest.artifact_id),
            "artifact_id": manifest.artifact_id,
            "artifact_kind": manifest.artifact_kind,
            "created_at": manifest.created_at,
            "workflow": {
                "run_id": workflow_context.get("run_id"),
                "workflow_name": workflow_context.get("workflow_name"),
                "step_name": workflow_context.get("step_name"),
                "step_index": workflow_context.get("step_index"),
                "workflow_step_key": workflow_context.get("workflow_step_key"),
            },
            "runtime": {
                "runner_kind": runner.get("kind"),
                "runtime_app_id": runner.get("runtime_app_id"),
                "volume_mappings": _report_volume_mappings(value, manifest),
            },
            "engine": engine,
            "input_artifact_refs": list(manifest.input_artifact_refs),
            "example_coverage": _summarize_example_coverage(manifest.example_coverage),
            "feature_names": sorted(manifest.storage_refs.get("features", {})),
            "label_names": sorted(manifest.storage_refs.get("labels", {})),
            "has_generations": "generations" in manifest.storage_refs,
            "primary_output": _primary_output_ref(manifest.storage_refs),
            "storage": _summarize_storage_refs(manifest.storage_refs),
            "summary": summary,
        }
    if isinstance(value, Mapping):
        return dict(value)
    return {"value": str(value)}


def _summarize_example_coverage(coverage: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "materialized": coverage.get("materialized"),
        "example_count": coverage.get("example_count"),
    }
    raw_keys = coverage.get("example_keys")
    if isinstance(raw_keys, Sequence) and not isinstance(raw_keys, str):
        payload["example_key_count"] = len(raw_keys)
    dataset_id = coverage.get("dataset_id")
    if dataset_id is not None:
        payload["dataset_id"] = dataset_id
    dataset_name = coverage.get("dataset_name")
    if dataset_name is not None:
        payload["dataset_name"] = dataset_name
    return payload


def _primary_output_ref(storage_refs: Mapping[str, Any]) -> dict[str, Any] | None:
    for key in ("report", "result", "summary", "generations", "manifest"):
        ref = storage_refs.get(key)
        if isinstance(ref, Mapping) and "path" in ref:
            return {
                "name": key,
                "store": ref.get("store"),
                "path": ref.get("path"),
                "format": ref.get("format"),
                "bytes": ref.get("bytes"),
            }
    return None


def _summarize_storage_refs(storage_refs: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, value in storage_refs.items():
        if isinstance(value, Mapping) and "path" in value:
            summary[str(name)] = {
                "store": value.get("store"),
                "path": value.get("path"),
                "format": value.get("format"),
                "bytes": value.get("bytes"),
            }
            continue
        if isinstance(value, Mapping):
            items: dict[str, Any] = {}
            for item_name, item_value in value.items():
                if isinstance(item_value, Mapping) and "path" in item_value:
                    items[str(item_name)] = {
                        "store": item_value.get("store"),
                        "path": item_value.get("path"),
                        "format": item_value.get("format"),
                        "bytes": item_value.get("bytes"),
                    }
            summary[str(name)] = {
                "count": len(items),
                "items": items,
            }
    return summary


def _report_volume_mappings(value: Any, manifest: Any) -> list[dict[str, Any]]:
    mappings: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    store = getattr(value, "store", None)
    if store is not None and getattr(store, "kind", None) == "modal_volume":
        try:
            from pipelines_v2.storage.modal import modal_volume_mount_path

            volume_name = str(getattr(store, "name"))
            mount_path = str(modal_volume_mount_path(str(getattr(store, "root"))))
            key = (volume_name, mount_path)
            if key not in seen:
                seen.add(key)
                mappings.append(
                    {
                        "name": volume_name,
                        "mount_path": mount_path,
                        "role": "artifact_store",
                    }
                )
        except Exception:
            pass

    runner = getattr(manifest, "runner", {})
    resources = runner.get("resources") if isinstance(runner, Mapping) else None
    volumes = resources.get("volumes") if isinstance(resources, Mapping) else None
    if isinstance(volumes, Sequence) and not isinstance(volumes, str):
        for volume in volumes:
            if not isinstance(volume, Mapping):
                continue
            volume_name = volume.get("name")
            mount_path = volume.get("mount_path")
            if volume_name is None or mount_path is None:
                continue
            key = (str(volume_name), str(mount_path))
            if key in seen:
                continue
            seen.add(key)
            mappings.append(
                {
                    "name": str(volume_name),
                    "mount_path": str(mount_path),
                    "role": "runner_resource",
                }
            )
    return mappings


def resolve_values_map(source: Any, *, label: str) -> Mapping[str, Any]:
    if isinstance(source, (LabelSet, CaseSet, ArtifactLabelRef)):
        return source.resolve_values()
    raise SpecValidationError(f"Expected a label/case ref for {label}, got {type(source).__name__}")


def reference_example_keys(source: Any, *, label: str) -> list[str]:
    if hasattr(source, "resolve_example_keys"):
        return sorted(str(key) for key in source.resolve_example_keys())
    values = resolve_values_map(source, label=label)
    return sorted(str(key) for key in values)


def coerce_token_sections(raw: Any) -> dict[str, list[int]] | None:
    if not isinstance(raw, Mapping):
        return None
    return {
        str(name): [int(position) for position in positions]
        for name, positions in raw.items()
        if isinstance(positions, Sequence)
    }


def make_label_payload(name: str, values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "label",
        "name": name,
        "values": {str(key): value for key, value in values.items()},
    }


def label_payload_from_grouped_source(
    *,
    name: str,
    source: Any,
    grouped_example_keys: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    values = resolve_values_map(source, label=name)
    grouped_values: dict[str, Any] = {}
    for output_key, example_keys in grouped_example_keys.items():
        selected = [values[key] for key in example_keys]
        if not selected:
            raise SpecValidationError(f"PairDeltaSpec label {name!r} had no source values for {output_key!r}")
        if not all_equal(selected):
            raise SpecValidationError(
                f"PairDeltaSpec label {name!r} is not constant across the selected source examples for {output_key!r}"
            )
        grouped_values[output_key] = selected[0]
    return make_label_payload(name, grouped_values)


def all_equal(values: Sequence[Any]) -> bool:
    if not values:
        return True
    baseline = stable_hash(values[0])
    return all(stable_hash(value) == baseline for value in values[1:])


def coerce_mapping_value(value: Any, *, label: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            parsed = json.loads(stripped)
            if isinstance(parsed, Mapping):
                return parsed
    raise SpecValidationError(f"{label} must be a mapping or JSON object string")


def coerce_transform_result(value: Any) -> TransformResult:
    if isinstance(value, TransformResult):
        return value
    if not isinstance(value, Mapping):
        raise SpecValidationError(
            "Transform builder must return TransformResult or a mapping with keys like "
            "'payload', 'labels', 'metadata', and 'example_keys'"
        )
    return TransformResult(
        payload=coerce_mapping_field(value.get("payload", {}), label="TransformResult payload"),
        labels=coerce_nested_mapping_field(value.get("labels", {}), label="TransformResult labels"),
        metadata=coerce_mapping_field(value.get("metadata", {}), label="TransformResult metadata"),
        example_keys=tuple(value.get("example_keys")) if value.get("example_keys") is not None else None,
    )


def coerce_mapping_field(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpecValidationError(f"{label} must be a mapping")
    return {str(key): item for key, item in value.items()}


def coerce_nested_mapping_field(value: Any, *, label: str) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise SpecValidationError(f"{label} must be a mapping")
    normalized: dict[str, Mapping[str, Any]] = {}
    for key, item in value.items():
        if not isinstance(item, Mapping):
            raise SpecValidationError(f"{label}[{key!r}] must be a mapping")
        normalized[str(key)] = {str(inner_key): inner_value for inner_key, inner_value in item.items()}
    return normalized


def coerce_transform_label_payload(name: str, values: Mapping[str, Any]) -> dict[str, Any]:
    if "kind" in values and values.get("kind") == "label" and "values" in values:
        raw_values = values.get("values")
        if not isinstance(raw_values, Mapping):
            raise SpecValidationError(f"Transform label payload {name!r} must contain mapping 'values'")
        return make_label_payload(name, raw_values)
    return make_label_payload(name, values)


def infer_transform_example_keys(labels: Mapping[str, Mapping[str, Any]]) -> list[str] | None:
    if not labels:
        return None
    key_sets = [
        {str(key) for key in payload.get("values", {})}
        for payload in labels.values()
        if isinstance(payload, Mapping)
    ]
    if not key_sets:
        return None
    intersection = set.intersection(*(set(keys) for keys in key_sets))
    return sorted(intersection)


def report_example_keys(inputs: Sequence[Any]) -> list[str] | None:
    keys: set[str] = set()
    found_any = False
    for item in inputs:
        if not isinstance(item, (CaptureArtifact, OperationArtifact)):
            continue
        coverage = item.manifest().example_coverage
        raw_keys = coverage.get("example_keys")
        if not isinstance(raw_keys, Sequence) or isinstance(raw_keys, str):
            continue
        keys.update(str(key) for key in raw_keys)
        found_any = True
    if not found_any:
        return None
    return sorted(keys)


def collapse_matrix_by_group(
    X: NDArray[np.float32],
    example_keys: Sequence[str],
    groups: NDArray[np.object_],
    *,
    positive_keys: set[str],
    negative_keys: set[str],
) -> tuple[NDArray[np.float32], list[str]]:
    grouped: dict[str, list[int]] = {}
    for index, group in enumerate(groups.tolist()):
        grouped.setdefault(str(group), []).append(index)

    rows: list[NDArray[np.float32]] = []
    row_keys: list[str] = []
    for group_key, indices in grouped.items():
        member_keys = [example_keys[index] for index in indices]
        is_positive = any(key in positive_keys for key in member_keys)
        is_negative = any(key in negative_keys for key in member_keys)
        if is_positive and is_negative:
            raise SpecValidationError(f"DirectionSpec group {group_key!r} mixes positive and negative members")
        if not is_positive and not is_negative:
            continue
        rows.append(X[np.asarray(indices, dtype=np.int64)].mean(axis=0))
        row_keys.append(f"{'positive' if is_positive else 'negative'}::{group_key}")

    if not rows:
        return np.empty((0, X.shape[1]), dtype=np.float32), []
    return np.stack(rows, axis=0).astype(np.float32), row_keys
