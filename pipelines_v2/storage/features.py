"""Feature payload encoding helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_TENSOR_REF_KEY = "__tensor_key__"
_TENSOR_ROW_KEY = "__tensor_row__"


@dataclass(frozen=True, slots=True)
class FeatureCodec:
    kind: str
    storage_format: str
    encode: Callable[..., tuple[dict[str, Any], dict[str, NDArray[Any]]]]
    decode: Callable[[dict[str, Any], dict[str, NDArray[Any]]], dict[str, Any]]
    merge: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None


_FEATURE_CODECS: dict[str, FeatureCodec] = {}
_FEATURE_FORMATS: dict[str, FeatureCodec] = {}


def register_feature_codec(
    kind: str,
    *,
    storage_format: str,
    encode: Callable[..., tuple[dict[str, Any], dict[str, NDArray[Any]]]],
    decode: Callable[[dict[str, Any], dict[str, NDArray[Any]]], dict[str, Any]],
    read_aliases: tuple[str, ...] = (),
    merge: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
) -> None:
    """Register a tensor codec and all accepted wire formats together.

    Encoders receive tensor_key_prefix and must use it for bundle-wide uniqueness.
    This supports new dense quantities without allowing unknown arrays into JSON.
    """
    from pipelines_v2.core.registry import register_kind

    formats = (storage_format, *read_aliases)
    if not callable(encode) or not callable(decode) or (merge is not None and not callable(merge)):
        raise TypeError("Feature encoder, decoder and optional merge must be callable")
    if len(set(formats)) != len(formats) or any(
        not isinstance(fmt, str) or not fmt.strip() or fmt != fmt.strip() or fmt == "json" or fmt in _FEATURE_FORMATS
        for fmt in formats
    ):
        raise ValueError("Feature storage format is invalid, reserved, or already registered")
    codec = FeatureCodec(kind, storage_format, encode, decode, merge)
    register_kind(_FEATURE_CODECS, kind, codec, label="Feature")
    for fmt in formats:
        _FEATURE_FORMATS[fmt] = codec


def _contains_tensor(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def feature_storage_format(payload: dict[str, Any]) -> str:
    feature_kind = str(payload.get("kind") or "")
    codec = _FEATURE_CODECS.get(feature_kind)
    if codec is not None:
        return codec.storage_format
    storage = payload.get("storage", {})
    requested = storage.get("format") if isinstance(storage, dict) else None
    if _contains_tensor(payload) or requested not in (None, "json", "inline"):
        raise ValueError(f"Unregistered dense feature kind: {feature_kind!r}; register a tensor codec")
    return "json"


def write_capture_features(store: Any, artifact_id: str, features: dict[str, dict[str, Any]]) -> dict[str, Any]:
    storage_refs: dict[str, Any] = {}
    metadata_by_name: dict[str, dict[str, Any]] = {}
    format_by_name: dict[str, str] = {}
    has_tensors_by_name: dict[str, bool] = {}
    shared_tensors: dict[str, NDArray[Any]] = {}

    for feature_index, (name, payload) in enumerate(features.items()):
        format = feature_storage_format(payload)
        if format == "json":
            storage_refs[name] = store.write_json(artifact_id, f"features/{name}.json", payload)
            continue

        metadata, tensors = encode_feature_payload(payload, tensor_key_prefix=f"feature_{feature_index}_")
        if not tensors and payload.get("kind") in {"residual", "moe_routing"}:
            # Preserve historical empty built-in refs. Their generic decoder is identity here.
            storage_refs[name] = store.write_json(artifact_id, f"features/{name}.json", metadata)
            continue
        metadata_by_name[name] = metadata
        format_by_name[name] = format
        has_tensors_by_name[name] = bool(tensors)
        collisions = set(shared_tensors) & set(tensors)
        if collisions:
            raise ValueError(f"Feature tensor keys collide: {sorted(collisions)}")
        shared_tensors.update(tensors)

    shared_tensor_ref: dict[str, Any] | None = None
    if shared_tensors:
        shared_tensor_ref = store.write_safetensors(artifact_id, "features/feature_tensors.safetensors", shared_tensors)

    for name, metadata in metadata_by_name.items():
        metadata_ref = store.write_json(artifact_id, f"features/{name}.metadata.json", metadata)
        tensor_ref = shared_tensor_ref if has_tensors_by_name[name] else None
        tensor_bytes = tensor_ref["bytes"] if tensor_ref is not None else 0
        storage_refs[name] = {
            "store": metadata_ref["store"],
            "name": metadata_ref.get("name"),
            "format": format_by_name[name],
            "path": metadata_ref["path"],
            "metadata_path": metadata_ref["path"],
            "tensor_path": tensor_ref["path"] if tensor_ref is not None else None,
            "metadata_bytes": metadata_ref["bytes"],
            "tensor_bytes": tensor_bytes,
            "bytes": metadata_ref["bytes"] + tensor_bytes,
        }

    return storage_refs


def encode_feature_payload(
    payload: dict[str, Any],
    *,
    tensor_key_prefix: str = "",
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    format = feature_storage_format(payload)
    if format == "json":
        return payload, {}

    feature_kind = str(payload.get("kind") or "")
    metadata, tensors = _FEATURE_CODECS[feature_kind].encode(payload, tensor_key_prefix=tensor_key_prefix)
    if not isinstance(metadata, dict) or metadata.get("kind") != feature_kind:
        raise ValueError("Feature encoder returned an incompatible kind")
    if any(not isinstance(key, str) or not key.startswith(tensor_key_prefix) for key in tensors):
        raise ValueError("Feature tensor keys must honor tensor_key_prefix")
    return metadata, tensors


def encode_tensor_feature_payload(
    payload: dict[str, Any], *, tensor_key_prefix: str = "",
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Generic ndarray codec for registered dense feature families."""
    feature_kind = str(payload.get("kind") or "")
    normalized = _normalized_feature_metadata(payload)
    tensors: dict[str, NDArray[Any]] = {}
    tensor_index = 0

    def replace(node: Any, *, path: tuple[str, ...]) -> Any:
        nonlocal tensor_index

        if isinstance(node, np.ndarray):
            tensor_key = f"{tensor_key_prefix}tensor_{tensor_index}"
            tensor_index += 1
            tensors[tensor_key] = _coerce_tensor(node, feature_kind=feature_kind, path=path, payload=payload)
            return {_TENSOR_REF_KEY: tensor_key}
        if isinstance(node, np.generic):
            return node.item()
        if isinstance(node, dict):
            return {str(key): replace(value, path=path + (str(key),)) for key, value in node.items()}
        if isinstance(node, tuple):
            return [replace(value, path=path + (str(index),)) for index, value in enumerate(node)]
        if isinstance(node, list):
            return [replace(value, path=path + (str(index),)) for index, value in enumerate(node)]
        return node

    return replace(normalized, path=()), tensors


def decode_feature_payload(metadata: dict[str, Any], tensors: dict[str, NDArray[Any]]) -> dict[str, Any]:
    def restore(node: Any) -> Any:
        if isinstance(node, dict):
            if _TENSOR_REF_KEY in node:
                tensor = tensors[str(node[_TENSOR_REF_KEY])]
                if _TENSOR_ROW_KEY in node:
                    return tensor[int(node[_TENSOR_ROW_KEY])]
                if set(node) == {_TENSOR_REF_KEY}:
                    return tensor
            return {str(key): restore(value) for key, value in node.items()}
        if isinstance(node, list):
            return [restore(value) for value in node]
        return node

    payload = restore(metadata)
    if not isinstance(payload, dict):
        raise TypeError("Decoded feature payload must be a mapping")
    return payload


def load_feature_payload(store: Any, ref: dict[str, Any]) -> dict[str, Any]:
    format = ref.get("format", "json")
    if format == "json":
        logger.info("Loading JSON feature payload path=%s", ref.get("path"))
        payload = store.read_json_ref(ref)
        if not isinstance(payload, dict):
            raise TypeError("Feature payload must be a mapping")
        return payload
    if format in _FEATURE_FORMATS:
        logger.info(
            "Loading tensor-backed feature metadata format=%s metadata_path=%s tensor_path=%s tensor_bytes=%s",
            format,
            ref.get("metadata_path"),
            ref.get("tensor_path"),
            ref.get("tensor_bytes"),
        )
        metadata = store.read_json_ref(
            {
                "store": ref["store"],
                "name": ref.get("name"),
                "path": ref["metadata_path"],
                "format": "json",
                "bytes": ref.get("metadata_bytes"),
            }
        )
        if not isinstance(metadata, dict):
            raise TypeError("Feature metadata must be a mapping")
        logger.info(
            "Loaded feature metadata format=%s top_level_keys=%s",
            format,
            sorted(str(key) for key in metadata)[:12],
        )
        tensors = store.read_safetensors_ref(
            {
                "store": ref["store"],
                "name": ref.get("name"),
                "path": ref["tensor_path"],
                "format": "safetensors",
                "bytes": ref.get("tensor_bytes"),
            }
        ) if ref.get("tensor_path") is not None else {}
        logger.info(
            "Loaded feature tensor bundle format=%s tensor_count=%s tensor_keys_preview=%s",
            format,
            len(tensors),
            sorted(str(key) for key in tensors)[:8],
        )
        codec = _FEATURE_FORMATS[format]
        if metadata.get("kind") != codec.kind:
            raise ValueError("Feature metadata kind does not match its registered format")
        payload = codec.decode(metadata, tensors)
        if not isinstance(payload, dict) or payload.get("kind") != codec.kind:
            raise ValueError("Feature decoder returned an incompatible kind")
        layers = payload.get("layers") if isinstance(payload, dict) else None
        logger.info(
            "Decoded feature payload format=%s layer_count=%s",
            format,
            len(layers) if isinstance(layers, dict) else None,
        )
        return payload
    raise ValueError(f"Unsupported feature ref format: {format}")


def _encode_residual_payload(
    payload: dict[str, Any],
    *,
    tensor_key_prefix: str,
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    metadata = _normalized_feature_metadata(payload)
    layers = metadata.get("layers")
    if not isinstance(layers, dict):
        return metadata, {}

    encoded = {key: value for key, value in metadata.items() if key != "layers"}
    encoded_layers: dict[str, Any] = {}
    tensors: dict[str, NDArray[Any]] = {}

    for layer, records in layers.items():
        layer_name = str(layer)
        if not isinstance(records, dict):
            encoded_layers[layer_name] = records
            continue

        encoded_records: dict[str, Any] = {}
        grouped_values: dict[tuple[str, tuple[int, ...]], list[tuple[str, NDArray[Any]]]] = {}
        for example_key, record in records.items():
            example_name = str(example_key)
            if not isinstance(record, dict):
                encoded_records[example_name] = record
                continue

            record_metadata: dict[str, Any] = {}
            for key, value in record.items():
                if key != "values":
                    record_metadata[str(key)] = _jsonify_feature_metadata(value)
                    continue
                array = _coerce_tensor(
                    value,
                    feature_kind="residual",
                    path=("layers", layer_name, example_name, "values"),
                    payload=payload,
                )
                grouped_values.setdefault((array.dtype.str, tuple(int(dim) for dim in array.shape)), []).append(
                    (example_name, array)
                )
            encoded_records[example_name] = record_metadata

        for group_index, ((_dtype, shape), items) in enumerate(grouped_values.items()):
            tensor_key = f"{tensor_key_prefix}layer_{_safe_tensor_key_part(layer_name)}_values_{group_index}"
            tensors[tensor_key] = np.stack([array for _example_key, array in items], axis=0)
            for row_index, (example_name, _array) in enumerate(items):
                encoded_records.setdefault(example_name, {})["values"] = {
                    _TENSOR_REF_KEY: tensor_key,
                    _TENSOR_ROW_KEY: row_index,
                    "shape": list(shape),
                }

        encoded_layers[layer_name] = encoded_records

    encoded["layers"] = encoded_layers
    return encoded, tensors


def _normalized_feature_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    feature_kind = str(payload.get("kind") or "")
    if feature_kind == "residual":
        storage = dict(payload.get("storage", {}))
        storage["format"] = "safetensors"
        return {
            **payload,
            "storage": storage,
        }
    return dict(payload)


def _jsonify_feature_metadata(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonify_feature_metadata(item) for item in value]
    if isinstance(value, list):
        return [_jsonify_feature_metadata(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonify_feature_metadata(item) for key, item in value.items()}
    return value


def _safe_tensor_key_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)


def _coerce_tensor(
    value: NDArray[Any],
    *,
    feature_kind: str,
    path: tuple[str, ...],
    payload: dict[str, Any],
) -> NDArray[Any]:
    if feature_kind == "residual" and path and path[-1] == "values":
        storage = dict(payload.get("storage", {}))
        dtype_name = str(storage.get("dtype", "float16"))
        return np.asarray(value, dtype=_numpy_dtype(dtype_name))
    return np.asarray(value)


def _numpy_dtype(name: str) -> Any:
    normalized = name.lower()
    if normalized == "float16":
        return np.float16
    if normalized == "float32":
        return np.float32
    if normalized == "bfloat16":
        return np.float32
    raise ValueError(f"Unsupported tensor storage dtype: {name}")


def merge_feature_payloads(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Merge shards only when their registered codec defines the layout semantics."""
    kind = left.get("kind")
    if kind != right.get("kind"):
        raise ValueError("Cannot merge feature payloads of different kinds")
    codec = _FEATURE_CODECS.get(str(kind))
    if codec is None or codec.merge is None:
        raise ValueError(f"Feature kind {kind!r} has no registered shard merge; use one shard")
    result = codec.merge(left, right)
    if not isinstance(result, dict) or result.get("kind") != kind:
        raise ValueError("Feature merger returned an incompatible kind")
    return result


def _merge_legacy_layers(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    for layer, records in dict(right.get("layers", {})).items():
        left.setdefault("layers", {}).setdefault(str(layer), {}).update(dict(records))
    return left


# Explicit built-ins. Historical residual v1 refs remain readable.
register_feature_codec(
    "residual", storage_format="residual_safetensors_v2",
    encode=_encode_residual_payload, decode=decode_feature_payload,
    read_aliases=("residual_safetensors_v1",), merge=_merge_legacy_layers,
)
register_feature_codec(
    "moe_routing", storage_format="moe_routing_safetensors_v1",
    encode=encode_tensor_feature_payload, decode=decode_feature_payload, merge=_merge_legacy_layers,
)

# Image/latent codecs are core contracts; no image engine or Pillow import here.
from pipelines_v2.storage.image_features import (  # noqa: E402
    decode_spatial_payload, encode_spatial_payload, merge_spatial_payloads,
)
for _spatial_kind in ("image", "latent"):
    register_feature_codec(
        _spatial_kind, storage_format=f"{_spatial_kind}_safetensors_v1",
        encode=encode_spatial_payload, decode=decode_spatial_payload, merge=merge_spatial_payloads,
    )

from pipelines_v2.storage.native_features import merge_native_payloads, encode_native_payload, decode_native_payload
register_feature_codec('native_tensor', storage_format='native_tensor_safetensors_v1', encode=encode_native_payload, decode=decode_native_payload, merge=merge_native_payloads)

from pipelines_v2.representation.sae.codec import register_codec as _register_sae_codec
_register_sae_codec()

register_feature_codec('crosscoder_tensors',storage_format='crosscoder_safetensors_v1',encode=encode_tensor_feature_payload,decode=decode_feature_payload)
