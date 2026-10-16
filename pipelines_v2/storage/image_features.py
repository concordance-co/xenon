"""Validated image/latent tensor codecs; full arrays never expand into JSON."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from pipelines_v2.data.images import (
    DiffusionCoordinates, ImageContract, _digest, _freeze, _json_copy, _positive_int,
)

_MAX_TENSOR_BYTES = 256 * 1024 * 1024
_DTYPES = {'uint8', 'float16', 'float32', 'float64'}


def tensor_content_digest(values: np.ndarray) -> str:
    if not isinstance(values, np.ndarray) or values.dtype.name not in _DTYPES or not values.dtype.isnative:
        raise ValueError('Image/latent tensors require a supported native numeric ndarray')
    if values.nbytes > _MAX_TENSOR_BYTES:
        raise ValueError('Image/latent tensor exceeds the 256 MiB codec unit limit; partition it')
    descriptor = json.dumps({'dtype': values.dtype.name, 'shape': list(values.shape)}, sort_keys=True).encode()
    digest = hashlib.sha256(descriptor + b'\0')
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SpatialTensor(ImageContract):
    kind: str
    content_digest: str
    shape: tuple[int, ...]
    axes: tuple[str, ...]
    dtype: str
    preprocessing_digest: str
    origin_ref: Mapping[str, Any]
    component_digest: str | None = None
    color_space: str | None = None
    value_range: tuple[float, float] | None = None
    trajectory: DiffusionCoordinates | None = None
    schema_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if self.kind not in {'image', 'latent'}:
            raise ValueError('SpatialTensor kind must be image or latent')
        _digest(self.content_digest, 'content_digest')
        _digest(self.preprocessing_digest, 'preprocessing_digest')
        object.__setattr__(self, 'shape', tuple(self.shape))
        object.__setattr__(self, 'axes', tuple(self.axes))
        object.__setattr__(self, 'origin_ref', _freeze(_json_copy(dict(self.origin_ref))))
        if len(self.shape) != len(self.axes) or not self.shape or len(set(self.axes)) != len(self.axes):
            raise ValueError('Tensor axes and shape must align with unique axis names')
        for dim in self.shape:
            _positive_int(dim, 'tensor dimension')
        if self.dtype not in _DTYPES:
            raise ValueError('Unsupported tensor dtype')
        import math
        if math.prod(self.shape) * np.dtype(self.dtype).itemsize > _MAX_TENSOR_BYTES:
            raise ValueError('Declared tensor exceeds 256 MiB codec unit limit')
        if not self.origin_ref.get('content_digest') and not (
            self.origin_ref.get('artifact_id') and self.origin_ref.get('scientific_identity')
        ):
            raise ValueError('Origin requires content digest or artifact ID plus scientific identity')
        for key in ('content_digest', 'scientific_identity'):
            if self.origin_ref.get(key):
                _digest(self.origin_ref[key], f'origin_ref.{key}')
        if self.component_digest is not None:
            _digest(self.component_digest, 'component_digest')
        if isinstance(self.trajectory, Mapping):
            object.__setattr__(self, 'trajectory', DiffusionCoordinates.from_dict(self.trajectory))
        elif self.trajectory is not None and not isinstance(self.trajectory, DiffusionCoordinates):
            raise TypeError('trajectory must be DiffusionCoordinates')
        if self.kind == 'image':
            if self.axes not in {('y', 'x', 'channel'), ('channel', 'y', 'x')}:
                raise ValueError('Images require HWC or CHW axes')
            channels = {'L': 1, 'RGB': 3, 'RGBA': 4}.get(self.color_space)
            if channels != self.shape[self.axes.index('channel')]:
                raise ValueError('Image channels disagree with color space')
            if self.trajectory is not None:
                raise ValueError('Trajectory coordinates belong on latents, not image tensors')
            if self.value_range is None or len(self.value_range) != 2:
                raise ValueError('Image value range must be explicit')
            bounds = tuple(self.value_range)
            if any(type(v) not in (int, float) or not np.isfinite(v) for v in bounds) or bounds[0] >= bounds[1]:
                raise ValueError('Image value range must be finite and increasing')
            object.__setattr__(self, 'value_range', bounds)
        else:
            if self.axes not in {('channel', 'y', 'x'), ('batch', 'channel', 'y', 'x')}:
                raise ValueError('Latents require CHW or BCHW axes')
            if self.dtype not in {'float16', 'float32', 'float64'} or self.component_digest is None:
                raise ValueError('Latents require floating dtype and immutable model component digest')
            if self.color_space is not None or self.value_range is not None:
                raise ValueError('Latents cannot inherit image color/range semantics')

    def validate(self, values: np.ndarray) -> None:
        if not isinstance(values, np.ndarray) or values.shape != self.shape or values.dtype.name != self.dtype:
            raise ValueError('Tensor shape/dtype disagrees with descriptor')
        if not np.isfinite(values).all():
            raise ValueError('Image/latent tensors must contain finite values')
        if self.value_range is not None and ((values < self.value_range[0]).any() or (values > self.value_range[1]).any()):
            raise ValueError('Image tensor lies outside declared value range')
        if tensor_content_digest(values) != self.content_digest:
            raise ValueError('Tensor content digest mismatch')

    @classmethod
    def from_array(cls, values: np.ndarray, **kwargs: Any) -> 'SpatialTensor':
        result = cls(content_digest=tensor_content_digest(values), shape=values.shape,
                     dtype=values.dtype.name, **kwargs)
        result.validate(values)
        return result


def _validate_payload(payload: dict[str, Any]) -> None:
    if set(payload) != {'kind', 'schema_version', 'records'} or type(payload['schema_version']) is not int or payload['schema_version'] != 1:
        raise ValueError('Spatial feature payload requires kind/schema_version=1/records')
    if payload['kind'] not in {'image', 'latent'} or not isinstance(payload['records'], dict):
        raise ValueError('Invalid spatial feature kind or records')
    total_bytes = 0
    for key, record in payload['records'].items():
        if not isinstance(key, str) or not key or not isinstance(record, dict) or set(record) != {'descriptor', 'values'}:
            raise ValueError('Each spatial record requires a stable key, descriptor and tensor values')
        descriptor = SpatialTensor.from_dict(record['descriptor'])
        if descriptor.kind != payload['kind']:
            raise ValueError('Mixed image/latent records are invalid')
        descriptor.validate(record['values'])
        total_bytes += record['values'].nbytes
    if total_bytes > _MAX_TENSOR_BYTES:
        raise ValueError('Spatial feature bundle exceeds 256 MiB; partition it')


def encode_spatial_payload(payload: dict[str, Any], *, tensor_key_prefix: str = '') -> Any:
    from pipelines_v2.storage.features import encode_tensor_feature_payload
    _validate_payload(payload)
    return encode_tensor_feature_payload(payload, tensor_key_prefix=tensor_key_prefix)


def decode_spatial_payload(metadata: dict[str, Any], tensors: dict[str, Any]) -> dict[str, Any]:
    from pipelines_v2.storage.features import decode_feature_payload
    payload = decode_feature_payload(metadata, tensors)
    _validate_payload(payload)
    return payload


def merge_spatial_payloads(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    _validate_payload(left)
    _validate_payload(right)
    if left['kind'] != right['kind'] or set(left['records']) & set(right['records']):
        raise ValueError('Cannot merge mixed kinds or duplicate spatial record keys')
    result = {**left, 'records': {**left['records'], **right['records']}}
    if sum(record['values'].nbytes for record in result['records'].values()) > _MAX_TENSOR_BYTES:
        raise ValueError('Spatial feature bundle exceeds 256 MiB; partition it')
    # Both inputs have been validated; their disjoint union preserves descriptors.
    return result


def spatial_summary(payload: dict[str, Any], *, max_records: int = 16, max_bytes: int = 16_384) -> dict[str, Any]:
    """Metadata-only summary from an already available payload; never reads a store."""
    _positive_int(max_records, 'max_records')
    _positive_int(max_bytes, 'max_bytes')
    _validate_payload(payload)
    records = payload['records']
    summary = {'kind': payload['kind'], 'record_count': len(records), 'records': {
        key: {name: record['descriptor'][name] for name in ('content_digest', 'shape', 'axes', 'dtype', 'origin_ref')}
        for key, record in list(records.items())[:max_records]
    }}
    if len(json.dumps(summary, allow_nan=False).encode()) > max_bytes:
        raise ValueError('Spatial summary exceeds its independent byte budget')
    return summary
