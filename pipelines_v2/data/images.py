"""Versioned image inputs and spatial/trajectory coordinates (C08).

Preparation reads bounded local files. Execution verifies prepared bytes at their
local or mounted-volume location; it never downloads a remote reference.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from pipelines_v2.core.types import stable_hash


def _digest(value: str, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{field} must be a lowercase SHA256 digest')


def _positive_int(value: int, field: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f'{field} must be a positive integer')


def _json_copy(value: Any) -> Any:
    def plain(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {key: plain(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [plain(val) for val in item]
        if isinstance(item, ImageContract):
            return item.to_dict()
        return item
    return json.loads(json.dumps(plain(value), allow_nan=False, sort_keys=True))


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(val) for key, val in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(val) for val in value)
    return value


class ImageContract:
    schema_version: ClassVar[int] = 1

    def to_dict(self) -> dict[str, Any]:
        return {'schema_version': self.schema_version, **_json_copy({f.name: getattr(self, f.name) for f in fields(self)})}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Any:
        value = dict(payload)
        version = value.pop('schema_version', None)
        if type(version) is not int or version != 1:
            raise ValueError(f'Unsupported {cls.__name__} schema_version: {version!r}')
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown {cls.__name__} fields: {sorted(unknown)}')
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ImageInput(ImageContract):
    image_id: str
    content_digest: str
    store_ref: Mapping[str, Any]
    media_type: str
    shape: tuple[int, ...]
    color_space: str
    preprocessing_digest: str
    layout: str = 'HWC'
    value_range: tuple[float, float] = (0, 255)
    dtype: str = 'uint8'
    schema_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.image_id, str) or not self.image_id.strip():
            raise ValueError('image_id must be nonempty')
        _digest(self.content_digest, 'content_digest')
        _digest(self.preprocessing_digest, 'preprocessing_digest')
        object.__setattr__(self, 'shape', tuple(self.shape))
        object.__setattr__(self, 'value_range', tuple(self.value_range))
        object.__setattr__(self, 'store_ref', _freeze(_json_copy(dict(self.store_ref))))
        if self.media_type not in {'image/png', 'image/jpeg'}:
            raise ValueError('Only prepared PNG and JPEG images are supported')
        if self.layout != 'HWC' or self.dtype != 'uint8' or self.value_range != (0, 255):
            raise ValueError('Prepared decoded image contract is HWC uint8 in [0,255]')
        channels = {'L': 1, 'RGB': 3, 'RGBA': 4}.get(self.color_space)
        if len(self.shape) != 3 or channels != self.shape[2]:
            raise ValueError('Image shape and color space disagree')
        for dim in self.shape:
            _positive_int(dim, 'image dimension')
        ref = self.store_ref
        if ref.get('store') not in {'local', 'modal_volume'}:
            raise ValueError('Image ref must name a local file or mounted Modal volume')
        if not isinstance(ref.get('path'), str) or not Path(ref['path']).is_absolute():
            raise ValueError('Image ref requires an absolute execution path')
        if ref['store'] == 'modal_volume' and not ref.get('name'):
            raise ValueError('Modal image ref requires a volume name')
        _positive_int(ref.get('bytes'), 'encoded image bytes')

    def semantic_dict(self) -> dict[str, Any]:
        """Locations and display IDs do not change scientific input identity."""
        result = self.to_dict()
        result.pop('image_id')
        result.pop('store_ref')
        return result

    def verify_at_execution(self, *, roots: tuple[Path, ...], max_bytes: int = 16 * 1024 * 1024,
                            max_pixels: int = 16_777_216) -> bytes:
        path = Path(self.store_ref['path']).resolve()
        if not roots or not any(path.is_relative_to(Path(root).resolve()) for root in roots):
            raise ValueError('Image reference lies outside allowed local/mounted roots')
        raw = _read_bounded(path, max_bytes)
        if len(raw) != self.store_ref['bytes'] or hashlib.sha256(raw).hexdigest() != self.content_digest:
            raise ValueError('Prepared image content changed; prepare and identify it again')
        media, shape, color = _decode_description(raw, max_pixels)
        if (media, shape, color) != (self.media_type, self.shape, self.color_space):
            raise ValueError('Decoded image does not match its descriptor')
        return raw


def _read_bounded(path: Path, max_bytes: int) -> bytes:
    _positive_int(max_bytes, 'max_bytes')
    with path.open('rb') as stream:
        raw = stream.read(max_bytes + 1)
    if not raw or len(raw) > max_bytes:
        raise ValueError('Encoded image exceeds its byte budget or is empty')
    return raw


def _decode_description(raw: bytes, max_pixels: int) -> tuple[str, tuple[int, ...], str]:
    _positive_int(max_pixels, 'max_pixels')
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Image preparation/validation requires Pillow; install the vision extra with uv sync --extra vision") from exc
    with Image.open(io.BytesIO(raw)) as image:
        if image.format not in {'PNG', 'JPEG'} or image.mode not in {'L', 'RGB', 'RGBA'}:
            raise ValueError('Prepare a non-animated L/RGB/RGBA PNG or JPEG first')
        if getattr(image, 'n_frames', 1) != 1 or image.width * image.height > max_pixels:
            raise ValueError('Image frame/pixel budget exceeded')
        # Avoid silently choosing adapter-specific orientation or normalization.
        if image.getexif().get(274, 1) != 1:
            raise ValueError('Apply EXIF orientation explicitly during preparation')
        image.load()
        return ('image/png' if image.format == 'PNG' else 'image/jpeg',
                (image.height, image.width, len(image.getbands())), image.mode)


def prepare_image(path: str | Path, *, cache_root: str | Path, preprocessing: Mapping[str, Any],
                  image_id: str | None = None, max_bytes: int = 16 * 1024 * 1024,
                  max_pixels: int = 16_777_216) -> ImageInput:
    """Prepare a local file into a content-addressed cache; no URL fetching."""
    if '://' in str(path):
        raise ValueError('Download images in an explicit preparation step; URLs are not execution inputs')
    if not preprocessing:
        raise ValueError('Declare preprocessing, including an explicit identity recipe if appropriate')
    raw = _read_bounded(Path(path), max_bytes)
    media, shape, color = _decode_description(raw, max_pixels)
    digest = hashlib.sha256(raw).hexdigest()
    root = Path(cache_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / (digest + ('.png' if media == 'image/png' else '.jpg'))
    if target.is_symlink():
        raise ValueError('Prepared cache target cannot be a symlink')
    descriptor = ImageInput(image_id or digest[:24], digest,
        {'store': 'local', 'path': str(target), 'bytes': len(raw)}, media, shape, color,
        stable_hash(_json_copy(dict(preprocessing))))
    fd, temp = tempfile.mkstemp(prefix='.image-', dir=root)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
    finally:
        Path(temp).unlink(missing_ok=True)
    return descriptor


def execution_input_identity(example: Any) -> str:
    """Discriminated multimodal identity; retain historical text shard identity."""
    images = tuple(getattr(example, 'images', ()))
    if not images:
        return str(example.prompt_hash)
    return stable_hash({'kind': 'image_input_v1', 'text': example.prompt,
                        'images': [image.semantic_dict() for image in images]})


def execution_shard_index(example: Any, count: int) -> int:
    _positive_int(count, 'shard count')
    identity = execution_input_identity(example)
    try:
        value = int(identity[:16], 16)
    except ValueError:
        value = int(stable_hash(identity)[:16], 16)
    return value % count


@dataclass(frozen=True, slots=True)
class SpatialSelector(ImageContract):
    coordinate_space: str
    shape_hw: tuple[int, int]
    interpolation: str = 'nearest'
    axis_order: tuple[str, str] = ('y', 'x')
    channel_axis: int | None = None
    schema_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'shape_hw', tuple(self.shape_hw))
        object.__setattr__(self, 'axis_order', tuple(self.axis_order))
        if self.coordinate_space not in {'input', 'preprocessed', 'latent', 'feature'}:
            raise ValueError('Unknown spatial coordinate space')
        if self.axis_order != ('y', 'x') or len(self.shape_hw) != 2:
            raise ValueError('Spatial masks use explicit (y,x) axes')
        for size in self.shape_hw:
            _positive_int(size, 'spatial dimension')
        if self.interpolation not in {'nearest', 'bilinear'}:
            raise ValueError('Mask interpolation must be nearest or bilinear')
        if self.channel_axis is not None and (type(self.channel_axis) is not int or self.channel_axis < 0):
            raise ValueError('channel_axis must be a nonnegative tensor axis or None')

    def validate_mask(self, mask: Any) -> None:
        import numpy as np
        if not isinstance(mask, np.ndarray) or mask.shape != self.shape_hw or mask.dtype.kind not in 'buf':
            raise ValueError('Mask must be a numeric/bool ndarray matching its spatial shape')
        if not np.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
            raise ValueError('Mask values must be finite in [0,1]')


@dataclass(frozen=True, slots=True)
class SpatialMapping(ImageContract):
    """Pixel-center affine map: target=(source+0.5)*scale-0.5+offset."""
    source: SpatialSelector
    target: SpatialSelector
    scale_yx: tuple[float, float]
    offset_yx: tuple[float, float] = (0.0, 0.0)
    schema_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        for name in ('source', 'target'):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                object.__setattr__(self, name, SpatialSelector.from_dict(value))
            elif not isinstance(value, SpatialSelector):
                raise TypeError(f'{name} must be a SpatialSelector')
        for name in ('scale_yx', 'offset_yx'):
            value = tuple(getattr(self, name))
            if len(value) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
                raise ValueError('Spatial affine values must be two finite numbers')
            object.__setattr__(self, name, value)
        if min(self.scale_yx) <= 0:
            raise ValueError('Spatial scales must be positive')

    def to_dict(self) -> dict[str, Any]:
        return {'schema_version': 1, 'source': self.source.to_dict(), 'target': self.target.to_dict(),
                'scale_yx': list(self.scale_yx), 'offset_yx': list(self.offset_yx)}

    def map_points(self, points_yx: Any) -> Any:
        import numpy as np
        points = np.asarray(points_yx, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 2 or not np.isfinite(points).all():
            raise ValueError('Points must have a final (y,x) axis and finite coordinates')
        return (points + .5) * self.scale_yx - .5 + self.offset_yx

    def map_mask(self, mask: Any) -> Any:
        """Resample with explicit zero padding; never infer a coordinate transform."""
        import numpy as np
        self.source.validate_mask(mask)
        y, x = np.indices(self.target.shape_hw, dtype=float)
        sy = (y - self.offset_yx[0] + .5) / self.scale_yx[0] - .5
        sx = (x - self.offset_yx[1] + .5) / self.scale_yx[1] - .5
        def sample(iy: Any, ix: Any) -> Any:
            inside = (iy >= 0) & (ix >= 0) & (iy < mask.shape[0]) & (ix < mask.shape[1])
            return np.where(inside, mask[np.clip(iy, 0, mask.shape[0]-1), np.clip(ix, 0, mask.shape[1]-1)], 0)
        if self.target.interpolation == 'nearest':
            result = sample(np.floor(sy+.5).astype(int), np.floor(sx+.5).astype(int))
        else:
            iy, ix = np.floor(sy).astype(int), np.floor(sx).astype(int)
            dy, dx = sy-iy, sx-ix
            result = ((1-dy)*(1-dx)*sample(iy,ix) + dy*(1-dx)*sample(iy+1,ix)
                      + (1-dy)*dx*sample(iy,ix+1) + dy*dx*sample(iy+1,ix+1))
        self.target.validate_mask(result)
        return result


@dataclass(frozen=True, slots=True)
class DiffusionCoordinates(ImageContract):
    prompt: str
    negative_prompt: str
    initial_noise_digest: str
    initial_noise_ref: Mapping[str, Any]
    generator_state_ref: Mapping[str, Any]
    scheduler_name: str
    scheduler_config_digest: str
    step_index: int
    timestep_value: float
    denoising_window: tuple[int, int]
    cfg_enabled: bool
    branch: str
    guidance_scale: float
    schema_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        _digest(self.initial_noise_digest, 'initial_noise_digest')
        _digest(self.scheduler_config_digest, 'scheduler_config_digest')
        object.__setattr__(self, 'denoising_window', tuple(self.denoising_window))
        for name in ('initial_noise_ref', 'generator_state_ref'):
            ref = _json_copy(dict(getattr(self, name)))
            if not ref or not ref.get('path') or ref.get('store') not in {'local', 'modal_volume'}:
                raise ValueError(f'{name} requires an explicit payload reference')
            if not Path(str(ref['path'])).is_absolute():
                raise ValueError(f'{name} requires an absolute execution path')
            _positive_int(ref.get('bytes'), f'{name}.bytes')
            if ref['store'] == 'modal_volume' and not ref.get('name'):
                raise ValueError(f'{name} requires a Modal volume name')
            object.__setattr__(self, name, _freeze(ref))
        if not isinstance(self.prompt, str) or not isinstance(self.negative_prompt, str) or not self.scheduler_name:
            raise ValueError('Prompts must be strings and scheduler name must be explicit')
        if (type(self.step_index) is not int or self.step_index < 0 or len(self.denoising_window) != 2
            or any(type(v) is not int or v < 0 for v in self.denoising_window)
            or not self.denoising_window[0] <= self.step_index < self.denoising_window[1]):
            raise ValueError('step_index must be inside the half-open denoising window')
        if type(self.timestep_value) not in (int, float) or not math.isfinite(self.timestep_value):
            raise ValueError('Scheduler timestep value must be finite (distinct from step index)')
        if type(self.guidance_scale) not in (int, float) or not math.isfinite(self.guidance_scale) or self.guidance_scale < 0:
            raise ValueError('guidance_scale must be finite and nonnegative')
        if type(self.cfg_enabled) is not bool or self.branch not in {'conditional', 'unconditional', 'both'}:
            raise ValueError('CFG requires an explicit boolean and branch enum')
        if not self.cfg_enabled and self.branch != 'conditional':
            raise ValueError('CFG-disabled trajectories have only a conditional branch')
