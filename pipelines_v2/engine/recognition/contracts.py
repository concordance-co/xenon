"""Pinned ViT recognition coordinates; distinct from diffusion/image generation."""
from dataclasses import dataclass, asdict
from pipelines_v2.core.types import stable_hash

REVISION = '3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3'
REPOSITORY = 'google/vit-base-patch16-224'
ADAPTER = 'vit-base-patch16-recognition-v1'
FILES = {'config.json': 'abdf73cc4ce79495a83f32a8c9265daa7a1363f5a3efa4704c88a6e7f45ad19d',
    'model.safetensors': '1cea07110a4a47edc51420b2dda6f3b8b58e7256e8f44b4ea6aa9696162ccb5d',
    'preprocessor_config.json': '90d1ae427c27d2b9dcb01f6a62ba96aa67aac2de1697c59cd212055b5035c4a2'}
PREPROCESSING = {'adapter': ADAPTER, 'processor_sha256': FILES['preprocessor_config.json'],
    'color': 'RGB-only-no-conversion', 'resize_hw': [224, 224], 'interpolation': 'Pillow bilinear',
    'crop': None, 'rescale': 1 / 255, 'mean': [.5, .5, .5], 'std': [.5, .5, .5], 'output': 'NCHW float32'}
MODEL_IDENTITY = stable_hash(FILES)


def token_geometry(token, input_hw):
    """Half-open edge boxes; nominal patch origin, not a post-attention receptive field."""
    if type(token) is not int or not 0 <= token < 197 or len(input_hw) != 2 or any(type(n) is not int or n < 1 for n in input_hw):
        raise ValueError('Invalid ViT token/input shape')
    if token == 0:
        return {'token': 0, 'kind': 'class', 'patch_yx': None, 'preprocessed_xyxy': None, 'input_xyxy': None}
    y, x = divmod(token - 1, 14)
    h, w = input_hw
    box = [x * 16, y * 16, (x + 1) * 16, (y + 1) * 16]
    return {'token': token, 'kind': 'patch', 'patch_yx': [y, x], 'preprocessed_xyxy': box,
        'input_xyxy': [box[0] * w / 224, box[1] * h / 224, box[2] * w / 224, box[3] * h / 224],
        'mapping': 'direct resize edge coordinates; bilinear samples use half-pixel centers',
        'scope': 'patch origin only; attention mixes global image context'}


@dataclass(frozen=True)
class ViTSite:
    tokens: tuple[int, ...]
    channels: tuple[int, ...]
    layer: int = 0
    model_identity: str = MODEL_IDENTITY
    adapter_version: str = ADAPTER
    quantity: str = 'residual_post_block'
    axes: tuple[str, ...] = ('batch', 'token', 'channel')
    read_or_write: str = 'read'
    phase: str = 'image'
    dtype: str = 'float32'
    schema_version: int = 1
    kind: str = 'vit_site'

    def __post_init__(self):
        for name, upper, ceiling in (('tokens', 197, 16), ('channels', 768, 16)):
            values = tuple(getattr(self, name)); object.__setattr__(self, name, values)
            if not 1 <= len(values) <= ceiling or len(set(values)) != len(values) or any(type(v) is not int or not 0 <= v < upper for v in values):
                raise ValueError(f'Explicit unique bounded {name} required')
        object.__setattr__(self, 'axes', tuple(self.axes))
        if type(self.layer) is not int or not 0 <= self.layer < 12:
            raise ValueError('ViT layer out of range')
        if (self.model_identity, self.adapter_version, self.quantity, self.axes, self.phase, self.dtype, self.schema_version, self.kind) != (MODEL_IDENTITY, ADAPTER, 'residual_post_block', ('batch','token','channel'), 'image', 'float32', 1, 'vit_site') or self.read_or_write not in ('read', 'write'):
            raise ValueError('Unsupported ViT coordinate identity/schema')

    @property
    def module_path(self): return f'vit.encoder.layer.{self.layer}'
    def to_dict(self): return dict(asdict(self), module_path=self.module_path)
    @classmethod
    def from_dict(cls, payload):
        p = dict(payload); path = p.pop('module_path', None); value = cls(**p)
        if path is not None and path != value.module_path: raise ValueError('ViT module path disagrees with layer')
        return value
