"""Versioned coordinates for explicitly supported native tensors (C04)."""
from dataclasses import dataclass, field
from typing import ClassVar
from pipelines_v2.core.types import EngineCapability, SpecValidationError
from pipelines_v2.operations.common.tokens import TokenSelector
from pipelines_v2.operations.common.schemas import TensorStorage

QWEN_VERSION = 'qwen3-dense-native-v1'
GPT2_VERSION = 'gpt2-head-native-v1'
MLP_QUANTITIES = ('gate_preactivation', 'gate_activation', 'up_projection', 'mlp_product')
HEAD_QUANTITY = 'attention_head_output'


@dataclass(frozen=True, slots=True)
class NativeSelection:
    tokens: TokenSelector = field(default_factory=TokenSelector.last)
    neurons: tuple[int, ...] = ()
    heads: tuple[int, ...] = ()

    def __post_init__(self):
        if self.tokens.kind not in ('last','slice','full_sequence'):
            raise SpecValidationError('Native v1 supports last/slice/full_sequence prompt tokens')
        for name in ('neurons', 'heads'):
            values = tuple(getattr(self, name))
            if len(set(values)) != len(values) or any(type(v) is not int or v < 0 for v in values):
                raise SpecValidationError('Native coordinates must be unique nonnegative integers')
            object.__setattr__(self, name, values)
        if bool(self.neurons) == bool(self.heads):
            raise SpecValidationError('Select a nonempty explicit neuron OR head set')

    @classmethod
    def from_dict(cls, p):
        return cls(tokens=TokenSelector.from_dict(p['tokens']), neurons=tuple(p.get('neurons', ())), heads=tuple(p.get('heads', ())))


@dataclass(frozen=True, slots=True)
class NativeSite:
    name: str
    model_identity: str
    adapter_version: str
    module_path: str
    layer: int
    quantity: str
    selection: NativeSelection
    axes: tuple[str, ...]
    read_or_write: str = 'read'
    phase: str = 'prompt'
    dtype: str = 'float32'
    schema_version: int = 1
    kind: ClassVar[str] = 'native_site'

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise SpecValidationError('Unsupported native site version')
        if not self.name or not self.model_identity or type(self.layer) is not int or self.layer < 0:
            raise SpecValidationError('Native sites require identity, name and nonnegative layer')
        head = self.quantity == HEAD_QUANTITY
        if not head and self.quantity not in MLP_QUANTITIES:
            raise SpecValidationError('Unsupported native quantity')
        expected = ('batch', 'token', 'head', 'head_dim') if head else ('batch', 'token', 'neuron')
        if tuple(self.axes) != expected or bool(self.selection.heads) != head:
            raise SpecValidationError('Native quantity, axes and selector disagree')
        version = GPT2_VERSION if head else QWEN_VERSION
        module = f'transformer.h.{self.layer}.attn.c_proj' if head else f'model.layers.{self.layer}.mlp'
        if self.adapter_version != version or self.module_path != module:
            raise SpecValidationError('Native adapter/module coordinate mismatch')
        if self.read_or_write not in ('read', 'write') or self.phase != 'prompt' or self.dtype != 'float32':
            raise SpecValidationError('Native v1 supports prompt FP32 read/write only')
        object.__setattr__(self, 'axes', expected)

    @property
    def layers(self): return (self.layer,)
    @property
    def tokens(self): return self.selection.tokens
    @property
    def storage(self): return TensorStorage(dtype=self.dtype)

    def required_capabilities(self):
        return {EngineCapability.NATIVE_HEAD_CAPTURE if self.quantity == HEAD_QUANTITY else EngineCapability.NATIVE_MLP_CAPTURE}

    @classmethod
    def from_dict(cls, p):
        data = dict(p); data.pop('kind', None)
        data['selection'] = NativeSelection.from_dict(data['selection'])
        data['axes'] = tuple(data['axes'])
        return cls(**data)

    def empty_payload(self):
        from pipelines_v2.core.types import to_primitive
        return {'kind':'native_tensor', 'coordinate':to_primitive(self), 'storage':{'dtype':self.dtype,'format':'safetensors'},
                'layers':{str(self.layer):{}}}


@dataclass(frozen=True, slots=True)
class ExpertLocalCoordinates:
    """Observed routed rows only. This schema grants no real MoE capability."""
    expert: int
    token_indices: tuple[int, ...]
    mixture_weights: tuple[float, ...]
    token_count: int
    schema_version: int = 1

    def __post_init__(self):
        import math
        object.__setattr__(self, 'token_indices', tuple(self.token_indices))
        object.__setattr__(self, 'mixture_weights', tuple(self.mixture_weights))
        if type(self.schema_version) is not int or self.schema_version != 1 or type(self.expert) is not int or self.expert < 0 or type(self.token_count) is not int or self.token_count < 0:
            raise SpecValidationError('Invalid expert-local coordinate identity')
        if len(self.token_indices) != len(self.mixture_weights) or len(set(self.token_indices)) != len(self.token_indices):
            raise SpecValidationError('Routed token/weight alignment mismatch')
        if any(type(t) is not int or not 0 <= t < self.token_count for t in self.token_indices):
            raise SpecValidationError('Routed token outside declared coverage')
        if any(type(w) not in (int,float) or not math.isfinite(w) or not 0 < w <= 1 for w in self.mixture_weights):
            raise SpecValidationError('Only positive observed mixture weights are valid')

    @classmethod
    def from_dict(cls,p):
        return cls(expert=p['expert'],token_indices=tuple(p['token_indices']),mixture_weights=tuple(p['mixture_weights']),token_count=p['token_count'],schema_version=p.get('schema_version',1))
