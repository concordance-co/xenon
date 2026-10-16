"""Pretrained SAE import, encode/decode and original-model feature-splice specs."""
from dataclasses import dataclass, field
from typing import Any, ClassVar
from pipelines_v2.core.types import OperationSpec
from pipelines_v2.operations.common._shared import analysis_runtime_spec, spec_value_from_dict
from pipelines_v2.representation.sae.dictionary import REVISION, MODEL_REVISION, WEIGHTS_SHA256


@dataclass(frozen=True)
class SAEDictionaryImportSpec(OperationSpec):
    cache_root: str = ''
    model_path: str = ''
    coordinate_evidence: dict = field(default_factory=dict)
    revision: str = REVISION
    weights_sha256: str = WEIGHTS_SHA256
    kind: ClassVar[str] = 'sae_dictionary_import'
    def __post_init__(self):
        if not self.cache_root or not self.model_path: raise ValueError('Prepared SAE and original-model cache paths required')
        if self.revision != REVISION or self.weights_sha256 != WEIGHTS_SHA256: raise ValueError('Unsupported dictionary checkpoint')
    def runtime_spec(self): return analysis_runtime_spec()
    @classmethod
    def from_dict(cls,payload): return cls(**{k:v for k,v in payload.items() if k!='kind'})


@dataclass(frozen=True)
class SAEEncodeSpec(OperationSpec):
    dictionary: Any = None
    feature: Any = None
    kind: ClassVar[str] = 'sae_encode'
    def __post_init__(self):
        if self.dictionary is None or self.feature is None: raise ValueError('Dictionary and original-model residual feature required')
    def runtime_spec(self): return analysis_runtime_spec()
    @classmethod
    def from_dict(cls,payload): return cls(schema_version=payload.get('schema_version',1), dictionary=spec_value_from_dict(payload['dictionary']),feature=spec_value_from_dict(payload['feature']))


@dataclass(frozen=True)
class SAEDecodeSpec(SAEEncodeSpec):
    kind: ClassVar[str] = 'sae_decode'


@dataclass(frozen=True)
class SAEFeatureSpliceSpec(OperationSpec):
    dictionary: Any = None
    engine: Any = None
    prompts: tuple[str,...] = ()
    feature_edits: tuple[tuple[int,float],...] = ()
    dataset: Any = None
    max_examples: int = 10000
    max_prompt_tokens: int = 1024
    verify_oracle: bool = False
    kind: ClassVar[str] = 'sae_feature_splice'
    def __post_init__(self):
        from pipelines_v2.engine.torch.engine import TorchEngine
        import math
        object.__setattr__(self,'prompts',tuple(self.prompts))
        object.__setattr__(self,'feature_edits',tuple(tuple(item) for item in self.feature_edits))
        if self.dictionary is None or not isinstance(self.engine,TorchEngine): raise ValueError('Dictionary and original GPT2 TorchEngine required')
        if self.engine.revision!=MODEL_REVISION: raise ValueError('Dictionary requires pinned original GPT2 revision')
        from pipelines_v2.data.datasets import Dataset
        if self.dataset is not None and (not isinstance(self.dataset,Dataset) or self.prompts): raise ValueError('Provide a Dataset or prompts, never both')
        for budget in (self.max_examples,self.max_prompt_tokens):
            if type(budget) is not int or budget<=0: raise ValueError('SAE budgets must be positive integers')
        rows=self.examples()
        if not rows or len(rows)>self.max_examples or any(not isinstance(e.prompt,str) or not e.prompt.strip() or e.images for e in rows): raise ValueError('SAE requires nonempty text examples within max_examples')
        if not self.feature_edits or len({i for i,_ in self.feature_edits})!=len(self.feature_edits): raise ValueError('SAE edits require distinct features')
        if any(type(i) is not int or not 0<=i<24576 or not math.isfinite(a) for i,a in self.feature_edits): raise ValueError('Invalid feature edit')
    def examples(self):
        from pipelines_v2.data.datasets import Example
        return tuple(self.dataset.examples) if self.dataset is not None else tuple(Example(key=f'prompt_{i}',prompt=p) for i,p in enumerate(self.prompts))
    def bound_engine(self): return self.engine
    def runtime_spec(self): return self.engine.runtime_spec()
    @classmethod
    def from_dict(cls,payload):
        from pipelines_v2.engine.torch.engine import TorchEngine
        return cls(schema_version=payload.get('schema_version',1),dictionary=spec_value_from_dict(payload['dictionary']),engine=TorchEngine.from_dict(payload['engine']),prompts=tuple(payload.get('prompts',())),feature_edits=tuple(tuple(v) for v in payload['feature_edits']),dataset=spec_value_from_dict(payload['dataset']) if payload.get('dataset') is not None else None,**{k:payload[k] for k in ('max_examples','max_prompt_tokens','verify_oracle') if k in payload})
