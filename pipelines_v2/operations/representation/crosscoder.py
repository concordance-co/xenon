"""Small aligned crosscoder capture, checkpoint fit/import, encoding and edits."""
from dataclasses import dataclass,field
from typing import Any,ClassVar
from pipelines_v2.data.datasets import Dataset
from pipelines_v2.representation.crosscoder.settings import Execution,Limits
from pipelines_v2.core.types import OperationSpec
from pipelines_v2.operations.common._shared import spec_value_from_dict
from pipelines_v2.representation.crosscoder.fitting import FitConfig
from pipelines_v2.representation.crosscoder.reference import runtime_spec


@dataclass(frozen=True, kw_only=True)
class _CrosscoderSpec(OperationSpec):
    execution:dict=field(default_factory=lambda:vars(Execution()))
    limits:dict=field(default_factory=lambda:vars(Limits()))
    def _validate_version(self):
        object.__setattr__(self,'execution',vars(Execution(**self.execution)))
        object.__setattr__(self,'limits',vars(Limits(**self.limits)))
        if type(self.schema_version)is not int or self.schema_version!=1:raise ValueError("Unsupported crosscoder schema")
    def runtime_spec(self):return runtime_spec(Execution(**self.execution).runtime_profile)
    @classmethod
    def from_dict(cls,payload):
        values={k:spec_value_from_dict(v) for k,v in payload.items() if k!='kind'}
        return cls(**values)


@dataclass(frozen=True)
class CrosscoderPairCaptureSpec(_CrosscoderSpec):
    model_paths:tuple[str,str]=('','')
    prompts:tuple[str,...]=()
    dataset:Dataset|None=None
    train_keys:tuple[str,...]=()
    layer:int=12
    train_count:int=3
    kind:ClassVar[str]='crosscoder_pair_capture'
    def runtime_secrets(self):return self.dataset.runtime_secrets() if self.dataset is not None else ()
    def __post_init__(self):
        self._validate_version()
        object.__setattr__(self,'model_paths',tuple(self.model_paths));object.__setattr__(self,'prompts',tuple(self.prompts))
        if type(self.schema_version)is not int or self.schema_version!=1:raise ValueError('Unsupported crosscoder schema')
        if len(self.model_paths)!=2 or any(not p for p in self.model_paths):raise ValueError('Prepared ordered base/instruct paths required')
        object.__setattr__(self,'train_keys',tuple(self.train_keys))
        if self.dataset is not None and (not isinstance(self.dataset,Dataset) or self.prompts):raise ValueError('Supply Dataset or prompts, not both')
        if self.dataset is None and (len(self.prompts)<2 or any(not isinstance(p,str) or not p.strip() for p in self.prompts)):raise ValueError('At least two nonempty raw prompts required')
        count=len(self.prompts) if self.dataset is None else (None if self.dataset.is_deferred else len(self.dataset.examples))
        if count is not None and count>Limits(**self.limits).max_examples:raise ValueError('Example budget exceeded')
        if type(self.layer)is not int or not 0<=self.layer<24 or type(self.train_count)is not int or self.train_count<1 or (not self.train_keys and count is not None and self.train_count>=count):raise ValueError('Explicit valid layer and disjoint prompt split required')


@dataclass(frozen=True)
class CrosscoderFitSpec(_CrosscoderSpec):
    pairs:Any=None
    reference_path:str=''
    config:dict=field(default_factory=lambda:vars(FitConfig()))
    resume_from:Any=None
    stop_after:int|None=None
    kind:ClassVar[str]='crosscoder_fit'
    def __post_init__(self):
        self._validate_version()
        object.__setattr__(self,'config',vars(FitConfig(**self.config)))
        if self.pairs is None or not self.reference_path:raise ValueError('Paired capture and prepared pinned reference required')
        if self.stop_after is not None and (type(self.stop_after)is not int or not 0<self.stop_after<=self.config['steps']):raise ValueError('Invalid stop_after')


@dataclass(frozen=True)
class CrosscoderImportSpec(_CrosscoderSpec):
    checkpoint:Any=None
    reference_path:str=''
    kind:ClassVar[str]='crosscoder_import'
    def __post_init__(self):
        self._validate_version()
        if self.checkpoint is None or not self.reference_path:raise ValueError('Completed safe checkpoint and pinned reference required')


@dataclass(frozen=True)
class CrosscoderEncodeSpec(_CrosscoderSpec):
    dictionary:Any=None
    feature:Any=None
    reference_path:str=''
    kind:ClassVar[str]='crosscoder_encode'
    def __post_init__(self):
        self._validate_version()
        if self.dictionary is None or self.feature is None or not self.reference_path:raise ValueError('Dictionary, feature and pinned reference required')


@dataclass(frozen=True)
class CrosscoderDecodeSpec(CrosscoderEncodeSpec):
    kind:ClassVar[str]='crosscoder_decode'


@dataclass(frozen=True)
class CrosscoderFeatureSpliceSpec(_CrosscoderSpec):
    dictionary:Any=None
    pairs:Any=None
    model_paths:tuple[str,str]=('','')
    reference_path:str=''
    feature_index:int=0
    amount:float=1.0
    kind:ClassVar[str]='crosscoder_feature_splice'
    def __post_init__(self):
        self._validate_version()
        import math
        object.__setattr__(self,'model_paths',tuple(self.model_paths))
        if self.dictionary is None or self.pairs is None or not self.reference_path or len(self.model_paths)!=2 or any(not p for p in self.model_paths):raise ValueError('Dictionary, pair capture and original prepared models required')
        if type(self.feature_index)is not int or self.feature_index<0 or type(self.amount)not in(int,float) or not math.isfinite(self.amount):raise ValueError('One finite feature edit required')
