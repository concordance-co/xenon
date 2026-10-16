from dataclasses import dataclass
from typing import ClassVar
from pipelines_v2.core.types import OperationSpec
from pipelines_v2.runtime.circuit_tracer import CircuitTracerRuntimeSpec
from pipelines_v2.operations.graphs.contracts import digest
from .prepare import pair_lock


@dataclass(frozen=True,slots=True)
class CircuitTraceSpec(OperationSpec):
    runtime: CircuitTracerRuntimeSpec | None = None
    model_path: str = ''
    dictionary_path: str = ''
    prompt: str = 'Hello'
    max_feature_nodes: int = 2
    max_tokens: int = 2
    timeout_seconds: int = 1000
    memory_bytes: int = 56*1024**3
    output_bytes: int = 32*1024**2
    kind: ClassVar[str] = 'circuit_trace'

    def __post_init__(self):
        if self.schema_version!=1 or not isinstance(self.runtime,CircuitTracerRuntimeSpec):raise ValueError('Circuit trace requires its isolated locked runtime')
        if not self.model_path or not self.dictionary_path or not isinstance(self.prompt,str) or not self.prompt.strip():raise ValueError('Trace needs explicit prepared assets and a bounded prompt')
        for value in (self.max_feature_nodes,self.max_tokens,self.timeout_seconds,self.memory_bytes,self.output_bytes):
            if type(value)is not int or value<=0:raise ValueError('Trace requires positive finite node/token/time/memory/output budgets')

    @property
    def request(self):return {'schema_version':1,'pair':pair_lock(),'tool_identity':self.runtime.tool_identity,'prompt':self.prompt,
        'max_feature_nodes':self.max_feature_nodes,'max_tokens':self.max_tokens,'batch_size':1,'max_n_logits':1,'dtype':'float32','backend':'transformerlens','seed':0,'tf32':False}
    def verify_inputs(self):
        from .prepare import verify_files
        for role,path in [('model',self.model_path),('dictionary',self.dictionary_path)]:
            verify_files(path,pair_lock()[role]['files'])

    def runtime_spec(self):return self.runtime
    def semantic_dict(self):return {'kind':self.kind,'schema_version':1,'request':self.request}
    @classmethod
    def from_dict(cls,payload):
        p=dict(payload);p.pop('kind',None);p['runtime']=CircuitTracerRuntimeSpec.from_dict(p['runtime']);return cls(**p)
