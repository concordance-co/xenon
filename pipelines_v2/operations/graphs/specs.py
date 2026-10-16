"""Small original-model head interventions with explicit pair and target contracts."""
from dataclasses import dataclass, field, asdict
from typing import ClassVar
from pipelines_v2.core.types import OperationSpec, to_primitive
from pipelines_v2.engine.native.engine import NativeTorchEngine
from pipelines_v2.operations.common.native import NativeSite, HEAD_QUANTITY


@dataclass(frozen=True)
class RoutePair:
    key: str
    template: str
    prompt: str
    correct: str
    foil: str
    opposite: str
    same_label: str
    unmatched: str
    source: dict

    def __post_init__(self):
        if not all((self.key,self.template,self.prompt,self.correct,self.foil,self.opposite,self.same_label,self.unmatched,self.source)):
            raise ValueError('Route pair requires explicit provenance and all control prompts')
        if self.correct==self.foil or self.prompt==self.opposite or self.prompt==self.unmatched:
            raise ValueError('Degenerate route pair')


@dataclass(frozen=True, slots=True)
class NativeRouteSpec(OperationSpec):
    engine: NativeTorchEngine | None = None
    sites: tuple[NativeSite, ...] = ()
    pairs: tuple[RoutePair, ...] = ()
    proposal_templates: tuple[str, ...] = ()
    proposal_reason: str = 'predeclared engineering fixture; not an inferred circuit'
    ig_steps: int = 0
    max_prompt_tokens: int = 1024
    max_work_units: int = 100000
    kind: ClassVar[str] = 'native_route'

    def __post_init__(self):
        if self.schema_version!=1 or not isinstance(self.engine,NativeTorchEngine) or self.engine.architecture!='gpt2':
            raise ValueError('Native route v1 requires the pinned GPT-2 engine')
        if not self.sites or not self.pairs:
            raise ValueError('Native route needs sites and pairs')
        if type(self.max_work_units) is not int or self.max_work_units<=0 or len(self.sites)*len(self.pairs)*(self.ig_steps+8)>self.max_work_units:
            raise ValueError('Native route exceeds explicit max_work_units budget')
        if any(s.quantity!=HEAD_QUANTITY or s.tokens.kind!='last' or s.read_or_write!='read' for s in self.sites):
            raise ValueError('Route v1 requires explicit last-prompt native head read sites')
        if len({s.name for s in self.sites})!=len(self.sites) or len({p.key for p in self.pairs})!=len(self.pairs):
            raise ValueError('Duplicate route sites/pairs')
        if not self.proposal_templates or set(self.proposal_templates)&{p.template for p in self.pairs}:
            raise ValueError('Proposal and held-out templates must be declared and disjoint')
        if type(self.ig_steps)is not int or self.ig_steps<0 or type(self.max_prompt_tokens)is not int or self.max_prompt_tokens<1:
            raise ValueError('Invalid finite candidate/token budget')

    def bound_engine(self):return self.engine
    def runtime_spec(self):return self.engine.runtime_spec()

    @classmethod
    def from_dict(cls,payload):
        p=dict(payload);p.pop('kind',None)
        p['engine']=NativeTorchEngine.from_dict(p['engine'])
        p['sites']=tuple(NativeSite.from_dict(s) for s in p['sites'])
        p['pairs']=tuple(RoutePair(**v) for v in p['pairs'])
        p['proposal_templates']=tuple(p['proposal_templates'])
        return cls(**p)
