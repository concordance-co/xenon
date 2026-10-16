"""Opt-in, serializable SDXL image operations for the existing workflow runner."""
from __future__ import annotations
from dataclasses import dataclass,field,asdict
from pathlib import Path
from typing import Any,ClassVar

from pipelines_v2.core.types import OperationSpec
from pipelines_v2.engine.diffusers.assets import model_lock,checkpoint_path,verify_checkpoint
from pipelines_v2.engine.diffusers.hooks import DenoisingHook
from pipelines_v2.engine.diffusers.session import ImageRequest
from pipelines_v2.storage.artifacts import OperationArtifact,artifact_from_dict


@dataclass(frozen=True,slots=True)
class SDXLEngine:
    cache_root: str = '/models'
    kind: ClassVar[str] = 'sdxl'
    dtype: ClassVar[str] = 'float16'

    def __post_init__(self):
        if not Path(self.cache_root).is_absolute():raise ValueError('SDXL cache root must be an absolute execution mount')

    def identity(self):
        lock=model_lock()
        return {'kind':self.kind,'model_id':lock['repository'],'revision':lock['revision'],'cache_root':self.cache_root,
                'dtype':self.dtype,'attention_backend':'torch_sdpa','scheduler':'EulerDiscreteScheduler',
                'compile':False,'offload':False,'batch_size':1}

    def semantic_identity(self):
        result=self.identity();result.pop('cache_root');return result

    @classmethod
    def from_dict(cls,payload):
        engine=cls(cache_root=payload.get('cache_root','/models'))
        if set(payload)-set(engine.identity()) or any(payload[key]!=value for key,value in engine.identity().items() if key in payload):
            raise ValueError('Only the pinned SDXL FP16/Euler eager configuration is supported')
        return engine

    @property
    def revision(self):return model_lock()['revision']
    @property
    def model_id(self):return model_lock()['repository']
    def resolved_model_path(self):return str(checkpoint_path(self.cache_root))
    def capabilities(self):return set()  # No text-generation or universal training claim.
    def runtime_spec(self):
        from pipelines_v2.engine.diffusers.artifacts import runtime_spec
        return runtime_spec()
    def observed_components(self):return verify_checkpoint(self.resolved_model_path())
    def planning_errors(self,spec):
        return () if isinstance(spec,DiffusionSpec) else ('SDXL requires a diffusion image operation',)


@dataclass(frozen=True)
class DiffusionSpec(OperationSpec):
    engine: SDXLEngine = field(default_factory=SDXLEngine)
    images: tuple[ImageRequest,...] = ()
    hooks: tuple[DenoisingHook,...] = ()
    noise_from: OperationArtifact | Any | None = None
    replacement_from: OperationArtifact | Any | None = None
    kind: ClassVar[str]='diffusion_generate'

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version!=1:raise ValueError('Unsupported diffusion spec version')
        object.__setattr__(self,'images',tuple(self.images));object.__setattr__(self,'hooks',tuple(self.hooks))
        if not isinstance(self.engine,SDXLEngine):raise TypeError('DiffusionSpec needs SDXLEngine')
        if not self.images or len(self.images)>4 or any(not isinstance(row,ImageRequest) for row in self.images):raise ValueError('Initial SDXL envelope supports1..4 named images')
        keys=[row.key for row in self.images]
        if len(set(keys))!=len(keys):raise ValueError('Image keys must be unique')
        if any(not isinstance(hook,DenoisingHook) for hook in self.hooks):raise TypeError('Use explicit DenoisingHook values')
        if len(self.hooks)>4:raise ValueError('Initial SDXL adapter supports up to4 hooks')
        for request in self.images:
            for hook in self.hooks:
                if max(hook.steps)>=request.steps:raise ValueError('Hook step exceeds image denoising trajectory')
                if request.guidance_scale<=1 and hook.branch!='conditional':raise ValueError('CFG-disabled request has only conditional branch')
        if self.kind=='diffusion_generate' and self.hooks:raise ValueError('Generation has no hooks; use DiffusionCaptureSpec or DiffusionInterventionSpec')
        if self.kind=='diffusion_capture' and (not self.hooks or any(h.operator!='capture' for h in self.hooks)):raise ValueError('Capture requires read-only internal hooks')
        if self.kind=='diffusion_intervene' and (not self.hooks or all(h.operator=='capture' for h in self.hooks) or self.noise_from is None):
            raise ValueError('Intervention needs a write hook and explicit shared noise_from baseline artifact')
        if any(h.operator=='replace' for h in self.hooks) and self.replacement_from is None:raise ValueError('Replacement needs a matched capture artifact')

    def bound_engine(self):return self.engine
    def runtime_spec(self):return self.engine.runtime_spec()
    def to_dict(self):
        value=super().to_dict();value['engine']=self.engine.identity();return value
    def semantic_dict(self):
        value=super().semantic_dict();value['engine']=self.engine.semantic_identity()
        for name in ('noise_from','replacement_from'):
            artifact=getattr(self,name)
            if artifact is not None and callable(getattr(artifact,'manifest',None)):
                manifest=artifact.manifest();provenance=manifest.provenance()
                value[name]={'scientific_identity':provenance.scientific_identity if provenance else 'legacy',
                    'payloads':sorted((item['sha256'],item['bytes']) for item in manifest.metadata.get('completion_v1',{}).get('payloads',()))}
        return value
    @classmethod
    def from_dict(cls,payload):
        expected={'schema_version','kind','engine','images','hooks','noise_from','replacement_from'}
        if set(payload)-expected or payload.get('kind')!=cls.kind:raise ValueError('Unsupported diffusion operation payload')
        return cls(schema_version=payload.get('schema_version'),engine=SDXLEngine.from_dict(payload['engine']),
            images=tuple(ImageRequest(**row) for row in payload['images']),hooks=tuple(DenoisingHook(**hook) for hook in payload.get('hooks',())),
            noise_from=artifact_from_dict(payload['noise_from']) if payload.get('noise_from') else None,
            replacement_from=artifact_from_dict(payload['replacement_from']) if payload.get('replacement_from') else None)


@dataclass(frozen=True)
class DiffusionCaptureSpec(DiffusionSpec):
    kind: ClassVar[str]='diffusion_capture'


@dataclass(frozen=True)
class DiffusionInterventionSpec(DiffusionSpec):
    kind: ClassVar[str]='diffusion_intervene'


DiffusionGenerationSpec=DiffusionSpec
DIFFUSION_KINDS=frozenset({'diffusion_generate','diffusion_capture','diffusion_intervene'})


def load_diffusion_spec(payload):
    cls={'diffusion_generate':DiffusionSpec,'diffusion_capture':DiffusionCaptureSpec,
         'diffusion_intervene':DiffusionInterventionSpec}.get(payload.get('kind'))
    if cls is None:raise ValueError('Unknown diffusion spec')
    return cls.from_dict(payload)


def requires_context(spec):
    raise RuntimeError('SDXL operations require the Modal diffusion executor and artifact context')
