"""Deliberately narrow native engine: pinned GPT-2 heads / Qwen3-0.6B MLP."""
from dataclasses import dataclass
from pathlib import Path
import json
from pipelines_v2.core.types import EngineCapability, SpecValidationError
from pipelines_v2.engine.base import EngineCaptureResult, EngineGenerationResult, EngineInterventionResult
from pipelines_v2.engine.torch.engine import TorchEngine, checkpoint_identity
from pipelines_v2.operations.common.native import NativeSite, NativeSelection, GPT2_VERSION, QWEN_VERSION, HEAD_QUANTITY
from pipelines_v2.operations.interventions.native import NativeCoordinatePatch
from .session import NativeSession

REVISIONS={'gpt2':'607a30d783dfa663caf39e06633721c8d4cfcd7e','qwen3':'c1899de289a04d12100db370d81485cdf75e47ca'}


@dataclass(frozen=True,slots=True)
class NativeTorchEngine:
    model_id: str
    revision: str
    architecture: str
    device: str='cpu'
    dtype: str='float32'
    attention_impl: str='eager'
    kind: str='torch_native'

    def __post_init__(self):
        if self.kind!='torch_native' or self.architecture not in REVISIONS or self.revision!=REVISIONS[self.architecture]:
            raise ValueError('Native engine requires the pinned GPT-2 or Qwen3-0.6B revision')
        if self.device not in {'cpu','cuda'} or (self.dtype,self.attention_impl)!=('float32','eager'):
            raise ValueError('Native engine currently supports CPU/CUDA FP32 eager only')

    @property
    def adapter_version(self): return GPT2_VERSION if self.architecture=='gpt2' else QWEN_VERSION
    def identity(self):
        return {'kind':self.kind,'model_id':self.model_id,'revision':self.revision,'architecture':self.architecture,
                'device':self.device,'dtype':self.dtype,'attention_impl':self.attention_impl,'adapter_version':self.adapter_version}
    semantic_identity=identity
    runtime_spec=TorchEngine.runtime_spec
    resolved_model_path=TorchEngine.resolved_model_path
    scratch_token_counts=TorchEngine.scratch_token_counts

    @classmethod
    def from_dict(cls,p):
        data=dict(p);version=data.pop('adapter_version',None);engine=cls(**data)
        if version is not None and version!=engine.adapter_version:raise ValueError('Native engine adapter mismatch')
        return engine

    def capabilities(self):
        return {EngineCapability.GENERATION,EngineCapability.TARGET_LOGPROBS,EngineCapability.LOGPROBS,
                EngineCapability.ACTIVATION_PATCHING,EngineCapability.REQUEST_SCOPED_INTERVENTIONS,
                EngineCapability.NATIVE_HEAD_CAPTURE if self.architecture=='gpt2' else EngineCapability.NATIVE_MLP_CAPTURE}
    def supported_patch_operators(self):return frozenset({'native_coordinate'})

    def site(self,name,quantity,layer,*,neurons=(),heads=(),tokens=None,model_identity=None):
        from pipelines_v2.operations.common.tokens import TokenSelector
        if model_identity is not None:
            import re
            if not isinstance(model_identity,str) or not re.fullmatch(r'[0-9a-f]{64}',model_identity):
                raise ValueError('Explicit model_identity must be a checkpoint SHA256 digest')
        head=quantity==HEAD_QUANTITY
        return NativeSite(name=name,model_identity=model_identity or checkpoint_identity(self.model_id),adapter_version=self.adapter_version,
            module_path=f'transformer.h.{layer}.attn.c_proj' if head else f'model.layers.{layer}.mlp',layer=layer,quantity=quantity,
            selection=NativeSelection(tokens=tokens or TokenSelector.last(),neurons=tuple(neurons),heads=tuple(heads)),
            axes=('batch','token','head','head_dim') if head else ('batch','token','neuron'))

    def planning_errors(self,spec,*,check_assets=True):
        errors=[]
        config={};identity=None
        if check_assets:
            try:
                config=json.loads((Path(self.model_id)/'config.json').read_text())
                if config['model_type']!=self.architecture:errors.append('Prepared model architecture does not match native engine')
                if self.architecture=='qwen3' and any(config.get(k)!=v for k,v in {'hidden_act':'silu','hidden_size':1024,'intermediate_size':3072,'num_hidden_layers':28,'num_attention_heads':16,'num_key_value_heads':8}.items()):
                    errors.append('Only the pinned dense Qwen3-0.6B configuration is supported')
                identity=checkpoint_identity(self.model_id)
            except (OSError,ValueError,KeyError):
                return ('Prepared native model cache is incomplete; download separately',)
        patch=getattr(spec,'patch',None)
        if patch is not None and not isinstance(patch,NativeCoordinatePatch):errors.append('Native engine supports NativeCoordinatePatch only')
        sites=list(getattr(spec,'sites',()))
        if isinstance(patch,NativeCoordinatePatch):sites.append(patch.write_site.coordinate)
        for site in sites:
            if not isinstance(site,NativeSite):errors.append('Native engine requires typed native sites');continue
            if (check_assets and site.model_identity!=identity) or site.adapter_version!=self.adapter_version:errors.append('Native site model/adapter identity mismatch')
            if (site.quantity==HEAD_QUANTITY)!=(self.architecture=='gpt2'):errors.append('Native quantity unsupported by architecture')
            count=config.get('n_layer',config.get('num_hidden_layers',0))
            width=config.get('n_head',0) if self.architecture=='gpt2' else config.get('intermediate_size',0)
            if check_assets and (site.layer>=count or any(v>=width for v in (*site.selection.heads,*site.selection.neurons))):errors.append('Native coordinate out of bounds')
            if site in getattr(spec,'sites',()) and site.read_or_write!='read':errors.append('Capture requires native read coordinates')
        generation=getattr(spec,'generation',None)
        if generation is not None:
            if generation.temperature!=0 or generation.top_p!=1 or generation.top_k!=-1:errors.append('Native generation is greedy only')
            if generation.enabled and generation.max_tokens is None:errors.append('Native generation needs bounded max_tokens')
            if generation.capture_generated_tokens or generation.capture_reasoning or generation.chat_tools or generation.tool_choice is not None or generation.structured_output is not None:
                errors.append('Native engine supports plain text prompt-side capture only')
        if getattr(spec,'prompt_metadata_builder',None) is not None:errors.append('Native adapter does not support custom prompt metadata')
        if any(getattr(e,'images',()) for e in getattr(getattr(spec,'dataset',None),'examples',())):errors.append('Native text adapter rejects images')
        return tuple(errors)

    def _validate(self,spec):
        errors=self.planning_errors(spec)
        if errors:raise SpecValidationError('; '.join(errors))

    def open_session(self):
        from transformers import AutoTokenizer, GPT2LMHeadModel, Qwen3ForCausalLM
        config=json.loads((Path(self.model_id)/'config.json').read_text())
        if config.get('model_type')!=self.architecture:raise ValueError('Prepared native architecture mismatch')
        tokenizer=AutoTokenizer.from_pretrained(self.model_id,local_files_only=True)
        cls=GPT2LMHeadModel if self.architecture=='gpt2' else Qwen3ForCausalLM
        model=cls.from_pretrained(self.model_id,local_files_only=True,attn_implementation='eager',dtype='float32')
        return NativeSession(model,tokenizer,model_identity=checkpoint_identity(self.model_id), device=self.device)

    def capture(self,spec,*,session=None,**_):
        self._validate(spec)
        if session is None:
            with self.open_session() as owned:return self.capture(spec,session=owned)
        features={site.name:site.empty_payload() for site in spec.sites};metadata=[];generations=[]
        for example in spec.dataset.examples:
            row,values=session.greedy(example.prompt,max_tokens=int(spec.generation.max_tokens or 0) if spec.generation.enabled else 0,
                target_token_ids=spec.generation.target_token_ids,capture=tuple(spec.sites))
            count=row['prompt_token_count'];metadata.append({'example_key':example.key,'token_count':count,'prompt_token_count':count})
            for site in spec.sites:
                features[site.name]['layers'][str(site.layer)][example.key]={'tokens':site.tokens.resolve(count),'values':values[site.name][0].cpu().numpy(),'token_count':count,
                    'units':list(site.selection.heads or site.selection.neurons),
                    **({'head_dim':int(values[site.name][0].shape[-1])} if site.quantity==HEAD_QUANTITY else {})}
            if spec.generation.enabled:generations.append({'example_key':example.key,**row})
        return EngineCaptureResult(features=features,generations=generations,metadata={'backend':'torch_native','adapter_version':self.adapter_version,
            'model_identity':session.model_identity,'example_metadata':metadata,'chat_template':None,'thinking':False})

    def generate(self,spec,*,session=None,**_):
        self._validate(spec)
        if session is None:
            with self.open_session() as owned:return self.generate(spec,session=owned)
        from pipelines_v2.operations.interventions.runtime import resolve_generation_examples
        rows=[]
        for example in resolve_generation_examples(spec):
            row,_=session.greedy(example.prompt,max_tokens=spec.generation.max_tokens,target_token_ids=spec.generation.target_token_ids)
            rows.append({'example_key':example.key,'example':example.to_dict(),**row})
        return EngineGenerationResult(rows=rows,metadata={'backend':'torch_native','adapter_version':self.adapter_version})

    def intervene(self,spec,*,session=None,**_):
        self._validate(spec)
        if session is None:
            with self.open_session() as owned:return self.intervene(spec,session=owned)
        from pipelines_v2.operations.interventions.runtime import resolve_patched_generation_targets
        rows=[]
        for example in resolve_patched_generation_targets(spec):
            row,_=session.greedy(example.prompt,max_tokens=spec.generation.max_tokens,target_token_ids=spec.generation.target_token_ids,edits=(spec.patch,))
            rows.append({'example_key':example.key,'example':example.to_dict(),'status':'ok','case_key':example.key,
                'target_tokens':spec.patch.target_tokens.resolve(row['prompt_token_count']),
                'patch_stats':{'operator':'native_coordinate','mode':spec.patch.mode,'quantity':spec.patch.write_site.coordinate.quantity},**row})
        return EngineInterventionResult(rows=rows,summary={'example_count':len(rows),'patched_count':len(rows)},metadata={'backend':'torch_native','adapter_version':self.adapter_version})
