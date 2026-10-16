"""One bounded official GPT-2/IOI MAttr method/seed unit."""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any,ClassVar
from pipelines_v2.core.types import OperationSpec
from pipelines_v2.runtime.external_tool import ExternalToolRuntimeSpec
from .contracts import EVALUATOR,digest,validate_request_identity,locked_assets


def official_tool_lock():
    return json.loads(Path(__file__).with_name('official_mattr_lock.json').read_text())


def _file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def asset_identity(root,names):
    return {name:_file_hash(Path(root)/name) for name in names}


def _verify_files(root,expected):
    for name,sha in expected.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or not re.fullmatch('[0-9a-f]{64}',sha):
            raise ValueError('Invalid locked asset descriptor')
        if _file_hash(Path(root)/name)!=sha:
            raise ValueError(f'Locked asset content mismatch: {name}')


@dataclass(frozen=True,slots=True)
class MAttrSpec(OperationSpec):
    request: dict[str,Any] = field(default_factory=dict)
    runtime: ExternalToolRuntimeSpec | None = None
    model_path: str = ''
    dataset_path: str = ''
    work_root: str | None = None
    timeout_seconds: int = 600
    memory_bytes: int = 8*1024**3
    output_bytes: int = 16*1024**2
    kind:ClassVar[str]='external_mattr'

    def __post_init__(self):
        r=self.request
        validate_request_identity(r)
        if self.schema_version!=1 or r.get('schema_version')!=1 or r.get('evaluator')!=EVALUATOR:
            raise ValueError('Unsupported MAttr request/evaluator version')
        if r.get('task')!='ioi' or r.get('model',{}).get('name') not in {'gpt2','qwen2.5'}:
            raise ValueError('MAttr v1 supports pinned GPT-2 and Qwen2.5 IOI only')
        if not isinstance(self.runtime,ExternalToolRuntimeSpec) or r.get('tool_lock')!=self.runtime.tool_identity:
            raise ValueError('MAttr request/tool lock mismatch')
        for source in ('model','data'):
            if not re.fullmatch('[0-9a-f]{40}',r[source].get('revision','')) or not r[source].get('files'):
                raise ValueError('MAttr model/data require immutable revisions and file hashes')
        if set(r['data']['files'])!={'train.json','dev.json'}:
            raise ValueError('MAttr requires official train and validation data files')
        required_model={'config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt'}
        if not required_model<=set(r['model']['files']):
            raise ValueError('Incomplete model/tokenizer assets')
        if r.get('orientation')!='iso_base_clean_source_s2_io_flip' or r.get('basis')!='attention_head_mlp_input' or r.get('include_input') is not True:
            raise ValueError('Unsupported basis/orientation/input-node convention')
        fit=r['fitting'];evaluation=r['evaluation']
        if set(fit)!={'steps','lr','optimizer','adam_eps','T','n_iters','masking','k_schedule','train_batch_size','k_avg','cf_cache_gb'}:
            raise ValueError('Unknown/missing fitting settings')
        if {k:v for k,v in fit.items() if k not in {'steps','lr'}}!={'optimizer':'adam','adam_eps':1e-8,'T':.5,'n_iters':30,'masking':'topk','k_schedule':'uniform','train_batch_size':1,'k_avg':1,'cf_cache_gb':0}:
            raise ValueError('Unsupported fitting policy')
        if not isinstance(fit['steps'],int) or isinstance(fit['steps'],bool) or fit['steps']<1:
            raise ValueError('Fitting steps must be positive')
        if not isinstance(fit['lr'],(float,int)) or not 0<fit['lr']<=1:
            raise ValueError('Invalid learning rate')
        if evaluation!={'split':'validation','train_split':'train','examples':evaluation.get('examples'),'batch_size':1,'grid':'official_ten_point'} or type(evaluation['examples']) is not int or evaluation['examples']<1:
            raise ValueError('Unsupported held-out evaluation')
        if not isinstance(r.get('seed'),int) or isinstance(r['seed'],bool) or not 0<=r['seed']<2**32:
            raise ValueError('Invalid seed')
        for value in (self.timeout_seconds,self.memory_bytes,self.output_bytes):
            if type(value) is not int or value<=0:
                raise ValueError('External budgets must be finite positive integers')
        json.dumps(r,allow_nan=False)

    @classmethod
    def gpt2_ioi(cls,*,model_path,dataset_path,runtime,steps=2,eval_examples=2,seed=42,
                 model_revision='607a30d783dfa663caf39e06633721c8d4cfcd7e',data_revision='e5f3468f3af4c0883be35cd3bced8c711c95d286',**budgets):
        model_files=asset_identity(model_path,('config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt'))
        data_files=asset_identity(dataset_path,('train.json','dev.json'))
        request={'schema_version':1,'task':'ioi','model':{'name':'gpt2','revision':model_revision,'files':model_files},
            'data':{'repository':'mib-bench/ioi','revision':data_revision,'files':data_files},'evaluator':EVALUATOR,
            'tool_lock':runtime.tool_identity,'basis':'attention_head_mlp_input','orientation':'iso_base_clean_source_s2_io_flip',
            'include_input':True,'seed':seed,'dtype':'float32',
            'fitting':{'steps':steps,'lr':.05,'optimizer':'adam','adam_eps':1e-8,'T':.5,'n_iters':30,'masking':'topk','k_schedule':'uniform','train_batch_size':1,'k_avg':1,'cf_cache_gb':0},
            'evaluation':{'split':'validation','train_split':'train','examples':eval_examples,'batch_size':1,'grid':'official_ten_point'}}
        return cls(request=request,runtime=runtime,model_path=str(model_path),dataset_path=str(dataset_path),**budgets)

    def verify_inputs(self):
        _verify_files(self.model_path,self.request['model']['files']);_verify_files(self.dataset_path,self.request['data']['files'])
        validate_ioi_files(self.dataset_path, self.request['evaluation']['examples'])

    @classmethod
    def qwen25_ioi(cls, *, model_path, dataset_path, runtime, steps=2, eval_examples=2, seed=42, **budgets):
        """Official upstream BF16 policy; this is not numerically matched to GPT-2 FP32."""
        # Reuse the supported request shape without hashing or opening the wrong model.
        assets=locked_assets('qwen2.5')
        request={'schema_version':1,'task':'ioi','model':{'name':'qwen2.5',**assets['model']},
            'data':{'repository':'mib-bench/ioi',**assets['data']},'evaluator':EVALUATOR,
            'tool_lock':runtime.tool_identity,'basis':'attention_head_mlp_input','orientation':'iso_base_clean_source_s2_io_flip',
            'include_input':True,'seed':seed,'dtype':'bfloat16',
            'fitting':{'steps':steps,'lr':.05,'optimizer':'adam','adam_eps':1e-8,'T':.5,'n_iters':30,'masking':'topk','k_schedule':'uniform','train_batch_size':1,'k_avg':1,'cf_cache_gb':0},
            'evaluation':{'split':'validation','train_split':'train','examples':eval_examples,'batch_size':1,'grid':'official_ten_point'}}
        return cls(request=request,runtime=runtime,model_path=str(model_path),dataset_path=str(dataset_path),**budgets)

    @classmethod
    def from_ioi_files(cls, *, model, model_path, dataset_path, runtime, steps=2, eval_examples=2, seed=42, **budgets):
        """Caller-owned IOI-schema train/dev files, identified by actual content."""
        assets=locked_assets(model)
        files=asset_identity(dataset_path,('train.json','dev.json'))
        validate_ioi_files(dataset_path,eval_examples)
        request={'schema_version':1,'task':'ioi','model':{'name':model,**assets['model']},
            'data':{'repository':'local-content:'+digest(files),'revision':digest(files)[:40],'files':files},
            'evaluator':EVALUATOR,'tool_lock':runtime.tool_identity,'basis':'attention_head_mlp_input',
            'orientation':'iso_base_clean_source_s2_io_flip','include_input':True,'seed':seed,
            'dtype':'float32' if model=='gpt2' else 'bfloat16',
            'fitting':{'steps':steps,'lr':.05,'optimizer':'adam','adam_eps':1e-8,'T':.5,'n_iters':30,'masking':'topk','k_schedule':'uniform','train_batch_size':1,'k_avg':1,'cf_cache_gb':0},
            'evaluation':{'split':'validation','train_split':'train','examples':eval_examples,'batch_size':1,'grid':'official_ten_point'}}
        return cls(request=request,runtime=runtime,model_path=str(model_path),dataset_path=str(dataset_path),**budgets)

    def runtime_spec(self):return self.runtime

    def semantic_dict(self):
        return {'kind':self.kind,'schema_version':1,'request':self.request,'runtime_tool_identity':self.runtime.tool_identity}

    @classmethod
    def from_dict(cls,payload):
        values={k:v for k,v in payload.items() if k!='kind'}
        values['runtime']=ExternalToolRuntimeSpec.from_dict(values['runtime'])
        return cls(**values)


def validate_ioi_files(root, eval_examples):
    """Validate the pinned upstream reader's required schema before model startup."""
    for name in ('train.json','dev.json'):
        rows=json.loads((Path(root)/name).read_text())
        if not isinstance(rows,list) or not rows: raise ValueError('IOI splits must be nonempty JSON row arrays')
        if name=='dev.json' and len(rows)<eval_examples: raise ValueError('Requested evaluation exceeds dev rows')
        for row in rows:
            metadata=row.get('metadata',{})
            counterfactual=row.get('s2_io_flip_counterfactual',{})
            if not isinstance(row.get('prompt'),str) or not row['prompt'].strip() or not isinstance(counterfactual.get('prompt'),str) or not counterfactual['prompt'].strip():
                raise ValueError('IOI rows require base and s2_io_flip_counterfactual prompts')
            if any(not isinstance(metadata.get(k),str) or not metadata[k].strip() for k in ('indirect_object','subject','random_c')):
                raise ValueError('IOI metadata requires indirect_object, subject and random_c')
