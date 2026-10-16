"""Narrow original Qwen2.5 dense residual adapter, CPU/CUDA FP32 eager."""
from contextlib import contextmanager
import gc
from pathlib import Path
import threading
from .settings import Execution,Limits

BASE_REVISION='060db6499f32faf8b98477b0a26969ef7d8b9987'
INSTRUCT_REVISION='7ae557604adf67be50417f59c2c2f167def9a775'
REPOSITORIES=('Qwen/Qwen2.5-0.5B','Qwen/Qwen2.5-0.5B-Instruct')
REVISIONS=(BASE_REVISION,INSTRUCT_REVISION)


def model_identity(path,side):
    from pipelines_v2.engine.torch.engine import checkpoint_identity
    from pipelines_v2.core.provenance import content_hash
    from pipelines_v2.runtime.readiness import local_checkpoint_components
    path=Path(path)
    import hashlib,json
    locked=json.loads(Path(__file__).with_name('model_lock.json').read_text())['models'][side]
    for name,expected in locked['files'].items():
        digest=hashlib.sha256()
        with (path/name).open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
        if digest.hexdigest()!=expected:raise ValueError('Pinned Qwen model/tokenizer bytes differ: '+name)
    components,_=local_checkpoint_components(path,requested_revision=REVISIONS[side])
    return {'repository':REPOSITORIES[side],'revision':REVISIONS[side],
            'checkpoint_digest':checkpoint_identity(path),'components':components}


class QwenSession:
    def __init__(self,path,*,side,execution=None,limits=None):
        self.execution=execution or Execution();self.limits=limits or Limits()
        import torch
        from tqdm import tqdm
        tqdm.set_lock(threading.RLock())
        from transformers import AutoTokenizer,Qwen2ForCausalLM,Qwen2Config
        config=Qwen2Config.from_pretrained(path,local_files_only=True)
        if config.model_type!='qwen2' or config.hidden_size!=896 or config.num_hidden_layers!=24 or config.hidden_act!='silu':
            raise ValueError('Only original dense Qwen2.5-0.5B coordinates are supported')
        self.identity=model_identity(path,side)
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model=Qwen2ForCausalLM.from_pretrained(path,config=config,local_files_only=True,use_safetensors=True,
            dtype=torch.float32,attn_implementation='eager').to(self.execution.device).eval()
        self.model.requires_grad_(False);self._lock=threading.RLock()

    def tokens(self,prompt):
        ids=self.tokenizer.encode(prompt,add_special_tokens=False)
        if not 1<=len(ids)<=min(self.limits.max_tokens_per_example,self.model.config.max_position_embeddings):raise ValueError('Prompt exceeds token/model-context budget')
        return ids

    def forward(self,ids,*,layer=12,edit=None):
        import torch
        if type(layer)is not int or not 0<=layer<24:raise ValueError('Invalid residual layer')
        inputs=torch.tensor([ids],dtype=torch.long,device=self.execution.device);mask=torch.ones_like(inputs);capture={}
        def before(module,args):
            value=args[0]
            if edit is not None:
                replacement=edit(value)
                if replacement.shape!=value.shape or replacement.dtype!=value.dtype or replacement.device!=value.device or not torch.isfinite(replacement).all():
                    raise ValueError('Residual edit changed type/shape or produced nonfinite values')
                value=replacement
            capture['values']=value.detach().clone()
            return (value,*args[1:])
        with self._lock,torch.inference_mode():
            handle=self.model.model.layers[layer].register_forward_pre_hook(before)
            try:output=self.model(input_ids=inputs,attention_mask=mask,position_ids=torch.arange(len(ids),device=self.execution.device)[None],use_cache=False).logits
            finally:handle.remove()
        return output,capture['values']

    def parity(self,ids,*,layer=12):
        import torch
        inputs=torch.tensor([ids],device=self.execution.device);mask=torch.ones_like(inputs);positions=torch.arange(len(ids),device=self.execution.device)[None]
        with torch.inference_mode():direct=self.model(input_ids=inputs,attention_mask=mask,position_ids=positions,use_cache=False,output_hidden_states=True)
        logits,capture=self.forward(ids,layer=layer)
        torch.testing.assert_close(logits,direct.logits,rtol=1e-4,atol=1e-5)
        torch.testing.assert_close(capture,direct.hidden_states[layer],rtol=1e-4,atol=1e-5)
        noop,_=self.forward(ids,layer=layer,edit=lambda value:value)
        if not torch.equal(noop,logits):raise ValueError('No-op changes Qwen logits')
        delta=torch.linspace(-.01,.01,896,device=self.execution.device)
        def edit(value):
            result=value.clone();result[:,-1]+=delta;return result
        patched,_=self.forward(ids,layer=layer,edit=edit)
        def independent(module,args):
            value=args[0].clone();value[0,-1]=value[0,-1]+delta;return (value,*args[1:])
        handle=self.model.model.layers[layer].register_forward_pre_hook(independent)
        try:
            with torch.inference_mode():expected=self.model(input_ids=inputs,attention_mask=mask,position_ids=positions,use_cache=False).logits
        finally:handle.remove()
        torch.testing.assert_close(patched,expected,rtol=1e-4,atol=1e-5)
        return {'direct_logit_max_abs':float((logits-direct.logits).abs().max()),
                'capture_max_abs':float((capture-direct.hidden_states[layer]).abs().max()),
                'patch_max_abs':float((patched-expected).abs().max()),'noop_max_abs':float((noop-logits).abs().max()),
                'rtol':1e-4,'atol':1e-5,'passed':True}

    def close(self):
        self.model=None;gc.collect()
        if self.execution.device=="cuda":
            import torch
            torch.cuda.empty_cache()
