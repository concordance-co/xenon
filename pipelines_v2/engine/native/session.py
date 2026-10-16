"""Scoped exact GPT-2 head and dense Qwen3 SwiGLU forwards; no auto-hooking."""
from contextlib import contextmanager
import threading
from types import MethodType
from pipelines_v2.operations.common.native import HEAD_QUANTITY, MLP_QUANTITIES, GPT2_VERSION, QWEN_VERSION
from pipelines_v2.engine.torch.session import TorchSession


class NativeSession(TorchSession):
    def __init__(self, model, tokenizer, *, model_identity, device='cpu'):
        import torch
        from pipelines_v2.runtime.numerics import configure_torch_threads
        configure_torch_threads()
        architecture = model.config.model_type
        if architecture not in ('gpt2','qwen3') or (architecture=='qwen3' and model.config.hidden_act!='silu'):
            raise ValueError('Native adapter supports GPT-2 heads and dense Qwen3 silu MLP only')
        self.model=model.to(device=device,dtype=torch.float32).eval()
        self.model.config._attn_implementation='eager'
        self.model.requires_grad_(False)
        self.tokenizer=tokenizer
        if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
        self.model_identity=model_identity
        self.architecture=architecture
        self.adapter_version=GPT2_VERSION if architecture=='gpt2' else QWEN_VERSION
        self._lock=threading.RLock();self.closed=False

    @property
    def context_length(self):
        return self.model.config.max_position_embeddings

    def encode(self,prompts):
        import torch
        rows=[self.tokenizer.encode(p,add_special_tokens=False) for p in prompts]
        if not rows or any(not row or len(row)>self.context_length for row in rows):
            raise ValueError('Native prompts must be nonempty and fit the context')
        ids=torch.full((len(rows),max(map(len,rows))),self.tokenizer.pad_token_id,dtype=torch.long)
        mask=torch.zeros_like(ids)
        for i,row in enumerate(rows): ids[i,:len(row)]=torch.tensor(row);mask[i,:len(row)]=1
        return ids.to(self.model.device),mask.to(self.model.device)

    def validate_site(self,site):
        if site.model_identity != self.model_identity or site.adapter_version != self.adapter_version:
            raise ValueError('Native site model/adapter identity mismatch')
        count=self.model.config.num_hidden_layers
        width=self.model.config.num_attention_heads if self.architecture=='gpt2' else self.model.config.intermediate_size
        if site.layer>=count or any(i>=width for i in (*site.selection.heads,*site.selection.neurons)):
            raise ValueError('Native layer/head/neuron coordinate out of bounds')
        if (site.quantity==HEAD_QUANTITY) != (self.architecture=='gpt2'):
            raise ValueError('Unsupported native quantity for architecture')

    @contextmanager
    def hooks(self, *, capture=(), edits=(), attention_mask, detach=True):
        import torch
        captures={};handles=[];restores=[]
        if len({s.name for s in capture}) != len(capture):raise ValueError("Duplicate native capture names")
        for site in (*capture,*(edit.write_site.coordinate for edit in edits)):self.validate_site(site)
        counts=[int(v) for v in attention_mask.sum(-1)]
        def positions(site,row):return site.tokens.resolve(counts[row])
        def selected(value,site,row):
            tokens=positions(site,row);units=site.selection.heads or site.selection.neurons
            return value[row,tokens][:,list(units)]
        def apply(value,quantity,layer):
            for edit in edits:
                site=edit.write_site.coordinate
                if (site.quantity,site.layer)!=(quantity,layer):continue
                updated=value.clone()
                for row in range(value.shape[0]):
                    tokens=positions(site,row);units=site.selection.heads or site.selection.neurons
                    current=selected(value,site,row)
                    replacement=torch.as_tensor(edit.value,dtype=value.dtype,device=value.device)
                    if replacement.ndim and tuple(replacement.shape)!=tuple(current.shape):
                        raise ValueError('Native replacement must be scalar or match selected token/unit tensor exactly')
                    replacement=current*replacement if edit.mode=='scale' else replacement.expand_as(current)
                    for ti,token in enumerate(tokens): updated[row,token,list(units)]=replacement[ti]
                value=updated
            for site in capture:
                if (site.quantity,site.layer)==(quantity,layer):
                    captures[site.name]=[(selected(value,site,row).detach().clone() if detach else selected(value,site,row)) for row in range(value.shape[0])]
            return value
        layers={s.layer for s in capture}|{e.write_site.coordinate.layer for e in edits}
        try:
            for layer in sorted(layers):
                if self.architecture=='gpt2':
                    module=self.model.transformer.h[layer].attn.c_proj
                    heads=self.model.config.num_attention_heads
                    def head_pre(module,args,layer=layer,heads=heads):
                        original=args[0];value=original.reshape(*original.shape[:2],heads,-1)
                        return (apply(value,HEAD_QUANTITY,layer).reshape_as(original),*args[1:])
                    handles.append(module.register_forward_pre_hook(head_pre))
                else:
                    module=self.model.model.layers[layer].mlp
                    had_local='forward' in module.__dict__;original_local=module.__dict__.get('forward')
                    restores.append((module,had_local,original_local))
                    def mlp_forward(module,x,layer=layer):
                        gate=apply(module.gate_proj(x),'gate_preactivation',layer)
                        activated=apply(module.act_fn(gate),'gate_activation',layer)
                        up=apply(module.up_proj(x),'up_projection',layer)
                        product=apply(activated*up,'mlp_product',layer)
                        return module.down_proj(product)
                    module.forward=MethodType(mlp_forward,module)
            yield captures
        finally:
            for handle in handles:handle.remove()
            for module,had_local,original in reversed(restores):
                if had_local:module.forward=original
                else:del module.forward

    def forward(self,input_ids=None,attention_mask=None,*,inputs_embeds=None,capture=(),edits=(),differentiable=False):
        import torch
        if self.closed:raise RuntimeError('Native session is closed')
        input_ids=input_ids.to(self.model.device) if input_ids is not None else None
        inputs_embeds=inputs_embeds.to(self.model.device) if inputs_embeds is not None else None
        attention_mask=attention_mask.to(self.model.device) if attention_mask is not None else None
        shape=input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
        if attention_mask is None:attention_mask=torch.ones(shape,dtype=torch.long,device=self.model.device)
        if tuple(attention_mask.shape)!=tuple(shape) or not torch.all((attention_mask==0)|(attention_mask==1)) or torch.any(attention_mask.sum(-1)==0):
            raise ValueError('Native attention mask must align with nonempty rows')
        # Explicit right padding only; selectors use unpadded prompt coordinates.
        if torch.any(attention_mask[:,1:]>attention_mask[:,:-1]):raise ValueError('Native adapter supports right padding only')
        position_ids=attention_mask.long().cumsum(-1)-1;position_ids.masked_fill_(attention_mask==0,0)
        with self._lock,torch.set_grad_enabled(differentiable),self.hooks(capture=capture,edits=edits,attention_mask=attention_mask,detach=not differentiable) as values:
            logits=self.model(input_ids=input_ids,attention_mask=attention_mask,inputs_embeds=inputs_embeds,position_ids=position_ids,use_cache=False,return_dict=True).logits
            return logits,values

    def greedy(self,prompt,*,max_tokens,target_token_ids=(),capture=(),edits=()):
        import torch
        ids,mask=self.encode([prompt]);prompt_count=ids.shape[1]
        if prompt_count+max_tokens>self.context_length:raise ValueError('Native generation exceeds context')
        if any(type(t)is not int or not 0<=t<self.model.config.vocab_size for t in target_token_ids):raise ValueError('Target outside vocabulary')
        # Freeze prompt token positions while generation extends the input.
        from dataclasses import replace
        from pipelines_v2.operations.common.tokens import TokenSelector
        fixed=[]
        for edit in edits:
            site=edit.write_site.coordinate
            positions=site.tokens.resolve(prompt_count)
            if not positions:raise ValueError('Native edit selected no prompt tokens')
            selector=TokenSelector.slice(positions[0],positions[-1]+1)
            coordinate=replace(site,selection=replace(site.selection,tokens=selector))
            fixed.append(replace(edit,target_tokens=selector,write_site=replace(edit.write_site,coordinate=coordinate)))
        logits,values=self.forward(ids,mask,capture=capture,edits=tuple(fixed))
        targets={str(t):float(torch.log_softmax(logits[0,-1],-1)[t]) for t in target_token_ids}
        tokens=[];finish='length'
        for index in range(max_tokens):
            token=int(logits[0,-1].argmax());tokens.append(token)
            if token==self.tokenizer.eos_token_id:finish='stop';break
            if index+1<max_tokens:
                ids=torch.cat([ids,torch.tensor([[token]],device=ids.device)],1);mask=torch.ones_like(ids)
                logits,_=self.forward(ids,mask,edits=tuple(fixed))
        row={'generated_text':self.tokenizer.decode(tokens),'generated_token_ids':tokens,'prompt_token_count':prompt_count,'generated_token_count':len(tokens),'finish_reason':finish}
        if target_token_ids:row['target_logprobs']=targets
        return row,values
