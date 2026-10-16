from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from pipelines_v2.api import NativeSite, NativeSelection, ExpertLocalCoordinates, TokenSelector
from pipelines_v2.operations.common.native import GPT2_VERSION,QWEN_VERSION,MLP_QUANTITIES,HEAD_QUANTITY


def session_for(architecture):
    torch=pytest.importorskip('torch');tr=pytest.importorskip('transformers')
    from pipelines_v2.engine.native import NativeSession
    class Tokenizer:
        pad_token_id=31;eos_token_id=30
        def encode(self,p,**_):return list(map(int,p.split()))
        def decode(self,t):return ' '.join(map(str,t))
    with torch.random.fork_rng():
        torch.manual_seed(86)
        if architecture=='gpt2':
            model=tr.GPT2LMHeadModel(tr.GPT2Config(vocab_size=32,n_positions=32,n_embd=16,n_layer=2,n_head=2,resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.,attn_implementation='eager'))
        else:
            model=tr.Qwen3ForCausalLM(tr.Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=4,max_position_embeddings=32,attention_dropout=0.,attn_implementation='eager'))
    return NativeSession(model,Tokenizer(),model_identity='tiny-'+architecture)


def site_for(s,quantity,*,units=(0,2),tokens=None):
    head=quantity==HEAD_QUANTITY
    return NativeSite(name=quantity,model_identity=s.model_identity,adapter_version=s.adapter_version,
        module_path='transformer.h.0.attn.c_proj' if head else 'model.layers.0.mlp',layer=0,quantity=quantity,
        selection=NativeSelection(tokens=tokens or TokenSelector.last(),heads=units if head else (),neurons=() if head else units),
        axes=('batch','token','head','head_dim') if head else ('batch','token','neuron'))


def test_qwen_quantities_match_independent_linear_silu_product():
    import torch
    with session_for('qwen3') as s:
        ids,mask=s.encode(['1 2 3','4 5'])
        sites=tuple(site_for(s,q) for q in MLP_QUANTITIES)
        actual,captures=s.forward(ids,mask,capture=sites)
        mlp=s.model.model.layers[0].mlp;seen=[]
        handle=mlp.register_forward_pre_hook(lambda m,a:seen.append(a[0].detach().clone()))
        with torch.no_grad():expected=s.model(ids,attention_mask=mask,position_ids=(mask.cumsum(-1)-1).masked_fill(mask==0,0),use_cache=False).logits
        handle.remove();x=seen[0]
        gate=torch.nn.functional.linear(x,mlp.gate_proj.weight);up=torch.nn.functional.linear(x,mlp.up_proj.weight)
        independent={'gate_preactivation':gate,'gate_activation':torch.nn.functional.silu(gate),'up_projection':up,'mlp_product':torch.nn.functional.silu(gate)*up}
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for q,values in independent.items():
            for row,last in enumerate((2,1)):
                torch.testing.assert_close(captures[q][row],values[row,last:last+1,[0,2]],rtol=0,atol=0)
        _,reordered=s.forward(ids.flip(0),mask.flip(0),capture=sites)
        for q in MLP_QUANTITIES:torch.testing.assert_close(reordered[q][0],captures[q][1])
        assert 'forward' not in mlp.__dict__


def test_gpt2_head_vectors_equal_attention_times_values_before_projection():
    import torch,math
    with session_for('gpt2') as s:
        ids,mask=s.encode(['1 2 3']);site=site_for(s,HEAD_QUANTITY,units=(0,1),tokens=TokenSelector.full_sequence())
        _,capture=s.forward(ids,mask,capture=(site,))
        attn=s.model.transformer.h[0].attn;seen=[]
        h=attn.register_forward_pre_hook(lambda m,a:seen.append(a[0].detach().clone()))
        with torch.no_grad():s.model(ids,use_cache=False)
        h.remove()
        q,k,v=attn.c_attn(seen[0]).split(16,dim=-1)
        q,k,v=(t.reshape(1,3,2,8).transpose(1,2) for t in (q,k,v))
        scores=q@k.transpose(-1,-2)/math.sqrt(8)
        scores=scores.masked_fill(~torch.ones(3,3,dtype=torch.bool).tril(),torch.finfo(scores.dtype).min)
        expected=(scores.softmax(-1)@v).transpose(1,2)[0]
        torch.testing.assert_close(capture[HEAD_QUANTITY][0],expected,atol=1e-6,rtol=1e-5)
        assert capture[HEAD_QUANTITY][0].shape==(3,2,8)
        assert not attn.c_proj._forward_pre_hooks


def test_native_codec_roundtrip_and_strict_chunk_merge(tmp_path):
    from pipelines_v2.core.types import to_primitive
    from pipelines_v2.operations.capture.specs import capture_site_from_dict
    from pipelines_v2.storage.features import write_capture_features,load_feature_payload,merge_feature_payloads
    from pipelines_v2.storage.local import LocalArtifactStore
    with session_for('qwen3') as s:
        site=site_for(s,'mlp_product');assert capture_site_from_dict(to_primitive(site))==site
        payload=site.empty_payload();payload['layers']['0']['a']={'tokens':[1],'token_count':2,'units':[0,2],'values':np.ones((1,2),dtype=np.float32)}
        store=LocalArtifactStore(tmp_path);ref=write_capture_features(store,'a',{'native':payload})['native']
        decoded=load_feature_payload(store,ref);np.testing.assert_array_equal(decoded['layers']['0']['a']['values'],payload['layers']['0']['a']['values'])
        with pytest.raises(ValueError,match='overlap'):merge_feature_payloads(payload,payload)
        bad={**payload,'coordinate':{**payload['coordinate'],'model_identity':'other'}}
        with pytest.raises(ValueError,match='identity'):merge_feature_payloads(payload,bad)
        p=to_primitive(site);p['schema_version']=99
        with pytest.raises(ValueError,match='version'):capture_site_from_dict(p)


def test_expert_local_only_observes_routed_tokens_and_mixture_weights():
    import torch
    x=torch.tensor([[-2.,3.],[1.,-4.],[2.,5.]])
    # Two selected experts; token 1 never routes to expert 0, so it has no row.
    coordinates=ExpertLocalCoordinates(expert=0,token_indices=(0,2),mixture_weights=(.75,.25),token_count=3)
    local=x[list(coordinates.token_indices)]*2
    contribution=torch.zeros_like(x)
    contribution[list(coordinates.token_indices)]=local*torch.tensor(coordinates.mixture_weights)[:,None]
    torch.testing.assert_close(contribution,torch.tensor([[-3.,4.5],[0.,0.],[1.,2.5]]))
    assert 1 not in coordinates.token_indices and local.shape[0]==2
    with pytest.raises(ValueError):ExpertLocalCoordinates(0,(0,0),(.5,.5),3)
    with pytest.raises(ValueError):ExpertLocalCoordinates(0,(1,),(0.,),3)


@pytest.mark.parametrize('architecture,quantity,units',[('gpt2',HEAD_QUANTITY,(1,)),('qwen3','mlp_product',(2,))])
def test_native_padding_and_batch_order_match_single_rows(architecture,quantity,units):
    import torch
    with session_for(architecture) as s:
        site=site_for(s,quantity,units=units)
        prompts=['1 2 3','4 5'];ids,mask=s.encode(prompts)
        logits,batched=s.forward(ids,mask,capture=(site,))
        _,reversed_rows=s.forward(ids.flip(0),mask.flip(0),capture=(site,))
        for row,prompt in enumerate(prompts):
            single,single_capture=s.forward(*s.encode([prompt]),capture=(site,))
            torch.testing.assert_close(batched[quantity][row],single_capture[quantity][0],atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(reversed_rows[quantity][1-row],batched[quantity][row],atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(logits[row,len(prompt.split())-1],single[0,-1],atol=1e-6,rtol=1e-5)


def test_native_planning_rejects_revision_and_out_of_range_without_loading(tmp_path):
    import json
    from pipelines_v2.api import NativeTorchEngine,CaptureSpec,Dataset,Example
    from pipelines_v2.engine.native.engine import REVISIONS
    with pytest.raises(ValueError,match='pinned'):NativeTorchEngine(str(tmp_path),'f'*40,'qwen3')
    (tmp_path/'config.json').write_text(json.dumps({'model_type':'gpt2','n_layer':2,'n_head':2}))
    (tmp_path/'model.safetensors').write_bytes(b'no model load in this static failure fixture')
    engine=NativeTorchEngine(str(tmp_path),REVISIONS['gpt2'],'gpt2')
    site=engine.site('bad',HEAD_QUANTITY,3,heads=(2,))
    spec=CaptureSpec(engine=engine,dataset=Dataset.from_examples([Example(key='a',prompt='one')]),sites=(site,))
    assert 'Native coordinate out of bounds' in engine.planning_errors(spec)
