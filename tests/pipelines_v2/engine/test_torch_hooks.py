import pytest

from pipelines_v2.engine.torch.engine import residual_edit


def test_identity_and_full_state_replacement_match_independent_forward(tiny_torch_session):
    import torch
    s=tiny_torch_session; ids,mask=s.encode(['1 4 5'])
    base,_=s.forward(ids,mask)
    noop,_=s.forward(ids,mask,edits={('resid_post',0):residual_edit('scale',[2],scale=1.,hidden_size=16)})
    torch.testing.assert_close(noop,base,rtol=0,atol=0)
    replacement=torch.linspace(-1,1,32).reshape(2,16)
    actual,_=s.forward(ids,mask,edits={('resid_post',0):residual_edit('interchange',[0,2],values=replacement,hidden_size=16)})
    def oracle(module,args,out):
        hidden=out[0] if isinstance(out,tuple) else out
        changed=hidden.clone();changed[:,0,:]=replacement[0];changed[:,2,:]=replacement[1]
        return (changed,*out[1:]) if isinstance(out,tuple) else changed
    handle=s.model.transformer.h[0].register_forward_hook(oracle)
    try:
        with torch.no_grad():
            expected=s.model(ids,attention_mask=mask,use_cache=False).logits
    finally:handle.remove()
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
    assert float((actual-base).abs().max())>1e-3


def test_hook_cleanup_after_failure_and_repeated_calls(tiny_torch_session):
    import torch,weakref,gc
    s=tiny_torch_session;ids,mask=s.encode(['1 3'])
    def broken(value):raise RuntimeError('injected')
    with pytest.raises(RuntimeError,match='injected'):
        s.forward(ids,mask,edits={('resid_pre',1):broken})
    assert all(not block._forward_hooks and not block._forward_pre_hooks for block in s.model.transformer.h)
    references=[]
    for _ in range(8):
        logits,captured=s.forward(ids,mask,capture=(('resid_post',0),))
        references.append(weakref.ref(captured['resid_post',0]))
        assert logits.grad_fn is None
    del logits,captured;gc.collect()
    assert all(ref() is None for ref in references)
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in s.model.parameters())
    assert not s.model.training


def test_torch_interchange_preserves_each_donor_token_and_checks_alignment(tiny_torch_session,tmp_path):
    from types import SimpleNamespace
    from dataclasses import replace
    import torch
    from pipelines_v2.api import TorchEngine,Dataset,Example,GenerationSpec,PatchedGenerationSpec,InterchangePatch,ResidualInterventionSite,TokenSelector
    from pipelines_v2.engine.torch.session import ADAPTER_VERSION
    s=tiny_torch_session
    engine=TorchEngine(str(tmp_path),'a'*40)
    dataset=Dataset.from_examples([Example(key='target',prompt='1 2 3',labels={'arm':'target','pair':'p'}),Example(key='donor',prompt='6 7 8',labels={'arm':'donor','pair':'p'})])
    _,states=s.forward(*s.encode(['6 7 8']),capture=(('resid_post',0),))
    values=states['resid_post',0][0].tolist()
    payload={'kind':'activation_bank_result','model_identity':s.model_identity,'adapter_version':ADAPTER_VERSION,'site':'resid_post',
             'layers':{'0':{'donor':{'tokens':[0,1,2],'token_count':3,'values':values}}}}
    patch=InterchangePatch(activation_bank=SimpleNamespace(result=lambda:payload),write_site=ResidualInterventionSite('resid_post',(0,)),target_tokens=TokenSelector.full_sequence(),donor_tokens=TokenSelector.full_sequence())
    spec=PatchedGenerationSpec(engine=engine,dataset=dataset,patch=patch,pair_by=dataset.labels('pair'),target_when=dataset.labels('arm').equals('target'),donor_when=dataset.labels('arm').equals('donor'),generation=GenerationSpec(enabled=True,max_tokens=1,target_token_ids=(2,7)))
    result=engine.intervene(spec,session=s).rows[0]
    def independent(module,args,out):
        replacement=states['resid_post',0].clone()
        return (replacement,*out[1:]) if isinstance(out,tuple) else replacement
    h=s.model.transformer.h[0].register_forward_hook(independent)
    try:
        with torch.no_grad():
            expected=torch.log_softmax(s.model(s.encode(['1 2 3'])[0],use_cache=False).logits[0,-1],dim=-1)
    finally:h.remove()
    assert result['target_logprobs']['2']==pytest.approx(float(expected[2]),abs=1e-5)
    assert result['target_logprobs']['7']==pytest.approx(float(expected[7]),abs=1e-5)
    payload['model_identity']='different-checkpoint'
    with pytest.raises(ValueError,match='checkpoint'):
        engine.intervene(spec,session=s)
    payload['model_identity']=s.model_identity
    payload['layers']['0']['donor']['tokens']=[0,2]
    with pytest.raises(ValueError,match='missing requested token'):
        engine.intervene(spec,session=s)


def test_add_direction_matches_independent_raw_vector_edit(tiny_torch_session,tmp_path):
    from types import SimpleNamespace
    import torch
    from pipelines_v2.api import TorchEngine,Dataset,Example,GenerationSpec,PatchedGenerationSpec,AddDirectionPatch,ResidualInterventionSite
    s=tiny_torch_session;engine=TorchEngine(str(tmp_path),'a'*40)
    vector=torch.linspace(-.2,.2,16)
    direction=SimpleNamespace(result=lambda:{'kind':'direction_result','layers':{'0':{'raw_vector':vector.tolist()}}})
    dataset=Dataset.from_examples([Example(key='a',prompt='1 4 5')])
    spec=PatchedGenerationSpec(engine=engine,dataset=dataset,patch=AddDirectionPatch(direction=direction,write_site=ResidualInterventionSite('resid_pre',(0,)),strength=.7),generation=GenerationSpec(enabled=True,max_tokens=1,target_token_ids=(3,)))
    actual=engine.intervene(spec,session=s).rows[0]['target_logprobs']['3']
    def independent(module,args):
        changed=args[0].clone();changed[:,-1,:]+=.7*vector;return (changed,*args[1:])
    h=s.model.transformer.h[0].register_forward_pre_hook(independent)
    try:
        with torch.no_grad():
            expected=torch.log_softmax(s.model(s.encode(['1 4 5'])[0],use_cache=False).logits[0,-1],dim=-1)[3]
    finally:h.remove()
    assert actual==pytest.approx(float(expected),abs=1e-5)
