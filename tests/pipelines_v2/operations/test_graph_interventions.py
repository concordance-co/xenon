from dataclasses import replace
import pytest
from tests.pipelines_v2.engine.test_native_sites import session_for,site_for
from pipelines_v2.operations.common.native import HEAD_QUANTITY
from pipelines_v2.operations.graphs import NativeRouteSpec,RoutePair,AttributionGraph
from pipelines_v2.engine.native import NativeTorchEngine
from pipelines_v2.operations.graphs.execution import evaluate_routes


def fixture_spec(session,ig_steps=0):
    return NativeRouteSpec(engine=NativeTorchEngine('/unused','607a30d783dfa663caf39e06633721c8d4cfcd7e','gpt2'),
        sites=(site_for(session,HEAD_QUANTITY,units=(1,)),),
        pairs=(RoutePair('heldout-0','heldout-template','1 2 3','4','5','2 1 3','1 3 2','6 7 8',{'fixture':True}),),
        proposal_templates=('proposal-template',),ig_steps=ig_steps)


def test_native_effect_matches_independent_forward_and_identity_is_exact():
    import torch
    with session_for('gpt2') as session:
        result=evaluate_routes(fixture_spec(session,ig_steps=2),session)
        graph=AttributionGraph.from_dict(result['graph'])
        ids,mask=session.encode(['1 2 3'])
        def explicit(module,args):
            value=args[0].clone();value[:,-1,8:16]=0
            return (value,*args[1:])
        handle=session.model.transformer.h[0].attn.c_proj.register_forward_pre_hook(explicit)
        try:
            with torch.no_grad():logits=session.model(ids,use_cache=False).logits
        finally:handle.remove()
        zero=next(e for e in graph.evidence if e.arm=='zero')
        assert zero.observed==pytest.approx(float(logits[0,-1,4]-logits[0,-1,5]),abs=1e-7)
        identity=next(e for e in graph.evidence if e.arm=='identity')
        assert identity.observed==identity.baseline
        assert identity.collateral['max_logit_change']==0
        assert {e.arm for e in graph.evidence}=={'identity','zero','opposite','same_label','unmatched'}
        assert all(e.scientific_verdict=='not_tested' for e in graph.evidence)
        assert graph.candidates[0]['basis']=='input_token_embedding'
        assert not session.model.transformer.h[0].attn.c_proj._forward_pre_hooks


def test_embedding_gradients_match_analytical_linear_oracle():
    import torch
    from pipelines_v2.operations.graphs.gradients import embedding_candidates
    x=torch.tensor([[[1.,2.],[3.,4.]]],dtype=torch.float64)
    weights=torch.tensor([[[2.,-1.],[.5,3.]]],dtype=torch.float64)
    result=embedding_candidates(lambda value:(value*weights).sum()+9,x,steps=4)
    assert result['input_times_gradient']==pytest.approx([0.,13.5])
    assert result['integrated_gradients']==pytest.approx([0.,13.5])
    assert result['target_delta']==pytest.approx(13.5)
    assert abs(result['completeness_residual'])<1e-12


def test_spec_codec_and_disjoint_template_guard():
    from pipelines_v2.operations.registry import operation_spec_from_dict
    with session_for('gpt2') as session:
        spec=fixture_spec(session)
        assert operation_spec_from_dict(spec.to_dict())==spec
        with pytest.raises(ValueError,match='disjoint'):replace(spec,proposal_templates=('heldout-template',))
        with pytest.raises(ValueError,match='budget'):replace(spec,ig_steps=17,max_work_units=1)


def test_identity_failure_is_reported_rejected_without_scientific_promotion():
    from unittest.mock import patch
    with session_for('gpt2') as session:
        original=session.forward
        def faulty(*args,**kwargs):
            logits,values=original(*args,**kwargs)
            if kwargs.get('edits'):
                logits=logits.clone();logits[0,-1,4]+=.01
            return logits,values
        with patch.object(session,'forward',faulty):payload=evaluate_routes(fixture_spec(session),session)
        assert payload['engineering_status']=='rejected'
        assert payload['scientific_verdict']=='not_tested'
        assert next(e for e in payload['graph']['evidence'] if e['arm']=='identity')['engineering_status']=='rejected'
