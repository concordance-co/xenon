from dataclasses import replace
import pytest
from tests.pipelines_v2.engine.test_native_sites import session_for,site_for
from pipelines_v2.api import NativeCoordinatePatch,NativeInterventionSite,TokenSelector
from pipelines_v2.operations.common.native import HEAD_QUANTITY,MLP_QUANTITIES


def patch_for(site,mode='replace',value=-2.):
    return NativeCoordinatePatch(write_site=NativeInterventionSite.from_capture(site),target_tokens=site.tokens,mode=mode,value=value)


@pytest.mark.parametrize('quantity',MLP_QUANTITIES)
def test_qwen_selected_edit_matches_rewritten_forward_and_preserves_neighbors(quantity):
    import torch
    from types import MethodType
    with session_for('qwen3') as s:
        ids,mask=s.encode(['1 2 3']);selected=site_for(s,quantity,units=(2,))
        all_sites=tuple(site_for(s,q,units=tuple(range(24)),tokens=TokenSelector.full_sequence()) for q in MLP_QUANTITIES)
        baseline,original=s.forward(ids,mask,capture=all_sites)
        noop,_=s.forward(ids,mask,edits=(patch_for(selected,'scale',1.),))
        torch.testing.assert_close(noop,baseline,rtol=0,atol=0)
        actual,observed=s.forward(ids,mask,capture=all_sites,edits=(patch_for(selected),))
        mlp=s.model.model.layers[0].mlp;original_forward=mlp.forward
        def explicit(module,x):
            gate=module.gate_proj(x)
            if quantity=='gate_preactivation':gate=gate.clone();gate[:,-1,2]=-2.
            activated=torch.nn.functional.silu(gate)
            if quantity=='gate_activation':activated=activated.clone();activated[:,-1,2]=-2.
            up=module.up_proj(x)
            if quantity=='up_projection':up=up.clone();up[:,-1,2]=-2.
            product=activated*up
            if quantity=='mlp_product':product=product.clone();product[:,-1,2]=-2.
            return module.down_proj(product)
        mlp.forward=MethodType(explicit,mlp)
        try:
            with torch.no_grad():expected=s.model(ids,use_cache=False).logits
        finally:del mlp.forward
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        before=original[quantity][0];after=observed[quantity][0]
        assert after[-1,2]==-2.
        mask_units=torch.ones_like(after,dtype=torch.bool);mask_units[-1,2]=False
        torch.testing.assert_close(after[mask_units],before[mask_units],rtol=0,atol=0)
        if quantity=='mlp_product':torch.testing.assert_close(original['gate_activation'][0],observed['gate_activation'][0],rtol=0,atol=0)
        if quantity=='gate_preactivation':
            expected_product=torch.nn.functional.silu(observed['gate_preactivation'][0])*observed['up_projection'][0]
            torch.testing.assert_close(observed['mlp_product'][0],expected_product,rtol=0,atol=0)
        assert 'forward' not in mlp.__dict__


def test_head_ablation_is_selected_and_matches_independent_projection_input_hook():
    import torch
    with session_for('gpt2') as s:
        ids,mask=s.encode(['1 2 3']);site=site_for(s,HEAD_QUANTITY,units=(1,))
        read=site_for(s,HEAD_QUANTITY,units=(0,1),tokens=TokenSelector.full_sequence())
        _,before=s.forward(ids,mask,capture=(read,))
        actual,after=s.forward(ids,mask,capture=(read,),edits=(patch_for(site,'scale',0.),))
        def explicit(module,args):
            out=args[0].clone();out[:,-1,8:16]=0;return (out,*args[1:])
        h=s.model.transformer.h[0].attn.c_proj.register_forward_pre_hook(explicit)
        try:
            with torch.no_grad():expected=s.model(ids,use_cache=False).logits
        finally:h.remove()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(after[HEAD_QUANTITY][0][:,0],before[HEAD_QUANTITY][0][:,0],rtol=0,atol=0)
        assert torch.count_nonzero(after[HEAD_QUANTITY][0][-1,1])==0


@pytest.mark.parametrize('architecture,quantity',[('gpt2',HEAD_QUANTITY),('qwen3','mlp_product')])
def test_native_gradients_cleanup_and_identity_failure(architecture,quantity):
    import torch
    with session_for(architecture) as s:
        ids,mask=s.encode(['1 2 3']);site=site_for(s,quantity,units=(0,))
        embeddings=s.model.get_input_embeddings()(ids).detach().requires_grad_(True)
        logits,capture=s.forward(inputs_embeds=embeddings,attention_mask=mask,capture=(site,),differentiable=True)
        capture[quantity][0].sum().backward()
        assert embeddings.grad is not None and embeddings.grad.abs().sum()>0
        assert all(p.grad is None and not p.requires_grad for p in s.model.parameters())
        with pytest.raises(ValueError,match='identity mismatch'):s.forward(ids,mask,capture=(replace(site,model_identity='wrong'),))
        with pytest.raises(ValueError,match='match selected'):s.forward(ids,mask,edits=(patch_for(site,value=[1.,2.,3.,4.,5.]),))
        if architecture=='qwen3':assert all('forward' not in layer.mlp.__dict__ for layer in s.model.model.layers)
        else:assert all(not layer.attn.c_proj._forward_pre_hooks for layer in s.model.transformer.h)
        s.forward(ids,mask)


def test_patch_codec_roundtrip_and_unadvertised_operator_rejection():
    from pipelines_v2.operations.interventions.recipes import ActivationPatchSpec
    from pipelines_v2.core.types import to_primitive
    with session_for('qwen3') as s:
        patch=patch_for(site_for(s,'gate_activation',units=(2,)))
        assert ActivationPatchSpec.from_dict(to_primitive(patch))==patch
        with pytest.raises(ValueError):replace(patch,mode='project')
        with pytest.raises(ValueError):replace(patch,target_tokens=TokenSelector.full_sequence())
