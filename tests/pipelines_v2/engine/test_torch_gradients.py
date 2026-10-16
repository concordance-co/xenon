import pytest


def test_differentiable_input_and_residual_mask_finite_difference(tiny_torch_session):
    import torch
    s=tiny_torch_session;ids,mask=s.encode(['1 2 3'])
    embeds=s.model.transformer.wte(ids).detach().requires_grad_(True)
    scale=torch.tensor(.8,requires_grad=True)
    def edit(value):
        output=value.clone();output[:,-1,:]=value[:,-1,:]*scale;return output
    logits,_=s.forward(inputs_embeds=embeds,attention_mask=mask,edits={('resid_post',0):edit},differentiable=True)
    score=logits[0,-1,4]-logits[0,-1,7]
    derivative,input_gradient=torch.autograd.grad(score,(scale,embeds))
    assert bool(input_gradient.isfinite().all()) and float(input_gradient.abs().sum())>0
    epsilon=1e-3
    # Match only the selected last coordinate in the independent finite difference.
    def evaluate(value):
        def multiply(hidden):
            changed=hidden.clone();changed[:,-1,:]*=value;return changed
        logits,_=s.forward(ids,mask,edits={('resid_post',0):multiply})
        return float(logits[0,-1,4]-logits[0,-1,7])
    finite=(evaluate(.8+epsilon)-evaluate(.8-epsilon))/(2*epsilon)
    assert float(derivative)==pytest.approx(finite,rel=2e-2,abs=2e-5)
    assert all(parameter.grad is None for parameter in s.model.parameters())
