import pytest
import torch
from pipelines_v2.engine.diffusers.hooks import DenoisingHook,internal_hooks


class TinyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__();self.site=torch.nn.Identity()
    def forward(self,sample,timestep):
        return self.site(sample+2)*3


def test_selected_step_cfg_half_spatial_cells_and_downstream_effect():
    model=TinyUNet();x=torch.arange(4*2*3*4,dtype=torch.float32).reshape(4,2,3,4)
    hook=DenoisingHook('site',(1,),(2,3,4),branch='conditional',operator='ablate',spatial_yx=(1,3,1,3))
    with internal_hooks(model,[hook],batch_size=2,cfg_enabled=True,timesteps=[99,42]) as state:
        first=model(x,99);second=model(x,42)
    assert torch.equal(first,(x+2)*3)
    expected=(x+2)*3;expected[2:,:,1:3,1:3]=0
    assert torch.equal(second,expected)
    assert torch.equal(state['captures'][(0,1)],(x+2)[2:])
    assert state['events'][0]['timestep_value']==42
    assert state['events'][0]['step_index']==1
    assert not model.site._forward_hooks and not model._forward_pre_hooks


@pytest.mark.parametrize('cfg,branch,rows',[(True,'conditional',slice(2,4)),(True,'unconditional',slice(0,2)),(True,'both',slice(0,4)),(False,'conditional',slice(0,2))])
def test_addition_matches_analytical_forward_and_preserves_other_branches(cfg,branch,rows):
    model=TinyUNet();x=torch.ones((4 if cfg else 2,2,3,4))
    hook=DenoisingHook('site',(0,),(2,3,4),branch=branch,operator='add',direction=(1,-2),strength=.5)
    with internal_hooks(model,[hook],batch_size=2,cfg_enabled=cfg,timesteps=[7]):y=model(x,7)
    expected=(x+2)*3;expected[rows]+=torch.tensor([1.5,-3])[None,:,None,None]
    assert torch.equal(y,expected)


def test_noop_and_matched_original_replacement_equal_direct_forward():
    model=TinyUNet();x=torch.randn(2,2,3,4);baseline=model(x,99)
    capture=DenoisingHook('site',(0,),(2,3,4),branch='both')
    with internal_hooks(model,[capture],batch_size=1,cfg_enabled=True,timesteps=[99]) as state:
        assert torch.equal(model(x,99),baseline)
    for hook,donor in [(DenoisingHook('site',(0,),(2,3,4),branch='both',operator='add',direction=(1,2),strength=0),{}),
                       (DenoisingHook('site',(0,),(2,3,4),branch='both',operator='replace'),state['captures'])]:
        with internal_hooks(model,[hook],batch_size=1,cfg_enabled=True,timesteps=[99],replacements=donor):
            assert torch.equal(model(x,99),baseline)


@pytest.mark.parametrize('failure',['shape','branch','path','step','replacement','exception'])
def test_wrong_layout_rejected_and_hooks_removed_on_every_exit(failure):
    model=TinyUNet();hook=DenoisingHook('site',(0,),(2,3,4))
    cfg=True;timesteps=[99]
    if failure=='branch':hook=DenoisingHook('site',(0,),(2,3,4),branch='unconditional');cfg=False
    if failure=='path':hook=DenoisingHook('missing',(0,),(2,3,4))
    if failure=='step':hook=DenoisingHook('site',(1,),(2,3,4))
    if failure=='replacement':hook=DenoisingHook('site',(0,),(2,3,4),operator='replace')
    with pytest.raises((ValueError,RuntimeError)):
        with internal_hooks(model,[hook],batch_size=1,cfg_enabled=cfg,timesteps=timesteps):
            if failure=='exception':raise RuntimeError('injected')
            model(torch.ones(2,2,3,5),99)
    assert not model.site._forward_hooks and not model._forward_pre_hooks


def test_timestep_not_step_index_and_missing_invocation_rejected():
    model=TinyUNet();hook=DenoisingHook('site',(0,),(2,3,4))
    with pytest.raises(ValueError,match='timestep'):
        with internal_hooks(model,[hook],batch_size=1,cfg_enabled=True,timesteps=[999]):model(torch.ones(2,2,3,4),0)
    with pytest.raises(ValueError,match='Incomplete'):
        with internal_hooks(model,[hook],batch_size=1,cfg_enabled=True,timesteps=[999]):pass
