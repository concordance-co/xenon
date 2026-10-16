"""SDXL UNet module-output hooks with explicit step, CFG and spatial axes."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any

ADAPTER_VERSION = 'sdxl-euler-unet-v1'


@dataclass(frozen=True)
class DenoisingHook:
    module_path: str
    steps: tuple[int, ...]
    shape_chw: tuple[int, int, int]
    branch: str = 'conditional'
    operator: str = 'capture'
    spatial_yx: tuple[int, int, int, int] | None = None
    direction: tuple[float, ...] = ()
    strength: float = 1.0

    def __post_init__(self):
        object.__setattr__(self,'steps',tuple(self.steps))
        object.__setattr__(self,'shape_chw',tuple(self.shape_chw))
        object.__setattr__(self,'direction',tuple(self.direction))
        if self.spatial_yx is not None:object.__setattr__(self,'spatial_yx',tuple(self.spatial_yx))
        if not self.module_path or self.module_path.startswith('.'):
            raise ValueError('Declare a UNet module output path')
        if not self.steps or len(set(self.steps)) != len(self.steps) or any(type(x) is not int or x<0 for x in self.steps):
            raise ValueError('Denoising steps must be unique nonnegative indices')
        if len(self.shape_chw)!=3 or any(type(x) is not int or x<=0 for x in self.shape_chw):
            raise ValueError('Declare positive channel,y,x shape')
        if self.branch not in {'conditional','unconditional','both'}:
            raise ValueError('Unsupported CFG branch')
        if self.operator not in {'capture','add','ablate','replace'}:
            raise ValueError('Unsupported internal hook operator')
        if type(self.strength) not in (int,float) or not math.isfinite(self.strength):
            raise ValueError('Hook strength must be finite')
        if self.operator=='add' and (len(self.direction)!=self.shape_chw[0] or not all(math.isfinite(x) for x in self.direction)):
            raise ValueError('Addition needs one finite direction value per channel')
        if self.operator!='add' and self.direction:
            raise ValueError('Direction is only meaningful for addition')
        if self.spatial_yx is not None:
            y0,y1,x0,x1=self.spatial_yx
            if any(type(x) is not int for x in self.spatial_yx) or not (0<=y0<y1<=self.shape_chw[1] and 0<=x0<x1<=self.shape_chw[2]):
                raise ValueError('Spatial bounds must lie within declared feature coordinates')

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)

    def site(self, model_identity, *, dtype='float16'):
        return {'kind':'diffusion_unet','schema_version':1,'model_identity':model_identity,
                'adapter_version':ADAPTER_VERSION,'module_path':self.module_path,'layer':None,
                'quantity':'module_output','selection':{'steps':self.steps,'branch':self.branch,
                    'spatial_yx':self.spatial_yx,'coordinate_space':'feature'},
                'axes':['batch','channel','y','x'],'read_or_write':'read' if self.operator=='capture' else 'write',
                'phase':'denoising','dtype':dtype}


@contextmanager
def internal_hooks(unet, hooks, *, batch_size, cfg_enabled, timesteps, replacements=None):
    """No Diffusers import. Each UNet forward verifies the declared branch layout.

    Hook outputs are cloned before writes; unselected cells retain exact values.
    Stored captures are pre-edit full branch tensors, keyed by hook and step.
    """
    import torch
    hooks=tuple(hooks); replacements=replacements or {}
    modules=dict(unet.named_modules()); handles=[]
    state={'step':-1,'timestep':None,'captures':{},'events':[],'calls':{}}
    expected_batch=batch_size*(2 if cfg_enabled else 1)
    if type(batch_size) is not int or batch_size<1:raise ValueError('Positive prompt batch size required')
    for i,hook in enumerate(hooks):
        if hook.module_path not in modules:raise ValueError(f'Unknown UNet site: {hook.module_path}')
        if max(hook.steps)>=len(timesteps):raise ValueError('Hook step outside scheduler trajectory')
        if not cfg_enabled and hook.branch!='conditional':raise ValueError('CFG-disabled inputs only have conditional branch')
        if hook.operator=='replace':
            for step in hook.steps:
                donor=replacements.get((i,step))
                rows=batch_size*(2 if hook.branch=='both' else 1)
                if not isinstance(donor,torch.Tensor) or tuple(donor.shape)!=(rows,*hook.shape_chw) or not torch.isfinite(donor).all():
                    raise ValueError('Replacement requires finite matched full branch BCHW activations for every step')

    def before(module,args,kwargs):
        sample=args[0] if args else kwargs['sample']
        timestep=args[1] if len(args)>1 else kwargs['timestep']
        state['step']+=1
        index=state['step']
        if sample.ndim!=4 or sample.shape[0]!=expected_batch or index>=len(timesteps):
            raise ValueError('Unsupported UNet batch/CFG invocation layout')
        value=float(timestep.item() if hasattr(timestep,'item') else timestep)
        expected=float(timesteps[index])
        if value!=expected:raise ValueError('UNet call does not match declared scheduler step/timestep')
        state['timestep']=value

    def callback(i,hook):
        def apply(module,args,output):
            if not isinstance(output,torch.Tensor) or tuple(output.shape)!=(expected_batch,*hook.shape_chw):
                raise ValueError(f'UNet site {hook.module_path} output does not match declared BCHW axes')
            step=state['step']
            if step not in hook.steps:return output
            key=(i,step)
            if key in state['calls']:raise ValueError('Site fired more than once for a denoising step')
            state['calls'][key]=1
            rows=slice(None) if hook.branch=='both' else slice(batch_size,None) if cfg_enabled and hook.branch=='conditional' else slice(0,batch_size)
            branch=output[rows]
            state['captures'][key]=branch.detach().cpu().clone()
            event={'hook_index':i,'step_index':step,'timestep_value':state['timestep'],'branch':hook.branch,
                   'shape':list(output.shape),'operator':hook.operator,'changed_cells':0}
            state['events'].append(event)
            if hook.operator=='capture':return output
            y0,y1,x0,x1=hook.spatial_yx or (0,hook.shape_chw[1],0,hook.shape_chw[2])
            result=output.clone()
            target=result[rows,:,y0:y1,x0:x1]
            if hook.operator=='add':
                direction=torch.tensor(hook.direction,device=output.device,dtype=output.dtype)[None,:,None,None]
                target.add_(hook.strength*direction)
            elif hook.operator=='ablate':target.zero_()
            else:target.copy_(replacements[key].to(output.device,output.dtype)[:,:,y0:y1,x0:x1])
            if not torch.isfinite(result).all():raise ValueError('Intervention produced nonfinite activations')
            event['changed_cells']=int((result!=output).count_nonzero().item())
            return result
        return apply
    try:
        handles.append(unet.register_forward_pre_hook(before,with_kwargs=True))
        for i,hook in enumerate(hooks):handles.append(modules[hook.module_path].register_forward_hook(callback(i,hook)))
        yield state
        if state['step']!=len(timesteps)-1:raise ValueError('Incomplete denoising trajectory')
        if set(state['calls'])!={(i,step) for i,hook in enumerate(hooks) for step in hook.steps}:
            raise ValueError('Requested internal hooks did not fire')
    finally:
        for handle in reversed(handles):handle.remove()
