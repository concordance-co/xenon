"""Original-activation-preserving feature deltas; reconstruction is a separate arm."""
from __future__ import annotations
import numpy as np
from .dictionary import centered


def splice(dictionary, original, *, mode='delta', feature_edits=None):
    x = np.asarray(original, dtype=np.float32)
    if x.shape[-1] != dictionary.d_in or not np.isfinite(x).all(): raise ValueError('Invalid original residual')
    edits = feature_edits or {}
    if mode == 'noop':
        if edits: raise ValueError('No-op cannot contain feature edits')
        return x.copy()  # No encode/decode/recentering roundoff in the control.
    if mode not in {'delta', 'reconstruct'}: raise ValueError('Use noop, delta or reconstruct')
    z = dictionary.encode_original(x)
    delta = np.zeros_like(x)
    for index, amount in edits.items():
        if type(index) is not int or not 0 <= index < dictionary.d_sae or not np.isfinite(amount):
            raise ValueError('Feature edit needs valid index and finite additive amount')
        delta += np.float32(amount) * dictionary.weights['W_dec'][index]
    if mode == 'delta': return x + centered(delta)
    return centered(dictionary.decode(z)+delta) + x.mean(-1, keepdims=True)


def torch_splice(dictionary, *, mode='delta', feature_edits=None, token_positions=None):
    """Scoped FP32 edit on the original residual, on CPU or CUDA."""
    import torch
    edits = feature_edits or {}
    if mode not in {'noop','delta','reconstruct'} or (mode=='noop' and edits):
        raise ValueError('Invalid splice mode/edits')
    if any(type(i) is not int or not 0<=i<dictionary.d_sae or not np.isfinite(a) for i,a in edits.items()):
        raise ValueError('Invalid feature edit')
    cached={}
    def edit(value):
        if value.dtype != torch.float32: raise ValueError('SAE splice requires FP32')
        if mode=='noop': return value.clone()
        if value.device not in cached:
            # Transfer dictionary once per scoped callback; no CPU activation roundtrip.
            cached[value.device]={k:torch.as_tensor(v,device=value.device) for k,v in dictionary.weights.items()}
        weights=cached[value.device]
        original=value if token_positions is None else value[:,list(token_positions)]
        delta=torch.zeros(dictionary.d_in,device=value.device,dtype=value.dtype)
        for index,amount in edits.items(): delta.add_(weights['W_dec'][index],alpha=float(amount))
        if mode=='delta': changed=original+(delta-delta.mean())
        else:
            mean=original.mean(-1,keepdim=True)
            codes=torch.relu((original-mean-weights['b_dec'])@weights['W_enc']+weights['b_enc'])
            decoded=codes@weights['W_dec']+weights['b_dec']+delta
            changed=decoded-decoded.mean(-1,keepdim=True)+mean
        if token_positions is None:return changed
        result=value.clone();result[:,list(token_positions)]=changed
        return result
    return edit
