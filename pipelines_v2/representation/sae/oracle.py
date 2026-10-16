"""Optional isolated SAELens/TransformerLens oracle, never a runtime dependency.

Run with the pinned reference stack; checkpoint loading is strictly offline.
AutoConfig redirection supplies the exact prepared config to TL's config lookup,
which otherwise accesses the Hub even when hf_model is already supplied.
No reference tensor arithmetic or weight conversion is replaced.
"""
import json
from pathlib import Path
from .dictionary import (SAEDictionary, centered, ADAPTER_VERSION, MODEL_REVISION,
    SITE, WEIGHTS_SHA256, verify_original_model, file_digest, CONFIG_SHA256)


def run_oracle(model_path, sae_path):
    import importlib.metadata
    import numpy as np
    import torch
    from unittest.mock import patch
    from transformers import GPT2LMHeadModel, AutoTokenizer
    from transformer_lens import HookedTransformer
    from sae_lens.saes.standard_sae import StandardSAE, StandardSAEConfig
    from safetensors.numpy import load_file
    versions={name:importlib.metadata.version(name) for name in ('torch','transformers','transformer-lens')}
    import sae_lens
    versions['sae-lens']=sae_lens.__version__
    if versions!={'torch':'2.12.0','transformers':'5.9.0','transformer-lens':'3.2.1','sae-lens':'6.54.5'}:
        raise ValueError(f'Unmeasured reference stack: {versions}')
    model_digest=verify_original_model(model_path)
    if file_digest(Path(sae_path)/'sae_weights.safetensors')!=WEIGHTS_SHA256 or file_digest(Path(sae_path)/'cfg.json')!=CONFIG_SHA256:
        raise ValueError('Reference dictionary bytes mismatch')
    torch.set_num_threads(4)
    hf=GPT2LMHeadModel.from_pretrained(model_path,local_files_only=True,attn_implementation='eager').eval()
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    with patch('transformer_lens.loading_from_pretrained.AutoConfig.from_pretrained',return_value=hf.config):
        tl=HookedTransformer.from_pretrained('gpt2-small',hf_model=hf,tokenizer=tokenizer,device='cpu',dtype=torch.float32,
            fold_ln=True,center_writing_weights=True,center_unembed=True,fold_value_biases=True,refactor_factored_attn_matrices=False)
    weights=load_file(str(Path(sae_path)/'sae_weights.safetensors'))
    adapter=SAEDictionary(weights,{'architecture':'standard','activation':'relu','normalization':'none'})
    reference=StandardSAE(StandardSAEConfig(d_in=768,d_sae=24576,device='cpu',dtype='float32',normalize_activations='none',apply_b_dec_to_input=True)).eval()
    reference.load_state_dict({name:torch.from_numpy(value) for name,value in weights.items()},strict=True)
    rows=[]
    for prompt in ('The capital of France is','A blue bird flew over the'):
        ids=tokenizer.encode(prompt,return_tensors='pt',add_special_tokens=False)
        captured=[]
        handle=hf.transformer.h[6].register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
        try:
            with torch.no_grad():
                hf_logits=hf(ids).logits
                tl_logits,cache=tl.run_with_cache(ids,names_filter=[SITE])
                target=cache[SITE]; mapped=centered(captured[0].numpy())
                encoded=adapter.encode(mapped); ref_encoded=reference.encode(target)
                ref_same=reference.encode(torch.from_numpy(mapped))
                decoded=adapter.decode(encoded); ref_decoded=reference.decode(torch.from_numpy(encoded))
        finally:handle.remove()
        assert torch.allclose(torch.from_numpy(mapped),target,atol=5e-4,rtol=1e-4), 'HF→TL activation map failed'
        assert np.allclose(encoded,ref_encoded.numpy(),atol=0.005,rtol=1e-4), 'Mapped encoding differs from reference coordinates'
        assert np.allclose(encoded,ref_same.numpy(),atol=0.005,rtol=1e-4), 'Encoder math differs from SAELens'
        assert np.allclose(decoded,ref_decoded.numpy(),atol=0.005,rtol=1e-4), 'Decoder math differs from SAELens'
        rows.append({'prompt':prompt,'tensor_max_abs':float(np.abs(mapped-target.numpy()).max()),
            'mapped_encode_max_abs':float(np.abs(encoded-ref_encoded.numpy()).max()),
            'encode_max_abs':float(np.abs(encoded-ref_same.numpy()).max()),
            'decode_max_abs':float(np.abs(decoded-ref_decoded.numpy()).max()),
            'centered_logit_max_abs':float(np.abs(centered(hf_logits.numpy())-centered(tl_logits.numpy())).max())})
    return {'adapter_version':ADAPTER_VERSION,'model_checkpoint_digest':model_digest,'model_revision':MODEL_REVISION,
        'dictionary_weights_sha256':WEIGHTS_SHA256,'site':SITE,'coordinate_map':'per_token_center','passed':True,
        'prompt_count':len(rows),'reference_versions':versions,'rows':rows,
        **{name:max(row[name] for row in rows) for name in ('tensor_max_abs','encode_max_abs','decode_max_abs','mapped_encode_max_abs')}}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--sae',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=run_oracle(args.model,args.sae)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
