"""Tool-interpreter-only bridge. No upstream torch objects enter control Python."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify_files(root,files):
    for name,expected in files.items():
        path=Path(root)/name
        if path.stat().st_size!=expected['bytes']:raise ValueError('Pinned asset size mismatch: '+name)
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
        if h.hexdigest()!=expected['sha256']:raise ValueError('Pinned asset digest mismatch: '+name)


def check_runtime(request):
    import circuit_tracer
    root=Path(circuit_tracer.__file__).parent.parent
    for name,expected in request['tool_lock']['sources'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise ValueError('Circuit-tracer source mismatch: '+name)
    versions={}
    for line in request['requirements'].splitlines():
        if '==' not in line:continue
        name,version=line.split('==',1);actual=importlib.metadata.version(name)
        if actual!=version:raise ValueError('Tool dependency mismatch: '+name)
        versions[name]=actual
    direct=json.loads(importlib.metadata.distribution('circuit-tracer').read_text('direct_url.json'))
    if direct.get('vcs_info',{}).get('commit_id')!=request['tool_lock']['revision']:raise ValueError('Circuit-tracer source commit mismatch')
    from circuit_tracer import ReplacementModel,attribute
    return versions


def run(request):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from circuit_tracer import ReplacementModel,attribute
    from circuit_tracer.transcoder import load_transcoder_set
    from circuit_tracer.graph import compute_graph_scores
    versions=check_runtime(request)
    scientific=request['scientific'];pair=scientific['pair'];output=Path(request['output_dir'])
    for role in ('model','dictionary'):verify_files(request[role+'_path'],pair[role]['files'])
    # Cache only symlinks to the already verified immutable snapshot. No Hub fallback.
    hub=output/'offline-hub';snapshot=hub/'models--Qwen--Qwen3-0.6B'/'snapshots'/pair['model']['revision'];snapshot.mkdir(parents=True)
    for name in pair['model']['files']:(snapshot/name).symlink_to(Path(request['model_path'])/name)
    torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('This exact tracing profile requires CUDA; no CPU fallback')
    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    dtype=torch.float32
    tokenizer=AutoTokenizer.from_pretrained(request['model_path'],local_files_only=True)
    hf=AutoModelForCausalLM.from_pretrained(request['model_path'],local_files_only=True,torch_dtype=dtype,attn_implementation='eager').eval()
    transcoders=load_transcoder_set({i:str(Path(request['dictionary_path'])/f'layer_{i}.safetensors') for i in range(28)},
        scan_name=pair['dictionary']['repository']+'@'+pair['dictionary']['revision'],feature_input_hook='mlp.hook_in',feature_output_hook='mlp.hook_out',
        dtype=dtype,device=torch.device('cuda'),lazy_encoder=False,lazy_decoder=False)
    model=ReplacementModel.from_pretrained_and_transcoders(pair['model']['repository'],transcoders,backend='transformerlens',
        hf_model=hf,tokenizer=tokenizer,revision=pair['model']['revision'],cache_dir=str(hub),local_files_only=True,
        device=torch.device('cuda'),dtype=dtype,fold_value_bias=False,refactor_factored_attn_matrices=False)
    tokens=model.ensure_tokenized(scientific['prompt'])
    if tokens.numel()>scientific['max_tokens']:raise ValueError('Prompt exceeds frozen trace token ceiling')
    with torch.no_grad():
        baseline=model(tokens).detach()
        hf=hf.to('cuda');native=hf(tokens[None],use_cache=False).logits.detach()
        parity=torch.allclose(baseline,native,rtol=1e-4,atol=1e-5)
        parity_error=float((baseline-native).abs().max())
    del hf,native;torch.cuda.empty_cache()
    graph=attribute(tokens,model,max_n_logits=1,batch_size=1,max_feature_nodes=scientific['max_feature_nodes'],offload=None,verbose=True)
    graph.to_pt(str(output/'upstream_graph.pt'))
    replacement_score,completeness_score=compute_graph_scores(graph)
    # One diagnostic forward through PLTs without error correction, retaining the ignored BOS unchanged.
    cache={};hooks=[]
    for layer in range(28):
        def read(value,hook,layer=layer):cache[layer]=value
        def replace(value,hook,layer=layer):
            x=cache.pop(layer);reconstruction=transcoders[layer].decode(transcoders[layer].encode(x),x)
            reconstruction[:,0]=value[:,0]
            return reconstruction
        hooks.extend([(f'blocks.{layer}.mlp.hook_in',read),(f'blocks.{layer}.mlp.hook_out',replace)])
    with torch.no_grad():replaced=model.run_with_hooks(tokens,fwd_hooks=hooks)
    p=torch.log_softmax(baseline[0,-1],-1);q=torch.log_softmax(replaced[0,-1],-1)
    fidelity={'scope':'same traced prompt only; not held-out qualification','without_error_correction':True,
        'max_logit_error':float((baseline-replaced).abs().max()),'next_token_kl':max(0.,float((p.exp()*(p-q)).sum())),
        'graph_replacement_score':replacement_score,'graph_completeness_score':completeness_score}
    original={'status':'unvalidated','reason':'No matched-control campaign; only optional same-prompt engineering edit',
        'hf_transformerlens_parity':{'passed':bool(parity),'max_logit_error':parity_error,'rtol':1e-4,'atol':1e-5}}
    # Upstream feature_intervention adds a decoder delta to the original MLP output,
    # with all subsequent original computations free to change. No HF feature-name join.
    if parity and len(graph.selected_features):
        selected=int(graph.selected_features[0]);layer,pos,feature=map(int,graph.active_features[selected]);value=float(graph.activation_values[selected])
        with torch.no_grad():
            identity,_=model.feature_intervention(tokens,[(layer,pos,feature,value)],freeze_attention=False,return_activations=False)
            ablated,_=model.feature_intervention(tokens,[(layer,pos,feature,0.)],freeze_attention=False,return_activations=False)
        token=graph.logit_targets[0].vocab_idx
        original.update({'scope':'original TransformerLens model, one selected PLT decoder direction, no attention/normalization freezes',
            'coordinate':{'layer':layer,'position':pos,'feature':feature},'identity_max_logit_error':float((baseline-identity).abs().max()),
            'target_token_id':token,'baseline_logit':float(baseline[0,-1,token]),'ablated_logit':float(ablated[0,-1,token]),
            'matched_controls':'not_run','scientific_verdict':'not_tested'})
    raw={'schema_version':1,'producer':'circuit-tracer-v0.5.3','request_identity':digest(scientific),'scientific_request':scientific,
        'model_identity':digest(pair['model']),'dictionary_identity':digest(pair['dictionary']),'dictionary_type':'PLT','layer_map':pair['dictionary']['layer_map'],
        'input_string':graph.input_string,'input_tokens':graph.input_tokens.tolist(),'n_layers':int(graph.cfg.n_layers),
        'active_features':graph.active_features.tolist(),'activation_values':graph.activation_values.tolist(),'selected_features':graph.selected_features.tolist(),
        'adjacency_matrix':graph.adjacency_matrix.tolist(),'matrix_orientation':'target_rows_source_columns',
        'logit_targets':[x._asdict() for x in graph.logit_targets],'logit_probabilities':graph.logit_probabilities.tolist(),
        'scan_name':graph.scan_name,'fidelity':fidelity,'original_model_validation':original,
        'tool_packages':versions,'compute':{'elapsed_seconds':time.monotonic()-start,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'device':torch.cuda.get_device_name(),'dtype':'float32','tf32':False,'seed':0},
        'scientific_verdict':'not_tested'}
    (output/'upstream_graph.json').write_text(json.dumps(raw,allow_nan=False))
    binding={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ('upstream_graph.pt','upstream_graph.json')}
    (output/'producer_binding.json').write_text(json.dumps({'request_identity':digest(scientific),'outputs':binding}))
    print(json.dumps({'stage':'trace_complete','nodes':graph.adjacency_matrix.shape[0],'nonzero_edges':int(graph.adjacency_matrix.count_nonzero()),'compute':raw['compute']}))


if __name__=='__main__':
    request=json.loads(Path(sys.argv[2]).read_text())
    if sys.argv[1]=='check':print(json.dumps(check_runtime(request)))
    elif sys.argv[1]=='run':run(request)
    else:raise ValueError('Unknown bridge mode')
