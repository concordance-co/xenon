"""Standalone tool-interpreter entrypoint. Deliberately never imports Xenon."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import pathlib
import runpy
import sys
import time


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'value': None, 'undefined_reason': 'official evaluator returned a nonfinite metric'}
    if isinstance(value, dict):
        return {k: json_safe(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if hasattr(value, 'item'):
        return json_safe(value.item())
    return value


def export(output, model='gpt2'):
    import torch
    # Only read output of the just-completed, pinned producer; avoid general pickle loading.
    saved = torch.load(output/f'ioi_{model}_scores.pt', map_location='cpu', weights_only=True)
    payload = {'metrics': json_safe(saved['mib_results']), 'args': saved['args'],
               'train_time_seconds': saved['train_time_s'],
               'installed_packages': dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions()))}
    (output/'official_metrics.json').write_text(json.dumps(payload, allow_nan=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','export','check']);p.add_argument('request')
    a=p.parse_args();request=json.loads(pathlib.Path(a.request).read_text())
    sys.path.insert(0,str(pathlib.Path(request['tool_root'])/'src'))
    sys.path.insert(0,request['mib_root']);sys.path.insert(0,str(pathlib.Path(request['mib_root'])/'EAP-IG/src'))
    if a.mode=='check':
        import inspect
        from MIB_circuit_track.evaluation import evaluate_area_under_curve
        from eap.graph import Graph
        from matryoshka_attribution import learn_scores
        expected='return weighted_edge_counts, area_under, area_from_1, average, faithfulnesses, accuracies, acc_auc'
        if expected not in inspect.getsource(evaluate_area_under_curve):
            raise RuntimeError('Wrong MIB evaluator signature/source')
        print(json.dumps({'status':'passed','python':sys.version.split()[0]}));return
    output=pathlib.Path(request['output_dir']);output.mkdir(parents=True,exist_ok=True)
    model=request['scientific']['model']['name']
    model_path={'gpt2':'gpt2','qwen2.5':'Qwen/Qwen2.5-0.5B'}[model]
    observed_assets={}
    if a.mode=='run':
        for role,root in [('model',pathlib.Path.cwd()/model_path),('data',pathlib.Path.cwd()/'mib-bench/ioi')]:
            observed_assets[role]={}
            for name,expected in request['scientific'][role]['files'].items():
                h=hashlib.sha256()
                with (root/name).open('rb') as stream:
                    for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
                observed_assets[role][name]=h.hexdigest()
                if h.hexdigest()!=expected:raise RuntimeError('Actual tool input differs from request')
        # Validate upstream IOI filtering and prompt shape before loading weights.
        from transformers import AutoTokenizer,AutoConfig
        tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        config=AutoConfig.from_pretrained(model_path,local_files_only=True)
        context=getattr(config,'max_position_embeddings',getattr(config,'n_positions',None))
        split_counts={}
        for filename in ('train.json','dev.json'):
            rows=json.loads((pathlib.Path('mib-bench/ioi')/filename).read_text())
            kept=[row for row in rows if len({len(tokenizer.encode(' '+row['metadata'][k],add_special_tokens=False)) for k in ('indirect_object','subject','random_c')})==1]
            if not kept or (filename=='dev.json' and len(kept)<request['scientific']['evaluation']['examples']):
                raise ValueError('Requested IOI examples exceed rows surviving upstream tokenizer filter')
            if context and any(not 0<len(tokenizer.encode(text,add_special_tokens=False))<=context for row in kept for text in (row['prompt'],row['s2_io_flip_counterfactual']['prompt'])):
                raise ValueError('IOI prompt exceeds model context')
            split_counts[filename]=len(kept)
        import torch
        cuda=torch.cuda.is_available()
        if cuda:torch.cuda.reset_peak_memory_stats()
        cfg=request['scientific'];fit=cfg['fitting']
        argv=[str(pathlib.Path(request['tool_root'])/'scripts/mib/eval_mib.py'),
              '--mib-path',request['mib_root'],'--model',model,'--task','ioi',
              '--steps',str(fit['steps']),'--eval-examples',str(cfg['evaluation']['examples']),
              '--batch-size','1','--train-batch-size','1','--include-input','--mode','iso',
              '--masking','topk','--k-schedule','uniform','--lr',str(fit['lr']),
              '--seed',str(cfg['seed']),'--split','validation','--train-split','train',
              '--cf-cache-gb','0','--no-wandb','--output',str(output)]
        sys.argv=argv;start=time.monotonic();runpy.run_path(argv[0],run_name='__main__')
        device={'available_filtered_rows':split_counts,'evaluated_examples':cfg['evaluation']['examples'],'device':'cuda' if cuda else 'cpu','elapsed_seconds':time.monotonic()-start,
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if cuda else None,
                'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved() if cuda else None}
        (output/'tool_compute.json').write_text(json.dumps(device))
    export(output, model)
    if a.mode=='run':
        binding={'scientific_request':request['scientific'],'observed_assets':observed_assets,'outputs':{}}
        for name in (*(f'ioi_{model}_{suffix}' for suffix in ('importances.json','scores.pt','validation.pkl','trainlog.csv')),'official_metrics.json'):
            binding['outputs'][name]=hashlib.sha256((output/name).read_bytes()).hexdigest()
        (output/'producer_binding.json').write_text(json.dumps(binding,allow_nan=False))

if __name__=='__main__':main()
