"""Execute or import official outputs; raw bytes remain durable inside the result artifact."""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import tempfile
from pipelines_v2.runtime.external_tool import verify_tool_source,run_bounded,external_tool_environment
from .contracts import normalize_official_outputs, model_profile


def _unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate JSON node/field')
        result[key]=value
    return result


def import_official(spec,output,*,compute):
    output=Path(output)
    load=lambda p:json.loads(p.read_text(),object_pairs_hook=_unique_object)
    binding=load(output/'producer_binding.json')
    if binding['scientific_request']!=spec.request or binding['observed_assets']!={role:spec.request[role]['files'] for role in ('model','data')}:
        raise ValueError('Producer/model/data identity differs from import request')
    stem='ioi_'+spec.request['model']['name']
    official_files=tuple(stem+'_'+suffix for suffix in ('importances.json','scores.pt','validation.pkl','trainlog.csv'))
    expected_outputs={*official_files,'official_metrics.json'}
    if set(binding['outputs'])!=expected_outputs:raise ValueError('Incomplete producer output binding')
    for name,sha in binding['outputs'].items():
        if hashlib.sha256((output/name).read_bytes()).hexdigest()!=sha:raise ValueError('Producer output bytes changed')
    exported=load(output/'official_metrics.json')
    args=exported['args'];r=spec.request
    wanted={'model':r['model']['name'],'task':'ioi','steps':r['fitting']['steps'],'seed':r['seed'],'mode':'iso',
            'masking':'topk','k_schedule':'uniform','lr':r['fitting']['lr'],'include_input':True,
            'split':'validation','train_split':'train','eval_examples':r['evaluation']['examples'],'ablation':'patching',
            'optimizer':'adam','adam_eps':1e-8,'T':.5,'n_iters':30,'train_batch_size':1,'batch_size':1,'k_avg':1,'cf_cache_gb':0}
    if any(args.get(k)!=v for k,v in wanted.items()):
        raise ValueError('Official output arguments do not match immutable request')
    raw={};total=0
    for name in (*official_files,'official_metrics.json','stdout.log','stderr.log','run.log','producer_binding.json'):
        p=output/name
        if not p.is_file():continue
        data=p.read_bytes();total+=len(data)
        if total>spec.output_bytes:raise ValueError('Raw output exceeds persistence budget')
        raw[name]={'encoding':'base64','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'data':base64.b64encode(data).decode()}
    for required in official_files:
        if required not in raw:raise ValueError('Missing official raw output: '+required)
    runtime={'python_version':spec.runtime.tool_lock['python_version'],'tool_lock_identity':spec.runtime.tool_identity,
             'installed_packages':exported['installed_packages']}
    return normalize_official_outputs(load(output/(stem+'_importances.json')),exported['metrics'],
        request=r,compute={**compute,'training_seconds':exported['train_time_seconds'],'tool_runtime':runtime},raw_outputs=raw)


def run_mattr(spec):
    from pipelines_v2.operations.execution.common import OperationExecutionResult
    verify_tool_source(spec.runtime);spec.verify_inputs()
    root=Path(tempfile.mkdtemp(prefix='xenon-mattr-',dir=spec.work_root))
    inputs=root/'inputs';inputs.mkdir()
    model_link=inputs/model_profile(spec.request)['path'];model_link.parent.mkdir(parents=True,exist_ok=True)
    model_link.symlink_to(Path(spec.model_path).resolve(),target_is_directory=True)
    data=inputs/'mib-bench/ioi';data.mkdir(parents=True)
    for name in ('train.json','dev.json'):(data/name).symlink_to((Path(spec.dataset_path)/name).resolve())
    output=root/'output';output.mkdir()
    request={'tool_root':spec.runtime.tool_root,'mib_root':spec.runtime.mib_root,'output_dir':str(output),'scientific':spec.request}
    request_file=root/'request.json';request_file.write_text(json.dumps(request))
    env={**external_tool_environment(),'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','HF_DATASETS_CACHE':str(root/'datasets-cache'),
         'WANDB_MODE':'disabled','TOKENIZERS_PARALLELISM':'false','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    bridge=Path(__file__).with_name('bridge.py')
    compute=run_bounded([spec.runtime.tool_python,str(bridge),'run',str(request_file)],cwd=inputs,env=env,output_dir=output,
        timeout_seconds=spec.timeout_seconds,memory_bytes=spec.memory_bytes,output_bytes=spec.output_bytes)
    compute.update(json.loads((output/'tool_compute.json').read_text()))
    result=import_official(spec,output,compute=compute)
    # Raw bytes are retained durably in result.json. Failed attempts retain their directory.
    import shutil
    shutil.rmtree(root)
    return OperationExecutionResult(payload=result,metadata={'external_tool_v1':{'tool_lock':spec.runtime.tool_lock,
        'request_identity':result['request_identity'],'resume_semantics':result['resume_semantics'],'engineering_status':'tested',
        'scientific_verdict':'not_tested'}},example_coverage={'method_seed_units':[f'mattr:{spec.request["seed"]}'],
        'node_count':len(result['nodes']),'evaluation_examples':spec.request['evaluation']['examples']})
