"""Observe the isolated upstream interpreter, never substitute control-Python facts."""
import json
import subprocess
from pipelines_v2.core.provenance_observation import OperationFacts,file_set_identity
from pipelines_v2.runtime.external_tool import verify_tool_source,external_tool_environment

_PROBE='''import sys,platform,json,importlib.metadata,torch
cuda=torch.cuda.is_available()
device={'kind':'cuda' if cuda else 'cpu','torch':torch.__version__}
if cuda:
 p=torch.cuda.get_device_properties(0)
 device.update(name=p.name,compute_capability=[p.major,p.minor],total_memory=p.total_memory,cuda_runtime=torch.version.cuda)
print(json.dumps({'python':platform.python_version(),'platform':platform.system()+'-'+platform.machine(),
'packages':dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())),
'device':device,'matmul':{'float32_precision':torch.get_float32_matmul_precision(),'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'num_threads':torch.get_num_threads(),'num_interop_threads':torch.get_num_interop_threads()}}))
'''


def provenance_facts(spec,runner_identity):
    from pipelines_v2.core.provenance import content_hash
    try: verify_tool_source(spec.runtime)
    except subprocess.SubprocessError as exc: raise ValueError('Isolated tool source unavailable') from exc
    fields={key:file_set_identity(getattr(spec,key),spec.request[role]['files'],expected=spec.request[role]['files']) for key,role in (('model_path','model'),('dataset_path','data'))}
    env={**external_tool_environment(),'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    try:
        result=subprocess.run([spec.runtime.tool_python,'-c',_PROBE],env=env,capture_output=True,text=True,timeout=60,check=True)
        facts=json.loads(result.stdout)
    except (subprocess.SubprocessError,json.JSONDecodeError) as exc:
        raise ValueError('Isolated tool runtime observation failed') from exc
    runtime={'family':'modal' if runner_identity.get('kind')=='modal' else 'local','tool_family':'mattr-isolated-upstream','lock_digest':spec.runtime.tool_identity,'python':facts['python'],'platform':facts['platform'],
        'installed_package_digest':content_hash(facts['packages']),'image_identity':None,'image_reason':'shared_with_control_runtime' if runner_identity.get('kind')=='modal' else 'local-isolated-interpreter'}
    fields['runtime']={'tool_lock':spec.runtime.tool_identity,'observed_tool_runtime':runtime}
    if spec.work_root is not None:fields['work_root']={'role':'temporary-output-location'}
    numerics={'dtype':spec.request['dtype'],'device':facts['device'],'matmul':facts['matmul'],'attention_implementation':'upstream-pinned-eager',
        'compile':False,'determinism':'explicit-seed-upstream-optimizer; kernels-may-be-nondeterministic','policy_version':'official-mattr-v1'}
    return OperationFacts(field_identities=fields,model_components=({'role':'model','repository_id':'local-content:'+fields['model_path']['content_digest'],'requested_revision':spec.request['model']['revision'],'resolved_revision':'sha256:'+fields['model_path']['content_digest'],'file_set_digest':fields['model_path']['content_digest']},),runtime=runtime,numerics=numerics,
        seeds={'method':spec.request['seed']},methods={'upstream_tool':spec.runtime.tool_identity,'evaluator':spec.request['evaluator']})
