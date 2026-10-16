"""Observed pair/reference bytes and enforced numerical policy for C01 reuse."""
import json
from pathlib import Path
from .settings import Execution,observed_policy
from .reference import lock,verify_runtime


def provenance_facts(spec, runner_identity):
    from pipelines_v2.core.provenance_observation import OperationFacts,file_set_identity
    execution=Execution(**spec.execution)
    # Metadata-only verification, no weights, downloads or GPU allocation.
    verify_runtime(execution.runtime_profile)
    unresolved=()
    try:
        numerics=observed_policy(execution)
    except ValueError:
        if execution.device!='cuda':raise
        numerics={'dtype':execution.dtype,'device':'cuda','attention_implementation':'eager','compile':False,'determinism':'deterministic_algorithms','matmul':'highest; tf32_disabled'}
        unresolved=('numerics.cuda_device_unobserved',)
    fields={};components=[]
    if hasattr(spec,'model_paths'):
        pinned=json.loads(Path(__file__).with_name('model_lock.json').read_text())['models']
        identities=[]
        for side,(path,model) in enumerate(zip(spec.model_paths,pinned,strict=True)):
            identity=file_set_identity(path,model['files'],expected=model['files'])
            identities.append({'repository':model['repository'],'revision':model['revision'],**identity})
            components.append({'role':f'paired_model_{side}','repository_id':model['repository'],
                'requested_revision':model['revision'],'resolved_revision':model['revision'],'file_set_digest':identity['content_digest'],**identity})
        fields['model_paths']=identities
    if hasattr(spec,'reference_path'):
        reference=lock()
        fields['reference_path']=file_set_identity(spec.reference_path,reference['files'],expected=reference['files'])
    seeds={'crosscoder.fit':spec.config['seed']} if spec.kind=='crosscoder_fit' else {}
    return OperationFacts(field_identities=fields,model_components=tuple(components),numerics=numerics,seeds=seeds,unresolved=unresolved,
        methods={'reference_revision':lock()['revision'],'adapter':'qwen2.5-residual-fp32-v2',
                 'fit_sampler':'seeded-uniform-token-with-replacement-v1','execution_policy':spec.execution})
