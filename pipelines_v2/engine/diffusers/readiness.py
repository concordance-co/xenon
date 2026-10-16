"""SDXL-specific observed components and admission, using the shared C01/C02 types."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import os
from pipelines_v2.core.provenance import ProvenanceV1,content_hash,observed_runtime,source_snapshot
from pipelines_v2.engine.diffusers.assets import model_lock
from pipelines_v2.engine.diffusers.hooks import ADAPTER_VERSION


def numerical_policy(*,configure=False):
    import torch
    if not torch.cuda.is_available():return None
    if configure:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    from pipelines_v2.runtime.qualification import gpu_driver_version
    return {'dtype':'float16','attention_implementation':'torch_sdpa/AttnProcessor2_0',
        'device':{'name':torch.cuda.get_device_name(),'cuda_runtime':torch.version.cuda,'driver_version':gpu_driver_version()},
        'determinism':{'algorithms':torch.are_deterministic_algorithms_enabled(),'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG')},
        'compile':False,'matmul':{'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}}


def components_descriptor(observed):
    lock=model_lock()
    return [{'role':name,'repository_id':lock['repository']+'/'+name,'requested_revision':lock['revision'],
        'resolved_revision':lock['revision'] if observed else None,'file_set_digest':observed['components'][name] if observed else None}
        for name in sorted({file.split('/')[0] for file in lock['files']})]


def resolve_diffusion_provenance(spec,runner_identity=None,*,workspace_root=None,source=None,runtime=None,
                                 model_components=None,numerics=None,methods=None,seeds=None):
    from pipelines_v2.core.paths import find_workspace_root
    runner_identity=runner_identity or {'kind':'local'}
    source=dict(source) if source is not None else source_snapshot(Path(workspace_root or find_workspace_root()))
    unresolved=source.pop('unresolved',[])
    runtime=runtime or observed_runtime(family=runner_identity.get('kind','local'))
    upstream={}
    for name in ('noise_from','replacement_from'):
        artifact=getattr(spec,name)
        if artifact is not None:
            if not callable(getattr(artifact,'manifest',None)):
                unresolved.append('data.unresolved_workflow_input');upstream[name]=None;continue
            prior=artifact.manifest().provenance()
            if prior is None or not prior.identity_complete:unresolved.append('data.upstream_unresolved')
            upstream[name]=prior.scientific_identity if prior else None
    if numerics is None:
        unresolved.append('numerics.execution_host_observation')
        numerics={key:None for key in ('dtype','attention_implementation','device','determinism','compile','matmul')}
    requests=[asdict(row) for row in spec.images]
    return ProvenanceV1(source=source,runtime=runtime,model_components=model_components or components_descriptor(None),
        data={'content_digest':content_hash({'requests':requests,'upstream':upstream}),
              'ordered_example_key_digest':content_hash([row.key for row in spec.images]),'split_digest':content_hash([]),
              'preprocessing':{'prompt_2':'same_prompt','negative_prompt_2':'same_negative_prompt','crop':[0,0],
                  'original_and_target_size':'explicit_request','watermark':False},'upstream_identities':upstream},
        numerics=numerics,methods=methods or {'operation_version':spec.kind+':v1','hook_version':ADAPTER_VERSION,
            'evaluator':{'hooks':[hook.to_dict() for hook in spec.hooks],'scheduler':'pinned-EulerDiscreteScheduler',
                         'noise_generation':'cpu-torch-float32-then-float16-v1'}},
        seeds=seeds or {row.key:row.seed for row in spec.images},unresolved=unresolved)


def diffusion_envelope(spec,runner_identity):
    from pipelines_v2.runtime.admission import WorkloadEnvelope
    resources=runner_identity.get('resources',{})
    return WorkloadEnvelope(model_identity=spec.engine.semantic_identity(),runtime_lock=spec.runtime_spec().lock.identity,
        hardware_topology={'gpu':resources.get('gpu'),'independent_shards':resources.get('shard_count') or 1,'tensor_parallel':1,'pipeline_parallel':1},
        dtype='float16',attention_impl='torch_sdpa/AttnProcessor2_0',operation_kind=spec.kind,
        hook_set=tuple(content_hash(hook.to_dict()) for hook in spec.hooks),
        max_input_shape=[1,4,max(row.height for row in spec.images)//8,max(row.width for row in spec.images)//8],
        max_output_shape=[max(row.height for row in spec.images),max(row.width for row in spec.images),3],
        batch_size=1,capture_shape=[list(hook.shape_chw) for hook in spec.hooks],backward_or_optimizer=False,
        compile_config={'compile':False,'offload':False,'steps':[row.steps for row in spec.images],
                        'guidance':[row.guidance_scale for row in spec.images]},host_memory_limit=resources.get('memory_mb'),
        device_memory_limit=None,timeout=resources.get('timeout_seconds'))


def admit_diffusion(spec,runner_identity,*,workspace_root=None,image_identity=None,import_modules=True,
                    artifact_store=None,check_writer_registry=False,scope='execution',**unused):
    from pipelines_v2.runtime.readiness import AdmissionRecord,runtime_readiness,store_readiness,writer_registry_readiness,_check
    runtime,checks=runtime_readiness(spec.runtime_spec(),image_identity=image_identity,family='modal',import_modules=import_modules)
    checks=list(checks);checks.append(store_readiness(artifact_store))
    if check_writer_registry:checks.append(writer_registry_readiness())
    observed=None
    try:observed=spec.engine.observed_components()
    except (ValueError,OSError):checks.append(_check('sdxl.cache','failed','engine.cache_root','Pinned SDXL components unavailable or changed','Run explicit CPU cache preparation'))
    else:checks.append(_check('sdxl.cache','passed','engine.cache_root','All18 pinned SDXL component files verified'))
    numerics=numerical_policy(configure=True) if scope=='execution' else None
    record=resolve_diffusion_provenance(spec,runner_identity,workspace_root=workspace_root,runtime=runtime,
        model_components=components_descriptor(observed),numerics=numerics)
    return AdmissionRecord(tuple(checks),record,diffusion_envelope(spec,runner_identity).to_dict(),scope=scope)


def bind_diffusion_admission(spec,runner_identity,admitted,*,workspace_root=None):
    from pipelines_v2.runtime.readiness import AdmissionRecord
    if not isinstance(admitted,AdmissionRecord):admitted=AdmissionRecord.from_dict(admitted)
    if admitted.scope!='execution' or not admitted.ready or not admitted.provenance.identity_complete:
        raise ValueError('SDXL reuse needs complete execution-host admission')
    from pipelines_v2.runtime.admission import WorkloadEnvelope
    if diffusion_envelope(spec,runner_identity).identity!=WorkloadEnvelope.from_dict(admitted.envelope).identity:
        raise ValueError('SDXL workload envelope changed')
    body=admitted.provenance.to_dict()
    if Path(spec.engine.resolved_model_path()).is_dir() and components_descriptor(spec.engine.observed_components())!=body['model_components']:
        raise ValueError('SDXL cached components changed')
    expected=resolve_diffusion_provenance(spec,runner_identity,workspace_root=workspace_root,runtime=body['runtime'],
        model_components=body['model_components'],numerics=body['numerics'])
    if expected.scientific_identity!=admitted.provenance.scientific_identity:raise ValueError('SDXL request/source/input identity changed')
    return expected


def static_diffusion_checks(spec,runner_identity):
    from pipelines_v2.runtime.admission import AdmissionCheck
    resources=runner_identity.get('resources',{})
    gpu=resources.get('gpu');single=isinstance(gpu,str) and bool(gpu) and (':' not in gpu or gpu.rsplit(':',1)[1]=='1')
    checks=[]
    for name,okay,message in [
        ('placement',runner_identity.get('kind')=='modal','SDXL execution uses Modal; local CPU fixtures are tests only'),
        ('gpu',single,'Initial SDXL adapter requires one explicit GPU'),
        ('shards',(resources.get('shard_count') or 1)==1,'Initial SDXL adapter requires one independent shard'),
        ('containers',(resources.get('max_containers') or 1)==1,'Initial SDXL adapter requires one worker'),
    ]:
        checks.append(AdmissionCheck('sdxl.'+name,'static','passed' if okay else 'failed','runner.resources',message))
    return tuple(checks)
