"""C01/C03/C08 publication for complete SDXL images; tensors stay at the store."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from pipelines_v2.core.provenance import ProvenanceV1,content_hash,observed_runtime,source_snapshot
from pipelines_v2.core.types import utc_now_iso
from pipelines_v2.data.images import DiffusionCoordinates
from pipelines_v2.engine.diffusers.hooks import ADAPTER_VERSION
from pipelines_v2.engine.diffusers.session import array,tensor_digest
from pipelines_v2.storage.artifacts import ArtifactManifest
from pipelines_v2.storage.features import write_capture_features
from pipelines_v2.storage.finalization import finalize_manifest,atomic_replace
from pipelines_v2.storage.image_features import SpatialTensor


def runtime_spec():
    from pipelines_v2.engine.base import PythonRuntimeSpec
    from pipelines_v2.runtime.locks import RuntimeLock
    lock=RuntimeLock.from_dict(json.loads(Path(__file__).with_name('runtime_lock.json').read_text()))
    return PythonRuntimeSpec(python_version=lock.python_version,lock=lock,local_python_sources=('pipelines_v2',),
        env={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'})


def verify_runtime():
    import importlib.metadata,platform
    lock=runtime_spec().lock
    errors=list(lock.verify_target())
    for name,version in lock.packages.items():
        try:actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:actual=None
        if actual!=version:errors.append(f'{name}: expected {version}, observed {actual}')
    if errors:raise ValueError('SDXL locked runtime mismatch: '+'; '.join(errors))
    return lock


def image_provenance(request,hooks,noise,session,source_root):
    import torch
    lock=verify_runtime()
    source=source_snapshot(Path(source_root));unresolved=source.pop('unresolved',[])
    runtime=observed_runtime(family='modal')
    image_id=os.environ.get('MODAL_IMAGE_ID')
    if not image_id:raise ValueError('Immutable Modal image identity unavailable')
    runtime.update(lock_digest=lock.identity,lock_kind='resolved_requirements',image_identity=image_id,image_reason=None)
    components=[{'role':name,'repository_id':session.components['repository']+'/'+name,
        'requested_revision':session.components['revision'],'resolved_revision':session.components['revision'],
        'file_set_digest':digest} for name,digest in session.components['components'].items()]
    policy={'dtype':'float16','attention_implementation':'torch_sdpa/AttnProcessor2_0',
        'device':{'name':torch.cuda.get_device_name(),'cuda_runtime':torch.version.cuda},
        'determinism':{'algorithms':torch.are_deterministic_algorithms_enabled(),'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG')},
        'compile':False,'matmul':{'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}}
    data={'request':asdict(request),'noise':tensor_digest(noise)}
    result=ProvenanceV1(source=source,runtime=runtime,model_components=components,
        data={'content_digest':content_hash(data),'ordered_example_key_digest':content_hash([request.key]),
              'split_digest':content_hash([]),'preprocessing':{'resize':None,'original_size':[request.height,request.width],
                  'crop':[0,0],'target_size':[request.height,request.width],'watermark':False}},
        numerics=policy,methods={'operation_version':'sdxl-image-v1','hook_version':ADAPTER_VERSION,
            'evaluator':{'hooks':[hook.to_dict() for hook in hooks],'scheduler':session.scheduler_digest}},
        seeds={'initial_noise_cpu':request.seed,'initial_noise_digest':tensor_digest(noise)},unresolved=unresolved)
    if not result.identity_complete:raise ValueError(f'Incomplete SDXL execution identity: {result.to_dict()['unresolved']}')
    return result


def publish_image(store,artifact_id,request,hooks,noise,rng_state,result,provenance):
    """Called only after one complete image; incomplete trajectories publish no final."""
    import numpy as np
    from PIL import Image
    noise_ref=store.write_safetensors(artifact_id,'inputs/noise.safetensors',{'noise':array(noise)})
    rng_ref=store.write_safetensors(artifact_id,'inputs/generator.safetensors',{name:array(value) for name,value in rng_state.items()})
    metadata=result['metadata'];component=metadata['components']['components']['unet']
    preprocessing=content_hash(provenance.to_dict()['data']['preprocessing'])
    origin={'content_digest':provenance.to_dict()['data']['content_digest']}
    def trajectory(step,branch):
        return DiffusionCoordinates(prompt=request.prompt,negative_prompt=request.negative_prompt,
            initial_noise_digest=metadata['initial_noise_digest'],initial_noise_ref=noise_ref,generator_state_ref=rng_ref,
            scheduler_name=metadata['scheduler_name'],scheduler_config_digest=metadata['scheduler_config_digest'],
            step_index=step,timestep_value=metadata['timesteps'][step],denoising_window=(0,request.steps),
            cfg_enabled=metadata['cfg_enabled'],branch=branch,guidance_scale=request.guidance_scale)
    image=np.asarray(result['image'],dtype=np.float32)
    descriptor=SpatialTensor.from_array(image,kind='image',axes=('y','x','channel'),color_space='RGB',value_range=(0,1),
        preprocessing_digest=preprocessing,origin_ref=origin,component_digest=metadata['components']['components']['vae'])
    features={'images':{'kind':'image','schema_version':1,'records':{request.key:{'descriptor':descriptor.to_dict(),'values':image}}}}
    latents={}
    final=SpatialTensor.from_array(result['latents'],kind='latent',axes=('batch','channel','y','x'),preprocessing_digest=preprocessing,
        origin_ref=origin,component_digest=component,trajectory=trajectory(request.steps-1,'conditional'))
    latents['final']={'descriptor':final.to_dict(),'values':result['latents']}
    for (index,step),tensor in result['captures'].items():
        values=array(tensor);desc=SpatialTensor.from_array(values,kind='latent',axes=('batch','channel','y','x'),
            preprocessing_digest=preprocessing,origin_ref=origin,component_digest=component,trajectory=trajectory(step,hooks[index].branch))
        latents[f'hook_{index}_step_{step}']={'descriptor':desc.to_dict(),'values':values}
    features['latents']={'kind':'latent','schema_version':1,'records':latents}
    refs={'features':write_capture_features(store,artifact_id,features),'initial_noise':noise_ref,'generator_state':rng_ref}
    image_path=Path(store.root)/artifact_id/'image.png'
    pixels=Image.fromarray((image.clip(0,1)*255).round().astype('uint8'))
    atomic_replace(image_path,lambda temporary:pixels.save(temporary,format='PNG'))
    image_ref={'store':store.kind,'path':str(image_path),'bytes':image_path.stat().st_size,'format':'png'}
    if store.kind=='modal_volume':image_ref['name']=store.name
    refs['image']=image_ref
    refs['result']=store.write_json(artifact_id,'result.json',metadata)
    operation_hash=content_hash({'request':asdict(request),'hooks':[hook.to_dict() for hook in hooks]})
    manifest=ArtifactManifest(artifact_id=artifact_id,artifact_kind='diffusion_image',schema_version=1,
        operation_spec_hash=operation_hash,operation_semantic_hash=operation_hash,created_at=utc_now_iso(),
        engine={'kind':'sdxl','revision':metadata['components']['revision']},runner={'kind':'modal'},input_artifact_refs=(),
        example_coverage={'example_count':1,'example_keys':[request.key]},storage_refs=refs,
        metadata={**metadata,'provenance_v1':provenance.to_dict(),'sites':[hook.site(metadata['components']) for hook in hooks]})
    return finalize_manifest(store,manifest)
