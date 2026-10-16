"""Image-boundary durable SDXL executor; invoked by the existing Modal runner."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path

from pipelines_v2.core.provenance import ProvenanceV1,content_hash,context_provenance
from pipelines_v2.core.types import utc_now_iso
from pipelines_v2.storage.artifacts import ArtifactManifest,OperationArtifact
from pipelines_v2.storage.finalization import CheckpointJournal,validate_completion,finalize_manifest
from pipelines_v2.engine.diffusers.session import SDXLSession,tensor_digest
from pipelines_v2.engine.diffusers.artifacts import publish_image


def diffusion_image(artifact,key):
    """Return one image's typed artifact without fetching its image/capture bytes."""
    if artifact.manifest().artifact_kind not in {'diffusion_generate','diffusion_capture','diffusion_intervene'}:
        raise ValueError('Expected a diffusion run artifact')
    rows=artifact.result()['images']
    matches=[row for row in rows if row['key']==key]
    if len(matches)!=1:raise ValueError(f'Expected exactly one source image for {key!r}')
    manifest=ArtifactManifest.from_dict(artifact.store.read_json_ref(matches[0]['manifest_ref']))
    return OperationArtifact(manifest,artifact.store)


def _input_noise(spec,request):
    import torch
    if spec.noise_from is None:
        return SDXLSession.noise(None,request)
    source=diffusion_image(spec.noise_from,request.key)
    manifest=source.manifest();validate_completion(manifest,store=source.store)
    noise=torch.from_numpy(source.store.read_safetensors_ref(manifest.storage_refs['initial_noise'])['noise']).clone()
    state={name:torch.from_numpy(value).clone() for name,value in source.store.read_safetensors_ref(manifest.storage_refs['generator_state']).items()}
    if tensor_digest(noise)!=manifest.metadata['initial_noise_digest']:raise ValueError('Shared initial-noise content does not match its source')
    if tuple(noise.shape)!=(1,4,request.height//8,request.width//8):raise ValueError('Shared-noise shape differs from requested image')
    if set(state)!={'before','after'}:raise ValueError('Shared noise has no complete generator state')
    return noise,state


def _replacement(spec,request,session):
    import torch
    if not any(hook.operator=='replace' for hook in spec.hooks):return {}
    donor=diffusion_image(spec.replacement_from,request.key)
    manifest=donor.manifest();validate_completion(manifest,store=donor.store)
    records=donor.feature('latents').load()['records']
    sites=manifest.metadata.get('sites',[]);result={}
    for index,hook in enumerate(spec.hooks):
        if hook.operator!='replace':continue
        matches=[i for i,site in enumerate(sites) if site['module_path']==hook.module_path and site['selection']['branch']==hook.branch]
        if len(matches)!=1:raise ValueError('Replacement requires one matched module/branch in donor capture')
        for step in hook.steps:
            record=records.get(f'hook_{matches[0]}_step_{step}')
            if record is None:raise ValueError('Donor capture is missing a selected step')
            desc=record['descriptor'];trajectory=desc['trajectory']
            if desc['component_digest']!=session.components['components']['unet'] or trajectory['scheduler_config_digest']!=session.scheduler_digest:
                raise ValueError('Donor UNet or scheduler identity differs')
            if trajectory['step_index']!=step or trajectory['branch']!=hook.branch or trajectory['cfg_enabled']!=(request.guidance_scale>1):
                raise ValueError('Donor denoising coordinates differ')
            # Exact scheduler spacing is part of the match, not only step index.
            session.pipeline.scheduler.set_timesteps(request.steps,device=session.device)
            if trajectory['timestep_value']!=float(session.pipeline.scheduler.timesteps[step]):raise ValueError('Donor scheduler timestep differs')
            result[index,step]=torch.from_numpy(record['values']).clone()
    return result


def _child_provenance(parent,request,noise):
    body=parent.to_dict();body.pop('scientific_identity');body.pop('schema_version');body.pop('worker_compatibility')
    body['data']={**body['data'],'content_digest':content_hash({'request':asdict(request),'noise':tensor_digest(noise),
        'parent_workload':parent.scientific_identity}),'ordered_example_key_digest':content_hash([request.key]),
        'parent_workload_identity':parent.scientific_identity,'partition':{'image_key':request.key}}
    return ProvenanceV1(**body)


def execute_diffusion(*,spec,store,runner_config,workflow_context,execution_session=None,progress_callback=None):
    from pipelines_v2.runtime.qualification import CanaryGate
    kwargs=dict(spec=spec,store=store,runner_config=runner_config,workflow_context=workflow_context,
                execution_session=execution_session,progress_callback=progress_callback)
    if not workflow_context.get('qualification_required'):
        return _execute_diffusion(**kwargs)
    gate=CanaryGate(store=store,spec=spec,runner_config=runner_config,context=workflow_context,progress_callback=progress_callback)
    with gate.owner():
        return _execute_diffusion(**kwargs,qualification_gate=gate)


def _execute_diffusion(*,spec,store,runner_config,workflow_context,execution_session=None,progress_callback=None,qualification_gate=None):
    from pipelines_v2.runtime.remote_executor import _artifact_id_for,_load_existing_manifest,_fresh_id_if_final_exists
    provenance=context_provenance(workflow_context)
    if provenance is None or not provenance.identity_complete:raise ValueError('SDXL execution requires complete observed admission')
    artifact_id=_artifact_id_for(spec=spec,workflow_context=workflow_context)
    existing=_load_existing_manifest(store=store,artifact_id=artifact_id,spec=spec,provenance=provenance)
    if existing is not None:return existing.to_dict()
    artifact_id=_fresh_id_if_final_exists(store,artifact_id)
    journal=CheckpointJournal(store=store,artifact_id=artifact_id,semantic_hash=spec.semantic_hash(),provenance=provenance,
        expected={'unit_keys':[row.key for row in spec.images],'explicit_empty':False},shard={'index':0,'count':1})
    complete={unit['key']:unit for unit in journal.body['completed_units']}
    results=[];image_refs={};session=None
    for request in spec.images:
        child_id=artifact_id+'-image-'+content_hash(request.key)[:16]
        if request.key in complete:
            manifest=ArtifactManifest.from_dict(store.read_json_ref(complete[request.key]['refs']['manifest']))
            validate_completion(manifest,store=store)
        else:
            noise,state=_input_noise(spec,request)
            child_provenance=_child_provenance(provenance,request,noise)
            orphan=Path(store.root)/child_id/'manifest.json'
            manifest=None
            if orphan.exists():
                candidate=ArtifactManifest.from_dict(json.loads(orphan.read_text()))
                validate_completion(candidate,store=store)
                if candidate.provenance().scientific_identity!=child_provenance.scientific_identity:raise ValueError('Orphan image identity mismatch')
                manifest=candidate
            if manifest is None:
                if session is None:
                    cache=getattr(execution_session,'_diffusion_sessions',{}) if execution_session else {}
                    from pipelines_v2.runtime.qualification import session_key
                    key=session_key(provenance,runner_config) if qualification_gate else provenance.to_dict()['worker_compatibility']['identity']
                    session=cache.get(key)
                    if session is None:
                        # A scoped worker retains at most one incompatible model state.
                        if qualification_gate:
                            for prior in cache.values():prior.close()
                            cache.clear()
                        session=SDXLSession(spec.engine.resolved_model_path());cache[key]=session
                        if hasattr(execution_session,'model_loads'):execution_session.model_loads+=1
                    if execution_session is not None:execution_session._diffusion_sessions=cache
                replacements=_replacement(spec,request,session)
                result=session.generate(request,noise=noise,hooks=spec.hooks,replacements=replacements)
                result['metadata']['session_v1']={'session_id':getattr(session,'session_id',None),
                    'worker_id':getattr(execution_session,'worker_id',None),
                    'model_load_count':getattr(execution_session,'model_loads',1),'image_count':getattr(session,'images',None)}
                manifest=publish_image(store,child_id,request,spec.hooks,noise,state,result,child_provenance)
            path=Path(store.root)/child_id/'manifest.json'
            ref={'store':store.kind,'path':str(path),'bytes':path.stat().st_size,'format':'json'}
            if store.kind=='modal_volume':ref['name']=store.name
            journal.append(request.key,refs={'manifest':ref},coverage={'image_key':request.key,'initial_noise_digest':manifest.metadata['initial_noise_digest']})
            complete[request.key]=journal.body['completed_units'][-1]
            if progress_callback:progress_callback({'stage':'diffusion_image_complete','step_name':workflow_context.get('step_name'),'image_key':request.key,'artifact_id':child_id})
        if qualification_gate:qualification_gate.observe(manifest)
        image_refs[request.key]=dict(manifest.storage_refs)
        results.append({'key':request.key,'artifact_id':child_id,'manifest_ref':complete[request.key]['refs']['manifest'],
                        'initial_noise_digest':manifest.metadata['initial_noise_digest'],'image_ref':manifest.storage_refs['image']})
    journal.verify_full()
    refs={'images':image_refs,'result':store.write_json(artifact_id,'result.json',{'images':results})}
    final=ArtifactManifest(artifact_id=artifact_id,artifact_kind=spec.kind,schema_version=1,
        operation_spec_hash=spec.spec_hash(),operation_semantic_hash=spec.semantic_hash(),created_at=utc_now_iso(),
        engine=spec.engine.identity(),runner=runner_config,input_artifact_refs=tuple(value.id for value in (spec.noise_from,spec.replacement_from) if value is not None),
        example_coverage={'image_count':len(results),'example_count':len(results),'example_keys':[row.key for row in spec.images]},
        storage_refs=refs,metadata={'provenance_v1':provenance.to_dict(),'admission_v1':workflow_context.get('admission_v1'),
            'resume_semantics':'restart_at_image','loads':session.loads if session else 0,
            'qualification_v1':qualification_gate.receipt if qualification_gate else None,
            'session_v1':manifest.metadata.get('session_v1')},workflow_context=workflow_context)
    return finalize_manifest(store,final).to_dict()
