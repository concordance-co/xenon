import pytest
import torch
from pipelines_v2.engine.diffusers.session import ImageRequest,SDXLSession,tensor_digest
from pipelines_v2.engine.diffusers.assets import model_lock,verify_checkpoint


def test_fixed_noise_is_shared_content_with_replayable_cpu_generator():
    session=object.__new__(SDXLSession)
    request=ImageRequest('a','A red ceramic teapot',seed=17,width=64,height=64)
    noise,state=session.noise(request)
    generator=torch.Generator(device='cpu');generator.set_state(state['before'])
    independent=torch.randn((1,4,8,8),generator=generator,dtype=torch.float32).half()
    assert torch.equal(noise,independent)
    assert torch.equal(generator.get_state(),state['after'])
    clone=noise.clone();clone[0,0,0,0]+=1
    assert tensor_digest(noise)!=tensor_digest(clone)


def test_model_lock_names_every_actual_component_and_refuses_missing_bytes(tmp_path):
    lock=model_lock()
    assert len(lock['revision'])==40
    assert {name.split('/')[0] for name in lock['files']} >= {'unet','vae','text_encoder','text_encoder_2','tokenizer','tokenizer_2','scheduler'}
    with pytest.raises(ValueError,match='missing'):verify_checkpoint(tmp_path)


@pytest.mark.parametrize('change',[{'width':65},{'seed':True},{'steps':0},{'guidance_scale':float('nan')}])
def test_invalid_requests_fail_before_model_load(change):
    with pytest.raises(ValueError):ImageRequest('a','prompt',**change)


def test_complete_image_publication_roundtrips_real_c01_c03_c08(tmp_path):
    import numpy as np
    from pipelines_v2.api import LocalArtifactStore,ToyEngine,GenerationRunSpec,GenerationSpec,Dataset,Example
    from pipelines_v2.core.provenance import resolve_provenance
    from pipelines_v2.storage.finalization import validate_completion
    from pipelines_v2.storage.artifacts import OperationArtifact,ArtifactManifest
    from pipelines_v2.engine.diffusers.artifacts import publish_image
    from pipelines_v2.engine.diffusers.hooks import DenoisingHook
    import json
    spec=GenerationRunSpec(engine=ToyEngine(),dataset=Dataset.from_examples([Example(key='a',prompt='fixture')]),generation=GenerationSpec(enabled=True,max_tokens=1))
    provenance=resolve_provenance(spec)
    assert provenance.identity_complete
    request=ImageRequest('a','fixture',width=64,height=64,steps=2)
    session=object.__new__(SDXLSession);noise,rng=session.noise(request)
    hook=DenoisingHook('site',(0,),(2,3,4))
    result={'image':np.zeros((64,64,3),dtype=np.float32),'latents':noise.numpy().copy(),
        'captures':{(0,0):torch.zeros(1,2,3,4,dtype=torch.float16)},
        'metadata':{'components':{'revision':'a'*40,'components':{'unet':'b'*64,'vae':'c'*64}},
            'initial_noise_digest':tensor_digest(noise),'scheduler_name':'EulerDiscreteScheduler',
            'scheduler_config_digest':'d'*64,'cfg_enabled':True,'timesteps':[999.0,0.0]}}
    store=LocalArtifactStore(tmp_path)
    manifest=publish_image(store,'image',request,(hook,),noise,rng,result,provenance)
    assert validate_completion(manifest,store=store)
    restored=ArtifactManifest.from_dict(json.loads((tmp_path/'image'/'manifest.json').read_text()))
    assert restored.provenance().scientific_identity==provenance.scientific_identity
    handle=OperationArtifact(restored,store)
    assert handle.feature('images').load()['records']['a']['values'].shape==(64,64,3)
    capture=handle.feature('latents').load()['records']['hook_0_step_0']
    assert capture['descriptor']['trajectory']['step_index']==0
    assert capture['descriptor']['trajectory']['timestep_value']==999.0
    assert np.array_equal(capture['values'],np.zeros((1,2,3,4),dtype=np.float16))
    from PIL import Image
    with Image.open(tmp_path/'image'/'image.png') as image:assert image.size==(64,64)


def test_diffusion_specs_roundtrip_and_image_identity_changes():
    from pipelines_v2.engine.diffusers.specs import SDXLEngine,DiffusionGenerationSpec,DiffusionCaptureSpec,load_diffusion_spec
    from pipelines_v2.engine.diffusers.hooks import DenoisingHook
    from dataclasses import replace
    import json
    row=ImageRequest('scene','red teapot')
    spec=DiffusionGenerationSpec(images=(row,))
    assert load_diffusion_spec(json.loads(json.dumps(spec.to_dict())))==spec
    assert replace(spec,engine=SDXLEngine(cache_root='/other-model-mount')).semantic_hash()==spec.semantic_hash()
    for changed in (replace(row,prompt='blue teapot'),replace(row,seed=1),replace(row,steps=19)):
        assert replace(spec,images=(changed,)).semantic_hash()!=spec.semantic_hash()
    capture=DiffusionCaptureSpec(images=(row,),hooks=(DenoisingHook('mid_block.resnets.1',(10,),(1280,32,32)),))
    assert load_diffusion_spec(json.loads(json.dumps(capture.to_dict())))==capture
    with pytest.raises(ValueError):load_diffusion_spec({**spec.to_dict(),'schema_version':True})


def test_diffusion_intervention_requires_explicit_shared_noise():
    from pipelines_v2.engine.diffusers.specs import DiffusionInterventionSpec
    from pipelines_v2.engine.diffusers.hooks import DenoisingHook
    with pytest.raises(ValueError,match='shared noise'):
        DiffusionInterventionSpec(images=(ImageRequest('a','prompt'),),hooks=(DenoisingHook('site',(0,),(1,1,1),operator='ablate'),))


def test_image_boundary_resume_runs_only_missing_images_and_reuses_final(tmp_path,monkeypatch):
    import numpy as np
    from pipelines_v2.api import LocalArtifactStore,ToyEngine,GenerationRunSpec,GenerationSpec,Dataset,Example
    from pipelines_v2.core.provenance import resolve_provenance
    from pipelines_v2.storage.finalization import validate_completion
    from pipelines_v2.storage.artifacts import ArtifactManifest,OperationArtifact
    from pipelines_v2.engine.diffusers.specs import DiffusionGenerationSpec
    from pipelines_v2.engine.diffusers.execution import execute_diffusion,diffusion_image,_input_noise
    import pipelines_v2.engine.diffusers.execution as execution
    fixture=GenerationRunSpec(engine=ToyEngine(),dataset=Dataset.from_examples([Example(key='fixture',prompt='fixture')]),generation=GenerationSpec(enabled=True,max_tokens=1))
    provenance=resolve_provenance(fixture)
    context={'provenance_v1':provenance.to_dict(),'run_id':'wr-fixture','workflow_step_key':'images'}
    store=LocalArtifactStore(tmp_path);calls=[];failed=[False]
    class Session:
        loads=1
        def __init__(self,*args):pass
        def generate(self,request,*,noise,hooks,replacements):
            if request.key=='b' and not failed[0]:failed[0]=True;raise RuntimeError('interrupted between images')
            calls.append(request.key)
            return {'image':np.zeros((64,64,3),dtype=np.float32),'latents':noise.numpy().copy(),'captures':{},
                'metadata':{'components':{'revision':'a'*40,'components':{'unet':'b'*64,'vae':'c'*64}},
                    'initial_noise_digest':tensor_digest(noise),'scheduler_name':'EulerDiscreteScheduler',
                    'scheduler_config_digest':'d'*64,'cfg_enabled':True,'timesteps':[999.,0.]}}
        noise=SDXLSession.noise
    monkeypatch.setattr(execution,'SDXLSession',Session)
    spec=DiffusionGenerationSpec(images=tuple(ImageRequest(key,'fixture',width=64,height=64,steps=2) for key in ('a','b')))
    from pipelines_v2.operations.registry import _OPERATION_REGISTRATIONS,OperationRegistration
    monkeypatch.setitem(_OPERATION_REGISTRATIONS,spec.kind,OperationRegistration(spec.from_dict,'artifact',spec.kind,lambda _:None))
    kwargs=dict(spec=spec,store=store,runner_config={'kind':'test'},workflow_context=context)
    with pytest.raises(RuntimeError,match='interrupted'):execute_diffusion(**kwargs)
    from pipelines_v2.runtime.remote_executor import _artifact_id_for
    parent_id=_artifact_id_for(spec=spec,workflow_context=context)
    assert not (tmp_path/parent_id/'manifest.json').exists()
    manifest=ArtifactManifest.from_dict(execute_diffusion(**kwargs))
    assert calls==['a','b']
    assert validate_completion(manifest,store=store)
    assert execute_diffusion(**kwargs)['artifact_id']==manifest.artifact_id
    assert calls==['a','b']
    artifact=OperationArtifact(manifest,store)
    assert diffusion_image(artifact,'a').manifest().example_coverage['example_keys']==['a']
    from dataclasses import replace
    reused_noise,state=_input_noise(replace(spec,noise_from=artifact),spec.images[0])
    original,_=SDXLSession.noise(None,spec.images[0])
    assert torch.equal(original,reused_noise)
    assert set(state)=={'before','after'}
