from dataclasses import replace
import pytest
from pipelines_v2.engine.recognition import ViTSite, ViTRecognitionEngine, RecognitionSpec, token_geometry
from pipelines_v2.engine.recognition.contracts import PREPROCESSING, REVISION


def test_coordinate_roundtrip_and_original_image_mapping():
    site=ViTSite(tokens=(0,1,98,196),channels=(0,7))
    assert ViTSite.from_dict(site.to_dict())==site
    assert token_geometry(0,(128,192))['input_xyxy'] is None
    assert token_geometry(1,(128,192))['preprocessed_xyxy']==[0,0,16,16]
    mapping=token_geometry(98,(128,192))
    assert mapping['patch_yx']==[6,13]
    assert mapping['input_xyxy']==[208*192/224,96*128/224,192.,112*128/224]
    assert token_geometry(196,(128,192))['input_xyxy'][2:]==[192.,128.]
    with pytest.raises(ValueError): ViTSite(tokens=(197,),channels=(0,))
    with pytest.raises(ValueError): ViTSite(tokens=(1,1),channels=(0,))
    with pytest.raises(ValueError): ViTSite.from_dict(dict(site.to_dict(),schema_version=2))
    with pytest.raises(ValueError): ViTSite.from_dict(dict(site.to_dict(),module_path='wrong'))


def test_image_workflow_dataclass_roundtrip(tmp_path):
    from pipelines_v2.data.images import prepare_image
    from pipelines_v2.api import Dataset,Example
    from pipelines_v2.workflow.orchestrator import _resolve_step_refs
    from pipelines_v2.operations.registry import operation_spec_from_dict
    from PIL import Image
    raw=tmp_path/'input.png';Image.new('RGB',(32,16)).save(raw)
    image=prepare_image(raw,cache_root=tmp_path/'images',preprocessing=PREPROCESSING)
    site=ViTSite(tokens=(0,1),channels=(0,))
    spec=RecognitionSpec(engine=ViTRecognitionEngine('/prepared'),site=site,edit=replace(site,tokens=(1,),read_or_write='write'),
        dataset=Dataset.from_examples((Example(key='a',prompt='',images=(image,)),)),image_roots=(str(tmp_path),))
    assert operation_spec_from_dict(spec.to_dict()).to_dict()==spec.to_dict()
    assert _resolve_step_refs(spec,{}).to_dict()==spec.to_dict()
    spec.validate_images()
    with pytest.raises(ValueError,match='processor'):
        replace(spec,dataset=Dataset.from_examples((Example(key='a',prompt='',images=(replace(image,preprocessing_digest='0'*64),)),))).validate_images()
    with pytest.raises(ValueError,match='outside'):
        replace(spec,image_roots=(str(tmp_path/'other'),)).validate_images()


def test_scoped_hook_preserves_selected_edit_and_cleans_up():
    import torch
    from transformers import ViTConfig,ViTForImageClassification
    from pipelines_v2.engine.recognition.session import RecognitionSession
    with torch.random.fork_rng():
        torch.manual_seed(1)
        model=ViTForImageClassification(ViTConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=1,
            num_attention_heads=2,image_size=224,patch_size=16,num_labels=5,attn_implementation='eager'))
    with RecognitionSession(model,None) as session:
        pixels=torch.zeros(1,3,224,224);site=ViTSite(tokens=(0,1),channels=(0,1))
        baseline,capture=session.forward(pixels,site)
        identity,_=session.forward(pixels,site,edit=replace(site,read_or_write='write'),scale=1.)
        assert torch.equal(baseline,identity)
        _,altered=session.forward(pixels,site,edit=replace(site,tokens=(1,),channels=(0,),read_or_write='write'),scale=0.)
        assert altered[0,1,0]==0 and torch.equal(altered[0,0],capture[0,0]) and altered[0,1,1]==capture[0,1,1]
        block=model.vit.encoder.layer[0]
        failing=block.register_forward_hook(lambda *_: (_ for _ in ()).throw(RuntimeError('oracle failure')))
        try:
            with pytest.raises(RuntimeError,match='oracle failure'):session.forward(pixels,site)
            assert len(block._forward_hooks)==1
        finally:failing.remove()
        assert not block._forward_hooks


def test_pinned_assets_detect_mutation_before_model_load(tmp_path):
    from pipelines_v2.engine.recognition.engine import verify_assets
    (tmp_path/'config.json').write_text('{}')
    with pytest.raises(ValueError,match='bytes differ'):verify_assets(tmp_path)
    with pytest.raises(ValueError):ViTRecognitionEngine(str(tmp_path),revision='main')
