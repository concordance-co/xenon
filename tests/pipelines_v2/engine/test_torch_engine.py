from dataclasses import replace
import json
import subprocess
import sys
import pytest

from pipelines_v2.api import CaptureSpec, Dataset, Example, GenerationRunSpec, GenerationSpec, ResidualSite, TokenSelector, TorchEngine
from pipelines_v2.engine.registry import engine_from_dict


def test_tiny_forward_padding_and_reordering_match_direct_hf(tiny_torch_session):
    import torch
    s=tiny_torch_session
    ids,mask=s.encode(['2 4 6','8'])
    logits,capture=s.forward(ids,mask,capture=(('resid_pre',0),('resid_post',0)))
    direct={}
    handle=s.model.transformer.h[0].register_forward_hook(lambda module,args,out:direct.update(post=out[0].detach() if isinstance(out,tuple) else out.detach()))
    try:
        with torch.no_grad():
            reference=s.model(ids,attention_mask=mask,position_ids=(mask.cumsum(-1)-1).clamp(min=0),use_cache=False).logits
    finally:
        handle.remove()
    torch.testing.assert_close(logits,reference,rtol=1e-4,atol=1e-5)
    torch.testing.assert_close(capture['resid_post',0],direct['post'],rtol=1e-4,atol=1e-5)
    for row,prompt in enumerate(['2 4 6','8']):
        single,_=s.forward(*s.encode([prompt]))
        length=int(mask[row].sum())
        torch.testing.assert_close(logits[row,:length],single[0],rtol=1e-4,atol=1e-5)
    reverse,_=s.forward(ids.flip(0),mask.flip(0))
    torch.testing.assert_close(reverse.flip(0),logits,rtol=1e-4,atol=1e-5)


def test_real_target_logprobs_order_difference_and_capture_keys(tiny_torch_session,tmp_path):
    import torch
    import numpy as np
    engine=TorchEngine(model_id=str(tmp_path),revision='a'*40)
    dataset=Dataset.from_examples([Example(key='long',prompt='2 4 6'),Example(key='short',prompt='8')])
    generation=GenerationSpec(enabled=True,max_tokens=2,target_token_ids=(9,3))
    spec=GenerationRunSpec(engine=engine,dataset=dataset,generation=generation)
    result=engine.generate(spec,session=tiny_torch_session)
    for example,row in zip(dataset.examples,result.rows,strict=True):
        ids,mask=tiny_torch_session.encode([example.prompt])
        with torch.no_grad():
            logits=tiny_torch_session.model(ids,attention_mask=mask).logits[0,-1]
        expected=torch.log_softmax(logits,dim=-1)
        assert list(row['target_logprobs'])==['9','3']
        assert row['target_logprobs']['9']==pytest.approx(float(expected[9]),abs=1e-5)
        assert row['target_logprobs']['9']-row['target_logprobs']['3']==pytest.approx(float(expected[9]-expected[3]),abs=1e-5)
        assert len(row['target_logprobs'])==2
    capture=engine.capture(CaptureSpec(engine=engine,dataset=dataset,sites=(ResidualSite('r','resid_post',(0,),TokenSelector.full_sequence()),)),session=tiny_torch_session)
    rows=capture.features['r']['layers']['0']
    assert rows['short']['tokens']==[0] and rows['long']['tokens']==[0,1,2]
    assert np.asarray(rows['short']['values']).shape==(1,16)
    assert capture.features['r']['model_identity']=='tiny-fixed-checkpoint'
    assert engine_from_dict(engine.identity()).identity()==engine.identity()


def test_target_vocab_rejection_and_legacy_omission(tiny_torch_session,tmp_path):
    engine=TorchEngine(model_id=str(tmp_path),revision='a'*40)
    dataset=Dataset.from_examples([Example(key='a',prompt='1 2')])
    spec=GenerationRunSpec(engine=engine,dataset=dataset,generation=GenerationSpec(enabled=True,max_tokens=1))
    assert 'target_token_ids' not in spec.to_dict()['generation']
    assert 'target_token_ids' not in spec.semantic_dict()['generation']
    assert 'target_logprobs' not in engine.generate(spec,session=tiny_torch_session).rows[0]
    with pytest.raises(ValueError,match='vocabulary'):
        engine.generate(replace(spec,generation=replace(spec.generation,target_token_ids=(32,))),session=tiny_torch_session)
    for bad in [(-1,),(True,),(1,1),(1.5,)]:
        with pytest.raises(ValueError,match='unique nonnegative'):
            GenerationSpec(target_token_ids=bad)


def test_optional_model_imports_stay_lazy():
    process=subprocess.run([sys.executable,'-c',"import sys; from pipelines_v2.engine import engine_from_dict; engine_from_dict({'kind':'toy'}); assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],capture_output=True,text=True)
    assert process.returncode==0,process.stderr


def test_unsupported_target_and_scale_fail_without_modal_submission(monkeypatch,tmp_path):
    from pipelines_v2.api import ToyEngine,VLLMEngine,ModalRunner,ModalResources,ModalVolumeStore,FileCatalog,PatchedGenerationSpec,ScalePatch,ResidualInterventionSite
    monkeypatch.setattr('pipelines_v2.runtime.modal.run_on_modal',lambda **kwargs:pytest.fail('Must fail before GPU submission'))
    dataset=Dataset.from_examples([Example(key='a',prompt='test')])
    for engine in (ToyEngine(),VLLMEngine(model_id='gpt2')):
        runner=ModalRunner(resources=ModalResources(gpu='L4'),artifacts=ModalVolumeStore(name='test',root='/data/artifacts'),catalog=FileCatalog(tmp_path/'catalog'))
        requests=[GenerationRunSpec(engine=engine,dataset=dataset,generation=GenerationSpec(enabled=True,max_tokens=1,target_token_ids=(1,))),
                  PatchedGenerationSpec(engine=engine,dataset=dataset,generation=GenerationSpec(enabled=True,max_tokens=1),patch=ScalePatch(write_site=ResidualInterventionSite('resid_post',(0,)),scale=0.))]
        for request in requests:
            plan = runner.plan(request)
            if request.generation.target_token_ids:
                from pipelines_v2.api import EngineCapability
                assert EngineCapability.TARGET_LOGPROBS in plan.missing_capabilities
            else:
                assert any('scale' in error for error in plan.errors)
            with pytest.raises(Exception):
                runner.run(request)


def test_scale_codec_and_strict_prompt_only(tmp_path):
    from pipelines_v2.api import ScalePatch,PatchedGenerationSpec,ResidualInterventionSite,PatchApplication
    from pipelines_v2.operations.interventions import ActivationPatchSpec
    patch=ScalePatch(write_site=ResidualInterventionSite('resid_post',(0,)),scale=0.)
    assert ActivationPatchSpec.from_dict(patch.to_dict()).scale==0.
    with pytest.raises(ValueError,match='finite'):
        ScalePatch(scale=float('nan'))
    engine=TorchEngine(str(tmp_path),'a'*40)
    request=PatchedGenerationSpec(engine=engine,dataset=Dataset.from_examples([Example(key='a',prompt='1 2')]),generation=GenerationSpec(enabled=True,max_tokens=1),patch=replace(patch,application=PatchApplication.every_token()))
    assert any('decode' in error for error in engine.planning_errors(request))
    # Preserve positional compatibility of the longstanding enabled field.
    assert GenerationSpec(True,1).enabled is True
