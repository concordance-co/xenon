from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest

from pipelines_v2.representation.crosscoder.alignment import align_sides,validate_pairs
from pipelines_v2.representation.crosscoder.fitting import FitConfig,fit,CrosscoderDictionary,validate_checkpoint,tree_digest
from pipelines_v2.core.provenance import content_hash

REFERENCE='/private/tmp/xenon-crosscoder-reference'


def pairs():
    rng=np.random.default_rng(7);sides=[]
    for side in range(2):
        sides.append([{'key':str(i),'token_ids':[1,2,3],'positions':[0,1,2],'prompt_digest':str(i),
                       'values':rng.normal(size=(3,8)).astype(np.float32)} for i in range(4)])
    models=[{'checkpoint_digest':str(i),'revision':str(i)*40} for i in range(2)]
    return sides,models


def aligned():
    sides,models=pairs();return align_sides(sides,models,layer=12,train_keys=['0','1','2'])


@pytest.mark.parametrize('mutation',['swapped_rows','tokenization','missing','duplicate','positions'])
def test_alignment_rejects_invalid_pair(mutation):
    sides,models=pairs()
    if mutation=='swapped_rows':sides[1][0],sides[1][1]=sides[1][1],sides[1][0]
    elif mutation=='tokenization':sides[1][0]['token_ids']=[3,2,1]
    elif mutation=='missing':sides[1].pop()
    elif mutation=='duplicate':sides[1][1]['key']=sides[1][0]['key']
    else:sides[1][0]['positions']=[1,2,3]
    with pytest.raises(ValueError):align_sides(sides,models,layer=12,train_keys=['0','1','2'])


def test_changed_model_order_or_values_fails_identity():
    payload=aligned();changed=deepcopy(payload);changed['alignment']['model_sides'].reverse()
    with pytest.raises(ValueError,match='integrity'):validate_pairs(changed)
    changed=deepcopy(payload);changed['records']['0']['values']=changed['records']['0']['values'][:,::-1].copy()
    with pytest.raises(ValueError,match='content changed'):validate_pairs(changed)


def test_pinned_reference_resumed_fit_matches_uninterrupted_safe_state(tmp_path):
    pytest.importorskip('zstandard')
    if not Path(REFERENCE).exists():pytest.skip('Pinned reference preparation is a separate offline step')
    from pipelines_v2.storage.local import LocalArtifactStore
    from pipelines_v2.storage.features import write_capture_features,load_feature_payload
    config=FitConfig(width=4,steps=10,batch_size=4)
    values=aligned();full=fit(values,REFERENCE,config,verify_environment=False)
    partial=fit(values,REFERENCE,config,stop_after=5,verify_environment=False)
    store=LocalArtifactStore(tmp_path)
    refs=write_capture_features(store,'checkpoint',{'state':partial})
    restored=load_feature_payload(store,refs['state'])
    validate_checkpoint(restored)
    resumed=fit(values,REFERENCE,config,checkpoint=restored,verify_environment=False)
    assert resumed['checkpoint_digest']==full['checkpoint_digest']
    assert resumed['losses']==full['losses']
    dictionary=CrosscoderDictionary(resumed,REFERENCE)
    x=values['records']['0']['values'];codes=dictionary.encode(x);reconstruction=dictionary.decode(codes)
    assert codes.shape==(3,4) and reconstruction.shape==x.shape
    assert np.isfinite(reconstruction).all()
    # Independent encoder/decoder equations on this explicitly identity-normalized reference.
    model=dictionary.model
    expected=np.maximum(np.einsum('bld,ldf->bf',x,model.encoder.weight.detach().numpy())+model.encoder.bias.detach().numpy(),0)
    np.testing.assert_allclose(codes,expected,rtol=1e-5,atol=1e-6)
    np.testing.assert_allclose(reconstruction,np.einsum('bf,lfd->bld',codes,model.decoder.weight.detach().numpy())+model.decoder.bias.detach().numpy(),rtol=1e-5,atol=1e-6)
    changed=deepcopy(restored);changed['step']=6
    with pytest.raises(ValueError,match='content changed'):fit(values,REFERENCE,config,checkpoint=changed,verify_environment=False)
    with pytest.raises(ValueError,match='Foreign'):fit(values,REFERENCE,FitConfig(width=4,steps=10,batch_size=3),checkpoint=restored,verify_environment=False)


def test_operation_roundtrip_and_unknown_version_rejected():
    from pipelines_v2.api import CrosscoderPairCaptureSpec
    from pipelines_v2.operations import operation_spec_from_dict
    spec=CrosscoderPairCaptureSpec(model_paths=('/base','/instruct'),prompts=('a','b'),train_count=1)
    assert operation_spec_from_dict(spec.to_dict())==spec
    with pytest.raises(ValueError):operation_spec_from_dict({**spec.to_dict(),'schema_version':True})


def test_checked_in_workflow_loads_and_all_specs_roundtrip(monkeypatch):
    from pipelines_v2.cli import load_python_workflow_file
    from pipelines_v2.operations import operation_spec_from_dict
    for key,value in {'BASE':'/base','INSTRUCT':'/instruct','REFERENCE':REFERENCE}.items():
        monkeypatch.setenv('XENON_CROSSCODER_'+key,value)
    path=Path(__file__).resolve().parents[3]/'workspaces/qwen_crosscoder/phase_00/workflow.py'
    dataset,workflow,runners=load_python_workflow_file(path=path)
    assert len(workflow.steps)==8 and set(runners)=={'compute','report_local'}
    for step in workflow.steps:
        assert operation_spec_from_dict(step.spec.to_dict())==step.spec
        if step.spec.kind.startswith('crosscoder'):
            with pytest.raises(ValueError,match='schema'):
                operation_spec_from_dict({**step.spec.to_dict(),'schema_version':True})


def test_configurable_width_steps_rows_and_explicit_budgets():
    from pipelines_v2.api import CrosscoderPairCaptureSpec,CrosscoderFeatureSpliceSpec,CrosscoderFitSpec,Dataset,Example
    from pipelines_v2.representation.crosscoder.settings import Limits
    from pipelines_v2.representation.crosscoder.execution import capture_examples
    config=FitConfig(width=128,steps=101,batch_size=64)
    assert config.width==128 and config.steps==101
    fit_spec=CrosscoderFitSpec(pairs='capture',reference_path='/reference',config={'width':64})
    assert fit_spec.config['seed']==0 and fit_spec.config['steps']==20
    dataset=Dataset.from_examples(tuple(Example(key=str(i),prompt='example '+str(i)) for i in range(6)))
    spec=CrosscoderPairCaptureSpec(model_paths=('/base','/instruct'),dataset=dataset,train_keys=('0','1','2','3'))
    assert len(capture_examples(spec))==6
    CrosscoderFeatureSpliceSpec(dictionary='dict',pairs='pairs',model_paths=('/base','/instruct'),reference_path='/ref',feature_index=100)
    with pytest.raises(ValueError,match='budget'):CrosscoderPairCaptureSpec(model_paths=('/base','/instruct'),dataset=dataset,limits={'max_examples':5})
    with pytest.raises(ValueError):Limits(max_total_tokens=0)


def test_token_batch_gather_matches_indexing_without_corpus_concatenation():
    from pipelines_v2.representation.crosscoder.fitting import training_batch
    payload=aligned();keys=['0','1','2'];indices=np.array([8,0,5,5,3])
    actual=training_batch(payload,keys,np.array([0,3,6,9]),indices)
    expected=np.concatenate([payload['records'][k]['values'] for k in keys])[indices]
    np.testing.assert_array_equal(actual,expected)


def test_fit_allocation_budget_rejected_before_trainer():
    from pipelines_v2.representation.crosscoder.settings import Limits
    if not Path(REFERENCE).exists():pytest.skip('Prepared reference needed')
    pytest.importorskip('zstandard')
    with pytest.raises(ValueError,match='parameter'):
        fit(aligned(),REFERENCE,FitConfig(width=128),limits=Limits(max_parameter_bytes=1),verify_environment=False)


def test_fit_more_than_512_training_tokens_and_actual_feature_width():
    if not Path(REFERENCE).exists():pytest.skip('Prepared reference needed')
    pytest.importorskip('zstandard')
    rng=np.random.default_rng(19)
    sides=[[{'key':str(i),'token_ids':list(range(300)),'positions':list(range(300)),
             'prompt_digest':str(i),'values':rng.normal(size=(300,8)).astype(np.float32)}
            for i in range(3)] for _ in range(2)]
    payload=align_sides(sides,[{'checkpoint_digest':str(i),'revision':str(i)*40} for i in range(2)],layer=12,train_keys=['0','1'])
    state=fit(payload,REFERENCE,FitConfig(width=64,steps=2,batch_size=8),verify_environment=False)
    dictionary=CrosscoderDictionary(state,REFERENCE)
    assert dictionary.feature_delta(63,1).shape==(2,8)
    with pytest.raises(ValueError,match='feature delta'):dictionary.feature_delta(64,1)


def test_modal_profile_defaults_and_explicit_volume_merge(tmp_path):
    from pipelines_v2.representation.crosscoder.workflow import runner_specs_from_config
    config={'mode':'modal','output':str(tmp_path),'execution':{'device':'cuda','runtime_profile':'linux-x86_64-cuda'},
            'modal':{'artifact_volume':'artifacts','volumes':[{'name':'reference','mount_path':'/reference'}]}}
    resources=runner_specs_from_config(config)['compute'].resources
    assert resources.gpu=='L4'
    assert {mount.mount_path:mount.name for mount in resources.volumes}=={'/models':'xenon-models','/reference':'reference'}
    config['modal']['volumes']=[{'name':'wrong-models','mount_path':'/models'}]
    with pytest.raises(ValueError,match='Conflicting'):runner_specs_from_config(config)
