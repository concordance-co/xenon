from dataclasses import replace
import pytest
from pipelines_v2.api import TorchEngine,GenerationRunSpec,GenerationSpec,Dataset,Example,ModalRunner,ModalResources,ModalVolumeMount,ModalVolumeStore
from pipelines_v2.research.llm import dataset_from_config,build_workflow,runner_specs

REV='607a30d783dfa663caf39e06633721c8d4cfcd7e'

def spec():
    return GenerationRunSpec(engine=TorchEngine('/models/gpt2',REV,device='cuda'),dataset=Dataset.from_examples([Example(key='a',prompt='Hello')]),generation=GenerationSpec(enabled=True,max_tokens=1))

def test_remote_assets_are_deferred_but_allocation_and_mount_are_not(tmp_path):
    resources=ModalResources(gpu='L4',memory_mb=16384,timeout_seconds=300,volumes=(ModalVolumeMount('models','/models'),))
    def errors(r):return ModalRunner(resources=r,artifacts=ModalVolumeStore('data','/data/test'),workspace_root=tmp_path).plan(spec()).errors
    assert not errors(resources)
    assert any('GPU' in e for e in errors(replace(resources,gpu=None)))
    assert any('mount' in e for e in errors(replace(resources,volumes=())))
    assert any('memory' in e for e in errors(replace(resources,memory_mb=None)))

def test_config_dataset_identity_and_explicit_limit():
    rows=[{'key':str(i),'prompt':f'Text {i}'} for i in range(40)]
    dataset=dataset_from_config({'examples':rows,'max_examples':40})
    assert len(dataset.examples)==40
    with pytest.raises(ValueError,match='max_examples'):dataset_from_config({'examples':rows,'max_examples':39})
    with pytest.raises(ValueError,match='Duplicate'):dataset_from_config({'examples':rows+[rows[0]]})

def test_sae_generalized_dataset_and_feature_edits_roundtrip():
    from pipelines_v2.api import SAEFeatureSpliceSpec,StepRef
    from pipelines_v2.operations.registry import operation_spec_from_dict
    dataset=dataset_from_config({'examples':[{'key':str(i),'prompt':'Hello'} for i in range(40)]})
    original=SAEFeatureSpliceSpec(dictionary=StepRef('dictionary'),engine=TorchEngine('/models/gpt2',REV),dataset=dataset,feature_edits=((0,1.),(1,-2.),(2,3.),(3,4.)))
    loaded=operation_spec_from_dict(original.to_dict())
    assert loaded.to_dict()==original.to_dict()
    with pytest.raises(ValueError,match='max_examples'):replace(original,max_examples=39)

def test_mattr_custom_data_identity_and_workload(tmp_path):
    import json
    from pipelines_v2.operations.attribution.specs import MAttrSpec,official_tool_lock
    from pipelines_v2.runtime.external_tool import ExternalToolRuntimeSpec
    runtime=ExternalToolRuntimeSpec(tool_lock=official_tool_lock())
    rows=[{'prompt':'Mary and John. John gave to','metadata':{'indirect_object':'Mary','subject':'John','random_c':'Tom'},'s2_io_flip_counterfactual':{'prompt':'Mary and John. Mary gave to'}} for _ in range(40)]
    for filename in ('train.json','dev.json'):(tmp_path/filename).write_text(json.dumps(rows))
    original=MAttrSpec.from_ioi_files(model='gpt2',model_path='/prepared',dataset_path=tmp_path,runtime=runtime,steps=501,eval_examples=33,timeout_seconds=900)
    assert original.request['data']['repository'].startswith('local-content:')
    (tmp_path/'dev.json').write_text(json.dumps(rows[:32]))
    with pytest.raises(ValueError,match='exceeds'):MAttrSpec.from_ioi_files(model='gpt2',model_path='/prepared',dataset_path=tmp_path,runtime=runtime,eval_examples=33)


def test_remote_native_site_can_bind_prepared_digest_without_local_model(tmp_path):
    from pipelines_v2.api import NativeTorchEngine
    from pipelines_v2.operations.common.native import HEAD_QUANTITY
    engine=NativeTorchEngine('/models/unavailable-here',REV,'gpt2',device='cuda')
    site=engine.site('head',HEAD_QUANTITY,0,heads=(0,),model_identity='a'*64)
    assert site.model_identity=='a'*64
    with pytest.raises(ValueError,match='SHA256'):engine.site('head',HEAD_QUANTITY,0,heads=(0,),model_identity='mutable-alias')


def test_sae_import_requires_explicit_or_artifact_volume_mount(tmp_path):
    from pipelines_v2.api import SAEDictionaryImportSpec
    from pipelines_v2.runtime.admission import static_admission
    runner=ModalRunner(ModalResources(memory_mb=16384,timeout_seconds=300),ModalVolumeStore('data','/data/test'),workspace_root=tmp_path)
    def failures(path):
        operation=SAEDictionaryImportSpec(model_path=path+'/gpt2',cache_root=path+'/sae')
        return [c for c in static_admission(operation,runner.identity(),artifact_store=runner.artifacts,workspace_root=tmp_path) if c.check_id=='resources.sae_mount' and c.status=='failed']
    assert not failures('/data/prepared')
    assert len(failures('/unmounted'))==2


def test_crosscoder_artifact_mount_and_single_device_topology(tmp_path):
    from pipelines_v2.operations.representation.crosscoder import CrosscoderPairCaptureSpec
    from pipelines_v2.runtime.admission import static_admission
    operation=CrosscoderPairCaptureSpec(model_paths=('/data/models/base','/data/models/instruct'),dataset=Dataset.from_examples([Example(key='train',prompt='Hello'),Example(key='heldout',prompt='World')]),train_count=1)
    store=ModalVolumeStore('data','/data/test')
    def checks(resources):return static_admission(operation,{'kind':'modal','resources':resources},artifact_store=store,workspace_root=tmp_path)
    assert not [c for c in checks({'shard_count':1}) if c.check_id=='resources.crosscoder_mount' and c.status=='failed']
    for resources in ({'shard_count':2},{'gpu':'L4:2'}):
        assert any(c.check_id=='resources.crosscoder_topology' and c.status=='failed' for c in checks(resources))


@pytest.mark.parametrize('kind',['sae','crosscoder'])
def test_invalid_mounts_return_failed_plan_checks(tmp_path,kind):
    from pipelines_v2.api import SAEDictionaryImportSpec
    from pipelines_v2.operations.representation.crosscoder import CrosscoderPairCaptureSpec
    if kind=='sae':operation=SAEDictionaryImportSpec(model_path='/models/gpt2',cache_root='/models/sae')
    else:operation=CrosscoderPairCaptureSpec(model_paths=('/models/base','/models/instruct'),dataset=Dataset.from_examples([Example(key='a',prompt='Hi'),Example(key='b',prompt='Hey')]),train_count=1)
    resources=ModalResources(memory_mb=16384,timeout_seconds=600,volumes=(ModalVolumeMount('one','/models'),ModalVolumeMount('two','/models/subdir')))
    plan=ModalRunner(resources,ModalVolumeStore('data','/data/test'),workspace_root=tmp_path).plan(operation)
    assert any(c.status=='failed' and c.check_id=='resources.'+kind+'_mount' for c in plan.admission_checks)
