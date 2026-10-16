from dataclasses import replace
import json
import struct

import pytest

from pipelines_v2.api import Dataset, Example, GenerationRunSpec, GenerationSpec, LocalRunner, ToyEngine, VLLMEngine
from pipelines_v2.core.provenance import resolve_provenance
from pipelines_v2.runtime.admission import AdmissionCheck, WorkloadEnvelope, effective_configuration, static_admission, workload_envelope
from pipelines_v2.runtime.readiness import local_checkpoint_components, runtime_readiness


def spec(engine=None):
    return GenerationRunSpec(engine=engine or ToyEngine(), dataset=Dataset.from_examples([Example(key='a', prompt='hi')]), generation=GenerationSpec(enabled=True, max_tokens=1))


def test_checks_and_envelope_roundtrip_unknown_is_not_passed():
    check = AdmissionCheck('model', 'readiness', 'unresolved', 'model.revision', 'No resolved commit')
    assert AdmissionCheck.from_dict(json.loads(json.dumps(check.to_dict()))) == check
    envelope = workload_envelope(spec(), {'kind': 'local'})
    assert WorkloadEnvelope.from_dict(json.loads(json.dumps(envelope.to_dict()))).identity == envelope.identity
    plan = LocalRunner().plan(spec())
    assert plan.valid and not plan.ready and not plan.qualified
    with pytest.raises(ValueError):
        replace(check, phase='qualification', status='passed')


@pytest.mark.parametrize('value', [-1, 0, float('nan'), float('inf'), '4', True])
def test_invalid_resources_static_failure(value):
    checks = static_admission(spec(), {'kind': 'modal', 'resources': {'memory_mb': value}})
    assert next(c for c in checks if c.check_id == 'resources.memory_mb').status == 'failed'


def test_redaction_and_override_origins(monkeypatch):
    from pipelines_v2.engine.base import PythonRuntimeSpec
    monkeypatch.setattr(ToyEngine, 'runtime_spec', lambda self: PythonRuntimeSpec(env={'OMP_NUM_THREADS': '1', 'HF_TOKEN': 'never-leak'}))
    config = effective_configuration(spec(), {'kind': 'local', 'resources': {'env': {'OMP_NUM_THREADS': '2', 'API_KEY': 'also-secret'}}})
    assert config['environment']['OMP_NUM_THREADS'] == {'value': '2', 'origin': 'runner.resources.env'}
    assert 'never-leak' not in json.dumps(config) and 'also-secret' not in json.dumps(config)


def checkpoint(tmp_path):
    (tmp_path / 'config.json').write_text('{"model_type":"gpt2","torch_dtype":"float32"}')
    (tmp_path / 'tokenizer.json').write_text('{"fixture":"tokenizer"}')
    header = json.dumps({'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0,4]}}).encode()
    (tmp_path / 'model.safetensors').write_bytes(struct.pack('<Q',len(header)) + header + struct.pack('<f',1))
    return tmp_path


def test_checkpoint_hashes_bytes_and_rejects_incomplete_cache(tmp_path):
    root = checkpoint(tmp_path)
    before, _ = local_checkpoint_components(root)
    (root / 'tokenizer.json').write_text('{"fixture":"changed"}')
    after, _ = local_checkpoint_components(root)
    assert before[0] == after[0] and before[1] != after[1]
    (root / 'model.safetensors.index.json').write_text('{"weight_map":{"weight":"missing-shard.safetensors"}}')
    with pytest.raises(ValueError, match='incomplete'):
        local_checkpoint_components(root)


def test_truncated_checkpoint_fails_before_any_model_load(tmp_path):
    root = checkpoint(tmp_path)
    weights = root / 'model.safetensors'
    weights.write_bytes(weights.read_bytes()[:-1])
    with pytest.raises(ValueError, match='Truncated'):
        local_checkpoint_components(root)


def test_lock_inventory_mismatch_and_import_failure_are_not_ready(monkeypatch):
    from pipelines_v2.engine.base import PythonRuntimeSpec
    from pipelines_v2.runtime.locks import RuntimeLock
    import pipelines_v2.runtime.readiness as module
    lock = RuntimeLock('fixture', 'darwin-arm64', '3.13', 'missing-library==1\n', ('unimportable_module',), 'fixture')
    monkeypatch.setattr(module.importlib, 'import_module', lambda name: (_ for _ in ()).throw(ImportError('secret detail')))
    _, checks = runtime_readiness(PythonRuntimeSpec(lock=lock))
    assert [check.status for check in checks] == ['failed', 'failed']
    assert 'secret detail' not in json.dumps([check.to_dict() for check in checks])


def test_cpu_numerical_policies_are_production_resolvers():
    from pipelines_v2.api import ProbeSpec, BasisSpec, DirectionSpec, GeometrySpec
    for operation in (ProbeSpec(), BasisSpec(random_state=7), BasisSpec(svd_solver='full'), DirectionSpec(), GeometrySpec(method='lda')):
        assert resolve_provenance(operation).identity_complete
    assert not resolve_provenance(BasisSpec()).identity_complete
    before = resolve_provenance(ProbeSpec())
    assert resolve_provenance(ProbeSpec(seed=3)).scientific_identity != before.scientific_identity
    assert resolve_provenance(BasisSpec(random_state=1)).scientific_identity != resolve_provenance(BasisSpec(random_state=2)).scientific_identity


def test_admission_rebinds_actual_request_source_and_scope(tmp_path):
    from pipelines_v2.runtime.readiness import AdmissionRecord, admitted_provenance_for_request
    operation = spec()
    record = AdmissionRecord((AdmissionCheck('fixture', 'readiness', 'passed', 'fixture', 'Observed fixture'),),
        resolve_provenance(operation), workload_envelope(operation, {'kind':'local'}).to_dict())
    assert admitted_provenance_for_request(operation, {'kind':'local'}, record).scientific_identity == record.provenance.scientific_identity
    changed = replace(operation, dataset=Dataset.from_examples([Example(key='a',prompt='changed')]))
    with pytest.raises(ValueError, match='changed'):
        admitted_provenance_for_request(changed, {'kind':'local'}, record)
    with pytest.raises(ValueError, match='execution evidence'):
        admitted_provenance_for_request(operation, {'kind':'local'}, replace(record, scope='cpu_preflight'))
    payload = json.loads(json.dumps(record.to_dict()))
    assert AdmissionRecord.from_dict(payload).provenance.scientific_identity == record.provenance.scientific_identity
    payload['scope'] = 'qualified'
    with pytest.raises(ValueError):
        AdmissionRecord.from_dict(payload)


def test_writer_readiness_forces_access_and_reports_provisioning(monkeypatch):
    import sys
    from types import SimpleNamespace
    from pipelines_v2.runtime.readiness import writer_registry_readiness
    calls = []
    class Registry:
        @classmethod
        def from_name(cls, name, **kwargs):
            calls.append((name,kwargs))
            return cls()
        def get(self, key, default):
            raise PermissionError('secret access detail')
    monkeypatch.setitem(sys.modules,'modal',SimpleNamespace(Dict=Registry))
    check = writer_registry_readiness()
    assert check.status == 'failed' and 'modal dict create' in check.remediation
    assert calls == [('xenon-artifact-writers-v1', {'create_if_missing':False})]
    assert 'secret access detail' not in json.dumps(check.to_dict())


def test_raw_credentials_rejected_without_echo():
    operation = spec(VLLMEngine(model_id='private/model', extra={'hf_token':'private-never-echo'}))
    plan = LocalRunner().plan(operation)
    assert not plan.valid
    assert any('credential' in error for error in plan.errors)
    assert 'private-never-echo' not in json.dumps(plan.effective_config)


@pytest.mark.parametrize('gpu',['A100:0','A100:9','A10:5','typo-gpu'])
def test_bad_modal_gpu_never_submits(tmp_path, monkeypatch, gpu):
    from pipelines_v2.api import ModalRunner, ModalResources, ModalVolumeStore
    from pipelines_v2.core.types import SpecValidationError
    calls=[]
    monkeypatch.setattr('pipelines_v2.runtime.modal.run_on_modal',lambda **kw:calls.append(kw))
    runner=ModalRunner(resources=ModalResources(gpu=gpu),artifacts=ModalVolumeStore(name='fixture',root=str(tmp_path)))
    with pytest.raises(SpecValidationError):
        runner.run(spec())
    assert calls == []


def test_resource_typos_and_string_sizes_are_not_coerced(tmp_path):
    from pipelines_v2.api import ModalResources, ModalRunner, ModalVolumeStore
    with pytest.raises(ValueError,match='Unknown'):
        ModalResources.from_dict({'memroy_mb':8192})
    runner=ModalRunner(resources=ModalResources.from_dict({'memory_mb':'8192'}),artifacts=ModalVolumeStore(name='fixture',root=str(tmp_path)))
    assert not runner.plan(spec()).valid


def test_worker_rejects_observed_runtime_drift(monkeypatch):
    from pipelines_v2.runtime.readiness import AdmissionRecord, verify_admitted_execution
    from pipelines_v2.core.provenance import ProvenanceV1
    operation=spec()
    original=AdmissionRecord((AdmissionCheck('observed','readiness','passed','fixture','Observed'),),resolve_provenance(operation),workload_envelope(operation,{'kind':'local'}).to_dict())
    body=original.provenance.to_dict()
    body['runtime']['installed_package_digest']='0'*64
    body.pop('schema_version');body.pop('scientific_identity')
    changed=replace(original,provenance=ProvenanceV1(**body))
    monkeypatch.setattr('pipelines_v2.runtime.readiness.admit_execution',lambda *args,**kwargs:changed)
    with pytest.raises(ValueError,match='runtime'):
        verify_admitted_execution(operation,{'kind':'local'},original)


def test_model_policy_requires_explicit_effective_configuration(monkeypatch):
    import sys
    from types import SimpleNamespace
    from pipelines_v2.runtime.numerics import model_numerical_policy
    device=SimpleNamespace(name='A100',major=8,minor=0,total_memory=80*1024**3)
    torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:1,get_device_properties=lambda index:device),
        version=SimpleNamespace(cuda='13.0'),get_float32_matmul_precision=lambda:'highest',
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),cudnn=SimpleNamespace(allow_tf32=True)))
    monkeypatch.setitem(sys.modules,'torch',torch)
    engine=VLLMEngine(model_id='fixture',dtype='bfloat16',model_seed=0,enforce_eager=True,extra={'attention_backend':'FLASH_ATTN'})
    observed,seeds=model_numerical_policy(spec(engine))
    assert observed['dtype']=='bfloat16' and seeds['model.initialization']==0
    assert 'nondeterministic' in observed['determinism']
    assert model_numerical_policy(spec(replace(engine,enforce_eager=False)))[0] is None
    assert model_numerical_policy(spec(replace(engine,model_seed=None)))[0] is None
    assert model_numerical_policy(spec(replace(engine,dtype='float16')))[0] != observed


def test_persisted_admission_rejects_changed_numerical_environment():
    from pipelines_v2.runtime.readiness import AdmissionRecord, admitted_provenance_for_request
    operation=spec()
    identity={'kind':'local','resources':{'env':{'OMP_NUM_THREADS':'1'}}}
    original=AdmissionRecord((AdmissionCheck('observed','readiness','passed','fixture','Observed'),),
        resolve_provenance(operation,identity),workload_envelope(operation,identity).to_dict())
    with pytest.raises(ValueError,match='envelope'):
        admitted_provenance_for_request(operation,{'kind':'local','resources':{'env':{'OMP_NUM_THREADS':'2'}}},original)


def test_persisted_admission_rehashes_accessible_checkpoint(tmp_path):
    from pipelines_v2.runtime.readiness import AdmissionRecord, admitted_provenance_for_request
    root=checkpoint(tmp_path)
    operation=spec(VLLMEngine(model_id=str(root),dtype='float32',model_seed=0,extra={'attention_backend':'FLASH_ATTN'}))
    components,_=local_checkpoint_components(root)
    numerics=resolve_provenance(spec()).to_dict()['numerics']
    provenance=resolve_provenance(operation,model_components=components,numerics=numerics)
    assert provenance.identity_complete
    record=AdmissionRecord((AdmissionCheck('observed','readiness','passed','fixture','Observed'),),provenance,workload_envelope(operation,{'kind':'local'}).to_dict())
    admitted_provenance_for_request(operation,{'kind':'local'},record)
    (root/'tokenizer.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='bytes changed'):
        admitted_provenance_for_request(operation,{'kind':'local'},record)


def test_mutable_remote_admission_refresh_and_immutable_pinned_intent(tmp_path,monkeypatch):
    from pipelines_v2.api import ModalRunner, ModalResources, ModalVolumeStore
    from pipelines_v2.runtime.readiness import AdmissionRecord, requires_target_refresh
    from pipelines_v2.core.provenance import resolve_for_runner
    operation=spec(VLLMEngine(model_id='/remote/model',dtype='bfloat16',model_seed=0,extra={'attention_backend':'FLASH_ATTN'}))
    identity={'kind':'local'}
    components=[{'role':role,'repository_id':'fixture','requested_revision':None,'resolved_revision':'sha256:'+'a'*64,'file_set_digest':'a'*64} for role in ['model','tokenizer']]
    provenance=resolve_provenance(operation,identity,model_components=components,numerics=resolve_provenance(spec()).to_dict()['numerics'])
    runner=ModalRunner(resources=ModalResources(gpu='A100-80GB'),artifacts=ModalVolumeStore(name='fixture',root=str(tmp_path)))
    record=AdmissionRecord((AdmissionCheck('observed','readiness','passed','fixture','Observed'),),provenance,workload_envelope(operation,runner.identity()).to_dict())
    assert requires_target_refresh(operation,record)
    runner.admission_records[operation.semantic_hash()]=record
    refreshed=[]
    monkeypatch.setattr(ModalRunner,'refresh_admission_for_reuse',lambda self,spec,record:refreshed.append(spec) or False)
    observed=resolve_for_runner(operation,runner)
    assert refreshed==[operation] and not observed.identity_complete
    assert operation.semantic_hash() not in runner.admission_records
    pinned=replace(operation,engine=replace(operation.engine,revision='b'*40))
    pinned_components=[{**item,'requested_revision':'b'*40} for item in components]
    pinned_record=replace(record,provenance=resolve_provenance(pinned,model_components=pinned_components,numerics=resolve_provenance(spec()).to_dict()['numerics']))
    assert not requires_target_refresh(pinned,pinned_record)
