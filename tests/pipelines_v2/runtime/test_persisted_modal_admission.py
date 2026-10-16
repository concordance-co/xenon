"""Persisted worker admission is usable only after a fresh target observation."""
from dataclasses import replace
from types import SimpleNamespace
import copy

import pytest
from pipelines_v2.api import (Dataset, Example, GenerationRunSpec, GenerationSpec, ToyEngine,
    LocalRunner, LocalArtifactStore, ModalRunner, ModalResources, ModalVolumeStore)
from pipelines_v2.core.provenance import resolve_provenance, resolve_for_runner, content_hash, ProvenanceV1
from pipelines_v2.runtime.admission import AdmissionCheck, workload_envelope
from pipelines_v2.runtime.readiness import AdmissionRecord


@pytest.fixture
def case(tmp_path,monkeypatch):
    import pipelines_v2.core.provenance as core
    source={'repository_id':'fixture','head':'a'*40,'dirty_digest':'b'*64,'mounted_source_digest':'c'*64}
    monkeypatch.setattr(core,'source_snapshot',lambda *a:dict(source))
    monkeypatch.setattr('pipelines_v2.runtime.readiness.source_snapshot',lambda *a:dict(source))
    operation=GenerationRunSpec(engine=ToyEngine(),dataset=Dataset.from_examples([Example(key='a',prompt='hello')]),generation=GenerationSpec(enabled=True,max_tokens=1))
    runner=ModalRunner(ModalResources(gpu='A100-80GB',memory_mb=4096,timeout_seconds=120),ModalVolumeStore(name='fixture',root=str(tmp_path)))
    runtime={'family':'modal','lock_digest':'d'*64,'python':'fixture','platform':'linux',
        'installed_package_digest':'e'*64,'image_identity':'im-observed','image_reason':'observed'}
    provenance=resolve_provenance(operation,runner.identity(),runtime=runtime,
        numerics=resolve_provenance(operation).to_dict()["numerics"])
    assert provenance.identity_complete
    record=AdmissionRecord((AdmissionCheck('observed','readiness','passed','fixture','Observed'),),provenance,workload_envelope(operation,runner.identity()).to_dict())
    artifact=LocalRunner(artifacts=LocalArtifactStore(tmp_path/'local'),provenance_resolver=lambda s:provenance).run(operation)
    manifest=artifact.manifest();meta=copy.deepcopy(manifest.metadata);meta['admission_v1']=record.to_dict();manifest=replace(manifest,metadata=meta)
    catalog=SimpleNamespace(list_workflow_runs=lambda **kwargs:[SimpleNamespace(run_id='old')],
        list_workflow_steps=lambda run_id:[SimpleNamespace(status='completed',artifact_id=manifest.artifact_id)],load_artifact=lambda ident:manifest)
    runner.catalog=catalog
    calls=[]
    readiness=replace(record,scope='cpu_preflight')
    monkeypatch.setattr(ModalRunner,'readiness',lambda self,spec:calls.append(spec) or readiness)
    return runner,operation,record,readiness,calls,catalog,source


def test_fresh_process_restores_only_worker_numerics_after_target_refresh(case,monkeypatch):
    runner,spec,record,ready,calls,_,_=case
    body=ready.provenance.to_dict();body.pop('scientific_identity');body.pop('schema_version')
    body['numerics']={**body['numerics'],'device':'cpu-preflight-unobserved-gpu'}
    body['unresolved']=['numerics.device']
    monkeypatch.setattr(ModalRunner,'readiness',lambda self,spec:calls.append(spec) or replace(ready,provenance=ProvenanceV1(**body)))
    runner.prepare_execution_identity(spec)
    assert len(calls)==1
    assert resolve_for_runner(spec,runner).scientific_identity==record.provenance.scientific_identity
    assert runner.admission_records[spec.semantic_hash()].provenance.to_dict()['numerics']==record.provenance.to_dict()['numerics']
    # A later workflow preparation refreshes again, never treats an old CPU
    # check as perpetual permission to reuse a mutable target.
    runner.prepare_execution_identity(spec)
    assert len(calls)==2


@pytest.mark.parametrize('section', ['model_components','runtime','data','source'])
def test_target_changes_decline_reuse(case,monkeypatch,section):
    runner,spec,record,ready,calls,_,_=case
    body=ready.provenance.to_dict();body.pop('scientific_identity');body.pop('schema_version')
    if section=='model_components':body[section][0]['file_set_digest']='f'*64
    elif section=='runtime':body[section]['installed_package_digest']='f'*64
    elif section=='source':body[section]['mounted_source_digest']='f'*64
    else:body[section]['content_digest']='f'*64
    changed=replace(ready,provenance=ProvenanceV1(**body))
    monkeypatch.setattr(ModalRunner,'readiness',lambda *args:changed)
    runner.prepare_execution_identity(spec)
    assert spec.semantic_hash() not in runner.admission_records
    assert not resolve_for_runner(spec,runner).identity_complete


def test_changed_request_source_or_envelope_rejected_before_online_readiness(case):
    runner,spec,_,_,calls,_,source=case
    source['mounted_source_digest']='f'*64
    runner.prepare_execution_identity(spec)
    assert not calls and not runner.admission_records
    source['mounted_source_digest']='c'*64
    runner.resources=replace(runner.resources,gpu='H100')
    runner.prepare_execution_identity(spec)
    assert not calls and not runner.admission_records


@pytest.mark.parametrize('failure',['legacy','cpu_only','wrong_manifest_binding','bad_completion'])
def test_invalid_persisted_evidence_is_not_adopted(case,failure):
    runner,spec,record,_,calls,catalog,_=case
    manifest=catalog.load_artifact('x');meta=copy.deepcopy(manifest.metadata)
    if failure=='legacy':meta.pop('admission_v1')
    elif failure=='cpu_only':meta['admission_v1']['scope']='cpu_preflight'
    elif failure=='wrong_manifest_binding':
        body=record.provenance.to_dict();body.pop('scientific_identity');body.pop('schema_version');body['seeds']['changed']=1
        meta['admission_v1']['provenance_v1']=ProvenanceV1(**body).to_dict()
    else:meta['completion_v1']['integrity_hash']='0'*64
    catalog.load_artifact=lambda ident:replace(manifest,metadata=meta)
    runner.prepare_execution_identity(spec)
    assert not calls and not runner.admission_records


def test_remote_observer_assets_need_no_local_observation_after_fresh_target(case,monkeypatch):
    runner,spec,record,ready,calls,_,_=case
    from pipelines_v2.core import provenance_observation
    monkeypatch.setitem(provenance_observation._OBSERVERS,spec.kind,'fixture')
    monkeypatch.setattr(provenance_observation,'observe_operation',lambda *a:(_ for _ in ()).throw(AssertionError('Remote assets must not be probed on caller')))
    body=ready.provenance.to_dict();body.pop('scientific_identity');body.pop('schema_version')
    body['numerics']={**body['numerics'],'device':'cpu-not-gpu'};body['unresolved']=['numerics.cuda_device_unobserved']
    monkeypatch.setattr(ModalRunner,'readiness',lambda self,spec:calls.append(spec) or replace(ready,provenance=ProvenanceV1(**body)))
    runner.prepare_execution_identity(spec)
    assert len(calls)==1
    assert resolve_for_runner(spec,runner).scientific_identity==record.provenance.scientific_identity
    assert runner._admission_context(spec)=={}  # Catalog facts govern reuse, not fresh execution.


@pytest.mark.parametrize('changed',['data','methods','tool_runtime','incomplete_assets'])
def test_remote_observer_target_change_never_adopts_prior_gpu_facts(case,monkeypatch,changed):
    runner,spec,record,ready,_,_,_=case
    from pipelines_v2.core import provenance_observation
    monkeypatch.setitem(provenance_observation._OBSERVERS,spec.kind,'fixture')
    body=ready.provenance.to_dict();body.pop('scientific_identity');body.pop('schema_version')
    if changed=='incomplete_assets':body['unresolved']=['operation.observation_unavailable']
    elif changed=='tool_runtime':body['runtime']['operation_runtime']={'changed':'actual interpreter package inventory'}
    elif changed=='methods':body['methods']['evaluator']={'changed':'configured evaluator'}
    else:body['data']['content_digest']='f'*64
    monkeypatch.setattr(ModalRunner,'readiness',lambda *a:replace(ready,provenance=ProvenanceV1(**body)))
    runner.prepare_execution_identity(spec)
    assert not runner.admission_records and not runner.admission_readiness_records


def test_catalog_admission_is_reuse_only_but_explicit_admission_stays_strict(case):
    runner,spec,record,*_=case
    runner.prepare_execution_identity(spec)
    assert runner._admission_context(spec)=={}
    assert resolve_for_runner(spec,runner).scientific_identity==record.provenance.scientific_identity
    runner._reuse_only_admissions.clear()
    assert runner._admission_context(spec)['admission_v1']==record.to_dict()
