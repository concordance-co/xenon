"""Observed operation facts must identify bytes without weakening parent/reuse gates."""
from dataclasses import dataclass, replace
from typing import Any, ClassVar
from pathlib import Path
import copy
import json
import shutil

import pytest

from pipelines_v2.core.types import OperationSpec
from pipelines_v2.core.provenance import resolve_provenance, compare_provenance, content_hash
from pipelines_v2.core.provenance_observation import OperationFacts, file_set_identity
from pipelines_v2.api import (Dataset, Example, LabelMapSpec, LocalRunner, LocalArtifactStore,
    ReportSpec, TransformSpec, TransformBuilder)
from pipelines_v2.storage.artifacts import OperationArtifact

SOURCE = {'repository_id': 'fixture', 'head': 'a'*40, 'dirty_digest': 'b'*64, 'mounted_source_digest': 'c'*64}
RUNTIME = {'family':'local', 'lock_digest':'d'*64, 'python':'fixture-python', 'platform':'fixture',
           'installed_package_digest':'e'*64, 'image_identity':None, 'image_reason':'local process'}
NUMERICS = {'dtype':'float32', 'attention_implementation':'not_applicable', 'device':'cpu',
            'determinism':'explicit-seed', 'compile':False, 'matmul':'observed-fixture'}


@dataclass(frozen=True)
class FileOperation(OperationSpec):
    reference_path: str = ''
    scale: float = 1.0
    parent: Any = None
    kind: ClassVar[str] = 'observed_fixture'

    def runtime_spec(self):
        return None


def resolve(spec, **kwargs):
    return resolve_provenance(spec, source=SOURCE, runtime=RUNTIME, **kwargs)


@pytest.fixture
def observer(monkeypatch):
    from pipelines_v2.core import provenance_observation
    state = {'numerics':dict(NUMERICS), 'runtime':dict(RUNTIME), 'seed':3}
    def observe(spec, runner):
        if not isinstance(spec, FileOperation):
            return None
        return OperationFacts(field_identities={'reference_path':file_set_identity(spec.reference_path, ['weights.bin','config.json'])},
            numerics=state['numerics'], runtime=state['runtime'], seeds={'fit':state['seed']})
    monkeypatch.setattr(provenance_observation, 'observe_operation', observe)
    return state


@pytest.fixture
def files(tmp_path):
    root=tmp_path/'first';root.mkdir();(root/'weights.bin').write_bytes(b'actual fixed model bytes');(root/'config.json').write_text('{"width":3}')
    return root


def test_observed_files_relocate_but_actual_content_and_all_settings_invalidate(files, tmp_path, observer):
    one=FileOperation(reference_path=str(files));first=resolve(one);assert first.identity_complete
    copied=tmp_path/'moved';shutil.copytree(files,copied)
    assert compare_provenance(resolve(replace(one,reference_path=str(copied))),first).reusable
    assert str(files) not in json.dumps(first.to_dict())
    (copied/'weights.bin').write_bytes(b'changed actual bytes')
    assert not compare_provenance(resolve(replace(one,reference_path=str(copied))),first).reusable
    assert not compare_provenance(resolve(replace(one,scale=2.)),first).reusable
    for section,key,value in [('numerics','dtype','float64'),('runtime','installed_package_digest','f'*64)]:
        saved=observer[section][key];observer[section][key]=value
        assert not compare_provenance(resolve(one),first).reusable
        observer[section][key]=saved
    observer['seed']=4
    assert not compare_provenance(resolve(one),first).reusable


def test_explicit_file_list_never_uses_directory_location_or_unlisted_secret(files):
    initial=file_set_identity(files,['weights.bin','config.json']);(files/'.env').write_text('never-hashed-secret')
    assert file_set_identity(files,['config.json','weights.bin'])==initial
    with pytest.raises(ValueError):file_set_identity(files,['.env'])
    with pytest.raises(ValueError):file_set_identity(files,['../elsewhere'])
    with pytest.raises(ValueError):file_set_identity(files,[])
    with pytest.raises(ValueError):file_set_identity(files,['weights.bin'],expected={'weights.bin':'a'*64})


def artifact(tmp_path):
    runner=LocalRunner(artifacts=LocalArtifactStore(tmp_path/'artifacts'),provenance_resolver=resolve)
    dataset=Dataset.from_examples([Example(key='x',prompt='small fixture',labels={'a':'yes'})])
    return runner.run(LabelMapSpec(source=dataset.labels('a'),mapping={'yes':'ok'},output_name='mapped'))


def test_new_observer_cannot_hide_artifact_or_dataset_identity(files, tmp_path, monkeypatch):
    from pipelines_v2.core import provenance_observation
    parent=artifact(tmp_path)
    monkeypatch.setattr(provenance_observation,'observe_operation',lambda *a:OperationFacts(field_identities={'parent':{'content_digest':'a'*64}},numerics=NUMERICS))
    with pytest.raises(ValueError,match='cannot replace'):
        resolve(FileOperation(reference_path=str(files),parent=parent))


def test_changed_parent_payload_same_science_identity_invalidates(files, tmp_path, observer):
    parent=artifact(tmp_path);spec=FileOperation(reference_path=str(files),parent=parent);first=resolve(spec)
    assert first.identity_complete
    manifest=parent.manifest();metadata=copy.deepcopy(manifest.metadata);completion=metadata['completion_v1']
    completion['payloads'][0]['sha256']='1'*64
    completion['integrity_hash']=content_hash({k:v for k,v in completion.items() if k!='integrity_hash'})
    different=OperationArtifact(replace(manifest,metadata=metadata),parent.store)
    assert different.manifest().provenance().scientific_identity==manifest.provenance().scientific_identity
    assert not compare_provenance(resolve(replace(spec,parent=different)),first).reusable
    legacy=OperationArtifact(replace(manifest,metadata={k:v for k,v in metadata.items() if k!='completion_v1'}),parent.store)
    assert not resolve(replace(spec,parent=legacy)).identity_complete


def test_builtin_report_complete_only_with_identified_inputs(tmp_path):
    parent=artifact(tmp_path)
    record=resolve(ReportSpec(inputs=(parent,)))
    assert record.identity_complete, record.to_dict()['unresolved']
    assert not resolve(ReportSpec(inputs=(parent,),template='unknown-plugin')).identity_complete
    meta=copy.deepcopy(parent.manifest().metadata);meta.pop('provenance_v1')
    legacy=OperationArtifact(replace(parent.manifest(),metadata=meta),parent.store)
    assert not resolve(ReportSpec(inputs=(legacy,))).identity_complete


def test_allowlisted_summary_does_not_make_arbitrary_transform_complete(tmp_path):
    parent=artifact(tmp_path)
    known=TransformSpec(builder=TransformBuilder('pipelines_v2.operations.attribution.report:summarize_attribution',('pipelines_v2',)),inputs={'attribution':parent})
    assert resolve(known).identity_complete
    unknown=replace(known,builder=TransformBuilder('custom_module:transform',('pipelines_v2',)))
    assert not resolve(unknown).identity_complete
    side_effect=replace(known,builder=TransformBuilder('pipelines_v2.operations.graphs.report:graph_report_transform',('pipelines_v2',)))
    assert not resolve(side_effect).identity_complete


def test_runner_numerical_environment_invalidates(files, observer):
    spec=FileOperation(reference_path=str(files))
    one=resolve(spec,runner_identity={'kind':'local','resources':{'env':{'OMP_NUM_THREADS':'1','HF_TOKEN':'private-a'}}})
    two=resolve(spec,runner_identity={'kind':'local','resources':{'env':{'OMP_NUM_THREADS':'2','HF_TOKEN':'private-b'}}})
    assert one.identity_complete and two.identity_complete
    assert not compare_provenance(two,one).reusable
    assert 'private-' not in json.dumps(one.to_dict())


def test_nested_tool_runtime_shares_only_observed_enclosing_modal_image(files,observer):
    observer['runtime'].update(family='modal',image_identity=None,image_reason='shared_with_control_runtime')
    spec=FileOperation(reference_path=str(files))
    assert not resolve(spec).identity_complete
    outer={**RUNTIME,'family':'modal','image_identity':'im-observed','image_reason':'observed'}
    record=resolve_provenance(spec,source=SOURCE,runtime=outer)
    assert record.identity_complete
    assert record.to_dict()['runtime']['operation_runtime']['image_identity']=='im-observed'
    assert record.to_dict()['runtime']['operation_runtime']['installed_package_digest']==observer['runtime']['installed_package_digest']


def test_sae_import_observer_uses_full_component_schema_and_actual_bytes(tmp_path,monkeypatch):
    import hashlib, struct
    from pipelines_v2.representation.sae import dictionary
    from pipelines_v2.operations.representation.sae import SAEDictionaryImportSpec
    root=tmp_path/'assets';root.mkdir()
    (root/'config.json').write_text('{"model_type":"gpt2"}')
    (root/'tokenizer.json').write_text('{"fixture":"tokenizer"}')
    header=json.dumps({'weight':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
    (root/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+struct.pack('<f',1))
    (root/'cfg.json').write_text('{"fixture":"dictionary"}')
    (root/'sae_weights.safetensors').write_bytes(struct.pack('<Q',len(header))+header+struct.pack('<f',2))
    monkeypatch.setattr(dictionary,'verify_original_model',lambda path:'a'*64)
    monkeypatch.setattr(dictionary,'CONFIG_SHA256',hashlib.sha256((root/'cfg.json').read_bytes()).hexdigest())
    monkeypatch.setattr(dictionary,'WEIGHTS_SHA256',hashlib.sha256((root/'sae_weights.safetensors').read_bytes()).hexdigest())
    spec=SAEDictionaryImportSpec(cache_root=str(root),model_path=str(root))
    record=resolve(spec)
    assert record.identity_complete,record.to_dict()['unresolved']
    assert [c['role'] for c in record.to_dict()['model_components']]==['model','tokenizer','sae_dictionary']
    copied=tmp_path/'moved';shutil.copytree(root,copied)
    assert resolve(replace(spec,cache_root=str(copied),model_path=str(copied))).scientific_identity==record.scientific_identity
    (copied/'tokenizer.json').write_text('{"fixture":"changed tokenizer"}')
    assert resolve(replace(spec,cache_root=str(copied),model_path=str(copied))).scientific_identity!=record.scientific_identity
    (root/'sae_weights.safetensors').write_bytes(b'wrong bytes')
    assert not resolve(spec).identity_complete
