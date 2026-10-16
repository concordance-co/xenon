"""Explicit extensions registered after runner imports traverse the real dispatch paths."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import pytest

# Import runners BEFORE registration: a cached tuple would fail these tests.
from pipelines_v2.api import (
    CaptureSpec, Dataset, EngineCapability, Example, LocalRunner, ModalResources,
    ModalRunner, ModalVolumeStore, OperationSpec, PythonRuntimeSpec, TokenSelector,
    ToyEngine, register_capture_site, register_engine, register_feature_codec,
    register_intervention_site, register_operation,
)
from pipelines_v2.core.types import SpecValidationError, to_primitive
from pipelines_v2.engine.base import EngineCaptureResult
from pipelines_v2.operations.capture.specs import capture_site_from_dict
from pipelines_v2.operations.common._shared import spec_value_from_dict
from pipelines_v2.operations.execution import OperationExecutionResult
from pipelines_v2.operations.interventions.sites import InterventionSite, ResidualInterventionSite
from pipelines_v2.operations.registry import operation_spec_from_dict
from pipelines_v2.runtime.remote_executor import execute_remote, execute_remote_many
from pipelines_v2.storage import LocalArtifactStore
from pipelines_v2.storage.artifacts import CaptureArtifact, OperationArtifact, artifact_from_dict
from pipelines_v2.storage.features import (
    decode_feature_payload, encode_tensor_feature_payload, feature_storage_format,
    load_feature_payload,
)


@dataclass(frozen=True)
class ExtensionSite:
    name: str = 'test_units'
    tokens: TokenSelector = field(default_factory=TokenSelector.last)
    schema_version: int = 1
    quantity: str = 'test_only'
    kind: ClassVar[str] = 'test_extension_site'

    def required_capabilities(self):
        return {EngineCapability.RESIDUAL_CAPTURE}

    @classmethod
    def from_dict(cls, payload):
        return cls(name=payload['name'], tokens=TokenSelector.from_dict(payload['tokens']),
                   schema_version=payload['schema_version'], quantity=payload['quantity'])


@dataclass(frozen=True)
class ExtensionWriteSite(InterventionSite):
    kind: ClassVar[str] = 'test_extension_write'
    schema_version: int = 1

    @classmethod
    def from_dict(cls, payload):
        return cls(site=payload['site'], layers=tuple(payload['layers']))


@dataclass(frozen=True)
class ExtensionEngine(ToyEngine):
    def identity(self):
        return {**super().identity(), 'kind': 'test_extension_engine'}

    def capture(self, spec):
        return EngineCaptureResult(features={site.name: {
            'kind': 'test_extension_dense',
            'axes': ['example', 'unit'],
            'values': np.asarray([[len(ex.prompt), 2] for ex in spec.dataset.examples], dtype=np.float32),
        } for site in spec.sites})


@dataclass(frozen=True)
class ExtensionOperation(OperationSpec):
    feature: Any = None
    kind: ClassVar[str] = 'test_extension_operation'

    def runtime_spec(self):
        return PythonRuntimeSpec()

    @classmethod
    def from_dict(cls, payload):
        return cls(feature=spec_value_from_dict(payload['feature']))


def execute_extension(spec):
    values = spec.feature.load()['values']
    return OperationExecutionResult(payload={'total': float(np.sum(values))})


@pytest.fixture
def extensions(monkeypatch):
    # Restore registries after each case without adding a public unregister API.
    import pipelines_v2.engine.registry as engines
    import pipelines_v2.operations.registry as operations
    import pipelines_v2.operations.capture.specs as sites
    import pipelines_v2.operations.interventions.sites as writes
    import pipelines_v2.storage.features as features
    import pipelines_v2.runtime.recovery as recovery
    from pipelines_v2.storage.finalization import local_writer_lock
    monkeypatch.setattr(ModalVolumeStore, "writer_lock", local_writer_lock)
    monkeypatch.setattr(ModalVolumeStore, "commit_visibility", lambda self: None)
    for module, name in (
        (engines, '_ENGINE_LOADERS'), (operations, '_OPERATION_LOADERS'),
        (operations, '_OPERATION_REGISTRATIONS'), (sites, '_CAPTURE_SITE_LOADERS'),
        (sites, '_CAPTURE_SITE_EMPTY'), (writes, '_INTERVENTION_SITE_LOADERS'), (features, '_FEATURE_CODECS'),
        (features, '_FEATURE_FORMATS'), (recovery, '_CAPTURE_VALIDATORS'),
    ):
        monkeypatch.setattr(module, name, dict(getattr(module, name)))
    def validate(spec, site, payload, metadata):
        assert payload['values'].shape == (len(spec.dataset.examples), 2)
        return [{'example_key': row.key, 'site': site.name, 'units': [0, 1]} for row in spec.dataset.examples]
    recovery.register_capture_coverage_validator('test_extension_dense', validate)
    register_engine('test_extension_engine' , ExtensionEngine.from_dict)
    register_capture_site(ExtensionSite.kind, ExtensionSite.from_dict, empty_payload=lambda site: {
        'kind':'test_extension_dense', 'axes':['example','unit'], 'values':np.empty((0, 2), dtype=np.float32),
    })
    register_intervention_site(ExtensionWriteSite.kind, ExtensionWriteSite.from_dict)
    register_feature_codec('test_extension_dense', storage_format='test_dense_safetensors_v1',
                           encode=encode_tensor_feature_payload, decode=decode_feature_payload,
                           merge=lambda left,right:{**left,'values':np.concatenate([left['values'],right['values']],axis=0)})
    register_operation(ExtensionOperation.kind, loader=ExtensionOperation.from_dict,
                       executor=execute_extension, artifact_kind='test_measurement')


def test_late_extensions_capture_and_artifact_end_to_end(extensions, tmp_path, monkeypatch):
    local = LocalRunner(artifacts=LocalArtifactStore(tmp_path / 'local'))
    modal = ModalRunner(ModalResources(enable_workflow_batching=True),
                        ModalVolumeStore('test', str(tmp_path / 'remote')))
    monkeypatch.setattr('pipelines_v2.runtime.modal.run_on_modal', execute_remote)
    # run_many forwards workspace_root, which transport accepts but executor does not.
    def fake_many(**kwargs):
        kwargs.pop('workspace_root', None)
        return execute_remote_many(**kwargs)
    monkeypatch.setattr('pipelines_v2.runtime.modal.run_many_on_modal', fake_many)
    spec = CaptureSpec(engine=ExtensionEngine(),
                       dataset=Dataset.from_examples([Example(key='a', prompt='hello'), Example(key='b', prompt='hi')]),
                       sites=(ExtensionSite(),))
    restored = operation_spec_from_dict(json.loads(json.dumps(spec.to_dict())))
    assert isinstance(restored.engine, ExtensionEngine)
    assert isinstance(restored.sites[0], ExtensionSite)
    assert restored.to_dict() == spec.to_dict()
    assert local.plan(restored).valid and modal.plan(restored).valid
    for runner in (local, modal):
        capture = runner.run(restored)
        assert isinstance(capture, CaptureArtifact)
        assert capture.manifest().storage_refs['features']['test_units']['format'] == 'test_dense_safetensors_v1'
        np.testing.assert_array_equal(capture.feature('test_units').load()['values'], [[5, 2], [2, 2]])
        op = ExtensionOperation(feature=capture.feature('test_units'))
        op = operation_spec_from_dict(json.loads(json.dumps(op.to_dict())))
        assert runner.plan(op).estimated_artifacts() == ('test_measurement',)
        artifact = runner.run(op)
        assert isinstance(artifact, OperationArtifact)
        assert artifact.manifest().artifact_kind == 'test_measurement'
        assert artifact.result() == {'total': 11.0}
        assert artifact_from_dict(to_primitive(artifact)).result() == {'total': 11.0}
        if runner is modal:
            assert modal.workflow_batch_key(op) is not None
            assert [a.result() for a in modal.run_many([op, op])] == [{'total': 11.0}] * 2


def test_write_site_roundtrip_and_future_schema_rejected(extensions):
    site = ExtensionWriteSite(site='test', layers=(1,))
    assert InterventionSite.from_dict(to_primitive(site)) == site
    assert isinstance(InterventionSite.from_dict({'site':'resid_post','layers':[0]}), ResidualInterventionSite)
    for loader, payload in ((capture_site_from_dict, to_primitive(ExtensionSite())),
                            (InterventionSite.from_dict, to_primitive(site))):
        with pytest.raises(SpecValidationError, match='schema_version'):
            loader({**payload, 'schema_version': 2})
        with pytest.raises(SpecValidationError, match='kind'):
            loader({**payload, 'kind': 'unregistered.module:callable'})


def test_duplicate_registration_fails_without_overwriting(extensions):
    with pytest.raises(ValueError, match='already registered'):
        register_engine('test_extension_engine', lambda value: None)
    with pytest.raises(ValueError, match='already registered'):
        register_capture_site(ExtensionSite.kind, lambda value: None)
    with pytest.raises(ValueError, match='already registered'):
        register_intervention_site(ExtensionWriteSite.kind, lambda value: None)
    with pytest.raises(ValueError, match='already registered'):
        register_operation(ExtensionOperation.kind, loader=lambda value: None, executor=lambda value: None)
    with pytest.raises(ValueError, match='already registered'):
        register_feature_codec('other', storage_format='test_dense_safetensors_v1',
                               encode=encode_tensor_feature_payload, decode=decode_feature_payload)
    # A failed format registration must not reserve its feature name.
    register_feature_codec('other', storage_format='other_v1',
                           encode=encode_tensor_feature_payload, decode=decode_feature_payload)


def test_unknown_operation_fails_planning_and_payload_cannot_import(tmp_path):
    @dataclass(frozen=True)
    class Unknown(OperationSpec):
        kind: ClassVar[str] = 'unregistered.module:callable'
    local = LocalRunner(artifacts=LocalArtifactStore(tmp_path))
    assert not local.plan(Unknown()).valid
    with pytest.raises(SpecValidationError, match='Unregistered'):
        local.run(Unknown())
    with pytest.raises(ValueError, match='Unknown operation'):
        operation_spec_from_dict(Unknown().to_dict())


def test_unknown_dense_feature_does_not_fall_back_to_json(tmp_path):
    for payload in ({'kind':'unknown', 'values':np.ones((2, 3))},
                    {'kind':'unknown', 'storage':{'format':'safetensors'}, 'values':[[1, 2]]}):
        with pytest.raises(ValueError, match='Unregistered dense'):
            feature_storage_format(payload)
    assert feature_storage_format({'kind':'annotation','label':'small JSON'}) == 'json'
    with pytest.raises(ValueError, match='Unsupported feature ref format'):
        load_feature_payload(LocalArtifactStore(tmp_path), {'format':'unregistered_dense_v1'})


def test_target_logprobs_is_distinct_from_broad_logprobs():
    assert EngineCapability.TARGET_LOGPROBS not in ToyEngine().capabilities()
    assert EngineCapability.LOGPROBS in ToyEngine().capabilities()


def test_core_cli_plan_imports_do_not_import_optional_runtimes():
    script = r'''
import importlib.abc, sys
class RejectOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','diffusers','matplotlib','sklearn','vllm'}:
            raise AssertionError('Unexpected optional import: ' + fullname)
sys.meta_path.insert(0, RejectOptional())
import pipelines_v2.api
import pipelines_v2.cli
from pipelines_v2.api import CaptureSpec, Dataset, Example, GenerationSpec, LocalRunner, ToyEngine
spec = CaptureSpec(engine=ToyEngine(), dataset=Dataset.from_examples([Example(key='a',prompt='hi')]), generation=GenerationSpec(enabled=True,max_tokens=1))
assert LocalRunner().plan(spec).valid
import json
from pathlib import Path
from pipelines_v2.operations.registry import operation_spec_from_dict
fixtures = json.loads(Path('tests/pipelines_v2/fixtures/legacy_serialization_v1.json').read_text())
for fixture in fixtures['operations']:
    operation_spec_from_dict(fixture['input'])
'''
    result = subprocess.run([sys.executable, '-c', script], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('version', [True, 1.0, '1', 0, 2, None])
def test_new_site_schema_is_an_exact_supported_integer(extensions, version):
    for loader, payload in ((capture_site_from_dict, to_primitive(ExtensionSite())),
                            (InterventionSite.from_dict, to_primitive(ExtensionWriteSite(site='x', layers=(0,))))):
        with pytest.raises(SpecValidationError, match='schema_version'):
            loader({**payload, 'schema_version': version})


def test_registered_loaders_cannot_return_another_kind(extensions):
    from pipelines_v2.engine.registry import engine_from_dict
    register_engine('wrong_engine', lambda payload: ToyEngine())
    register_capture_site('wrong_site', lambda payload: ExtensionSite())
    register_intervention_site('wrong_write', lambda payload: ResidualInterventionSite(site='x', layers=(0,)))
    register_operation('wrong_operation', loader=lambda payload: ExtensionOperation(), executor=execute_extension)
    for loader, payload in ((engine_from_dict, {'kind':'wrong_engine'}),
                            (capture_site_from_dict, {'kind':'wrong_site'}),
                            (InterventionSite.from_dict, {'kind':'wrong_write'}),
                            (operation_spec_from_dict, {'kind':'wrong_operation'})):
        with pytest.raises(ValueError, match='incompatible'):
            loader(payload)


def test_feature_encoder_prefix_and_kind_contract(extensions, tmp_path):
    from pipelines_v2.storage.features import write_capture_features
    def bad_prefix(payload, *, tensor_key_prefix):
        return payload, {'shared': np.ones((1,))}
    register_feature_codec('bad_prefix', storage_format='bad_prefix_v1',
                           encode=bad_prefix, decode=decode_feature_payload)
    store = LocalArtifactStore(tmp_path)
    store.make_artifact_dir('test')
    with pytest.raises(ValueError, match='tensor_key_prefix'):
        write_capture_features(store, 'test', {'a':{'kind':'bad_prefix'}})
    def bad_kind(payload, *, tensor_key_prefix):
        return {'kind':'wrong'}, {tensor_key_prefix+'0':np.ones((1,))}
    register_feature_codec('bad_kind', storage_format='bad_kind_v1',
                           encode=bad_kind, decode=decode_feature_payload)
    with pytest.raises(ValueError, match='incompatible kind'):
        write_capture_features(store, 'test', {'a':{'kind':'bad_kind'}})


def test_feature_decoder_kind_contract(extensions, tmp_path):
    from pipelines_v2.storage.features import write_capture_features
    register_feature_codec('bad_decode', storage_format='bad_decode_v1',
                           encode=encode_tensor_feature_payload, decode=lambda metadata,tensors:{'kind':'wrong'})
    store = LocalArtifactStore(tmp_path)
    store.make_artifact_dir('test')
    refs = write_capture_features(store, 'test', {'a':{'kind':'bad_decode','values':np.ones((1,))}})
    with pytest.raises(ValueError, match='decoder returned an incompatible'):
        load_feature_payload(store, refs['a'])


def test_additive_default_fields_preserve_all_serialization_paths():
    from pipelines_v2.core.types import to_semantic_primitive
    @dataclass(frozen=True)
    class OldNested:
        value: int = 3
    @dataclass(frozen=True)
    class NewNested:
        value: int = 3
        optional: tuple[int, ...] = field(default_factory=tuple, metadata={'omit_if_default':True})
    @dataclass(frozen=True)
    class OldSpec(OperationSpec):
        value: Any = field(default_factory=OldNested)
    @dataclass(frozen=True)
    class NewSpec(OperationSpec):
        value: Any = field(default_factory=NewNested)
        optional: str | None = field(default=None, metadata={'omit_if_default':True})
    old, new = OldSpec(), NewSpec()
    assert to_primitive(OldNested()) == to_primitive(NewNested())
    assert to_semantic_primitive(OldNested()) == to_semantic_primitive(NewNested())
    assert old.to_dict() == new.to_dict()
    assert old.spec_hash() == new.spec_hash()
    assert old.semantic_dict() == new.semantic_dict()
    assert old.semantic_hash() == new.semantic_hash()
    assert new.spec_hash() != NewSpec(optional='changed').spec_hash()
    assert new.semantic_hash() != NewSpec(value=NewNested(optional=(1,))).semantic_hash()


def test_extension_shard_merge_including_empty_partition(extensions, tmp_path):
    from pipelines_v2.runtime.remote_executor import merge_remote_shards
    store = LocalArtifactStore(tmp_path)
    spec = CaptureSpec(engine=ExtensionEngine(), dataset=Dataset.from_examples([
        Example(key='a',prompt='hello'),Example(key='b',prompt='hi')]), sites=(ExtensionSite(),))
    context = {'run_id':'test_merge','workflow_step_key':'capture'}
    shards = [execute_remote(runner_config={'kind':'test'},store_config=store.identity(),spec_payload=spec.to_dict(),
                             workflow_context={**context,'execution_shard':{'index':index,'count':3}})
              for index in range(3)]
    assert any(shard['metadata'].get('empty_shard') for shard in shards)
    merged = merge_remote_shards(runner_config={'kind':'test'},store_config=store.identity(),
                                spec_payload=spec.to_dict(),shard_manifests=shards,workflow_context=context)
    payload = load_feature_payload(store,merged['storage_refs']['features']['test_units'])
    assert sorted(payload['values'].tolist()) == [[2,2],[5,2]]


def test_missing_merge_and_empty_semantics_fail_explicitly(extensions):
    from pipelines_v2.operations.capture.specs import empty_capture_site_payload
    from pipelines_v2.storage.features import merge_feature_payloads
    register_feature_codec('unmergeable',storage_format='unmergeable_v1',
                           encode=encode_tensor_feature_payload,decode=decode_feature_payload)
    with pytest.raises(ValueError,match='no registered shard merge'):
        merge_feature_payloads({'kind':'unmergeable','values':np.ones((1,))},
                               {'kind':'unmergeable','values':np.ones((1,))})
    with pytest.raises(SpecValidationError,match='no empty-shard'):
        empty_capture_site_payload(ExtensionWriteSite(site='x',layers=(0,)))


def test_metadata_only_registered_codec_still_decodes(extensions, tmp_path):
    from pipelines_v2.storage.features import write_capture_features
    def encode(payload, *, tensor_key_prefix):
        return {'kind':'metadata_only','encoded':payload['value']+1},{}
    def decode(metadata,tensors):
        assert tensors == {}
        return {'kind':'metadata_only','value':metadata['encoded']-1}
    register_feature_codec('metadata_only',storage_format='metadata_only_v1',encode=encode,decode=decode)
    store=LocalArtifactStore(tmp_path)
    store.make_artifact_dir('test')
    payload={'kind':'metadata_only','value':42}
    ref=write_capture_features(store,'test',{'a':payload,'b':{'kind':'test_extension_dense','values':np.ones((1,2))}})['a']
    assert ref['format']=='metadata_only_v1'
    assert ref['tensor_path'] is None
    assert load_feature_payload(store,ref)==payload
