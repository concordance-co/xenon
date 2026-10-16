from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from pipelines_v2.experiments import ExperimentPack
from pipelines_v2.core.types import SpecValidationError

ROOT = Path(__file__).resolve().parents[3]
TOY = ROOT / 'workspaces' / 'toy_pack'


def toy_module(path=TOY):
    spec = importlib.util.spec_from_file_location('pack_fixture', path / 'workflow.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pack_payload():
    return json.loads((TOY / 'pack.json').read_text())


def test_pack_roundtrip_and_workflow_binding(monkeypatch):
    monkeypatch.delenv('XENON_EXPERIMENT_PRESET', raising=False)
    payload = pack_payload()
    pack = ExperimentPack.from_dict(payload)
    payload['study_id'] = 'mutated'
    assert pack.to_dict()['study_id'] == 'toy-pack'
    assert ExperimentPack.from_dict(pack.to_dict()).identity == pack.identity
    assert pack.validate_workflow(toy_module().unchecked_workflow(), phase_dir=TOY) == 'smoke'
    monkeypatch.setenv('XENON_EXPERIMENT_PRESET', 'scale')
    with pytest.raises(SpecValidationError, match='disabled'):
        toy_module().build_workflow()
    monkeypatch.setenv('XENON_EXPERIMENT_PRESET', 'typo')
    with pytest.raises(SpecValidationError, match='Unknown experiment preset'):
        toy_module().build_workflow()


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(schema_version=2),
    lambda p: p.update(schema_version=True),
    lambda p: p.update(workflow='../workflow.py'),
    lambda p: p['split_policy'].update(held_out=['missing']),
    lambda p: p['metrics']['mapped_count'].update(direction='maybe'),
    lambda p: p['metrics']['mapped_count'].update(evaluator_digest='latest'),
    lambda p: p['controls'].update(shuffle={'purpose':'test','required':'yes'}),
    lambda p: p['presets']['smoke']['limits'].update(max_examples=-1),
    lambda p: p['presets']['smoke']['limits'].update(max_workers=True),
    lambda p: p['presets']['smoke']['limits'].update(max_gpu_hours=float('inf')),
    lambda p: p['presets']['smoke']['limits'].update(timeout_seconds=None),
    lambda p: p['presets']['reference'].update(paper_settings={}),
    lambda p: p.update(unknown='field'),
])
def test_malformed_metadata_fails_before_build(mutation):
    payload = pack_payload()
    mutation(payload)
    with pytest.raises((SpecValidationError, TypeError)):
        ExperimentPack.from_dict(payload)


@pytest.mark.parametrize('field', ['workflow_hash', 'models', 'data', 'outputs'])
def test_stale_metadata_rejects_built_workflow(field):
    payload = pack_payload()
    if field == 'workflow_hash':
        payload['presets']['smoke']['workflow_hash'] = '0' * 64
    elif field == 'models':
        payload['model_locks']['capture'] = '0' * 64
    elif field == 'data':
        payload['data_locks'][next(iter(payload['data_locks']))] = '0' * 64
    else:
        payload['expected_outputs']['labels'] = 'capture'
    with pytest.raises(SpecValidationError, match='match'):
        ExperimentPack.from_dict(payload).validate_workflow(toy_module().unchecked_workflow(), phase_dir=TOY, preset='smoke')


def test_source_changes_reject(tmp_path):
    shutil.copy(TOY / 'workflow.py', tmp_path / 'workflow.py')
    (tmp_path / 'workflow.py').write_text('changed source')
    with pytest.raises(SpecValidationError, match='Source lock mismatch'):
        ExperimentPack.from_dict(pack_payload()).validate_workflow(toy_module().unchecked_workflow(), phase_dir=tmp_path, preset='smoke')


def test_scale_requires_measured_verifier():
    payload = pack_payload()
    payload['presets']['scale']['enabled'] = True
    workflow = toy_module().unchecked_workflow()
    with pytest.raises(SpecValidationError, match='upstream qualification'):
        ExperimentPack.from_dict(payload).validate_workflow(workflow, phase_dir=TOY, preset='scale')
    payload['qualification_requirements'] = ['exact_envelope']
    payload['presets']['scale']['qualification_refs'] = ['qualification_immutable_id']
    pack = ExperimentPack.from_dict(payload)
    with pytest.raises(SpecValidationError, match='verifier'):
        pack.validate_workflow(workflow, phase_dir=TOY, preset='scale')
    with pytest.raises(SpecValidationError, match='stale'):
        pack.validate_workflow(workflow, phase_dir=TOY, preset='scale', qualification_check=lambda ref, plan: False)
    # Contract fixture only; no claim that this callback measures qualification.
    assert pack.validate_workflow(workflow, phase_dir=TOY, preset='scale', qualification_check=lambda ref, plan: True) == 'scale'


def test_toy_pack_cli_and_fresh_catalog_process(tmp_path):
    phase = tmp_path / 'phase'
    phase.mkdir()
    for name in ('workflow.py', 'pack.json'):
        shutil.copy(TOY / name, phase / name)
    catalog = phase / '.catalog'
    env = {**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONDONTWRITEBYTECODE':'1',
           'XENON_EXPERIMENT_PRESET':'smoke'}
    def cli(*args):
        process = subprocess.run([sys.executable, '-m', 'pipelines_v2.cli', 'workflow', *args,
                                  '--local-catalog-root', str(catalog)], env=env, cwd=ROOT,
                                 capture_output=True, text=True, timeout=90)
        assert process.returncode == 0, process.stderr + process.stdout
        return json.loads(process.stdout)
    planned = cli('plan', '--file', str(phase / 'workflow.py'))
    assert all(not step['errors'] for step in planned['steps'])
    ran = cli('run', '--file', str(phase / 'workflow.py'))
    run_id = ran['run_id']
    listed = cli('runs', '--file', str(phase / 'workflow.py'))
    assert run_id in json.dumps(listed)
    shown = cli('show', '--run-id', run_id)
    assert run_id in json.dumps(shown)
    regenerated = cli('report', run_id, '--catalog-postgres-env', 'XENON_UNUSED_FIXTURE_CATALOG')
    assert regenerated['artifact_kind'] == 'report'
    from pipelines_v2.storage import FileCatalog
    manifest = FileCatalog(catalog).load_artifact(ran['steps']['labels']['artifact_id'])
    assert manifest.provenance().scientific_identity
    from pipelines_v2.storage.artifacts import OperationArtifact
    from pipelines_v2.storage import LocalArtifactStore
    artifact = OperationArtifact(manifest, LocalArtifactStore(phase / 'artifacts'))
    assert artifact.result()['mapped_count'] == 2


def test_presets_bind_different_data_recipes_independently():
    from pipelines_v2.api import Dataset, Example
    from pipelines_v2.experiments.packs import workflow_locks
    module = toy_module()
    smoke = module.unchecked_workflow()
    reference_data = Dataset.from_examples((*module.build_dataset().examples,
        Example(key='toy-c', prompt='gamma', labels={'class':'a'}),
        Example(key='toy-d', prompt='delta', labels={'class':'b'})))
    reference = module.unchecked_workflow(reference_data)
    assert reference.semantic_hash() != smoke.semantic_hash()
    payload = pack_payload()
    models, datasets = workflow_locks(reference)
    profile = payload['presets']['reference']
    profile.update(enabled=True, workflow_hash=reference.semantic_hash())
    profile['limits']['max_examples'] = 4
    for bindings, registry, locks in (('model_bindings','model_locks',models), ('data_bindings','data_locks',datasets)):
        profile[bindings] = {path: 'reference/' + path for path in locks}
        payload[registry].update({'reference/' + path: value for path,value in locks.items()})
    pack = ExperimentPack.from_dict(payload)
    assert pack.validate_workflow(smoke, phase_dir=TOY, preset='smoke') == 'smoke'
    assert pack.validate_workflow(reference, phase_dir=TOY, preset='reference') == 'reference'
    profile['data_bindings'] = payload['presets']['smoke']['data_bindings']
    with pytest.raises(SpecValidationError, match='Data locks'):
        ExperimentPack.from_dict(payload).validate_workflow(reference, phase_dir=TOY, preset='reference')


def test_unregistered_output_has_actionable_pack_error(monkeypatch):
    import pipelines_v2.operations.registry as registry
    monkeypatch.setattr(registry, 'operation_registration', lambda kind: None)
    with pytest.raises(SpecValidationError, match='Unregistered operation kind'):
        ExperimentPack.from_dict(pack_payload()).validate_workflow(
            toy_module().unchecked_workflow(), phase_dir=TOY, preset='smoke')
