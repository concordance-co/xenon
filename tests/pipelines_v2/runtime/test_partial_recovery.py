"""Interrupted real filesystem capture/intervention recovery without GPU work."""
from pathlib import Path
import json
from dataclasses import replace

import pytest

from pipelines_v2.api import (Dataset, Example, ToyEngine, CaptureSpec, ResidualSite, TokenSelector,
    GenerationSpec, PatchedGenerationSpec, ProjectOutPatch, ResidualInterventionSite, ArtifactManifest)
from pipelines_v2.core.provenance import resolve_provenance
from pipelines_v2.runtime.remote_executor import execute_remote
from pipelines_v2.storage.finalization import IntegrityError, validate_completion
from pipelines_v2.storage.local import LocalArtifactStore


@pytest.fixture
def harness(tmp_path, monkeypatch):
    dataset = Dataset.from_examples([Example(key=str(i),prompt=f'prompt {i}') for i in range(3)])
    spec = CaptureSpec(engine=ToyEngine(), dataset=dataset,
        sites=(ResidualSite(name='resid',site='resid_post',layers=(0,),tokens=TokenSelector.last()),))
    provenance = resolve_provenance(spec)
    assert provenance.identity_complete
    monkeypatch.setattr('pipelines_v2.core.provenance.resolve_provenance',lambda *a,**kw:provenance)
    store = LocalArtifactStore(tmp_path)
    args = dict(runner_config={'kind':'fixture'}, store_config=store.identity(), spec_payload=spec.to_dict(),
                workflow_context={'run_id':'resume','workflow_step_key':'capture'})
    return spec, store, args


def test_capture_interruption_resumes_missing_units_exactly_once(harness, monkeypatch):
    spec, store, args = harness
    original = ToyEngine.capture
    calls = []
    def interrupted(engine, spec):
        calls.extend(row.key for row in spec.dataset.examples)
        if calls[-1] == '1':
            raise RuntimeError('killed mid capture')
        return original(engine,spec)
    monkeypatch.setattr(ToyEngine,'capture',interrupted)
    with pytest.raises(RuntimeError,match='killed'):
        execute_remote(**args)
    assert not list(Path(store.root).glob('*/manifest.json'))
    assert len(list(Path(store.root).glob('*/journal.json'))) == 1
    calls.clear()
    def recovered(engine,spec):
        calls.extend(row.key for row in spec.dataset.examples)
        return original(engine,spec)
    monkeypatch.setattr(ToyEngine,'capture',recovered)
    final = ArtifactManifest.from_dict(execute_remote(**args))
    assert calls == ['1','2']
    assert validate_completion(final,store=store)
    assert final.example_coverage['example_count'] == 3
    calls.clear()
    assert execute_remote(**args)['artifact_id'] == final.artifact_id
    assert calls == []


def test_capture_missing_site_and_token_never_publish_completion(harness, monkeypatch):
    spec, store, args = harness
    original = ToyEngine.capture
    def broken(engine,spec):
        result = original(engine,spec)
        record = result.features['resid']['layers']['0'][spec.dataset.examples[0].key]
        record['tokens'] = []
        record['values'] = record['values'][:0]
        return result
    monkeypatch.setattr(ToyEngine,'capture',broken)
    with pytest.raises(IntegrityError,match='selected capture tokens'):
        execute_remote(**args)
    assert not list(Path(store.root).glob('*/manifest.json'))


def test_corrupt_capture_checkpoint_is_rejected_before_model(harness, monkeypatch):
    spec, store, args = harness
    original = ToyEngine.capture
    def interrupted(engine,spec):
        if spec.dataset.examples[0].key == '1': raise RuntimeError('stop')
        return original(engine,spec)
    monkeypatch.setattr(ToyEngine,'capture',interrupted)
    with pytest.raises(RuntimeError): execute_remote(**args)
    journal = json.loads(next(Path(store.root).glob('*/journal.json')).read_text())
    Path(journal['completed_units'][0]['payloads'][0]['path']).write_text('corrupt')
    monkeypatch.setattr(ToyEngine,'capture',lambda *a:pytest.fail('model ran before integrity check'))
    with pytest.raises(IntegrityError,match='checksum'): execute_remote(**args)


def test_intervention_interruption_preserves_completed_targets(harness, monkeypatch):
    capture, store, args = harness
    spec = PatchedGenerationSpec(engine=ToyEngine(),dataset=capture.dataset,
        patch=ProjectOutPatch(write_site=ResidualInterventionSite(site='resid_post',layers=(0,)),
            target_tokens=TokenSelector.last(),subspace={'layers':{'0':{'components':[]}}}),
        generation=GenerationSpec(enabled=True,max_tokens=1))
    monkeypatch.setattr('pipelines_v2.engine.toy.load_subspace_source', lambda patch: {'layers': {'0': {'mean':[0]*4, 'scale':[1]*4, 'safe_scale':[1]*4, 'components':[[1,0,0,0]]}}})
    args['spec_payload'] = spec.to_dict()
    original, calls = ToyEngine.intervene, []
    def interrupted(engine,spec):
        key=spec.dataset.examples[0].key
        if key == '1': raise RuntimeError('stop')
        return original(engine,spec)
    monkeypatch.setattr(ToyEngine,'intervene',interrupted)
    with pytest.raises(RuntimeError):execute_remote(**args)
    def recovered(engine,spec):
        calls.extend(row.key for row in spec.dataset.examples)
        return original(engine,spec)
    monkeypatch.setattr(ToyEngine,'intervene',recovered)
    final=ArtifactManifest.from_dict(execute_remote(**args))
    assert calls == ['1','2']
    assert validate_completion(final,store=store)
    assert [row['example_key'] for row in store.read_json_ref(final.storage_refs['result'])['rows']] == ['0','1','2']


def test_unresolved_identity_never_reuses_prior_capture_units(harness, monkeypatch):
    spec, store, args = harness
    import pipelines_v2.core.provenance as module
    from pipelines_v2.core.provenance import ProvenanceV1
    original = module.resolve_provenance(spec).to_dict()
    for key in ('schema_version','scientific_identity'):
        original.pop(key)
    original['unresolved'] = ['fixture.unresolved_model']
    record = ProvenanceV1(**original)
    monkeypatch.setattr(module,'resolve_provenance',lambda *a,**kw:record)
    first = execute_remote(**args)
    second = execute_remote(**args)
    assert first['artifact_id'] != second['artifact_id']


def test_catalog_failure_after_finalization_repairs_without_capture(harness, monkeypatch):
    spec, store, args = harness
    from pipelines_v2.api import LocalRunner, FileCatalog, WorkflowSpec, WorkflowStep, WorkflowOrchestrator
    import pipelines_v2.core.provenance as module
    class Catalog(FileCatalog):
        fail = True
        def record_artifact(self, manifest):
            if self.fail:
                object.__setattr__(self, "fail", False)
                raise OSError('index failed after finalize')
            return super().record_artifact(manifest)
    catalog = Catalog(Path(store.root)/'catalog')
    runner = LocalRunner(artifacts=store,catalog=catalog,provenance_resolver=lambda spec:module.resolve_provenance(spec))
    workflow = WorkflowSpec(name='catalog-repair',steps=(WorkflowStep(name='capture',runner='local',spec=spec),))
    orchestrator = WorkflowOrchestrator(runners={'local':runner},workflow_catalog=catalog)
    with pytest.raises(OSError,match='index failed'):orchestrator.run(workflow)
    run = catalog.list_workflow_runs()[0]
    monkeypatch.setattr(ToyEngine,'capture',lambda *a:pytest.fail('repaired catalog loaded model'))
    result = orchestrator.run(workflow,resume_run_id=run.run_id)
    assert validate_completion(result.step('capture').manifest(),store=store)


def test_duplicate_generation_rows_never_publish_completion(harness, monkeypatch):
    from pipelines_v2.api import GenerationRunSpec
    capture, store, args = harness
    spec = GenerationRunSpec(engine=ToyEngine(), dataset=capture.dataset,
        generation=GenerationSpec(enabled=True, max_tokens=1))
    args['spec_payload'] = spec.to_dict()
    original = ToyEngine.generate
    def duplicate(engine, spec):
        result = original(engine, spec)
        return replace(result, rows=[*result.rows, result.rows[0]])
    monkeypatch.setattr(ToyEngine, 'generate', duplicate)
    with pytest.raises(IntegrityError, match='duplicate example keys'):
        execute_remote(**args)
    assert not list(Path(store.root).glob('*/manifest.json'))


def test_missing_residual_values_never_publish_completion(harness, monkeypatch):
    spec, store, args = harness
    original = ToyEngine.capture
    def missing(engine, spec):
        result = original(engine, spec)
        del result.features['resid']['layers']['0'][spec.dataset.examples[0].key]['values']
        return result
    monkeypatch.setattr(ToyEngine, 'capture', missing)
    with pytest.raises(IntegrityError, match='omitted residual values'):
        execute_remote(**args)
    assert not list(Path(store.root).glob('*/manifest.json'))
