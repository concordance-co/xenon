"""Adversarial C01 checks at real workflow lifecycle boundaries, without model/network work."""
from __future__ import annotations

from dataclasses import replace

import pytest

from pipelines_v2.api import (
    Dataset, Example, FileCatalog, LabelMapSpec, LocalArtifactStore, LocalRunner,
    StepRef, WorkflowOrchestrator, WorkflowSpec, WorkflowStep,
)
from pipelines_v2.core.provenance import KEY, resolve_provenance


class RecordingRunner(LocalRunner):
    def run(self, spec, *, workflow_context=None, progress_callback=None):
        self.calls.append(workflow_context.step_name if workflow_context else spec.kind)
        return super().run(spec, workflow_context=workflow_context, progress_callback=progress_callback)


@pytest.fixture
def harness(tmp_path):
    version = {'map': 'a' * 64}
    runtime = {
        'family': 'local', 'lock_digest': 'b' * 64, 'python': 'fixture-python',
        'platform': 'fixture-platform', 'installed_package_digest': 'c' * 64,
        'image_identity': None, 'image_reason': 'local process has no container image',
    }
    def resolver(spec):
        # Only the first operation changes code. Downstream invalidation must
        # come from its actual changed input, not a blanket source mismatch.
        digest = version['map'] if getattr(spec, 'output_name', None) == 'mapped' else 'd' * 64
        source = {'repository_id': 'fixture-repository', 'head': 'e' * 40,
                  'dirty_digest': digest, 'mounted_source_digest': digest}
        return resolve_provenance(spec, source=source, runtime=runtime)
    catalog = FileCatalog(tmp_path / 'catalog')
    store = LocalArtifactStore(tmp_path / 'artifacts')
    runner = RecordingRunner(artifacts=store, catalog=catalog, provenance_resolver=resolver)
    runner.calls = []
    dataset = Dataset.from_examples([
        Example(key='a', prompt='alpha', labels={'class':'positive'}),
        Example(key='b', prompt='beta', labels={'class':'negative'}),
    ], id='fixture-dataset')
    workflow = WorkflowSpec(name='provenance-boundary', steps=(
        WorkflowStep(name='map', runner='local', spec=LabelMapSpec(
            source=dataset.labels('class'), mapping={'positive':1,'negative':0}, output_name='mapped')),
        WorkflowStep(name='remap', runner='local', spec=LabelMapSpec(
            source=StepRef('map').label('mapped'), mapping={'1':'yes','0':'no'}, output_name='final')),
    ))
    orchestrator = WorkflowOrchestrator(runners={'local':runner}, workflow_catalog=catalog, max_parallelism=1)
    return version, runner, catalog, workflow, orchestrator


def _repeat(orchestrator, workflow, first, mode):
    if mode == 'resume':
        return orchestrator.run(workflow, resume_run_id=first.run_id)
    if mode == 'explicit':
        return orchestrator.run(workflow, reuse_from_run_id=first.run_id)
    return orchestrator.run(workflow, reuse_completed=True)


@pytest.mark.parametrize('mode', ['resume', 'explicit', 'latest'])
def test_unchanged_provenance_reuses_without_execution(harness, mode):
    _, runner, catalog, workflow, orchestrator = harness
    first = orchestrator.run(workflow)
    runner.calls.clear()
    second = _repeat(orchestrator, workflow, first, mode)
    assert runner.calls == []
    assert second.step('map').id == first.step('map').id
    assert second.step('remap').id == first.step('remap').id
    assert all(record.status in {'completed','reused'} for record in catalog.list_workflow_steps(second.run_id))


@pytest.mark.parametrize('mode', ['resume', 'explicit', 'latest'])
def test_stale_upstream_reexecutes_and_does_not_skip_downstream(harness, mode):
    version, runner, catalog, workflow, orchestrator = harness
    first = orchestrator.run(workflow)
    version['map'] = 'f' * 64
    runner.calls.clear()
    second = _repeat(orchestrator, workflow, first, mode)
    assert runner.calls == ['map', 'remap']
    assert second.step('map').id != first.step('map').id
    assert second.step('remap').id != first.step('remap').id
    rows = {record.step_name:record for record in catalog.list_workflow_steps(second.run_id)}
    assert rows['remap'].input_artifact_refs == (second.step('map').id,)
    assert rows['map'].status == rows['remap'].status == 'completed'
    assert second.step('remap').label('final').resolve_values() == {'a':'yes','b':'no'}


@pytest.mark.parametrize('mode', ['resume', 'explicit', 'latest'])
def test_legacy_upstream_cannot_certify_current_reuse(harness, mode):
    _, runner, catalog, workflow, orchestrator = harness
    first = orchestrator.run(workflow)
    manifest = first.step('map').manifest()
    legacy = replace(manifest, metadata={key:value for key,value in manifest.metadata.items() if key != KEY})
    catalog.record_artifact(legacy)
    runner.artifacts.write_json(legacy.artifact_id, 'manifest.json', legacy.to_dict())
    runner.calls.clear()
    second = _repeat(orchestrator, workflow, first, mode)
    assert runner.calls == ['map','remap']
    assert second.step('map').id != first.step('map').id
    assert second.step('remap').id != first.step('remap').id
    assert KEY in second.step('map').manifest().metadata


@pytest.mark.parametrize('changed', [False, True])
def test_running_record_recovery_checks_provenance_before_promoting_complete(harness, changed):
    version, runner, catalog, workflow, orchestrator = harness
    first = orchestrator.run(workflow)
    records = {record.step_name:record for record in catalog.list_workflow_steps(first.run_id)}
    # No artifact id on the running step: exercise find_artifact_for_workflow_step.
    catalog.record_workflow_step(replace(records['map'], status='running', artifact_id=None, finished_at=None))
    if changed:
        version['map'] = 'f' * 64
    runner.calls.clear()
    second = orchestrator.run(workflow, resume_run_id=first.run_id)
    if changed:
        assert runner.calls == ['map','remap']
        assert second.step('map').id != first.step('map').id
        assert second.step('remap').id != first.step('remap').id
    else:
        assert runner.calls == []
        assert second.step('map').id == first.step('map').id
        assert second.step('remap').id == first.step('remap').id
    assert all(record.status in {'completed','reused'} for record in catalog.list_workflow_steps(second.run_id))


@pytest.mark.parametrize('mode', ['resume', 'explicit', 'latest'])
def test_actual_prompt_bytes_invalidate_even_with_unchanged_supplied_prompt_hash(harness, mode):
    _, runner, _, workflow, orchestrator = harness
    first = orchestrator.run(workflow)
    first_step = workflow.steps[0]
    dataset = first_step.spec.source.dataset
    changed_dataset = replace(dataset, examples=tuple(
        replace(example, prompt='changed actual bytes') if example.key == 'a' else example
        for example in dataset.examples
    ))
    changed_step = replace(first_step, spec=replace(first_step.spec, source=changed_dataset.labels('class')))
    changed_workflow = replace(workflow, steps=(changed_step, workflow.steps[1]))
    # Legacy hashes trust the explicitly supplied prompt_hash. The independent
    # content observation must still catch this mismatch before selecting reuse.
    assert changed_workflow.semantic_hash() == workflow.semantic_hash()
    runner.calls.clear()
    second = _repeat(orchestrator, changed_workflow, first, mode)
    assert runner.calls == ['map','remap']
    assert second.step('map').id != first.step('map').id
    assert second.step('remap').id != first.step('remap').id


def test_heartbeat_does_not_publish_unfinished_results(harness, monkeypatch):
    from types import SimpleNamespace
    from pipelines_v2.workflow import orchestrator as module

    _, runner, _, workflow, orchestrator = harness
    events = []
    orchestrator.progress_sink = SimpleNamespace(emit=events.append)
    original_wait = module.wait
    seen = set()

    def heartbeat_before_completion(futures, **kwargs):
        key = frozenset(futures)
        if key not in seen:
            seen.add(key)
            return set(), set(futures)
        return original_wait(futures, **kwargs)

    monkeypatch.setattr(module, 'wait', heartbeat_before_completion)
    result = orchestrator.run(workflow)
    assert runner.calls == ['map', 'remap']
    assert [event.step_name for event in events if event.stage == 'heartbeat'] == ['map', 'remap']
    assert result.step('remap').label('final').resolve_values() == {'a': 'yes', 'b': 'no'}
