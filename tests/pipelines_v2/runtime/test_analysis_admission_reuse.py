"""Production resolvers, actual CPU operations and workflow reuse: no injected C01."""
from dataclasses import replace

import pytest

from pipelines_v2.api import (BasisSpec, CaptureSpec, CentroidSpec, Dataset, DirectionSpec, Example,
    FileCatalog, GeometrySpec, LocalArtifactStore, LocalRunner, LocalResources, ProbeSpec, ResidualSite,
    StepRef, SubspaceSpec, ToyEngine, WorkflowOrchestrator, WorkflowSpec, WorkflowStep)
from pipelines_v2.core.provenance import provenance_from_metadata


class RecordingRunner(LocalRunner):
    def run(self, spec, **kwargs):
        self.calls.append(spec.kind)
        return super().run(spec, **kwargs)


def setup(tmp_path, kind):
    dataset = Dataset.from_examples([Example(key=str(index), prompt=f'item {index}',
        labels={'class':str(index % 2), 'split':'train' if index < 8 else 'test'}) for index in range(12)])
    feature = StepRef('capture').feature('resid')
    operations = {
        'probe': ProbeSpec(feature=feature, labels=dataset.labels('class'), split=dataset.labels('split'), folds=2, baselines=['majority']),
        'basis': BasisSpec(feature=feature, components=2, random_state=7),
        'subspace': SubspaceSpec(feature=feature, components=2, random_state=7),
        'direction': DirectionSpec(feature=feature, positive=dataset.labels('class').equals('1'), negative=dataset.labels('class').equals('0')),
        'centroid': CentroidSpec(feature=feature, by=dataset.labels('class')),
        'geometry': GeometrySpec(feature=feature, components=2, random_state=7),
    }
    store, catalog = LocalArtifactStore(tmp_path/'artifacts'), FileCatalog(tmp_path/'catalog')
    runner = RecordingRunner(artifacts=store, catalog=catalog)
    runner.calls = []
    workflow = WorkflowSpec(name='production-numerics-'+kind, steps=(
        WorkflowStep(name='capture',runner='local',spec=CaptureSpec(engine=ToyEngine(hidden_size=4,num_layers=1),
            dataset=dataset, sites=[ResidualSite(name='resid',site='resid_post',layers=[0])])),
        WorkflowStep(name='analysis',runner='local',spec=operations[kind]),
    ))
    return runner, workflow, WorkflowOrchestrator(runners={'local':runner}, workflow_catalog=catalog,max_parallelism=1)


@pytest.mark.parametrize('kind',['probe','basis','subspace','direction','centroid','geometry'])
def test_supported_production_analysis_reuses(tmp_path, kind):
    runner, workflow, orchestrator = setup(tmp_path,kind)
    first = orchestrator.run(workflow)
    assert provenance_from_metadata(first.step('analysis').manifest().metadata).identity_complete
    runner.calls.clear()
    second = orchestrator.run(workflow, resume_run_id=first.run_id)
    assert runner.calls == []
    assert second.step('analysis').id == first.step('analysis').id
    third = orchestrator.run(workflow, reuse_completed=True)
    assert runner.calls == [] and third.step('analysis').id == first.step('analysis').id


@pytest.mark.parametrize('mutation',['seed','algorithm','environment','upstream'])
def test_effective_analysis_changes_invalidate_reuse(tmp_path, mutation):
    runner, workflow, orchestrator = setup(tmp_path,'basis')
    first = orchestrator.run(workflow)
    capture, analysis = workflow.steps
    if mutation == 'seed':
        analysis = replace(analysis,spec=replace(analysis.spec,random_state=8))
    elif mutation == 'algorithm':
        analysis = replace(analysis,spec=replace(analysis.spec,svd_solver='full'))
    elif mutation == 'environment':
        runner.resources = LocalResources(env={'OMP_NUM_THREADS':'2'})
    else:
        dataset = capture.spec.dataset
        capture = replace(capture,spec=replace(capture.spec,dataset=replace(dataset,examples=tuple(
            replace(row,prompt='different actual prompt') if row.key=='0' else row for row in dataset.examples))))
    runner.calls.clear()
    second = orchestrator.run(replace(workflow,steps=(capture,analysis)),reuse_from_run_id=first.run_id)
    assert 'basis' in runner.calls
    assert second.step('analysis').id != first.step('analysis').id


def test_unseeded_stochastic_analysis_executes_but_cannot_reuse(tmp_path):
    runner, workflow, orchestrator = setup(tmp_path,'basis')
    workflow = replace(workflow,steps=(workflow.steps[0],replace(workflow.steps[1],spec=replace(workflow.steps[1].spec,random_state=None))))
    first = orchestrator.run(workflow)
    assert not provenance_from_metadata(first.step('analysis').manifest().metadata).identity_complete
    runner.calls.clear()
    second = orchestrator.run(workflow,reuse_from_run_id=first.run_id)
    assert runner.calls == ['basis']
    assert second.step('analysis').id != first.step('analysis').id
