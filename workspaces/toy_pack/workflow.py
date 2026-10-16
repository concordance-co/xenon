"""Two-row engineering fixture. No scientific or GPU qualification claims."""
from pathlib import Path

from pipelines_v2.api import (CaptureSpec, Dataset, Example, LabelMapSpec, LocalArtifactStore,
                              LocalRunnerSpec, ReportSpec, ResidualSite, StepRef, ToyEngine,
                              WorkflowSpec, WorkflowStep)
from pipelines_v2.experiments import ExperimentPack
from pipelines_v2.storage import FileCatalog

ROOT = Path(__file__).resolve().parent


def build_dataset():
    return Dataset.from_examples((Example(key='toy-a', prompt='alpha', labels={'class': 'a'}),
                                  Example(key='toy-b', prompt='beta', labels={'class': 'b'})))


def unchecked_workflow(dataset=None):
    """Build ordinary specs; pack authoring can hash this without recursive loads."""
    dataset = dataset or build_dataset()
    return WorkflowSpec(name='experiment-pack-toy', steps=(
        WorkflowStep('capture', 'fixture_local', CaptureSpec(engine=ToyEngine(), dataset=dataset,
                                                            sites=(ResidualSite(name='resid', site='resid_post', layers=(0,)),))),
        WorkflowStep('labels', 'fixture_local', LabelMapSpec(source=dataset.labels('class'), mapping={'a': 0, 'b': 1})),
        WorkflowStep('report', 'report_local', ReportSpec(inputs=(StepRef('capture'), StepRef('labels')),
                                                        output_dir=str(ROOT / 'reports'))),
    ))


def build_workflow(dataset=None):
    workflow = unchecked_workflow(dataset)
    ExperimentPack.load(ROOT / 'pack.json').validate_workflow(workflow, phase_dir=ROOT)
    return workflow


def build_runner_specs():
    catalog = FileCatalog(ROOT / '.catalog')
    store = LocalArtifactStore(ROOT / 'artifacts')
    return {'fixture_local': LocalRunnerSpec(artifacts=store, catalog=catalog),
            'report_local': LocalRunnerSpec(artifacts=store, catalog=catalog)}
