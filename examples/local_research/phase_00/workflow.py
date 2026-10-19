"""Eight-example offline GPT-2 residual experiment; see README for preparation."""
from pathlib import Path
import os
from pipelines_v2.api import (TorchEngine, Dataset, CaptureSpec, GenerationRunSpec, GenerationSpec,
    PatchedGenerationSpec, ScalePatch, InterchangePatch, ResidualInterventionSite, ResidualSite,
    TokenSelector, TensorStorage, StepRef, ReportSpec, WorkflowSpec, WorkflowStep, ActivationBankSpec,
    TransformSpec, TransformBuilder, LocalRunnerSpec, LocalArtifactStore, FileCatalog)
from pipelines_v2.experiments import ExperimentPack
from examples.local_research.phase_00.fixtures import MODEL, REVISION, TOKENS, LAYER, PAIRS, examples
from examples.local_research.phase_00.analysis import summarize

ROOT = Path(__file__).resolve().parent


def build_dataset():
    return Dataset.from_examples(examples())


def unchecked_workflow(dataset=None):
    dataset = dataset or build_dataset()
    engine = TorchEngine(model_id=MODEL, revision=REVISION)
    generation = GenerationSpec(enabled=True, max_tokens=8, target_token_ids=TOKENS)
    site = ResidualInterventionSite('resid_post', (LAYER,))
    steps = [
        WorkflowStep('capture', 'scratch', CaptureSpec(engine=engine, dataset=dataset,
            sites=(ResidualSite('block5', 'resid_post', (LAYER,), tokens=TokenSelector.last(), storage=TensorStorage(dtype='float32')),))),
        WorkflowStep('bank', 'scratch', ActivationBankSpec(feature=StepRef('capture').feature('block5'), layers=(LAYER,))),
        WorkflowStep('baseline', 'scratch', GenerationRunSpec(engine=engine, dataset=dataset, generation=generation)),
    ]
    for name, scale in (('identity', 1.), ('zero', 0.)):
        steps.append(WorkflowStep(name, 'scratch', PatchedGenerationSpec(engine=engine, dataset=dataset,
            generation=generation, patch=ScalePatch(write_site=site, scale=scale))))
    for name in PAIRS:
        steps.append(WorkflowStep(name, 'scratch', PatchedGenerationSpec(engine=engine, dataset=dataset,
            generation=generation, patch=InterchangePatch(write_site=site, activation_bank=StepRef('bank'),
                target_tokens=TokenSelector.last(), donor_tokens=TokenSelector.last()),
            pair_by=dataset.labels(name), target_when=dataset.labels('role').equals('target'),
            donor_when=dataset.labels('role').equals('donor'))))
    names = ('baseline', 'identity', 'zero', *PAIRS)
    steps.append(WorkflowStep('summary', 'scratch', TransformSpec(builder=TransformBuilder.from_function(summarize,
        local_python_sources=('pipelines_v2', 'examples/local_research/phase_00')),
        inputs={name: StepRef(name) for name in names})))
    steps.append(WorkflowStep('report', 'scratch', ReportSpec(inputs=(StepRef('summary'), *(StepRef(name) for name in names)))))
    return WorkflowSpec(name='gpt2-local-research-phase00', steps=tuple(steps))


def build_workflow(dataset=None):
    workflow = unchecked_workflow(dataset)
    ExperimentPack.load(ROOT / 'pack.json').validate_workflow(workflow, phase_dir=ROOT)
    return workflow


def build_runner_specs():
    home = Path(os.environ.get('XENON_HOME', '~/.xenon')).expanduser()
    return {'scratch': LocalRunnerSpec(artifacts=LocalArtifactStore(home / 'pipelines_v2/artifacts'),
        catalog=FileCatalog(home / 'pipelines_v2/catalog'))}
