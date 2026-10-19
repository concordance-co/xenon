"""Four-prompt, three-neuron bounded native microscope plumbing."""
import os
from pathlib import Path
from pipelines_v2.api import (NativeTorchEngine, NativeInterventionSite, NativeCoordinatePatch,
    Dataset, Example, CaptureSpec, GenerationRunSpec, GenerationSpec, PatchedGenerationSpec,
    WorkflowSpec, WorkflowStep, ReportSpec, StepRef, TransformBuilder, TransformSpec,
    LocalRunnerSpec, LocalArtifactStore, FileCatalog)
from pipelines_v2.operations.neurons.summaries import QUANTITIES
from pipelines_v2.operations.neurons.transforms import summarize_capture, dossier_transform

REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
CONTEXTS = {
    'france': {'text': 'The capital of France is', 'document_id': 'fixture-france', 'split': 'discovery'},
    'numbers': {'text': 'Two plus three equals', 'document_id': 'fixture-numbers', 'split': 'discovery'},
    'weather': {'text': 'Rain falls from the', 'document_id': 'fixture-weather', 'split': 'heldout'},
    'code': {'text': 'In Python, a list is written with', 'document_id': 'fixture-code', 'split': 'heldout'},
}


def build_dataset():
    return Dataset.from_examples(tuple(Example(key=key, prompt=row['text'], labels={'role': 'edit' if key == 'france' else 'capture'}) for key, row in CONTEXTS.items()))


def build_workflow(dataset=None):
    dataset = dataset or build_dataset()
    engine = NativeTorchEngine(model_id=os.environ['XENON_NATIVE_MODEL_CACHE'], revision=REVISION, architecture='qwen3')
    sites = tuple(engine.site(q, q, 0, neurons=(0, 1, 2)) for q in QUANTITIES)
    write = NativeInterventionSite.from_capture(engine.site('edit', 'gate_preactivation', 0, neurons=(0,)))
    generation = GenerationSpec(enabled=True, max_tokens=2, target_token_ids=(10, 20))
    select = dataset.labels('role').equals('edit')
    steps = [WorkflowStep('capture', 'scratch', CaptureSpec(engine=engine, dataset=dataset, sites=sites)),
        WorkflowStep('baseline', 'scratch', GenerationRunSpec(engine=engine, dataset=dataset, generation=generation, select_when=select))]
    for name, scale in (('identity', 1.), ('edit', 0.)):
        steps.append(WorkflowStep(name, 'scratch', PatchedGenerationSpec(engine=engine, dataset=dataset, generation=generation,
            select_when=select, patch=NativeCoordinatePatch(write_site=write, target_tokens=write.coordinate.tokens, mode='scale', value=scale))))
    sources = ('pipelines_v2', 'examples/native_neurons/phase_00')
    steps.extend([
        WorkflowStep('summary', 'scratch', TransformSpec(builder=TransformBuilder.from_function(summarize_capture, local_python_sources=sources),
            inputs={'capture': StepRef('capture'), 'contexts': CONTEXTS, 'limits': {'k': 2, 'reservoir_size': 4, 'max_neurons': 3, 'max_observations': 12}})),
        WorkflowStep('dossiers', 'scratch', TransformSpec(builder=TransformBuilder.from_function(dossier_transform, local_python_sources=sources),
            inputs={'summary': StepRef('summary'), 'baseline': StepRef('baseline'), 'identity': StepRef('identity'), 'edit': StepRef('edit'),
                'manual_candidates': {'0': 'Unverified plumbing fixture: this coordinate varies across the supplied discovery contexts.'}, 'edited_neuron': 0})),
        WorkflowStep('report', 'scratch', ReportSpec(inputs=(StepRef('dossiers'),))),
    ])
    return WorkflowSpec(name='native-neuron-microscope-phase00', steps=tuple(steps))


def build_runner_specs():
    home = Path(os.environ.get('XENON_HOME', '~/.xenon')).expanduser()
    return {'scratch': LocalRunnerSpec(artifacts=LocalArtifactStore(home / 'pipelines_v2/artifacts'), catalog=FileCatalog(home / 'pipelines_v2/catalog'))}
