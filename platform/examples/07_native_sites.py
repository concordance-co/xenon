"""One prepared-model native capture + selected-coordinate edit; plain text, no chat template."""
import os
from pipelines_v2.api import (NativeTorchEngine, NativeInterventionSite, NativeCoordinatePatch,
    Dataset, Example, CaptureSpec, GenerationSpec, PatchedGenerationSpec, WorkflowSpec, WorkflowStep, ReportSpec, StepRef)
from pipelines_v2.engine.native.engine import REVISIONS


def build_dataset():
    return Dataset.from_examples((Example(key='native-smoke',prompt='The capital of France is'),))


def build_workflow(dataset):
    architecture=os.environ.get('XENON_NATIVE_ARCHITECTURE','qwen3')
    engine=NativeTorchEngine(model_id=os.environ['XENON_NATIVE_MODEL_CACHE'],revision=REVISIONS[architecture],architecture=architecture)
    quantities=('attention_head_output',) if architecture=='gpt2' else ('gate_preactivation','gate_activation','up_projection','mlp_product')
    sites=tuple(engine.site(quantity,quantity,0,heads=(0,) if architecture=='gpt2' else (),neurons=() if architecture=='gpt2' else (0,1,2)) for quantity in quantities)
    write=NativeInterventionSite.from_capture(sites[0])
    return WorkflowSpec(name='native-sites-'+architecture,steps=(
        WorkflowStep('capture','scratch',CaptureSpec(engine=engine,dataset=dataset,sites=sites)),
        WorkflowStep('edit','scratch',PatchedGenerationSpec(engine=engine,dataset=dataset,
            patch=NativeCoordinatePatch(write_site=write,target_tokens=write.coordinate.tokens,mode='scale',value=0.),
            generation=GenerationSpec(enabled=True,max_tokens=2,target_token_ids=(10,20)))),
        WorkflowStep('report','scratch',ReportSpec(inputs=(StepRef('capture'),StepRef('edit')))),
    ))


def build_runner_specs():
    from pathlib import Path
    from pipelines_v2.api import LocalRunnerSpec, LocalArtifactStore, FileCatalog
    home=Path(os.environ.get('XENON_HOME','~/.xenon')).expanduser()
    return {'scratch':LocalRunnerSpec(artifacts=LocalArtifactStore(home/'pipelines_v2/artifacts'),
        catalog=FileCatalog(home/'pipelines_v2/catalog'))}
