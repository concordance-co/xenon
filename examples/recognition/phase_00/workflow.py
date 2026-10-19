"""Two image fixtures through pinned ViT capture/control/edit and local report."""
import os,json
from pathlib import Path
from dataclasses import replace
from pipelines_v2.api import (ViTRecognitionEngine, ViTSite, RecognitionSpec, Dataset, Example,
    WorkflowStep, WorkflowSpec, ReportSpec, StepRef, LocalRunnerSpec, LocalArtifactStore, FileCatalog)
from pipelines_v2.data.images import ImageInput


def build_dataset():
    root=Path(os.environ['XENON_RECOGNITION_INPUTS'])
    images=[ImageInput.from_dict(row) for row in json.loads((root/'inputs.json').read_text())]
    return Dataset.from_examples(tuple(Example(key=image.image_id,prompt='',images=(image,)) for image in images))


def build_workflow(dataset=None):
    site=ViTSite(tokens=(0,1,98),channels=tuple(range(8)),layer=0)
    edit=replace(site,tokens=(98,),read_or_write='write')
    return WorkflowSpec(name='vit-recognition-microscope',steps=(
        WorkflowStep('recognition','scratch',RecognitionSpec(engine=ViTRecognitionEngine(os.environ['XENON_RECOGNITION_MODEL']),
            dataset=dataset or build_dataset(),site=site,edit=edit,image_roots=(str(Path(os.environ['XENON_RECOGNITION_INPUTS']).resolve()),))),
        WorkflowStep('report','scratch',ReportSpec(inputs=(StepRef('recognition'),))),
    ))


def build_runner_specs():
    home=Path(os.environ.get('XENON_HOME','~/.xenon')).expanduser()
    return {'scratch':LocalRunnerSpec(artifacts=LocalArtifactStore(home/'pipelines_v2/artifacts'),catalog=FileCatalog(home/'pipelines_v2/catalog'))}
