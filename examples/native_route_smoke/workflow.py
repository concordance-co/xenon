"""Two-example official IOI graph plumbing, using prepared offline GPT-2 only."""
import json
import os
from pathlib import Path
from pipelines_v2.api import (Dataset, NativeTorchEngine, NativeRouteSpec, RoutePair, WorkflowSpec, WorkflowStep,
    TransformSpec, TransformBuilder, StepRef, ReportSpec, LocalRunnerSpec, LocalArtifactStore, FileCatalog)
from pipelines_v2.operations.common.native import HEAD_QUANTITY
from pipelines_v2.operations.graphs.report import graph_report_transform
ROOT=Path(__file__).resolve().parent
REVISION='607a30d783dfa663caf39e06633721c8d4cfcd7e'


def build_dataset():return Dataset.from_examples(())


def build_workflow(dataset=None):
    model=os.environ.get('XENON_GPT2_MODEL')
    if not model:raise ValueError('Set XENON_GPT2_MODEL to the prepared pinned GPT-2 snapshot directory')
    engine=NativeTorchEngine(model_id=model,revision=REVISION,architecture='gpt2')
    pairs=json.loads((ROOT/'pairs.json').read_text())
    spec=NativeRouteSpec(engine=engine,sites=tuple(engine.site(f'layer5.head{h}',HEAD_QUANTITY,5,heads=(h,)) for h in (0,1)),
        pairs=tuple(RoutePair(**r) for r in pairs['pairs']),proposal_templates=tuple(pairs['proposal_templates']),ig_steps=4)
    output=os.environ.get('XENON_GRAPH_OUTPUT',str(Path.home()/'.xenon/reports/native-route'))
    return WorkflowSpec(name='native-route-gpt2-ioi-plumbing',steps=(
        WorkflowStep('route','local',spec),
        WorkflowStep('graph_report','local',TransformSpec(builder=TransformBuilder.from_function(graph_report_transform,local_python_sources=('pipelines_v2',)),
            inputs={'route':StepRef('route'),'output_dir':output})),
        WorkflowStep('report','local',ReportSpec(inputs=(StepRef('graph_report'),StepRef('route')))),))


def build_runner_specs():
    root=Path(os.environ.get('XENON_GRAPH_ARTIFACTS',str(Path.home()/'.xenon/pipelines_v2/native-route')))
    return {'local':LocalRunnerSpec(artifacts=LocalArtifactStore(root),catalog=FileCatalog(root/'catalog'))}
