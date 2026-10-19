"""Official MAttr smoke: two fitting steps, two held-out examples, no research claim."""
import json
import os
from pathlib import Path
from pipelines_v2.api import MAttrSpec,ExternalToolRuntimeSpec,official_tool_lock,WorkflowSpec,WorkflowStep,LocalRunnerSpec,LocalArtifactStore
from pipelines_v2.runtime.locks import core_runtime_lock
from pipelines_v2.storage import FileCatalog
from pipelines_v2.experiments import ExperimentPack

ROOT=Path(__file__).resolve().parent


def settings():
    path=os.environ.get('XENON_MATTR_CONFIG')
    if not path:raise ValueError('Set XENON_MATTR_CONFIG to execution.json from python -m pipelines_v2.operations.attribution prepare')
    return json.loads(Path(path).read_text())


def build_dataset():
    # The external harness reads its separately pinned IOI files; no rows cross interpreters.
    from pipelines_v2.api import Dataset
    return Dataset.from_examples(())


def unchecked_workflow(dataset=None):
    cfg=settings()
    runtime=ExternalToolRuntimeSpec(lock=core_runtime_lock(),local_python_sources=('pipelines_v2',),tool_lock=official_tool_lock(),
        **{k:cfg[k] for k in ('tool_root','mib_root','tool_python')})
    request=json.loads((ROOT/'request.json').read_text())
    spec=MAttrSpec(request=request,runtime=runtime,model_path=cfg['model_path'],dataset_path=cfg['dataset_path'])
    return WorkflowSpec(name='official-mattr-gpt2-ioi-smoke',steps=(WorkflowStep('attribution','isolated_local',spec),))


def build_workflow(dataset=None):
    workflow=unchecked_workflow()
    ExperimentPack.load(ROOT/'pack.json').validate_workflow(workflow,phase_dir=ROOT)
    return workflow


def build_runner_specs():
    root=Path(settings()['artifact_root'])
    return {'isolated_local':LocalRunnerSpec(artifacts=LocalArtifactStore(root),catalog=FileCatalog(root/'catalog'))}
