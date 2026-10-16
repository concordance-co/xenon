"""One official Qwen2.5 / IOI method-seed unit. No paper reproduction claim."""
import json
import os
from pathlib import Path

from pipelines_v2.api import (Dataset, ExternalToolRuntimeSpec, FileCatalog,
    LocalArtifactStore, LocalRunnerSpec, MAttrSpec, WorkflowSpec, WorkflowStep,
    official_tool_lock, TransformBuilder, TransformSpec, StepRef, OperationArtifact)
from pipelines_v2.runtime.locks import core_runtime_lock
from pipelines_v2.operations.attribution.report import summarize_attribution


def settings():
    name = os.environ.get('XENON_MATTR_CONFIG')
    if not name:
        raise ValueError('Set XENON_MATTR_CONFIG to prepared Qwen2.5 execution.json')
    return json.loads(Path(name).read_text())


def build_dataset():
    return Dataset.from_examples(())


def build_workflow(dataset=None):
    cfg = settings()
    runtime = ExternalToolRuntimeSpec(lock=core_runtime_lock(), local_python_sources=('pipelines_v2',),
        tool_lock=official_tool_lock(), **{k:cfg[k] for k in ('tool_root','mib_root','tool_python')})
    spec = MAttrSpec.qwen25_ioi(model_path=cfg['model_path'], dataset_path=cfg['dataset_path'],
        runtime=runtime, steps=2, eval_examples=2, seed=42, memory_bytes=12*1024**3)
    steps = []
    source_id = os.environ.get('XENON_MATTR_SOURCE_ARTIFACT')
    if source_id:
        root = Path(cfg['artifact_root'])
        manifest = FileCatalog(root/'catalog').load_artifact(source_id)
        if manifest is None or manifest.artifact_kind != 'external_attribution':
            raise ValueError('Source must be an existing cataloged external attribution artifact')
        source = OperationArtifact(manifest, LocalArtifactStore(root))
    else:
        steps.append(WorkflowStep('attribution','isolated_local',spec))
        source = StepRef('attribution')
    steps.append(WorkflowStep('summary','isolated_local',TransformSpec(
        builder=TransformBuilder.from_function(summarize_attribution,local_python_sources=('pipelines_v2',)),
        inputs={'attribution':source})))
    return WorkflowSpec(name='official-mattr-qwen25-ioi-smoke', steps=tuple(steps))


def build_runner_specs():
    root = Path(settings()['artifact_root'])
    return {'isolated_local':LocalRunnerSpec(artifacts=LocalArtifactStore(root), catalog=FileCatalog(root/'catalog'))}
