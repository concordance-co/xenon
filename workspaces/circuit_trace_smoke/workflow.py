"""One-prompt PLT graph through the regular Modal artifact workflow path."""
import os
from pathlib import Path
from pipelines_v2.api import (Dataset,CircuitTraceSpec,CircuitTracerRuntimeSpec,WorkflowSpec,WorkflowStep,StepRef,
    TransformSpec,TransformBuilder,ReportSpec,ModalRunnerSpec,ModalResources,ModalVolumeMount,ModalVolumeStore,
    LocalRunnerSpec,LocalArtifactStore,FileCatalog)
from pipelines_v2.runtime.locks import core_runtime_lock
from pipelines_v2.operations.circuit_tracer.prepare import pair_lock
from pipelines_v2.operations.circuit_tracer.report import trace_report_transform


def output_root():return Path(os.environ.get('XENON_TRACER_OUTPUT',str(Path.home()/'.xenon/reports/circuit-trace')))
def build_dataset():return Dataset.from_examples(())


def build_workflow(dataset=None):
    pair=pair_lock();base='/models/circuit-tracer-qwen3'
    runtime=CircuitTracerRuntimeSpec(lock=core_runtime_lock(),local_python_sources=('pipelines_v2',))
    trace=CircuitTraceSpec(runtime=runtime,model_path=f'{base}/model/{pair["model"]["revision"]}',dictionary_path=f'{base}/dictionary/{pair["dictionary"]["revision"]}')
    return WorkflowSpec(name='qwen3-plt-circuit-trace-plumbing',steps=(
        WorkflowStep('trace','trace_gpu',trace),
        WorkflowStep('graph_report','report_local',TransformSpec(builder=TransformBuilder.from_function(trace_report_transform,local_python_sources=('pipelines_v2',)),inputs={'trace':StepRef('trace'),'output_dir':str(output_root())})),
        WorkflowStep('report','report_local',ReportSpec(inputs=(StepRef('graph_report'),))),))


def build_runner_specs():
    output=output_root();catalog=FileCatalog(output/'catalog')
    return {'trace_gpu':ModalRunnerSpec(artifacts=ModalVolumeStore(name='xenon-data',root='/data/circuit-trace-plumbing'),catalog=catalog,
        resources=ModalResources(gpu='A100-80GB',cpu=8,memory_mb=65536,timeout_seconds=1200,client_timeout_seconds=1500,
            readiness_timeout_seconds=300,max_containers=1,shard_count=1,volumes=(ModalVolumeMount('xenon-models','/models'),))),
        'report_local':LocalRunnerSpec(artifacts=LocalArtifactStore(output/'artifacts'),catalog=catalog)}
