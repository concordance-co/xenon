"""Configurable paired-model workflow and CPU/Modal entry point."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from .fitting import FitConfig
from .settings import Execution,Limits


def build_workflow(*, dataset, model_paths, reference_path, fit=None, execution=None, limits=None,
                   train_keys=(), train_count=3, layer=12, checkpoint_step=None, feature_index=0, amount=10.0,
                   compute_runner='compute', report_runner='report_local', name='qwen2.5-crosscoder'):
    from pipelines_v2.api import WorkflowSpec,WorkflowStep,StepRef,ReportSpec
    from pipelines_v2.operations.representation.crosscoder import (
        CrosscoderPairCaptureSpec,CrosscoderFitSpec,CrosscoderImportSpec,
        CrosscoderEncodeSpec,CrosscoderDecodeSpec,CrosscoderFeatureSpliceSpec)
    config=asdict(FitConfig(**(fit or {})))
    common={'execution':asdict(Execution(**(execution or {}))), 'limits':asdict(Limits(**(limits or {})))}
    steps=[WorkflowStep('paired_capture',compute_runner,CrosscoderPairCaptureSpec(
        model_paths=tuple(model_paths),dataset=dataset,train_keys=tuple(train_keys),train_count=train_count,layer=layer,**common))]
    resume=None
    if checkpoint_step is not None:
        if type(checkpoint_step) is not int or not 0<checkpoint_step<config['steps']:
            raise ValueError('checkpoint_step must fall strictly inside the configured fitting schedule')
        steps.append(WorkflowStep('checkpoint',compute_runner,CrosscoderFitSpec(pairs=StepRef('paired_capture'),
            reference_path=reference_path,config=config,stop_after=checkpoint_step,**common)))
        resume=StepRef('checkpoint')
    steps.extend((
        WorkflowStep('fit',compute_runner,CrosscoderFitSpec(pairs=StepRef('paired_capture'),reference_path=reference_path,
            config=config,resume_from=resume,**common)),
        WorkflowStep('dictionary',compute_runner,CrosscoderImportSpec(checkpoint=StepRef('fit'),reference_path=reference_path,**common)),
        WorkflowStep('encode',compute_runner,CrosscoderEncodeSpec(dictionary=StepRef('dictionary'),
            feature=StepRef('paired_capture').feature('pairs'),reference_path=reference_path,**common)),
        WorkflowStep('decode',compute_runner,CrosscoderDecodeSpec(dictionary=StepRef('dictionary'),
            feature=StepRef('encode').feature('codes'),reference_path=reference_path,**common)),
        WorkflowStep('splice',compute_runner,CrosscoderFeatureSpliceSpec(dictionary=StepRef('dictionary'),
            pairs=StepRef('paired_capture'),model_paths=tuple(model_paths),reference_path=reference_path,
            feature_index=feature_index,amount=amount,**common),depends_on=('decode',)),
        WorkflowStep('report',report_runner,ReportSpec(inputs=(StepRef('paired_capture'),StepRef('fit'),StepRef('encode'),StepRef('splice')))),
    ))
    return WorkflowSpec(name=name,steps=tuple(steps))


def dataset_from_config(config):
    from pipelines_v2.data.datasets import Dataset,Example
    if 'dataset' in config and 'prompts' in config:raise ValueError('Specify dataset or prompts, not both')
    if 'dataset' in config:return Dataset.from_dict(config['dataset'])
    return Dataset.from_examples(tuple(Example(key=f'prompt_{i}',prompt=prompt) for i,prompt in enumerate(config['prompts'])))


def workflow_from_config(config,dataset=None):
    allowed={'dataset','prompts','model_paths','reference_path','fit','execution','limits','train_keys','train_count',
             'layer','checkpoint_step','feature_index','amount','name','output','mode','modal'}
    if set(config)-allowed:raise ValueError('Unknown crosscoder workflow fields: '+', '.join(sorted(set(config)-allowed)))
    kwargs={k:v for k,v in config.items() if k not in {'dataset','prompts','output','mode','modal'}}
    return build_workflow(dataset=dataset if dataset is not None else dataset_from_config(config),**kwargs)


def runner_specs_from_config(config):
    from pipelines_v2.api import LocalRunnerSpec,LocalArtifactStore,FileCatalog,ModalRunnerSpec,ModalResources,ModalVolumeMount,ModalVolumeStore
    output=Path(config['output']);catalog=FileCatalog(output/'catalog')
    local=LocalRunnerSpec(artifacts=LocalArtifactStore(output/'artifacts'),catalog=catalog)
    if config.get('mode','local-scratch')=='local-scratch':return {'compute':local,'report_local':local}
    if config['mode']!='modal':raise ValueError('mode must be local-scratch or modal')
    execution=Execution(**config.get('execution',{}))
    if execution.runtime_profile!='linux-x86_64-cuda':raise ValueError('Modal requires the explicit Linux runtime profile')
    options=dict(config.get('modal',{}));volume=options.pop('artifact_volume')
    artifact_root=options.pop('artifact_root','/vol/crosscoder-artifacts')
    model_volume=options.pop('model_volume','xenon-models');model_mount=options.pop('model_mount','/models')
    limits=Limits(**config.get('limits',{}))
    mounts={model_mount:ModalVolumeMount(model_volume,model_mount)}
    for value in options.pop('volumes',()):
        mount=value if isinstance(value,ModalVolumeMount) else ModalVolumeMount.from_dict(value)
        if mount.mount_path in mounts and mount!=mounts[mount.mount_path]:
            raise ValueError('Conflicting crosscoder volume mount: '+mount.mount_path)
        mounts[mount.mount_path]=mount
    defaults={'gpu':'L4' if execution.device=='cuda' else None,'cpu':4,'memory_mb':16384,
              'timeout_seconds':int(limits.timeout_seconds),'client_timeout_seconds':int(limits.timeout_seconds)+300,
              'max_containers':1,'shard_count':1}
    resources=ModalResources(**{**defaults,**options},volumes=tuple(mounts.values()))
    remote=ModalRunnerSpec(resources=resources,artifacts=ModalVolumeStore(volume,artifact_root),catalog=catalog)
    return {'compute':remote,'report_local':local}


def run_config(config,*,plan_only=False,resume_run_id=None):
    from pipelines_v2.workflow.orchestrator import WorkflowOrchestrator
    workflow=workflow_from_config(config);specs=runner_specs_from_config(config)
    runners={name:spec.to_runner() for name,spec in specs.items()}
    orchestrator=WorkflowOrchestrator(runners=runners,workflow_catalog=specs['report_local'].catalog,max_parallelism=1)
    plan=orchestrator.plan(workflow)
    if plan_only:return {'workflow':workflow.name,'steps':[{'name':step.name,'runner':step.runner,'depends_on':list(step.depends_on),'errors':list(step.execution.errors),'warnings':list(step.execution.warnings),'valid':step.execution.valid} for step in plan.steps]}
    for step in plan.steps:
        step.execution.require_capabilities();step.execution.require_checks()
    if config.get('mode','local-scratch')=='local-scratch':
        from pipelines_v2.runtime.scratch import LocalScratchProfile
        from pipelines_v2.runtime.scratch_supervision import ScratchLimits
        if resume_run_id:raise ValueError('Use explicit checkpoint input for fitting resume; scratch whole-run recovery only reindexes promoted artifacts')
        limits=Limits(**config.get('limits',{}));execution=Execution(**config.get('execution',{}))
        if execution.device!='cpu':raise ValueError('Offline scratch is CPU-only; select Modal for CUDA')
        output=Path(config['output'])
        profile=LocalScratchProfile(model_cache=Path(config['model_paths'][0]),artifact_root=output/'artifacts',catalog_root=output/'catalog',
            limits=ScratchLimits(max_examples=limits.max_examples,max_input_tokens=limits.max_tokens_per_example,max_generated_tokens=1,
                max_artifact_bytes=limits.max_output_bytes,timeout_seconds=limits.timeout_seconds,max_process_tree_rss_bytes=limits.max_process_rss_bytes))
        result=profile.run(workflow,runner_specs=specs)
    else:
        result=orchestrator.run(workflow,resume_run_id=resume_run_id)
    summary={'run_id':result.run_id,'scientific_verdict':'not_tested','steps':{
        name:{'artifact_id':artifact.id,'provenance_status':artifact.manifest().provenance_status,'result':artifact.result()}
        for name,artifact in result.step_results.items()}}
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--plan',action='store_true');parser.add_argument('--resume-run-id')
    args=parser.parse_args();result=run_config(json.loads(args.config.read_text()),plan_only=args.plan,resume_run_id=args.resume_run_id)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
