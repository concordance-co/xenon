"""Reusable LLM workflows. Inputs and budgets are explicit; preparation is separate."""
from pathlib import Path
import json


def dataset_from_config(config):
    from pipelines_v2.api import Dataset, Example
    rows=config.get('examples')
    if rows is None:
        rows=json.loads(Path(config['dataset_file']).read_text())
    if not isinstance(rows,list) or not rows: raise ValueError('Dataset must be a nonempty list of Example records')
    maximum=config.get('max_examples',10000)
    if type(maximum) is not int or maximum<1 or len(rows)>maximum: raise ValueError('Dataset exceeds max_examples')
    examples=tuple(Example.from_dict(row) for row in rows)
    if len({e.key for e in examples})!=len(examples):raise ValueError('Duplicate example keys')
    if any(not isinstance(e.prompt,str) or not e.prompt.strip() or e.images for e in examples):raise ValueError('LLM workflows require nonempty text prompts')
    return Dataset.from_examples(examples)


def runner_specs(config):
    from pipelines_v2.api import LocalRunnerSpec, LocalArtifactStore, ModalRunnerSpec, ModalResources, ModalVolumeMount, ModalVolumeStore
    # CLI fills standard local+shared catalog defaults for unset catalogs.
    local=LocalRunnerSpec(artifacts=LocalArtifactStore(config['local_artifact_root']))
    if config.get('backend','local')=='local':return {'compute':local,'report':local}
    if config['backend']!='modal':raise ValueError('backend must be local or modal')
    profile=dict(config['resources'])
    if any(not profile.get(k) for k in ('cpu','memory_mb','timeout_seconds')):raise ValueError('Modal requires explicit cpu, memory_mb and timeout_seconds')
    profile.setdefault('max_containers',1);profile.setdefault('shard_count',1)
    if profile['max_containers']!=1 or profile['shard_count']!=1:raise ValueError('These sequential workflows require one worker/shard')
    profile['volumes']=tuple(ModalVolumeMount(**m) for m in profile.get('volumes',[]))
    remote=ModalRunnerSpec(resources=ModalResources(**profile),artifacts=ModalVolumeStore(**config['artifact_store']))
    from dataclasses import replace
    preparation=replace(remote,resources=replace(remote.resources,gpu=None))
    return {'compute':remote,'prepare':preparation,'report':local}


def build_workflow(config):
    from pipelines_v2.api import (TorchEngine,NativeTorchEngine,CaptureSpec,ResidualSite,TokenSelector,TensorStorage,
        GenerationSpec,GenerationRunSpec,WorkflowSpec,WorkflowStep,StepRef,ReportSpec,
        SAEFeatureSpliceSpec,SAEDictionaryImportSpec,MAttrSpec,ExternalToolRuntimeSpec,official_tool_lock)
    from pipelines_v2.runtime.locks import load_runtime_lock,core_runtime_lock
    method=config['method'];steps=[]
    if method in {'residual','sae','native_capture'}:
        dataset=dataset_from_config(config)
        engine_config=dict(config['engine'])
        engine=(NativeTorchEngine if method=='native_capture' else TorchEngine)(**engine_config)
        if config.get('backend','local')=='local' and engine.device!='cpu':raise ValueError('Local profile is CPU; use Modal for CUDA')
        if method=='residual':
            sites=tuple(ResidualSite(name=f'layer{layer}',site=config.get('site','resid_post'),layers=(layer,),tokens=TokenSelector.last(),storage=TensorStorage(dtype='float32')) for layer in config.get('layers',[5]))
            steps.append(WorkflowStep('capture','compute',CaptureSpec(engine=engine,dataset=dataset,sites=sites)))
            steps.append(WorkflowStep('generation','compute',GenerationRunSpec(engine=engine,dataset=dataset,generation=GenerationSpec(enabled=True,max_tokens=config.get('max_new_tokens',8)))))
            report_inputs=(StepRef('generation'),)
        elif method=='native_capture':
            from pipelines_v2.operations.common.native import NativeSite
            sites=tuple(NativeSite.from_dict(s) for s in config['sites'])
            steps.append(WorkflowStep('capture','compute',CaptureSpec(engine=engine,dataset=dataset,sites=sites)))
            report_inputs=() # Bulk captures stay on their compute store.
            if config.get('native_edit') is not None:
                from pipelines_v2.api import PatchedGenerationSpec
                from pipelines_v2.operations.interventions.native import NativeCoordinatePatch,NativeInterventionSite
                edit=config['native_edit'];site=next(s for s in sites if s.name==edit['site'])
                patch=NativeCoordinatePatch(write_site=NativeInterventionSite.from_capture(site),target_tokens=site.tokens,mode=edit['mode'],value=edit['value'])
                generation=GenerationSpec(enabled=True,max_tokens=config.get('max_new_tokens',2))
                steps.extend((WorkflowStep('baseline','compute',GenerationRunSpec(engine=engine,dataset=dataset,generation=generation)),
                    WorkflowStep('intervention','compute',PatchedGenerationSpec(engine=engine,dataset=dataset,generation=generation,patch=patch))))
                report_inputs=(StepRef('baseline'),StepRef('intervention'))
        else:
            steps.append(WorkflowStep('dictionary','prepare' if config.get('backend')=='modal' else 'compute',SAEDictionaryImportSpec(**config['dictionary'])))
            steps.append(WorkflowStep('splice','compute',SAEFeatureSpliceSpec(dictionary=StepRef('dictionary'),engine=engine,dataset=dataset,
                feature_edits=tuple(tuple(v) for v in config['feature_edits']),max_examples=config.get('max_examples',10000),
                max_prompt_tokens=config.get('max_prompt_tokens',1024),verify_oracle=config.get('verify_oracle',False))))
            report_inputs=(StepRef('splice'),)
    elif method=='native_route':
        from pipelines_v2.api import NativeRouteSpec
        spec=NativeRouteSpec.from_dict(config['operation'])
        steps.append(WorkflowStep('route','compute',spec));report_inputs=(StepRef('route'),)
    elif method=='mattr':
        runtime=ExternalToolRuntimeSpec(lock=load_runtime_lock('core-linux') if config.get('backend')=='modal' else core_runtime_lock(),local_python_sources=('pipelines_v2',),tool_lock=official_tool_lock(),**config.get('tool_paths',{}))
        args=dict(config['attribution'])
        # Request can be prepared locally then paths remapped to mounted assets.
        spec=MAttrSpec(runtime=runtime,**args) if 'request' in args else MAttrSpec.from_ioi_files(runtime=runtime,**args)
        steps.append(WorkflowStep('attribution','compute',spec))
        from pipelines_v2.api import TransformSpec,TransformBuilder
        from pipelines_v2.operations.attribution.report import summarize_attribution
        steps.append(WorkflowStep('summary','prepare' if config.get('backend')=='modal' else 'compute',TransformSpec(builder=TransformBuilder.from_function(summarize_attribution,local_python_sources=('pipelines_v2',)),inputs={'attribution':StepRef('attribution')})))
        report_inputs=(StepRef('summary'),)
    else:raise ValueError('Unknown LLM method')
    if report_inputs:steps.append(WorkflowStep('report','report',ReportSpec(inputs=report_inputs)))
    return WorkflowSpec(name=config.get('name',f'llm-{method}'),steps=tuple(steps))
