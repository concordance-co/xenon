"""Offline GPT-2 capture, no-op and ablation. Set XENON_GPT2_CACHE first."""
from pipelines_v2.api import (TorchEngine,Dataset,Example,CaptureSpec,GenerationRunSpec,GenerationSpec,PatchedGenerationSpec,
    ScalePatch,ResidualInterventionSite,ResidualSite,TokenSelector,TensorStorage,StepRef,ReportSpec,WorkflowSpec,WorkflowStep)
import os
MODEL=os.environ['XENON_GPT2_CACHE']

def build_dataset():
    return Dataset.from_examples([Example(key='france',prompt='The capital of France is')])

def build_workflow(dataset):
    engine=TorchEngine(model_id=MODEL,revision='607a30d783dfa663caf39e06633721c8d4cfcd7e')
    generation=GenerationSpec(enabled=True,max_tokens=8,target_token_ids=(6342,3576))
    return WorkflowSpec(name='gpt2-local-residual-steering',steps=(
        WorkflowStep('capture','scratch',CaptureSpec(engine=engine,dataset=dataset,sites=(ResidualSite('block5','resid_post',(5,),tokens=TokenSelector.last(),storage=TensorStorage(dtype='float32')),))),
        WorkflowStep('baseline','scratch',GenerationRunSpec(engine=engine,dataset=dataset,generation=generation)),
        WorkflowStep('noop','scratch',PatchedGenerationSpec(engine=engine,dataset=dataset,generation=generation,patch=ScalePatch(write_site=ResidualInterventionSite('resid_post',(5,)),scale=1.))),
        WorkflowStep('ablation','scratch',PatchedGenerationSpec(engine=engine,dataset=dataset,generation=generation,patch=ScalePatch(write_site=ResidualInterventionSite('resid_post',(5,)),scale=0.))),
        WorkflowStep('report','scratch',ReportSpec(inputs=(StepRef('baseline'),StepRef('noop'),StepRef('ablation')))),
    ))
