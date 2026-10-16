"""Explicit supervised offline scratch workflow for one real SAE plumbing proof.

No dictionary training, remote calls, feature search or research-quality claim.
Prepared inputs and coordinate evidence must exist before this command starts.
"""
from pathlib import Path
import json
import os
from pipelines_v2.api import (Dataset, Example, CaptureSpec, ResidualSite, TokenSelector,
    TensorStorage, TorchEngine, WorkflowSpec, WorkflowStep, StepRef, ReportSpec)
from pipelines_v2.operations.representation.sae import SAEDictionaryImportSpec,SAEEncodeSpec,SAEDecodeSpec,SAEFeatureSpliceSpec
from pipelines_v2.representation.sae.dictionary import MODEL_REVISION
from pipelines_v2.representation.sae.execution import register
register()

PROMPTS=('The capital of France is','A blue bird flew over the')


def build_dataset():return Dataset.from_examples(tuple(Example(key=f'prompt_{i}',prompt=p) for i,p in enumerate(PROMPTS)))


def build_workflow(dataset=None):
    model=os.environ['XENON_SAE_MODEL'];cache=os.environ['XENON_SAE_CACHE'];evidence=json.loads(Path(os.environ['XENON_SAE_ALIGNMENT']).read_text())
    engine=TorchEngine(model_id=model,revision=MODEL_REVISION)
    return WorkflowSpec(name='gpt2-sae-plumbing',steps=(
        WorkflowStep('dictionary','scratch',SAEDictionaryImportSpec(cache_root=cache,model_path=model,coordinate_evidence=evidence)),
        WorkflowStep('capture','scratch',CaptureSpec(engine=engine,dataset=dataset or build_dataset(),sites=(ResidualSite(name='residual',site='resid_pre',layers=(6,),tokens=TokenSelector.full_sequence(),storage=TensorStorage(dtype='float32')),))),
        WorkflowStep('encode','scratch',SAEEncodeSpec(dictionary=StepRef('dictionary'),feature=StepRef('capture').feature('residual'))),
        WorkflowStep('decode','scratch',SAEDecodeSpec(dictionary=StepRef('dictionary'),feature=StepRef('encode').feature('codes'))),
        WorkflowStep('splice','scratch',SAEFeatureSpliceSpec(dictionary=StepRef('dictionary'),engine=engine,prompts=PROMPTS,feature_edits=((0,10.0),))),
        WorkflowStep('report','scratch',ReportSpec(inputs=(StepRef('dictionary'),StepRef('encode'),StepRef('decode'),StepRef('splice')))),
    ))


if __name__=='__main__':
    import argparse
    from pipelines_v2.runtime.scratch import LocalScratchProfile
    from pipelines_v2.runtime.scratch_supervision import ScratchLimits
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--sae',required=True);parser.add_argument('--alignment',required=True);parser.add_argument('--output',required=True);parser.add_argument('--recover')
    args=parser.parse_args()
    os.environ.update(XENON_SAE_MODEL=str(Path(args.model).resolve()),XENON_SAE_CACHE=str(Path(args.sae).resolve()),XENON_SAE_ALIGNMENT=str(Path(args.alignment).resolve()))
    profile=LocalScratchProfile(model_cache=Path(args.model),artifact_root=Path(args.output)/'artifacts',catalog_root=Path(args.output)/'catalog',
        limits=ScratchLimits(max_examples=8,max_input_tokens=128,max_generated_tokens=1,max_artifact_bytes=256*1024*1024,timeout_seconds=600,max_process_tree_rss_bytes=8*1024**3))
    result=profile.recover(args.recover) if args.recover else profile.run_file(__file__)
    outputs={name:{'artifact_id':artifact.id,'result':artifact.result() if callable(getattr(artifact,'result',None)) else {'kind':'capture','coverage':artifact.manifest().example_coverage}} for name,artifact in result.step_results.items()}
    Path(args.output).mkdir(parents=True,exist_ok=True)
    (Path(args.output)/'summary.json').write_text(json.dumps(outputs,indent=2)+'\n')
    print(json.dumps({'run_id':result.run_id,'artifacts':{name:v['artifact_id'] for name,v in outputs.items()}},indent=2))
