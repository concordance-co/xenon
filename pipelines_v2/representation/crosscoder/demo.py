"""Execute the checked-in tiny paired workflow under bounded offline supervision."""
import argparse,json,os
from pathlib import Path


def main():
    from pipelines_v2.runtime.scratch import LocalScratchProfile
    from pipelines_v2.runtime.scratch_supervision import ScratchLimits
    parser=argparse.ArgumentParser();parser.add_argument('--models',required=True);parser.add_argument('--reference',required=True);parser.add_argument('--output',required=True);parser.add_argument('--recover')
    parser.add_argument('--workflow',type=Path,help='Path to the checked-in qwen_crosscoder workflow.py (required outside a checkout)')
    args=parser.parse_args()
    workflow=args.workflow or Path(__file__).resolve().parents[3]/'workspaces/qwen_crosscoder/phase_00/workflow.py'
    if not args.recover and not workflow.is_file():
        parser.error('Crosscoder demo needs the repository recipe; pass --workflow /path/to/xenon/workspaces/qwen_crosscoder/phase_00/workflow.py')
    models=json.loads(Path(args.models).read_text())['models']
    os.environ.update(XENON_CROSSCODER_BASE=models[0]['path'],XENON_CROSSCODER_INSTRUCT=models[1]['path'],
        XENON_CROSSCODER_REFERENCE=str(Path(args.reference).resolve()),XENON_CROSSCODER_OUTPUT=str(Path(args.output).resolve()))
    profile=LocalScratchProfile(model_cache=Path(models[0]['path']),artifact_root=Path(args.output)/'artifacts',catalog_root=Path(args.output)/'catalog',
        limits=ScratchLimits(max_examples=4,max_input_tokens=128,max_generated_tokens=1,max_artifact_bytes=64*1024*1024,
                             timeout_seconds=300,max_process_tree_rss_bytes=8*1024**3))
    result=profile.recover(args.recover) if args.recover else profile.run_file(workflow.resolve())
    summary={'run_id':result.run_id,'scientific_verdict':'not_tested','steps':{
        name:{'artifact_id':artifact.id,'provenance_status':artifact.manifest().provenance_status,'result':artifact.result()}
        for name,artifact in result.step_results.items()}}
    Path(args.output).mkdir(exist_ok=True,parents=True)
    (Path(args.output)/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'run_id':result.run_id,'steps':{name:record['artifact_id'] for name,record in summary['steps'].items()}},indent=2))

if __name__=='__main__':main()
