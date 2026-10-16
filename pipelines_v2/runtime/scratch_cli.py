"""Thin CLI adapter. Workflow source executes only inside the offline child."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .scratch import LocalScratchProfile
from .scratch_supervision import ScratchLimits


def run_scratch_cli(*, file, limits_file=None, artifact_root=None, catalog_root=None, model_cache=None, run_id=None):
    limits = ScratchLimits.from_dict(json.loads(Path(limits_file).read_text())) if limits_file else ScratchLimits()
    profile = LocalScratchProfile(limits=limits, artifact_root=artifact_root, catalog_root=catalog_root, model_cache=model_cache)
    result = profile.run_file(file,run_id=run_id)
    print(json.dumps({'run_id':result.run_id,'workflow_hash':result.workflow_hash,
                      'execution_profile':profile.to_dict(),
                      'steps':{name:{'artifact_id':value.id,'artifact_kind':value.manifest().artifact_kind}
                               for name,value in result.step_results.items()}},indent=2))
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description='Run an existing WorkflowSpec with enforced offline scratch limits')
    parser.add_argument('--file',required=True)
    parser.add_argument('--limits-file')
    parser.add_argument('--artifact-root')
    parser.add_argument('--catalog-root')
    parser.add_argument('--model-cache')
    parser.add_argument('--run-id')
    return run_scratch_cli(**vars(parser.parse_args(argv)))


if __name__ == '__main__':
    raise SystemExit(main())
