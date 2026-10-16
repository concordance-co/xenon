"""Private subprocess entrypoint for LocalScratchProfile; not a second scheduler."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

from pipelines_v2.operations.reports import ReportSpec
from pipelines_v2.runtime.local import LocalResources, LocalRunner
from pipelines_v2.storage.local import FileCatalog
from pipelines_v2.workflow.orchestrator import WorkflowOrchestrator
from pipelines_v2.workflow.specs import WorkflowSpec, WorkflowStep
from .scratch import LocalScratchProfile, validate_scratch_workflow
from .scratch_store import BudgetedLocalArtifactStore
from .scratch_supervision import ScratchLimitError


class ScratchWorkerRunner(LocalRunner):
    """Normal runner execution inside the process-wide supervisor and store guard."""
    def __init__(self, *, profile, artifacts, catalog):
        super().__init__(resources=LocalResources(device='cpu'), artifacts=artifacts, catalog=catalog)
        self.profile = profile

    def identity(self):
        return {**super().identity(), 'scratch_profile':self.profile.to_dict()}

    def run(self, spec, *, workflow_context=None, progress_callback=None):
        validate_scratch_workflow(WorkflowSpec(steps=(WorkflowStep('operation','scratch',spec),)),
                                  self.profile.limits,model_cache=self.profile.model_cache,check_tokens=True)
        if isinstance(spec, ReportSpec):
            # Keep all writes inside the supervised attempt. Reports are relocated
            # with the artifact and revalidated before canonical publication.
            spec = replace(spec, output_dir=str(self.artifacts.root))
        artifact = super().run(spec, workflow_context=workflow_context, progress_callback=progress_callback)
        count = artifact.manifest().example_coverage.get('example_count')
        if count is not None and count > self.profile.limits.max_examples:
            raise ScratchLimitError('Scratch produced-example count exceeds limit')
        return artifact


def main(request_path):
    request_path = Path(request_path).resolve()
    attempt = request_path.parent
    request = json.loads(request_path.read_text())
    profile = LocalScratchProfile.from_dict(request['profile'])
    if 'workflow_file' in request:
        from pipelines_v2.cli import load_python_workflow_file
        _, workflow, runner_specs = load_python_workflow_file(path=request['workflow_file'])
        validate_scratch_workflow(workflow, profile.limits, model_cache=profile.model_cache, runner_specs=runner_specs)
    else:
        workflow = WorkflowSpec.from_dict(request['workflow'])
    validate_scratch_workflow(workflow, profile.limits, model_cache=profile.model_cache, check_tokens=True)
    store = BudgetedLocalArtifactStore(attempt / 'artifacts', max_new_bytes=profile.limits.max_artifact_bytes)
    catalog = FileCatalog(attempt / 'catalog')
    runner = ScratchWorkerRunner(profile=profile, artifacts=store, catalog=catalog)
    result = WorkflowOrchestrator(runners={step.runner:runner for step in workflow.steps},
                                  workflow_catalog=catalog, max_parallelism=1).run(workflow, new_run_id=request['run_id'])
    (attempt / 'result.json').write_text(json.dumps({name:artifact.manifest().to_dict() for name,artifact in result.step_results.items()},indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
