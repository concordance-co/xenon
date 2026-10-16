"""Two-image capture/derive plumbing gate. The larger research design is not run."""
from pipelines_v2.api import (SDXLEngine, ImageRequest, DenoisingHook, DiffusionCaptureSpec,
    StepRef, WorkflowSpec, WorkflowStep, TransformSpec, ReportSpec)
from workspaces.sdxl_steering.study import discovery_rows, SITE, STEP, SHAPE
from workspaces.sdxl_steering.plumbing import (
    KEY, PLUMBING_STEPS, builder, derive_pair_step, export_direction_step, output_root, build_runner_specs)


def build_workflow(dataset=None):
    import os
    from pathlib import Path
    from pipelines_v2.api import FileCatalog, OperationArtifact
    previous_run = os.environ.get('XENON_SDXL_DISCOVERY_RUN')
    if previous_run:
        catalog = FileCatalog(Path.home()/'.xenon/pipelines_v2/catalog')
        records = {record.step_name: record for record in catalog.list_workflow_steps(previous_run)}
        inputs = {}
        for name in ('cool', 'warm'):
            record = records.get(name)
            if record is None or record.status not in {'completed', 'reused'} or not record.artifact_id:
                raise ValueError(f'Prior discovery run has no completed {name} capture')
            manifest = catalog.load_artifact(record.artifact_id)
            if manifest is None or manifest.artifact_kind != 'diffusion_capture':
                raise ValueError(f'Prior discovery run has no cataloged {name} capture')
            inputs[name] = OperationArtifact(manifest, build_runner_specs()['images'].artifacts)
        return WorkflowSpec(name='sdxl-plumbing-direction-from-captures', steps=(
            WorkflowStep('direction', 'analysis', TransformSpec(builder=builder(derive_pair_step), inputs=inputs)),
            WorkflowStep('export', 'report_local', TransformSpec(builder=builder(export_direction_step), inputs={'direction': StepRef('direction')})),
            WorkflowStep('report', 'report_local', ReportSpec(inputs=(StepRef('export'),), output_dir=str(output_root()/'discovery-report'))),
        ))
    cool, warm = discovery_rows()[:2]
    engine = SDXLEngine()
    hook = DenoisingHook(SITE, PLUMBING_STEPS, SHAPE, branch='conditional')
    def request(row):
        return ImageRequest(KEY, row['prompt'], seed=row['seed'])
    return WorkflowSpec(name='sdxl-plumbing-discovery', steps=(
        WorkflowStep('cool', 'images', DiffusionCaptureSpec(engine=engine, images=(request(cool),), hooks=(hook,))),
        WorkflowStep('warm', 'images', DiffusionCaptureSpec(engine=engine, images=(request(warm),), hooks=(hook,), noise_from=StepRef('cool'))),
        WorkflowStep('direction', 'analysis', TransformSpec(builder=builder(derive_pair_step), inputs={'cool': StepRef('cool'), 'warm': StepRef('warm')})),
        WorkflowStep('export', 'report_local', TransformSpec(builder=builder(export_direction_step), inputs={'direction': StepRef('direction')})),
        WorkflowStep('report', 'report_local', ReportSpec(inputs=(StepRef('export'),), output_dir=str(output_root()/'discovery-report'))),
    ))


def build_dataset():
    from pipelines_v2.api import Dataset, Example
    return Dataset.from_examples(tuple(Example(key=row['key'], prompt=row['prompt']) for row in discovery_rows()[:2]))
