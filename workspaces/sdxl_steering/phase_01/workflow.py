"""Four-image same-noise steering plumbing proof, consuming the saved pair direction."""
import json
import os
from pathlib import Path
from pipelines_v2.api import (SDXLEngine, ImageRequest, DenoisingHook, DiffusionGenerationSpec,
    DiffusionInterventionSpec, StepRef, WorkflowSpec, WorkflowStep, TransformSpec, ReportSpec)
from workspaces.sdxl_steering.study import SITE, STEP, SHAPE, SPATIAL_REGIONS, held_out_rows
from workspaces.sdxl_steering.plumbing import (
    KEY, PLUMBING_STEPS, builder, compare_images_step, render_gallery_step, output_root, validate_direction, build_runner_specs)


def build_workflow(dataset=None, *, direction_payload=None, full_gut_check=True):
    if direction_payload is None:
        path = Path(os.environ.get('XENON_SDXL_DIRECTION', str(output_root()/'direction.json')))
        if path.stat().st_size > 65536:
            raise ValueError('Direction export exceeds 64 KiB cap')
        direction_payload = json.loads(path.read_text())
    direction = validate_direction(direction_payload)
    row = held_out_rows()[0]
    request = ImageRequest(KEY, row['prompt'], seed=row['seed'])
    engine = SDXLEngine()
    def hooks(strength):
        return tuple(DenoisingHook(SITE, PLUMBING_STEPS, SHAPE, branch='conditional', operator='add',
            spatial_yx=rect, direction=direction, strength=strength) for rect in SPATIAL_REGIONS)
    arms = (('warm4', 4.0), ('warm12', 12.0), ('reverse4', -4.0)) if full_gut_check else (('warm12', 12.0),)
    return WorkflowSpec(name='sdxl-plumbing-steering', steps=(
        WorkflowStep('baseline', 'images', DiffusionGenerationSpec(engine=engine, images=(request,))),
        *(WorkflowStep(name, 'images', DiffusionInterventionSpec(engine=engine, images=(request,), hooks=hooks(strength), noise_from=StepRef('baseline'))) for name, strength in arms),
        WorkflowStep('metrics', 'analysis', TransformSpec(builder=builder(compare_images_step), inputs={name: StepRef(name) for name in ('baseline', *(name for name, _ in arms))})),
        WorkflowStep('gallery', 'report_local', TransformSpec(builder=builder(render_gallery_step), inputs={'summary': StepRef('metrics')})),
        WorkflowStep('report', 'report_local', ReportSpec(inputs=(StepRef('gallery'),), output_dir=str(output_root()/'steering-report'))),
    ))


def build_dataset():
    from pipelines_v2.api import Dataset, Example
    return Dataset.from_examples(tuple(Example(key=row['key'], prompt=row['prompt']) for row in held_out_rows()[:1]))
