"""Six-image engineering proof; deliberately not the frozen research study."""
from __future__ import annotations
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import numpy as np
from pipelines_v2.api import (ModalRunnerSpec, ModalResources, ModalVolumeMount,
    ModalVolumeStore, LocalRunnerSpec, LocalArtifactStore, TransformBuilder)
from pipelines_v2.engine.diffusers.execution import diffusion_image
from .metrics import background_mask, image_metrics
from .study import SITE, STEP, SHAPE, design_digest

SCOPE = 'aggressive plumbing gut check only; one discovery pair, no research-effect claim'
PLUMBING_STEPS = tuple(range(8, 13))
KEY = 'plumbing_scene'


def output_root():
    return Path(os.environ.get('XENON_SDXL_OUTPUT', str(Path.home()/'.xenon/reports/sdxl-plumbing'))).resolve()


def build_runner_specs():
    store = ModalVolumeStore(name='xenon-data', root='/data/sdxl-plumbing')
    return {
        'images': ModalRunnerSpec(artifacts=store, resources=ModalResources(
            gpu='A100-80GB', cpu=8, memory_mb=65536, timeout_seconds=900,
            client_timeout_seconds=1200, readiness_timeout_seconds=480,
            max_containers=1, shard_count=1, enable_workflow_batching=True,
            volumes=(ModalVolumeMount('xenon-models', '/models'),))),
        'analysis': ModalRunnerSpec(artifacts=store, resources=ModalResources(
            cpu=4, memory_mb=16384, timeout_seconds=300, client_timeout_seconds=600,
            max_containers=1, shard_count=1)),
        'report_local': LocalRunnerSpec(artifacts=LocalArtifactStore(output_root()/'artifacts')),
    }


def builder(function):
    return TransformBuilder.from_function(function, local_python_sources=('pipelines_v2', 'workspaces/sdxl_steering'))


def derive_pair_step(cool, warm):
    return {'payload': derive_pair(cool, warm)}


def export_direction_step(direction):
    return {'payload': export_direction(direction)}


def compare_images_step(baseline, warm12, warm4=None, reverse4=None):
    return {'payload': compare_images(baseline, warm12, warm4, reverse4)}


def render_gallery_step(summary):
    return {'payload': render_gallery(summary)}


def derive_pair(cool, warm):
    children = [diffusion_image(artifact, KEY) for artifact in (cool, warm)]
    digests = [child.manifest().metadata['initial_noise_digest'] for child in children]
    if len(set(digests)) != 1:
        raise ValueError('Discovery pair does not share persisted initial noise')
    records = [child.feature('latents').load()['records'] for child in children]
    arrays = [np.stack([np.asarray(record[f'hook_0_step_{step}']['values'], dtype=np.float64) for step in PLUMBING_STEPS]) for record in records]
    if any(a.shape != (len(PLUMBING_STEPS), 1, *SHAPE) or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Invalid discovery capture shape or values')
    mask = background_mask(32, 32)
    direction = np.mean([delta[0, :, mask].mean(axis=0) for delta in arrays[1]-arrays[0]], axis=0)
    rms = float(np.sqrt(np.mean([np.mean(a[:, 0, :, mask]**2) for a in arrays])))
    delta_rms = float(np.sqrt(np.mean(direction**2)))
    if delta_rms <= 1e-12 or rms <= 1e-12:
        raise ValueError('Degenerate captured direction; cannot verify learned-direction path')
    direction = direction / delta_rms * rms
    result = {'schema_version': 1, 'scope': SCOPE, 'design_digest': design_digest(),
        'site': SITE, 'steps': list(PLUMBING_STEPS), 'direction': direction.tolist(), 'activation_rms': rms,
        'normalization': 'unit channel RMS times pair activation RMS', 'training_pairs': 1,
        'source_artifact_ids': [a.id for a in (cool, warm)],
        'source_image_ids': [a.id for a in children], 'initial_noise_digest': digests[0]}
    result['direction_digest'] = direction_digest(result['direction'])
    return result


def direction_digest(direction):
    return hashlib.sha256(json.dumps(direction, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_direction(payload):
    direction = np.asarray(payload['direction'], dtype=float)
    if payload.get('scope') != SCOPE or payload.get('design_digest') != design_digest():
        raise ValueError('Direction is not from this plumbing design')
    if direction.shape != (1280,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ValueError('Invalid channel direction')
    if payload.get('direction_digest') != direction_digest(payload['direction']):
        raise ValueError('Direction digest mismatch')
    if payload.get('site') != SITE or payload.get('steps') != list(PLUMBING_STEPS) or payload.get('training_pairs') != 1:
        raise ValueError('Direction coordinates mismatch')
    return tuple(float(x) for x in direction)


def export_direction(direction):
    payload = direction.result()
    validate_direction(payload)
    root = output_root(); root.mkdir(parents=True, exist_ok=True)
    path = root/'direction.json'
    path.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    return {'scope': SCOPE, 'direction_path': str(path), 'direction_artifact_id': direction.id,
            'direction_digest': payload['direction_digest']}


def compare_images(baseline, warm12, warm4=None, reverse4=None):
    from PIL import Image
    arms = [('baseline', baseline)]
    if warm4 is not None: arms.append(('warm4', warm4))
    arms.append(('warm12', warm12))
    if reverse4 is not None: arms.append(('reverse4', reverse4))
    children = [diffusion_image(artifact, KEY) for _, artifact in arms]
    metadata = [child.manifest().metadata for child in children]
    if len({meta['initial_noise_digest'] for meta in metadata}) != 1:
        raise ValueError('Intervention must use persisted baseline noise')
    arrays = [np.asarray(child.feature('images').load()['records'][KEY]['values']) for child in children]
    cells = []
    for label, child, pixels, meta in zip((name for name, _ in arms), children, arrays, metadata):
        preview = Image.fromarray((pixels.clip(0, 1)*255).round().astype('uint8'))
        preview.thumbnail((384, 384)); stream = io.BytesIO(); preview.save(stream, format='JPEG', quality=82)
        if len(stream.getvalue()) > 131072:
            raise ValueError('Preview exceeds 128 KiB cap')
        cells.append({'arm': label, 'artifact_id': child.id, 'status': 'complete',
            'metrics': image_metrics(pixels, arrays[0]), 'hook_events': meta.get('hook_events', []),
            'preview': 'data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode()})
    changed_by_arm = [sum(event.get('changed_cells', 0) for event in meta.get('hook_events', [])) for meta in metadata[1:]]
    changed = sum(changed_by_arm)
    if any(count <= 0 for count in changed_by_arm) or any(c['metrics']['max_pixel_difference'] <= 0 for c in cells[1:]):
        raise ValueError('No observed hook/output change; plumbing intervention proof failed')
    return {'schema_version': 1, 'scope': SCOPE, 'design_digest': design_digest(),
        'initial_noise_digest': metadata[0]['initial_noise_digest'], 'cells': cells,
        'changed_cells': changed, 'image_count': len(cells),
        'checks': {'same_persisted_noise': True, 'tensor_changed': True, 'image_changed': True}}


def render_gallery(summary):
    """Local presentation only: consumes bounded remote metrics and previews."""
    import html
    payload = summary.result()
    root = output_root(); root.mkdir(parents=True, exist_ok=True)
    cards = []
    titles = {'baseline': 'Baseline', 'warm4': 'Warm direction ×4',
              'warm12': 'Warm direction ×12', 'reverse4': 'Reverse direction ×4'}
    for cell in payload['cells']:
        arm = cell['arm']
        title = titles[arm]
        data = base64.b64decode(cell['preview'].split(',', 1)[1], validate=True)
        if len(data) > 131072:
            raise ValueError('Preview exceeds 128 KiB cap')
        (root/(arm+'.jpg')).write_bytes(data)
        metrics = cell['metrics']
        details = html.escape(json.dumps({k:v for k,v in cell.items() if k!='preview'}, indent=2))
        cards.append('<section><h2>'+title+'</h2><img alt="'+title+'" src="'+cell['preview']+'">'
            +'<p>Background warmth: <b>'+format(metrics['background_warmth'], '+.3f')+'</b><br>'
            +'Background pixel change: <b>'+format(metrics['background_rgb_mae'], '.3f')+'</b></p>'
            +'<details><summary>Run details</summary><pre>'+details+'</pre></details></section>')
    path = root/'gallery.html'
    path.write_text('<!doctype html><meta charset="utf-8"><title>SDXL steering gut check</title>'
        '<style>body{font:16px system-ui;margin:32px;background:#f5f4f0;color:#222}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:20px}'
        'section{background:white;padding:18px;border-radius:12px}img{width:100%;border-radius:6px}'
        'h2{font-size:18px}pre{white-space:pre-wrap;font-size:12px}summary{cursor:pointer}</style>'
        '<h1>Strong internal image steering</h1>'
        '<p>Same prompt, seed and initial noise. Direction captured from a blue/orange background pair; '
        'applied to the outer spatial region during five denoising steps.</p>'
        '<p>Warmth is mean red minus blue (−1 to +1). Pixel change is mean absolute RGB difference '
        'from baseline (0 to 1). These are image measurements, not quality scores.</p>'
        '<main>'+''.join(cards)+'</main><p>'+html.escape(SCOPE)+'</p>')
    (root/'metrics.json').write_text(json.dumps({**payload, 'cells': [{k:v for k,v in c.items() if k!='preview'} for c in payload['cells']]}, indent=2)+'\n')
    return {'scope': SCOPE, 'gallery_path': str(path), 'summary_artifact_id': summary.id,
            'checks': payload['checks'], 'image_count': payload['image_count']}
