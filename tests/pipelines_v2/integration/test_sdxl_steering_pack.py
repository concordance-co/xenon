"""Analytical checks for the study's primary color proxy and random control."""
import numpy as np
import pytest
from workspaces.sdxl_steering.metrics import background_mask, fit_direction, image_metrics, prompt_bootstrap
from workspaces.sdxl_steering.study import discovery_rows, held_out_rows, SPATIAL_REGIONS


def test_color_proxy_distinguishes_background_and_center_edits():
    baseline = np.full((8, 8, 3), .5)
    mask = background_mask(8, 8)
    background = baseline.copy()
    background[mask, 0] = .75
    background[mask, 2] = .25
    assert image_metrics(background, baseline)['background_warmth'] == .5
    assert image_metrics(background, baseline)['center_rgb_mae'] == 0
    center = baseline.copy()
    center[~mask, 0] = 1
    assert image_metrics(center, baseline)['background_warmth'] == 0
    assert image_metrics(center, baseline)['center_rgb_mae'] == pytest.approx(1/6)


def test_discovery_and_controls_are_disjoint_and_paired():
    discovery, heldout = discovery_rows(), held_out_rows()
    assert len(discovery) == len(heldout) == 16
    assert not {r['prompt'] for r in discovery} & {r['prompt'] for r in heldout}
    for i in range(8):
        pair = [r for r in discovery if r['pair'] == i]
        assert {r['label'] for r in pair} == {'warm', 'cool'}
        assert len({r['seed'] for r in pair}) == 1
        assert len([r for r in heldout if r['family'] == i]) == 2
    coverage = np.zeros((32, 32), dtype=int)
    for y0,y1,x0,x1 in SPATIAL_REGIONS:
        coverage[y0:y1,x0:x1] += 1
    assert np.array_equal(coverage, background_mask(32, 32))


def test_direction_is_paired_and_random_control_matches_norm():
    cool = np.zeros((1, 1280, 32, 32), dtype=np.float32)
    warm = np.broadcast_to(np.arange(1, 1281, dtype=np.float32)[None, :, None, None], cool.shape)
    result = fit_direction([(cool, warm)] * 8)
    direction, random = np.array(result['direction']), np.array(result['random_direction'])
    assert direction.shape == (1280,)
    assert np.allclose(direction / direction[0], np.arange(1, 1281))
    assert np.linalg.norm(direction) == pytest.approx(np.linalg.norm(random))
    assert result == fit_direction([(cool, warm)] * 8)


def test_prompt_bootstrap_keeps_seeds_together():
    result = prompt_bootstrap({i: [-1, 3] for i in range(8)}, draws=100)
    assert result['mean'] == 1
    assert result['ci95'] == [1, 1]


def _plumbing_direction():
    from workspaces.sdxl_steering.plumbing import SCOPE, PLUMBING_STEPS, direction_digest
    from workspaces.sdxl_steering.study import SITE, design_digest
    values = [1.0] * 1280
    return {'scope': SCOPE, 'design_digest': design_digest(), 'site': SITE,
            'steps': list(PLUMBING_STEPS), 'training_pairs': 1,
            'direction': values, 'direction_digest': direction_digest(values)}


def test_plumbing_workflows_keep_six_images_shared_noise_and_strong_signed_edits():
    from workspaces.sdxl_steering.phase_00.workflow import build_workflow as discovery
    from workspaces.sdxl_steering.phase_01.workflow import build_workflow as steering
    first = discovery()
    second = steering(direction_payload=_plumbing_direction(), full_gut_check=True)
    all_steps = (*first.steps, *second.steps)
    assert sum(len(getattr(s.spec, 'images', ())) for s in all_steps) == 6
    assert first.steps[1].spec.noise_from.step == 'cool'
    assert first.steps[0].spec.hooks[0].steps == (8, 9, 10, 11, 12)
    arms = second.steps[1:4]
    assert [s.spec.hooks[0].strength for s in arms] == [4, 12, -4]
    for step in arms:
        assert step.spec.noise_from.step == 'baseline'
        assert step.spec.images == second.steps[0].spec.images
        assert len(step.spec.hooks) == 4
        assert all(h.steps == (8, 9, 10, 11, 12) for h in step.spec.hooks)


def test_plumbing_direction_rejects_mutation():
    from workspaces.sdxl_steering.plumbing import validate_direction
    payload = _plumbing_direction()
    assert len(validate_direction(payload)) == 1280
    payload['direction'][0] = 2.0
    with pytest.raises(ValueError, match='digest mismatch'):
        validate_direction(payload)


def test_plumbing_capture_to_direction_to_export(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from workspaces.sdxl_steering import plumbing
    values = np.arange(1, 1281, dtype=np.float32)[None, :, None, None]
    class Child:
        def __init__(self, name): self.id = name
        def manifest(self): return SimpleNamespace(metadata={'initial_noise_digest': 'same'})
        def feature(self, name):
            value = np.broadcast_to(values if self.id == 'warm' else np.ones_like(values), (1, 1280, 32, 32))
            return SimpleNamespace(load=lambda: {'records': {f'hook_0_step_{step}': {'values': value} for step in plumbing.PLUMBING_STEPS}})
    monkeypatch.setattr(plumbing, 'diffusion_image', lambda a, k: a)
    from pipelines_v2.api import TransformSpec
    from pipelines_v2.operations.execution.derive import run_transform
    payload = run_transform(TransformSpec(builder=plumbing.builder(plumbing.derive_pair_step),
        inputs={'cool': Child('cool'), 'warm': Child('warm')})).payload
    direction = plumbing.validate_direction(payload)
    assert direction[0] == 0
    assert np.sqrt(np.mean(np.square(direction))) == pytest.approx(payload['activation_rms'])
    monkeypatch.setenv('XENON_SDXL_OUTPUT', str(tmp_path))
    result = run_transform(TransformSpec(builder=plumbing.builder(plumbing.export_direction_step),
        inputs={'direction': SimpleNamespace(id='direction-artifact', result=lambda: payload)})).payload
    assert (tmp_path/'direction.json').exists()
    assert result['direction_artifact_id'] == 'direction-artifact'


def test_plumbing_remote_summary_and_local_gallery(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from workspaces.sdxl_steering import plumbing
    class Child:
        def __init__(self, index): self.id = str(index); self.index = index
        def manifest(self):
            return SimpleNamespace(metadata={'initial_noise_digest': 'same', 'hook_events': [{'changed_cells': self.index}]})
        def feature(self, name):
            pixels = np.full((8, 8, 3), .5, dtype=np.float32)
            pixels[..., 0] += self.index * .1
            return SimpleNamespace(load=lambda: {'records': {plumbing.KEY: {'values': pixels}}})
    monkeypatch.setattr(plumbing, 'diffusion_image', lambda a, k: a)
    from pipelines_v2.api import TransformSpec
    from pipelines_v2.operations.execution.derive import run_transform
    result = run_transform(TransformSpec(builder=plumbing.builder(plumbing.compare_images_step),
        inputs={'baseline': Child(0), 'warm4': Child(1), 'warm12': Child(2), 'reverse4': Child(3)})).payload
    assert result['checks'] == {'same_persisted_noise': True, 'tensor_changed': True, 'image_changed': True}
    assert [c['arm'] for c in result['cells']] == ['baseline', 'warm4', 'warm12', 'reverse4']
    monkeypatch.setenv('XENON_SDXL_OUTPUT', str(tmp_path))
    rendered = run_transform(TransformSpec(builder=plumbing.builder(plumbing.render_gallery_step),
        inputs={'summary': SimpleNamespace(id='metrics-artifact', result=lambda: result)})).payload
    assert (tmp_path/'gallery.html').read_text().count('<section>') == 4
    assert (tmp_path/'metrics.json').exists()
    assert rendered['image_count'] == 4
