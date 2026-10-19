"""Explicit image proxies and prompt-cluster uncertainty; no human quality claim."""
from __future__ import annotations
import numpy as np


def background_mask(height, width):
    mask = np.ones((height, width), dtype=bool)
    mask[height//4:3*height//4, width//4:3*width//4] = False
    return mask


def image_metrics(image, baseline=None):
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 4:
        raise ValueError('Expected spatial RGB image')
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError('Image must contain finite RGB values in [0,1]')
    mask = background_mask(*image.shape[:2])
    red_blue = image[..., 0] - image[..., 2]
    edge_energy = (np.abs(np.diff(image, axis=0)).mean() + np.abs(np.diff(image, axis=1)).mean()) / 2
    result = {'background_warmth': float(red_blue[mask].mean()),
              'center_warmth': float(red_blue[~mask].mean()),
              'edge_energy': float(edge_energy),
              'near_black_fraction': float((image.max(axis=2) < .01).mean()),
              'near_white_fraction': float((image.min(axis=2) > .99).mean()),
              'channel_clipping_fraction': float(((image < .001) | (image > .999)).mean()),
              'prompt_adherence': None, 'human_quality': None}
    if baseline is not None:
        baseline = np.asarray(baseline, dtype=np.float64)
        if baseline.shape != image.shape or not np.isfinite(baseline).all():
            raise ValueError('Baseline must be finite and aligned')
        delta = np.abs(image-baseline)
        result.update(center_rgb_mae=float(delta[~mask].mean()),
                      background_rgb_mae=float(delta[mask].mean()),
                      max_pixel_difference=float(delta.max()))
    return result


def fit_direction(pairs):
    """Paired BCHW tensors from discovery only; spatial means retain channels."""
    from .study import RANDOM_DIRECTION_SEED, design_digest
    deltas, mean_squares = [], []
    for cool, warm in pairs:
        cool, warm = np.asarray(cool, dtype=np.float64), np.asarray(warm, dtype=np.float64)
        if cool.shape != (1, 1280, 32, 32) or warm.shape != cool.shape:
            raise ValueError('Expected aligned conditional BCHW tensors at the frozen site')
        if not np.isfinite(cool).all() or not np.isfinite(warm).all():
            raise ValueError('Nonfinite discovery activation')
        mask = background_mask(32, 32)
        deltas.append((warm-cool)[0, :, mask].mean(axis=0))
        mean_squares.extend([float((cool[0, :, mask]**2).mean()), float((warm[0, :, mask]**2).mean())])
    if len(deltas) != 8:
        raise ValueError('The frozen discovery set requires all eight matched pairs')
    direction = np.mean(deltas, axis=0)
    rms = float(np.sqrt(np.mean(mean_squares)))
    direction_rms = float(np.sqrt(np.mean(direction**2)))
    if direction.shape != (1280,) or direction_rms <= 1e-12 or rms <= 1e-12:
        raise ValueError('Degenerate direction; do not substitute a random target')
    direction = direction / direction_rms * rms
    random = np.random.default_rng(RANDOM_DIRECTION_SEED).standard_normal(1280)
    random = random / np.sqrt(np.mean(random**2)) * rms
    return {'schema_version': 1, 'design_digest': design_digest(), 'direction': direction.tolist(),
            'random_direction': random.tolist(), 'activation_rms': rms,
            'paired_difference_rms': direction_rms, 'training_pairs': 8}


def prompt_bootstrap(values_by_family, *, draws=10000):
    from .study import BOOTSTRAP_SEED
    means = np.asarray([np.mean(values_by_family[k]) for k in sorted(values_by_family)], dtype=float)
    if len(means) != 8 or not np.isfinite(means).all():
        raise ValueError('Require all eight prompt families for confirmatory uncertainty')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = means[rng.integers(0, len(means), size=(draws, len(means)))].mean(axis=1)
    return {'mean': float(means.mean()), 'ci95': np.quantile(samples, [.025, .975]).tolist(),
            'prompt_families': len(means), 'bootstrap_draws': draws}
