"""Frozen T17 design. Held-out data must not select the site or strength."""
from __future__ import annotations

import hashlib
import json

MODEL_REVISION = '462165984030d82259a11f4367a4eed129e94a7b'
SITE = 'mid_block.resnets.1'
STEP = 10
SHAPE = (1280, 32, 32)
STRENGTHS = (0.5, 1.0, 2.0)
HELD_OUT_SEEDS = (1001, 1002)
RANDOM_DIRECTION_SEED = 1729
BOOTSTRAP_SEED = 2718
# Four disjoint rectangles exclude the central 50% width/height region.
SPATIAL_REGIONS = ((0, 8, 0, 32), (24, 32, 0, 32), (8, 24, 0, 8), (8, 24, 24, 32))
DISCOVERY_OBJECTS = (
    'white ceramic mug', 'silver alarm clock', 'gray stone sphere', 'white porcelain bowl',
    'silver metal watering can', 'white plaster bust', 'gray fabric cube', 'white enamel kettle',
)
HELD_OUT_DESCRIPTIONS = (
    'A centered silver lantern resting on a shelf, an out-of-focus interior behind it.',
    'A white origami crane on a low pedestal, photographed in a softly lit room.',
    'A single gray seashell displayed on a small stand in front of a distant wall.',
    'A white ceramic vase on a console table, with a plain interior behind it.',
    'A silver bicycle bell sitting on a narrow ledge, against a blurred room.',
    'A small white toy sailboat centered on a plinth, an empty wall in the distance.',
    'A gray stone elephant figurine on a shelf, with softly blurred decor behind it.',
    'A white folded towel on a stool, photographed against a distant plain wall.',
)


def discovery_rows():
    return tuple({'key': f'discovery_{i}_{label}', 'pair': i, 'label': label,
                  'seed': i, 'prompt': f'Studio photograph of a centered {obj} on a table, '
                  f'a plain {color} backdrop, neutral white illumination on the object.'}
                 for i, obj in enumerate(DISCOVERY_OBJECTS)
                 for label, color in [('cool', 'cool blue'), ('warm', 'warm orange')])


def held_out_rows():
    return tuple({'key': f'heldout_{i}_seed{seed}', 'family': i, 'seed': seed,
                  'prompt': prompt + ' Natural photographic detail, neutral illumination.'}
                 for i, prompt in enumerate(HELD_OUT_DESCRIPTIONS) for seed in HELD_OUT_SEEDS)


def design():
    return {'schema_version': 1, 'model_revision': MODEL_REVISION,
            'site': SITE, 'step': STEP, 'shape': SHAPE, 'branch': 'conditional',
            'spatial_regions': SPATIAL_REGIONS, 'strengths': STRENGTHS,
            'direction_normalization': 'unit channel RMS times discovery activation RMS',
            'random_seed': RANDOM_DIRECTION_SEED, 'bootstrap_seed': BOOTSTRAP_SEED,
            'discovery': discovery_rows(), 'held_out': held_out_rows(),
            'primary_metric': 'mean sRGB(red-blue), outside central half-width/half-height rectangle',
            'primary_strength': 1.0, 'bootstrap_unit': 'prompt family; both seeds averaged first',
            'steps': 20, 'guidance_scale': 5.0, 'height': 1024, 'width': 1024,
            'limits': {'discovery_images': 16, 'held_out_images': 128,
                       'gpu_seconds': 7200, 'max_workers': 1}}


def design_digest():
    return hashlib.sha256(json.dumps(design(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
