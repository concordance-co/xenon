"""Explicit local synthetic-image preparation; no network/model download."""
import argparse
import json
from pathlib import Path
from pipelines_v2.data.images import prepare_image
from pipelines_v2.engine.recognition.contracts import PREPROCESSING


def prepare(root):
    import numpy as np
    from PIL import Image
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    y, x = np.indices((128,192))
    arrays = {'gradient':np.stack((x*255//191,y*255//127,(x+y)*255//318),-1).astype('uint8'),
        'checker':np.repeat((((x//16+y//16)%2)*255).astype('uint8')[...,None],3,axis=2)}
    descriptors = []
    for key, values in arrays.items():
        path=root/(key+'.png'); Image.fromarray(values).save(path)
        descriptors.append(prepare_image(path, cache_root=root/'images', preprocessing=PREPROCESSING,image_id=key).to_dict())
    (root/'inputs.json').write_text(json.dumps(descriptors,indent=2)+'\n')
    return root/'inputs.json'


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',required=True)
    print(prepare(p.parse_args().root))
