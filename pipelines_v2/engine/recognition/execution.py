"""Bounded model-backed operation with selected capture and obvious controls."""
from pathlib import Path
import io
from pipelines_v2.operations.execution.common import OperationExecutionResult
from pipelines_v2.storage.image_features import SpatialTensor
from .contracts import PREPROCESSING, MODEL_IDENTITY, token_geometry


def run_recognition(spec):
    import numpy as np
    import torch
    from PIL import Image
    spec.validate_images()
    rows, images = [], {}
    with spec.engine.open_session() as session:
        for example in spec.dataset.examples:
            image = example.images[0]
            pixels = session.preprocess(image, roots=spec.image_roots)
            baseline, values = session.forward(pixels, spec.site)
            identity, identity_values = session.forward(pixels, spec.site, edit=spec.edit, scale=1.)
            edited, edited_values = session.forward(pixels, spec.site, edit=spec.edit, scale=0.)
            if not torch.equal(baseline, identity) or not torch.equal(values, identity_values):
                raise ValueError('Recognition identity control differs from baseline')
            if any(not torch.isfinite(value).all() for value in (baseline, values, edited, edited_values)):
                raise ValueError('Recognition returned nonfinite values')
            predicted = int(baseline.argmax(-1))
            raw = image.verify_at_execution(roots=tuple(Path(p) for p in spec.image_roots), max_pixels=1048576)
            with Image.open(io.BytesIO(raw)) as original: array = np.array(original)
            descriptor = SpatialTensor.from_array(array, kind='image', axes=('y','x','channel'),
                preprocessing_digest=image.preprocessing_digest, origin_ref={'content_digest':image.content_digest},
                color_space='RGB', value_range=(0,255))
            images[example.key] = {'descriptor':descriptor.to_dict(), 'values':array}
            rows.append({'example_key':example.key, 'input':image.semantic_dict(), 'image_id':image.image_id,
                'geometry':[token_geometry(token, image.shape[:2]) for token in spec.site.tokens],
                'capture':values[0].tolist(), 'edited_capture':edited_values[0].tolist(),
                'baseline_logits':baseline[0].tolist(), 'edited_logits':edited[0].tolist(),
                'predicted_class':predicted, 'predicted_label':session.model.config.id2label[predicted],
                'edited_predicted_class':int(edited.argmax(-1)), 'identity_max_error':0.,
                'max_logit_change':float((baseline-edited).abs().max()),
                'baseline_class_logit_delta':float(edited[0,predicted]-baseline[0,predicted])})
    return OperationExecutionResult(payload={'schema_version':1,'kind':'recognition_result',
        'summary':{'image_count':len(rows),'identity_exact':True,'evidence_level':'engineering','scientific_verdict':'not_tested'},
        'model_identity':MODEL_IDENTITY,'preprocessing':PREPROCESSING,'site':spec.site.to_dict(),'edit':spec.edit.to_dict(),
        'operator':{'mode':'scale','identity':1.,'edit':0.,'scope':'selected block-output token/channel coordinates; downstream model recomputed'},
        'rows':rows,'limitations':['Synthetic fixture images, not a natural-image evaluation.',
            'Patch locations identify original embedding patches; attention makes later receptive fields global.',
            'No optimized visualizations, heldout response claim, training, image generation, or scientific interpretation.']},
        features={'input_images':{'schema_version':1,'kind':'image','records':images}},
        example_coverage={'example_count':len(rows),'example_keys':[r['example_key'] for r in rows]},
        metadata={'backend':'vit_recognition','adapter_version':spec.engine.adapter_version,'resume_semantics':'restart-operation'})
