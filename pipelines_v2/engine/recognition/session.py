"""Scoped ViT hooks preserve downstream recomputation and selected-token semantics."""
import io
from contextlib import nullcontext
from pathlib import Path
from .contracts import PREPROCESSING, ViTSite
from pipelines_v2.core.types import stable_hash


class RecognitionSession:
    def __init__(self, model, processor):
        self.model, self.processor = model.eval(), processor
        self.model.requires_grad_(False)
    def __enter__(self): return self
    def __exit__(self, *_): self.model = None; self.processor = None
    def preprocess(self, image, *, roots):
        import torch
        from PIL import Image
        if image.color_space != 'RGB' or image.preprocessing_digest != stable_hash(PREPROCESSING):
            raise ValueError('Recognition requires RGB input with pinned ViT preprocessing identity')
        raw = image.verify_at_execution(roots=tuple(Path(p) for p in roots), max_pixels=1048576)
        with Image.open(io.BytesIO(raw)) as decoded:
            pixels = self.processor(images=decoded, return_tensors='pt')['pixel_values']
        if pixels.shape != (1,3,224,224) or pixels.dtype != torch.float32 or not torch.isfinite(pixels).all():
            raise ValueError('Processor returned unexpected tensor')
        return pixels
    def forward(self, pixels, site: ViTSite, *, edit=None, scale=1., differentiable=False):
        import torch, math
        if site.read_or_write != 'read': raise ValueError('Capture requires read coordinate')
        if pixels.shape != (1,3,224,224) or pixels.dtype != torch.float32 or not torch.isfinite(pixels).all():
            raise ValueError('ViT session requires one finite 224x224 FP32 image')
        if edit is not None and (edit.read_or_write != 'write' or edit.layer != site.layer):
            raise ValueError('Edit must name a write coordinate at the captured block')
        if not math.isfinite(scale): raise ValueError('Edit scale must be finite')
        captured = {}
        def hook(module, args, output):
            # Transformers ViT layer returns a tensor; preserve exact output ABI.
            if not isinstance(output, torch.Tensor): raise TypeError('Unexpected ViT block output ABI')
            value = output
            if edit is not None:
                value = output.clone()
                for token in edit.tokens:
                    value[:,token,list(edit.channels)] = value[:,token,list(edit.channels)] * scale
            captured['values'] = value[:,list(site.tokens)][:,:,list(site.channels)]
            return value
        handle = self.model.vit.encoder.layer[site.layer].register_forward_hook(hook)
        try:
            with nullcontext() if differentiable else torch.no_grad():
                logits = self.model(pixel_values=pixels).logits
            return logits, captured['values']
        finally:
            handle.remove()
