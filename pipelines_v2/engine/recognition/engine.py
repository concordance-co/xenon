"""Offline, byte-pinned ViT classification engine; no generation interface."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
from pipelines_v2.core.provenance import content_hash
from pipelines_v2.engine.torch.engine import TorchEngine
from .contracts import ADAPTER, REVISION, REPOSITORY, FILES, PREPROCESSING


def verify_assets(root):
    root = Path(root)
    for name, expected in FILES.items():
        p = root / name
        if not p.is_file(): raise ValueError(f'Prepared ViT cache missing {name}; download separately')
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
        if h.hexdigest() != expected: raise ValueError(f'Pinned ViT bytes differ: {name}')
    return [{'role': role, 'repository_id': REPOSITORY, 'requested_revision': REVISION,
        'resolved_revision': REVISION, 'file_set_digest': content_hash({name: FILES[name] for name in names})}
        for role, names in (('model', ('config.json', 'model.safetensors')), ('processor', ('preprocessor_config.json',)), ('class_map', ('config.json',)))]


@dataclass(frozen=True)
class ViTRecognitionEngine:
    model_id: str
    revision: str = REVISION
    device: str = 'cpu'
    dtype: str = 'float32'
    attention_impl: str = 'eager'
    kind: str = 'vit_recognition'

    def __post_init__(self):
        if (self.revision,self.device,self.dtype,self.attention_impl,self.kind) != (REVISION,'cpu','float32','eager','vit_recognition'):
            raise ValueError('Recognition engine requires pinned ViT CPU FP32 eager')
    @property
    def adapter_version(self): return ADAPTER
    @property
    def architecture(self): return 'vit'
    def identity(self): return dict(asdict(self), adapter_version=ADAPTER, preprocessing=PREPROCESSING)
    semantic_identity = identity
    runtime_spec = TorchEngine.runtime_spec
    resolved_model_path = TorchEngine.resolved_model_path
    def observed_components(self): return verify_assets(self.model_id)
    def capabilities(self):
        from pipelines_v2.core.types import EngineCapability as C
        return {C.IMAGE_CLASSIFICATION, C.SPATIAL_TOKEN_CAPTURE, C.ACTIVATION_PATCHING, C.REQUEST_SCOPED_INTERVENTIONS}
    def supported_input_modalities(self): return frozenset({'image'})
    def scratch_token_counts(self, examples): return [197 for _ in examples]
    @classmethod
    def from_dict(cls, payload):
        p = dict(payload)
        if p.pop('adapter_version', ADAPTER) != ADAPTER or p.pop('preprocessing', PREPROCESSING) != PREPROCESSING:
            raise ValueError('Unsupported recognition adapter/processor')
        return cls(**p)
    def planning_errors(self, spec):
        try:
            self.observed_components()
            spec.validate_images()
        except (OSError, ValueError) as exc: return (str(exc),)
        return ()
    def open_session(self):
        from transformers import ViTImageProcessor, ViTForImageClassification
        from .session import RecognitionSession
        self.observed_components()
        processor = ViTImageProcessor.from_pretrained(self.model_id, local_files_only=True,
            do_resize=True, size={'height':224,'width':224}, resample=2, do_rescale=True,
            rescale_factor=1/255, do_normalize=True, image_mean=[.5]*3, image_std=[.5]*3)
        model = ViTForImageClassification.from_pretrained(self.model_id, local_files_only=True,
            dtype='float32', attn_implementation='eager')
        return RecognitionSession(model, processor)
