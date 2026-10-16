"""One bounded classification/capture/control operation in the ordinary workflow graph."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar
from pipelines_v2.core.types import OperationSpec, stable_hash
from pipelines_v2.data.datasets import Dataset
from .contracts import ViTSite, PREPROCESSING
from .engine import ViTRecognitionEngine


@dataclass(frozen=True)
class RecognitionSpec(OperationSpec):
    engine: ViTRecognitionEngine | None = None
    dataset: Dataset = field(default_factory=lambda: Dataset.from_examples(()))
    site: ViTSite | None = None
    edit: ViTSite | None = None
    image_roots: tuple[str, ...] = ()
    kind: ClassVar[str] = 'recognition_microscope'

    def __post_init__(self):
        object.__setattr__(self, 'image_roots', tuple(self.image_roots))
        if self.schema_version != 1 or not isinstance(self.engine, ViTRecognitionEngine): raise ValueError('Recognition v1 requires ViTRecognitionEngine')
        if not isinstance(self.site, ViTSite) or not isinstance(self.edit, ViTSite) or self.site.read_or_write != 'read' or self.edit.read_or_write != 'write' or self.edit.layer != self.site.layer:
            raise ValueError('Explicit aligned ViT read/write coordinates required')
        if not set(self.edit.tokens) <= set(self.site.tokens) or not set(self.edit.channels) <= set(self.site.channels):
            raise ValueError('Edited coordinates must be captured for inspection')
        if self.dataset.is_deferred or not 1 <= len(self.dataset.examples) <= 4 or len({e.key for e in self.dataset.examples}) != len(self.dataset.examples):
            raise ValueError('Recognition fixture requires 1–4 unique prepared images')
        if not self.image_roots or any(not Path(p).is_absolute() for p in self.image_roots): raise ValueError('Explicit absolute image roots required')
    def validate_images(self):
        for example in self.dataset.examples:
            if example.prompt != '' or len(example.images) != 1: raise ValueError('Recognition expects one image and empty text per example')
            image = example.images[0]
            if image.color_space != 'RGB' or image.preprocessing_digest != stable_hash(PREPROCESSING): raise ValueError('Recognition processor/input mismatch')
            image.verify_at_execution(roots=tuple(Path(p) for p in self.image_roots), max_pixels=1048576)
    def required_capabilities(self): return self.engine.capabilities()
    def bound_engine(self): return self.engine
    def runtime_spec(self): return self.engine.runtime_spec()
    @classmethod
    def from_dict(cls, payload):
        p = dict(payload); p.pop('kind', None)
        p['engine'] = ViTRecognitionEngine.from_dict(p['engine']); p['dataset'] = Dataset.from_dict(p['dataset'])
        p['site'] = ViTSite.from_dict(p['site']); p['edit'] = ViTSite.from_dict(p['edit'])
        return cls(**p)
