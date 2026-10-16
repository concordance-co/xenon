"""Optional pinned SDXL generation and internal denoising hooks (lazy dependencies)."""
from pipelines_v2.engine.diffusers.hooks import DenoisingHook
from pipelines_v2.engine.diffusers.session import ImageRequest, SDXLSession
__all__=['DenoisingHook','ImageRequest','SDXLSession']
