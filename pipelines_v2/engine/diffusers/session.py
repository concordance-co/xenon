"""Narrow offline SDXL base/Euler session; explicit shared noise across arms."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import time
import uuid
from pipelines_v2.runtime.qualification import RSSSampler
from typing import Any
from pipelines_v2.engine.diffusers.hooks import ADAPTER_VERSION, internal_hooks
from pipelines_v2.engine.diffusers.assets import verify_checkpoint
from pipelines_v2.storage.image_features import tensor_content_digest


def array(tensor):
    return tensor.detach().cpu().contiguous().numpy().copy()


def tensor_digest(tensor):
    return tensor_content_digest(array(tensor))


@dataclass(frozen=True)
class ImageRequest:
    key: str
    prompt: str
    negative_prompt: str = ''
    seed: int = 0
    width: int = 1024
    height: int = 1024
    steps: int = 20
    guidance_scale: float = 5.0

    def __post_init__(self):
        import math
        if not isinstance(self.key,str) or not self.key or not isinstance(self.prompt,str) or not isinstance(self.negative_prompt,str):raise ValueError('Explicit key and aligned prompt strings required')
        if type(self.seed) is not int or not 0<=self.seed<2**63:raise ValueError('Seed must be nonnegative int64')
        if any(type(x) is not int or x<64 or x>1024 or x%64 for x in (self.width,self.height)):raise ValueError('Initial SDXL adapter supports dimensions 64..1024 divisible by64')
        if type(self.steps) is not int or not 1<=self.steps<=50:raise ValueError('Steps must be1..50')
        if type(self.guidance_scale) not in (int,float) or not math.isfinite(self.guidance_scale) or self.guidance_scale<0:raise ValueError('Guidance must be finite and nonnegative')


class SDXLSession:
    def __init__(self, checkpoint, *, device='cuda'):
        import torch
        from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
        from diffusers.models.attention_processor import AttnProcessor2_0
        if device!='cuda' or not torch.cuda.is_available():raise ValueError('SDXL generation requires the admitted Modal CUDA worker')
        self.components=verify_checkpoint(checkpoint)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True)
        torch.cuda.reset_peak_memory_stats()
        started=time.monotonic()
        with RSSSampler() as loading_rss:
            self.pipeline=StableDiffusionXLPipeline.from_pretrained(str(checkpoint),torch_dtype=torch.float16,
                variant='fp16',use_safetensors=True,local_files_only=True,add_watermarker=False).to(device)
        if type(self.pipeline.scheduler) is not EulerDiscreteScheduler:raise ValueError('Only pinned EulerDiscreteScheduler is supported')
        if self.pipeline.unet.config.time_cond_proj_dim is not None:raise ValueError('Unsupported CFG time conditioning')
        self.pipeline.unet.set_attn_processor(AttnProcessor2_0())
        for model in (self.pipeline.unet,self.pipeline.vae,self.pipeline.text_encoder,self.pipeline.text_encoder_2):
            model.requires_grad_(False);model.eval()
        self.scheduler_config=dict(self.pipeline.scheduler.config)
        self.scheduler_digest=hashlib.sha256(json.dumps(self.scheduler_config,sort_keys=True).encode()).hexdigest()
        self.device=device;self.loads=1;self.images=0;self.session_id=uuid.uuid4().hex
        from pipelines_v2.runtime.qualification import gpu_driver_version
        driver=gpu_driver_version()
        self.loading={'peak_rss_bytes':loading_rss.peak,'peak_reserved_vram_bytes':torch.cuda.max_memory_reserved(),
                      'driver_version':driver,'seconds':time.monotonic()-started,'peak_vram_bytes':torch.cuda.max_memory_allocated(),
                      'device':torch.cuda.get_device_name(),'total_vram_bytes':torch.cuda.get_device_properties(0).total_memory}
        self.pipeline.set_progress_bar_config(disable=True)

    def close(self):
        import torch
        self.pipeline=None
        torch.cuda.empty_cache()

    def noise(self, request):
        import torch
        generator=torch.Generator(device='cpu').manual_seed(request.seed)
        before=generator.get_state().clone()
        noise=torch.randn((1,4,request.height//8,request.width//8),generator=generator,dtype=torch.float32).to(torch.float16)
        return noise,{'before':before,'after':generator.get_state().clone()}

    def generate(self, request, *, noise, hooks=(), replacements=None):
        import torch
        torch.cuda.reset_peak_memory_stats()
        with RSSSampler() as execution_rss:
            result=self._generate(request,noise=noise,hooks=hooks,replacements=replacements)
        result['metadata'].update(peak_rss_bytes=execution_rss.peak,seconds=execution_rss.seconds,
            peak_vram_bytes=torch.cuda.max_memory_allocated(),peak_reserved_vram_bytes=torch.cuda.max_memory_reserved())
        return result

    def _generate(self, request, *, noise, hooks=(), replacements=None):
        import torch
        from diffusers import EulerDiscreteScheduler
        if tuple(noise.shape)!=(1,4,request.height//8,request.width//8) or noise.dtype!=torch.float16 or not torch.isfinite(noise).all():
            raise ValueError('Supply finite shared FP16 standard-normal BCHW noise at requested dimensions')
        pipe=self.pipeline
        # Scheduler history, step counter, latents and RNG are fresh for each image.
        pipe.scheduler=EulerDiscreteScheduler.from_config(self.scheduler_config)
        pipe.scheduler.set_timesteps(request.steps,device=self.device)
        timesteps=pipe.scheduler.timesteps.detach().cpu().tolist()
        generator=torch.Generator(device='cpu').manual_seed(request.seed)
        cfg=request.guidance_scale>1
        with torch.inference_mode():
            conditioning=pipe.encode_prompt(prompt=[request.prompt],prompt_2=[request.prompt],
                negative_prompt=[request.negative_prompt],negative_prompt_2=[request.negative_prompt],
                device=self.device,num_images_per_prompt=1,do_classifier_free_guidance=cfg)
            names=('prompt_embeds','negative_prompt_embeds','pooled_prompt_embeds','negative_pooled_prompt_embeds')
            conditioning_hash={name:tensor_digest(value) for name,value in zip(names,conditioning) if value is not None}
            final={}
            def after_step(pipeline,index,timestep,kwargs):
                if index==request.steps-1:final['latents']=array(kwargs['latents'])
                return kwargs
            started=time.monotonic()
            with internal_hooks(pipe.unet,hooks,batch_size=1,cfg_enabled=cfg,timesteps=timesteps,replacements=replacements) as trace:
                output=pipe(**dict(zip(names,conditioning)),height=request.height,width=request.width,
                    num_inference_steps=request.steps,guidance_scale=request.guidance_scale,
                    generator=generator,latents=noise.clone().to(self.device),output_type='np',
                    callback_on_step_end=after_step,callback_on_step_end_tensor_inputs=['latents'],
                    original_size=(request.height,request.width),target_size=(request.height,request.width),
                    crops_coords_top_left=(0,0),negative_original_size=(request.height,request.width),
                    negative_target_size=(request.height,request.width),negative_crops_coords_top_left=(0,0))
            self.images+=1
            return {'image':output.images[0],'latents':final['latents'],'captures':trace['captures'],
                'metadata':{'adapter_version':ADAPTER_VERSION,'components':self.components,
                    'initial_noise_digest':tensor_digest(noise),'conditioning':conditioning_hash,
                    'scheduler_name':'EulerDiscreteScheduler','scheduler_config':self.scheduler_config,
                    'scheduler_config_digest':self.scheduler_digest,'timesteps':timesteps,
                    'cfg_enabled':cfg,'guidance_scale':request.guidance_scale,'hook_events':trace['events'],
                    'loads':self.loads,'images':self.images,'session_id':self.session_id,
                    'seconds':time.monotonic()-started,
                    'peak_vram_bytes':torch.cuda.max_memory_allocated(),'loading':self.loading,
                    'numerics':{'dtype':'float16','attention':'AttnProcessor2_0/torch_sdpa','tf32':False,
                        'deterministic_algorithms':True,'compile':False,'offload':False,'watermark':False}}}
