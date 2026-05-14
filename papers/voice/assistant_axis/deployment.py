"""Deployment spec for the Assistant Axis service."""

from __future__ import annotations

import os

from pipelines_v2.api import (
    DeploymentConfig,
    DeploymentSpec,
    DeploymentTargetSpec,
    HTTPEndpoint,
    ModalResources,
    ModalSecret,
    ModalVolumeMount,
    PythonRuntimeSpec,
    ServiceFactory,
)
from papers.voice.assistant_axis.runtime import MODEL_VOLUME_NAME, MODEL_VOLUME_PATH
from papers.voice.assistant_axis.service_runtime import DEFAULT_TRAITS, create_app


SERVICE_GPU_SPEC = os.getenv("ASSISTANT_AXIS_SERVICE_GPU", "B200:1")
SERVICE_GPU_MEMORY_UTILIZATION = os.getenv("ASSISTANT_AXIS_SERVICE_GPU_MEMORY_UTILIZATION", "0.99")


def build_deployment() -> DeploymentSpec:
    return DeploymentSpec(
        name="assistant-axis",
        service=ServiceFactory.from_function(
            create_app,
            local_python_sources=("pipelines_v2", "papers/voice/assistant_axis"),
        ),
        runtime=PythonRuntimeSpec(
            python_version="3.13",
            pip_packages=(
                "aiohttp>=3.13.3",
                "fastapi>=0.115.0",
                "huggingface_hub>=0.36.0",
                "numpy>=1.26.0",
                "openai>=2.35.0",
                "psycopg[binary]>=3.2.0",
                "pyarrow>=23.0.1",
                "pydantic>=2.7.0",
                "safetensors>=0.4.0",
                "torch>=2.4.0",
                "transformers>=4.51.0",
                "uvicorn[standard]>=0.30.0",
                "vllm",
            ),
            env={
                "HF_HOME": f"{MODEL_VOLUME_PATH}/hf_home",
                "TRANSFORMERS_CACHE": f"{MODEL_VOLUME_PATH}/hf_home/transformers",
                "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
                "VLLM_USE_DEEP_GEMM": "0",
                "XENON_ACTIVATION_PATCH_MAX_TOKENS": os.getenv(
                    "ASSISTANT_AXIS_SERVICE_PATCH_MAX_TOKENS",
                    "128",
                ),
                "ASSISTANT_AXIS_SERVICE_GPU": SERVICE_GPU_SPEC,
                "ASSISTANT_AXIS_SERVICE_GPU_MEMORY_UTILIZATION": SERVICE_GPU_MEMORY_UTILIZATION,
            },
            local_python_sources=("pipelines_v2", "papers/voice/assistant_axis"),
        ),
        endpoints=(HTTPEndpoint.asgi(label="api"),),
        config=DeploymentConfig(values={"default_traits": list(DEFAULT_TRAITS)}),
    )


def build_deployment_targets() -> dict[str, DeploymentTargetSpec]:
    return {
        "prod": DeploymentTargetSpec.modal(
            app_name="assistant-axis-service",
            resources=ModalResources(
                gpu=SERVICE_GPU_SPEC,
                cpu=8,
                memory_mb=96 * 1024,
                timeout_seconds=60 * 60,
                max_containers=int(os.getenv("ASSISTANT_AXIS_SERVICE_MAX_CONTAINERS", "1")),
                secrets=(
                    ModalSecret.from_env_var(
                        "HF_TOKEN",
                        secret_name=os.getenv("ASSISTANT_AXIS_HF_SECRET_NAME", "huggingface"),
                    ),
                ),
                volumes=(
                    ModalVolumeMount(
                        name=MODEL_VOLUME_NAME,
                        mount_path=MODEL_VOLUME_PATH,
                        create_if_missing=True,
                    ),
                ),
            ),
            min_containers=int(os.getenv("ASSISTANT_AXIS_SERVICE_MIN_CONTAINERS", "1")),
            startup_timeout_seconds=60 * 60,
            scaledown_window_seconds=int(os.getenv("ASSISTANT_AXIS_SERVICE_SCALEDOWN_WINDOW", "900")),
        ),
    }
