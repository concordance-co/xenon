"""Figure 3 numerical-semantics validation for Llama 3.3 70B emotion vectors."""

from __future__ import annotations

import os
from pathlib import Path

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    Example,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalVolumeMount,
    ModalVolumeStore,
    ReportSpec,
    ResidualSite,
    StepRef,
    TensorStorage,
    TokenSelector,
    TransformBuilder,
    TransformSpec,
    TransferPolicy,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)
from papers.voice.emotions.replication.validation import (
    build_numeric_semantics_validation,
    latest_emotion_space_result_path,
    numeric_semantics_prompt_rows,
    score_numeric_semantics_feature,
)
from papers.voice.storage import (
    ARTIFACT_VOLUME_NAME,
    MODEL_VOLUME_PATH,
    YORA_MODEL_VOLUME_NAME,
    local_vector_root,
    modal_vector_root,
)


WORKFLOW_NAME = "papers_voice_emotions_llama33_70b_numeric_semantics"
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_VOLUME_NAME = YORA_MODEL_VOLUME_NAME
DEFAULT_LAYER = 56
DEFAULT_LAYERS = (40, 44, 48, 52, 56, 60, 64)
DEFAULT_MAX_MODEL_LEN = 16384
DEFAULT_MAX_NUM_SEQS = 512
DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192
EMOTIONS = ("happy", "sad", "afraid", "calm")
MODAL_ARTIFACT_ROOT = modal_vector_root(
    "emotions",
    "llama-3.3-70b",
    "sofroniew-2026",
    "numeric-semantics",
)
LOCAL_ARTIFACT_ROOT = local_vector_root(
    "emotions",
    "llama-3.3-70b",
    "sofroniew-2026",
    "numeric-semantics",
)
DEFAULT_REPORT_DIR = f"papers/voice/emotions/replication/reports/{WORKFLOW_NAME}"


def selected_layer() -> int:
    layers = selected_layers()
    return DEFAULT_LAYER if DEFAULT_LAYER in layers else int(layers[0])


def selected_layers() -> tuple[int, ...]:
    raw = os.getenv("EMOTION_NUMERIC_LAYERS", "").strip()
    if raw:
        return tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    single_layer = os.getenv("EMOTION_NUMERIC_LAYER", "").strip()
    if single_layer:
        return (int(single_layer),)
    return DEFAULT_LAYERS


def vector_space_result_path() -> str:
    override = os.getenv("EMOTION_NUMERIC_VECTOR_SPACE_PATH")
    if override:
        return override
    return str(latest_emotion_space_result_path())


def validation_output_dir() -> str:
    override = os.getenv("EMOTION_NUMERIC_OUTPUT_DIR")
    if override:
        return override
    layers = selected_layers()
    if len(layers) == 1:
        return f"{DEFAULT_REPORT_DIR}/figure3_layer{layers[0]}"
    return f"{DEFAULT_REPORT_DIR}/figure3_layers_{'_'.join(str(layer) for layer in layers)}"


def build_dataset() -> Dataset:
    return Dataset.from_examples(
        [
            Example(
                key=str(row["key"]),
                prompt=list(row["prompt"]),
                labels=dict(row["labels"]),
            )
            for row in numeric_semantics_prompt_rows()
        ],
        name=f"{WORKFLOW_NAME}_figure3_prompts",
    )


def build_workflow() -> WorkflowSpec:
    layers = selected_layers()
    dataset = build_dataset()
    return WorkflowSpec(
        name=WORKFLOW_NAME,
        steps=(
            WorkflowStep(
                name="capture_numeric_prompts",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=_engine(),
                    dataset=dataset,
                    sites=(
                        ResidualSite(
                            name="assistant_prefill_residual",
                            site="resid_post",
                            layers=layers,
                            tokens=TokenSelector.last(),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                    ),
                ),
            ),
            WorkflowStep(
                name="score_numeric",
                runner="analysis_local",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        score_numeric_semantics_feature,
                        local_python_sources=("papers",),
                    ),
                    inputs={
                        "feature": StepRef("capture_numeric_prompts").feature("assistant_prefill_residual"),
                        "vector_space_path": vector_space_result_path(),
                        "concepts": EMOTIONS,
                        "layers": layers,
                        "metric": "cosine",
                    },
                ),
            ),
            WorkflowStep(
                name="figure3_validation",
                runner="analysis_local",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        build_numeric_semantics_validation,
                        local_python_sources=("papers",),
                    ),
                    inputs={
                        "scores": StepRef("score_numeric"),
                        "output_dir": validation_output_dir(),
                    },
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=(StepRef("score_numeric"), StepRef("figure3_validation")),
                    template="voice_emotions_llama70b_numeric_semantics",
                    output_dir=os.getenv("EMOTION_NUMERIC_REPORT_DIR", DEFAULT_REPORT_DIR),
                ),
            ),
        ),
    )


def build_runner_specs() -> dict[str, object]:
    artifact_store = ModalVolumeStore(
        name=ARTIFACT_VOLUME_NAME,
        root=os.getenv("EMOTION_NUMERIC_MODAL_ARTIFACT_ROOT", MODAL_ARTIFACT_ROOT),
        transfer_policy=TransferPolicy(allow_large_transfer=True),
    )
    local_artifact_root = Path(os.getenv("EMOTION_NUMERIC_LOCAL_ARTIFACT_ROOT", str(LOCAL_ARTIFACT_ROOT)))
    local_store = LocalArtifactStore(local_artifact_root)
    model_volume = ModalVolumeMount(
        name=MODEL_VOLUME_NAME,
        mount_path=MODEL_VOLUME_PATH,
        create_if_missing=True,
        commit_on_success=True,
    )
    shared_env = {
        "HF_HOME": f"{MODEL_VOLUME_PATH}/hf_home",
        "TRANSFORMERS_CACHE": f"{MODEL_VOLUME_PATH}/hf_home/transformers",
        "VLLM_CACHE_ROOT": os.getenv("EMOTION_NUMERIC_VLLM_CACHE_ROOT", MODEL_VOLUME_PATH),
        "TORCHINDUCTOR_CACHE_DIR": f"{MODEL_VOLUME_PATH}/torch_compile_cache",
        **_workflow_env(),
    }
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu=os.getenv("EMOTION_NUMERIC_CAPTURE_GPU", "H200:2"),
                cpu=_env_int("EMOTION_NUMERIC_CAPTURE_CPU", 16),
                memory_mb=_env_int("EMOTION_NUMERIC_CAPTURE_MEMORY_MB", 128 * 1024),
                timeout_seconds=_env_int("EMOTION_NUMERIC_CAPTURE_TIMEOUT_SECONDS", 60 * 60 * 6),
                max_containers=_env_optional_int("EMOTION_NUMERIC_CAPTURE_MAX_CONTAINERS") or 1,
                shard_count=_env_optional_int("EMOTION_NUMERIC_CAPTURE_SHARD_COUNT"),
                enable_workflow_batching=_env_bool("EMOTION_NUMERIC_CAPTURE_WORKFLOW_BATCHING", True),
                env=shared_env,
                volumes=(model_volume,),
            ),
            artifacts=artifact_store,
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=_env_int("EMOTION_NUMERIC_ANALYSIS_CPU", 8),
                memory_mb=_env_int("EMOTION_NUMERIC_ANALYSIS_MEMORY_MB", 64 * 1024),
                timeout_seconds=_env_int("EMOTION_NUMERIC_ANALYSIS_TIMEOUT_SECONDS", 60 * 60 * 2),
                enable_workflow_batching=_env_bool("EMOTION_NUMERIC_ANALYSIS_WORKFLOW_BATCHING", True),
                env=shared_env,
                volumes=(model_volume,),
            ),
            artifacts=artifact_store,
        ),
        "analysis_local": LocalRunnerSpec(artifacts=local_store),
        "report_local": LocalRunnerSpec(artifacts=local_store),
    }


def _engine() -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        model_path_root=MODEL_VOLUME_PATH,
        tensor_parallel_size=_env_int("EMOTION_NUMERIC_TENSOR_PARALLEL_SIZE", 2),
        max_model_len=_env_int("EMOTION_NUMERIC_MAX_MODEL_LEN", DEFAULT_MAX_MODEL_LEN),
        gpu_memory_utilization=float(os.getenv("EMOTION_NUMERIC_GPU_MEMORY_UTILIZATION", "0.90")),
        enforce_eager=False,
        enable_prefix_caching=True,
        max_num_seqs=_env_int("EMOTION_NUMERIC_MAX_NUM_SEQS", DEFAULT_MAX_NUM_SEQS),
        max_num_batched_tokens=_env_optional_int("EMOTION_NUMERIC_MAX_NUM_BATCHED_TOKENS")
        or DEFAULT_MAX_NUM_BATCHED_TOKENS,
        enable_chunked_prefill=_env_bool("EMOTION_NUMERIC_ENABLE_CHUNKED_PREFILL", True),
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _workflow_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key.startswith("EMOTION_NUMERIC_")}
