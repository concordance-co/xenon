from __future__ import annotations

"""Modal smoke workflow for Assistant Axis scoring on Llama 3.3 70B."""

from pathlib import Path

from pipelines_v2.api import (
    AssistantAxisPrecomputedCoordinateSpec,
    AssistantAxisScoreSpec,
    CaptureSpec,
    Dataset,
    Example,
    GenerationSpec,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    ResidualSite,
    StepRef,
    TensorStorage,
    TokenSelector,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)


MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_VOLUME_NAME = "yora-models"
MODEL_VOLUME_PATH = "/models"
ARTIFACT_VOLUME_NAME = "xenon-data"
MODAL_ARTIFACT_ROOT = "/data/artifacts/assistant_axis_llama33_70b_smoke"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "assistant_axis_llama33_70b_smoke"


def build_dataset() -> Dataset:
    return Dataset.from_examples(
        [
            Example(
                key="llama33_axis_smoke_0",
                prompt=[
                    {
                        "role": "user",
                        "content": "In two concise sentences, explain why clear assistant responses are useful.",
                    }
                ],
            )
        ],
        name="assistant_axis_llama33_70b_smoke",
    )


def _engine() -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        model_path_root=MODEL_VOLUME_PATH,
        max_model_len=2048,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.88,
        enforce_eager=False,
        max_num_seqs=1,
        enable_prefix_caching=True,
        add_generation_prompt=True,
    )


def build_runner_specs() -> dict[str, object]:
    hf_secret = ModalSecret.from_env_var("HF_TOKEN", secret_name="huggingface")
    model_mount = ModalVolumeMount(
        name=MODEL_VOLUME_NAME,
        mount_path=MODEL_VOLUME_PATH,
        create_if_missing=True,
        commit_on_success=True,
    )
    modal_store = ModalVolumeStore(name=ARTIFACT_VOLUME_NAME, root=MODAL_ARTIFACT_ROOT)
    shared_env = {
        "HF_HOME": f"{MODEL_VOLUME_PATH}/hf_home",
        "TRANSFORMERS_CACHE": f"{MODEL_VOLUME_PATH}/hf_home/transformers",
    }
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H100:2",
                timeout_seconds=60 * 60 * 2,
                max_containers=1,
                env=shared_env,
                secrets=(hf_secret,),
                volumes=(model_mount,),
            ),
            artifacts=modal_store,
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=4,
                memory_mb=16 * 1024,
                timeout_seconds=60 * 30,
                env=shared_env,
                secrets=(hf_secret,),
                volumes=(model_mount,),
            ),
            artifacts=modal_store,
        ),
        "report_local": LocalRunnerSpec(artifacts=LocalArtifactStore(LOCAL_ARTIFACT_ROOT)),
    }


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    return WorkflowSpec(
        name="assistant_axis_llama33_70b_smoke",
        steps=(
            WorkflowStep(
                name="capture_generated_residual",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=_engine(),
                    dataset=dataset,
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=24,
                        temperature=0.0,
                        top_p=1.0,
                        top_k=-1,
                        capture_generated_tokens=True,
                    ),
                    sites=(
                        ResidualSite(
                            name="residual_generated",
                            site="resid_post",
                            layers=(40,),
                            tokens=TokenSelector.section("generated"),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                    ),
                ),
            ),
            WorkflowStep(
                name="load_assistant_axis",
                runner="analysis_cpu",
                spec=AssistantAxisPrecomputedCoordinateSpec(
                    model_id=MODEL_ID,
                    token_env_var="HF_TOKEN",
                ),
            ),
            WorkflowStep(
                name="score_generated_against_axis",
                runner="analysis_cpu",
                spec=AssistantAxisScoreSpec(
                    feature=StepRef("capture_generated_residual").feature("residual_generated"),
                    axis=StepRef("load_assistant_axis"),
                    model_id=MODEL_ID,
                    summaries=("mean", "min", "max"),
                    emit_labels=True,
                ),
            ),
        ),
    )
