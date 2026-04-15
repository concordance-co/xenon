from __future__ import annotations

"""Minimal Modal workflow to inspect actual Qwen3 router layer capture."""

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    Example,
    GenerationSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalVolumeMount,
    ModalVolumeStore,
    MoERoutingSite,
    RoutingRecord,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)


MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"


def build_dataset() -> Dataset:
    return Dataset.from_examples(
        [
            Example(
                key="router_probe_buy_large",
                prompt=(
                    "SYSTEM\nChoose exactly one action.\n\n"
                    "STRATEGY\nBuy ALPHA immediately.\n\n"
                    "SETTINGS\nUse the largest size.\n"
                ),
            ),
            Example(
                key="router_probe_sell_small",
                prompt=(
                    "SYSTEM\nChoose exactly one action.\n\n"
                    "STRATEGY\nSell BETA immediately.\n\n"
                    "SETTINGS\nUse the smallest size.\n"
                ),
            ),
            Example(
                key="router_probe_hold_medium",
                prompt=(
                    "SYSTEM\nChoose exactly one action.\n\n"
                    "STRATEGY\nHold GAMMA for now.\n\n"
                    "SETTINGS\nUse a medium size.\n"
                ),
            ),
        ],
        name="router_layer_probe",
    )


def build_runner_specs() -> dict[str, object]:
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="A100-80GB",
                volumes=(ModalVolumeMount(name=MODEL_VOLUME_NAME, mount_path=MODEL_VOLUME_PATH),),
            ),
            artifacts=ModalVolumeStore(
                name="xenon-data",
                root="/data/artifacts/pipelines_v2_router_layer_probe",
            ),
        ),
    }


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    return WorkflowSpec(
        name="pipelines_v2_router_layer_probe",
        steps=(
            WorkflowStep(
                name="capture_router_probe",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=VLLMEngine(
                        model_id=MODEL_ID,
                        max_model_len=8192,
                        enforce_eager=False,
                        max_num_seqs=4,
                        enable_prefix_caching=False,
                    ),
                    dataset=dataset,
                    sites=[
                        MoERoutingSite(
                            name="router_probe",
                            layers=[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44],
                            record=[RoutingRecord.gate_logits(dtype="float16")],
                        )
                    ],
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=256,
                        temperature=0.0,
                    ),
                ),
            ),
        ),
    )
