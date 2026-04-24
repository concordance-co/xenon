from __future__ import annotations

import json
from pathlib import Path

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    FileCatalog,
    GenerationSpec,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalVolumeMount,
    ModalVolumeStore,
    ResidualSite,
    TensorStorage,
    TokenSelector,
    WorkflowSpec,
    WorkflowStep,
)

from projects.MOREBENCH.phase_03.specs import experiment_02_workflow as base


LOCAL_CATALOG_ROOT = Path("artifacts") / "morebench_phase03_experiment02_cross_language_prompt_probe_dense_catalog"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "morebench_phase03_experiment02_cross_language_prompt_probe_dense"
MODAL_ARTIFACT_ROOT = "/data/artifacts/morebench_phase_03_experiment02_cross_language_prompt_probe_dense"

TRANSFORM_RESULT_PATH = (
    Path("artifacts")
    / "morebench_phase03_experiment02_cross_language_prompt_probe_full"
    / "transform_33d92c1d07d0_339727c1"
    / "result.json"
)

DENSE_LAYERS = (16, 20, 22, 24, 26, 28, 32, 40)


def build_dataset() -> Dataset:
    payload = json.loads(TRANSFORM_RESULT_PATH.read_text(encoding="utf-8"))
    return Dataset.from_dict(payload["dataset"])


def build_runner_specs() -> dict[str, object]:
    catalog = FileCatalog(root=LOCAL_CATALOG_ROOT)
    modal_store = ModalVolumeStore(
        name="xenon-data",
        root=MODAL_ARTIFACT_ROOT,
    )
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H200",
                timeout_seconds=60 * 60 * 6,
                volumes=(ModalVolumeMount(name=base.MODEL_VOLUME_NAME, mount_path=base.MODEL_VOLUME_PATH),),
            ),
            artifacts=modal_store,
            catalog=catalog,
        ),
        "analysis_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(LOCAL_ARTIFACT_ROOT),
            catalog=catalog,
        ),
    }


def build_workflow() -> WorkflowSpec:
    dataset = build_dataset()
    return WorkflowSpec(
        name="morebench_phase03_experiment02_cross_language_prompt_probe_dense_capture",
        steps=(
            WorkflowStep(
                name="capture_prompt_eos_residual_dense",
                runner="capture_gpu",
                description="Capture prompt-final residuals on dense layers around the L24 dip for the translated full-30 prompt set.",
                spec=CaptureSpec(
                    engine=base._engine(max_num_seqs=8),
                    dataset=dataset,
                    sites=[
                        ResidualSite(
                            name="prompt_eos_residual",
                            site="resid_post",
                            layers=list(DENSE_LAYERS),
                            tokens=TokenSelector.last(),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        )
                    ],
                    generation=GenerationSpec(enabled=False),
                ),
            ),
        ),
    )
