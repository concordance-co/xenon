from __future__ import annotations

from pathlib import Path

from pipelines_v2.api import (
    CaptureSpec,
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

from projects.MOREBENCH.phase_03.specs import experiment_02_cross_language_pilot_workflow as pilot
from projects.MOREBENCH.phase_03.specs import experiment_02_workflow as base


LOCAL_CATALOG_ROOT = Path("artifacts") / "morebench_phase03_experiment02_cross_language_prompt_probe_catalog"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "morebench_phase03_experiment02_cross_language_prompt_probe"
MODAL_ARTIFACT_ROOT = "/data/artifacts/morebench_phase_03_experiment02_cross_language_prompt_probe"


def build_dataset():
    return pilot.build_dataset()


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
                timeout_seconds=60 * 60 * 4,
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
        name="morebench_phase03_experiment02_cross_language_prompt_probe",
        steps=(
            WorkflowStep(
                name="capture_prompt_eos_residual",
                runner="capture_gpu",
                description="Capture prompt-final residuals on the translated cross-language pilot prompts.",
                spec=CaptureSpec(
                    engine=base._engine(max_num_seqs=8),
                    dataset=dataset,
                    sites=[
                        ResidualSite(
                            name="prompt_eos_residual",
                            site="resid_post",
                            layers=list(base.CAPTURED_LAYERS),
                            tokens=TokenSelector.last(),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        )
                    ],
                    generation=GenerationSpec(enabled=False),
                ),
            ),
        ),
    )
