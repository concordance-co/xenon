"""Local validation workflow for completed Llama 70B emotion-vector assets."""

from __future__ import annotations

import os
from pathlib import Path

from pipelines_v2.api import (
    Dataset,
    LocalArtifactStore,
    LocalRunnerSpec,
    ReportSpec,
    StepRef,
    TransformBuilder,
    TransformSpec,
    WorkflowSpec,
    WorkflowStep,
)
from papers.voice.emotions.replication.validation import (
    build_paper_graph_validation,
    latest_emotion_space_result_path,
)


WORKFLOW_NAME = "papers_voice_emotions_llama33_70b_validation"
DEFAULT_SELECTED_LAYER = 52
DEFAULT_REPORT_DIR = f"papers/voice/emotions/replication/reports/{WORKFLOW_NAME}"
DEFAULT_LOCAL_ARTIFACT_ROOT = f"{DEFAULT_REPORT_DIR}/artifacts"


def emotion_space_result_path() -> str:
    override = os.getenv("EMOTION_VALIDATION_EMOTION_SPACE_PATH")
    if override:
        return override
    return str(latest_emotion_space_result_path())


def selected_layer() -> int:
    return int(os.getenv("EMOTION_VALIDATION_LAYER", str(DEFAULT_SELECTED_LAYER)))


def validation_output_dir() -> str:
    override = os.getenv("EMOTION_VALIDATION_OUTPUT_DIR")
    if override:
        return override
    return f"{DEFAULT_REPORT_DIR}/paper_graphs_layer{selected_layer()}"


def build_dataset() -> Dataset:
    return Dataset.from_examples((), name=f"{WORKFLOW_NAME}_local_assets")


def build_workflow() -> WorkflowSpec:
    return WorkflowSpec(
        name=WORKFLOW_NAME,
        steps=(
            WorkflowStep(
                name="paper_graph_validation",
                runner="analysis_local",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        build_paper_graph_validation,
                        local_python_sources=("papers",),
                    ),
                    inputs={
                        "emotion_space_path": emotion_space_result_path(),
                        "selected_layer": selected_layer(),
                        "pca_components": 3,
                        "output_dir": validation_output_dir(),
                    },
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=(StepRef("paper_graph_validation"),),
                    template="voice_emotions_llama70b_validation",
                    output_dir=os.getenv("EMOTION_VALIDATION_REPORT_DIR", DEFAULT_REPORT_DIR),
                ),
            ),
        ),
    )


def build_runner_specs() -> dict[str, object]:
    artifact_root = Path(os.getenv("EMOTION_VALIDATION_LOCAL_ARTIFACT_ROOT", DEFAULT_LOCAL_ARTIFACT_ROOT))
    artifact_store = LocalArtifactStore(artifact_root)
    return {
        "analysis_local": LocalRunnerSpec(artifacts=artifact_store),
        "report_local": LocalRunnerSpec(artifacts=artifact_store),
    }
