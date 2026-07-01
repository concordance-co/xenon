"""Score traces against the packaged Llama 3.3 70B Assistant Axis asset."""

from __future__ import annotations

import os

from pipelines_v2.api import Dataset, WorkflowSpec

from papers.voice.assistant_axis.assets import (
    DEFAULT_SCORE_WORKFLOW_NAME,
    build_trace_scoring_workflow,
    default_traits,
)
from papers.voice.assistant_axis.runtime import byot_dataset_from_env, env_list, runner_specs


WORKFLOW_NAME = DEFAULT_SCORE_WORKFLOW_NAME


def build_dataset() -> Dataset:
    return byot_dataset_from_env(workflow_name=WORKFLOW_NAME)


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    traits = env_list("ASSISTANT_AXIS_ASSET_TRAITS", default_traits())
    return build_trace_scoring_workflow(
        dataset=dataset or build_dataset(),
        traits=traits,
        workflow_name=WORKFLOW_NAME,
        include_assistant_axis=True,
    )


def build_runner_specs() -> dict[str, object]:
    return runner_specs(workflow_name=os.getenv("ASSISTANT_AXIS_WORKFLOW_NAME", WORKFLOW_NAME))
