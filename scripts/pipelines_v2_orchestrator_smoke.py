from __future__ import annotations

"""Minimal Modal-ready capture+probe workflow for pipelines_v2.

This file is intentionally only workflow definition:

- `build_dataset()`
- `build_workflow(dataset=None)`

The CLI is responsible for constructing runners and executing the workflow.
"""

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    Example,
    ModalResources,
    ModalRunnerSpec,
    ModalVolumeStore,
    PromptMetadataBuilder,
    ProbeSpec,
    ResidualSite,
    StepRef,
    TokenPooling,
    TokenSelector,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def build_prompt_metadata(rendered_prompt: str) -> dict[str, object]:
    """Return explicit section spans for the smoke prompt format."""
    return {
        "token_sections": {
            "STRATEGY": _section_char_span(rendered_prompt, "STRATEGY", next_header="SETTINGS"),
            "SETTINGS": _section_char_span(rendered_prompt, "SETTINGS"),
        }
    }


def _section_char_span(rendered_prompt: str, header: str, *, next_header: str | None = None) -> dict[str, int]:
    marker = f"{header}\n"
    start = rendered_prompt.index(marker) + len(marker)
    if next_header is None:
        end = len(rendered_prompt)
    else:
        end = rendered_prompt.index(f"\n\n{next_header}\n", start)
    return {"char_start": start, "char_end": end}


def build_dataset() -> Dataset:
    """Return a tiny balanced in-memory dataset for a capture+probe smoke run."""
    return Dataset.from_examples(
        [
            Example(
                key="ex_pos_train",
                prompt=(
                    "SYSTEM\nChoose exactly one action.\n\n"
                    "STRATEGY\nBuy ALPHA immediately.\n\n"
                    "SETTINGS\nUse the largest size.\n"
                ),
                labels={"class": "positive", "split": "train"},
                cases={"pair_id": "pair_1"},
                case_key="pair_1",
            ),
            Example(
                key="ex_neg_train",
                prompt=(
                    "SYSTEM\nChoose exactly one action.\n\n"
                    "STRATEGY\nObserve only.\n\n"
                    "SETTINGS\nDo not trade.\n"
                ),
                labels={"class": "negative", "split": "train"},
                cases={"pair_id": "pair_2"},
                case_key="pair_2",
            ),
            Example(
                key="ex_pos_test",
                prompt=(
                    "SYSTEM\nChoose exactly one action.\n\n"
                    "STRATEGY\nBuy BETA immediately.\n\n"
                    "SETTINGS\nUse the largest size.\n"
                ),
                labels={"class": "positive", "split": "test"},
                cases={"pair_id": "pair_3"},
                case_key="pair_3",
            ),
            Example(
                key="ex_neg_test",
                prompt=(
                    "SYSTEM\nChoose exactly one action.\n\n"
                    "STRATEGY\nHold cash.\n\n"
                    "SETTINGS\nDo not trade.\n"
                ),
                labels={"class": "negative", "split": "test"},
                cases={"pair_id": "pair_4"},
                case_key="pair_4",
            ),
        ],
        name="pipelines_v2_modal_capture_probe_smoke",
    )


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    """Return a minimal two-step workflow: Modal capture, then Modal probe."""
    dataset = dataset or build_dataset()
    return WorkflowSpec(
        name="pipelines_v2_modal_capture_probe_smoke",
        steps=(
            WorkflowStep(
                name="capture",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=VLLMEngine(
                        model_id=MODEL_ID,
                        max_model_len=2048,
                        enforce_eager=False,
                        max_num_seqs=4,
                    ),
                    dataset=dataset,
                    prompt_metadata_builder=PromptMetadataBuilder.from_function(build_prompt_metadata),
                    sites=[
                        ResidualSite(
                            name="resid_prompt_tokens",
                            site="resid_post",
                            layers=[0, 6, 12, 18],
                            tokens=TokenSelector.full_sequence(),
                        )
                    ],
                ),
            ),
            WorkflowStep(
                name="probe",
                runner="analysis_cpu",
                spec=ProbeSpec(
                    feature=StepRef("capture").feature("resid_prompt_tokens"),
                    labels=dataset.labels("class"),
                    split=dataset.labels("split"),
                    tokens=TokenSelector.section("STRATEGY"),
                    pooling=TokenPooling.mean(),
                    folds=2,
                    baselines=["majority", "shuffled_label"],
                    metrics=["accuracy", "balanced_accuracy", "selectivity"],
                ),
            ),
        ),
    )


def build_runner_specs() -> dict[str, object]:
    """Return the named runner specs consumed by the workflow steps."""
    artifact_store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts/pipelines_v2_modal_capture_probe_smoke",
    )
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="L4",
                timeout_seconds=3600,
            ),
            artifacts=artifact_store,
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=4,
                memory_mb=8 * 1024,
                timeout_seconds=3600,
            ),
            artifacts=artifact_store,
        ),
    }
