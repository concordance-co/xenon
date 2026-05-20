from __future__ import annotations

"""Score one fake trace against released Llama 3.3 70B Assistant Axis traits."""

from pathlib import Path
from typing import Any, Mapping

from pipelines_v2.api import (
    AssistantAxisPrecomputedCoordinateSpec,
    AssistantAxisTraitCoordinateSpec,
    CaptureSpec,
    Dataset,
    Example,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    ProjectionSpec,
    ReportSpec,
    ResidualSite,
    SectionSelector,
    StepRef,
    TensorStorage,
    TokenSelector,
    TransformBuilder,
    TransformResult,
    TransformSpec,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)


WORKFLOW_NAME = "assistant_axis_llama33_70b_trait_trace_score"
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_VOLUME_NAME = "yora-models"
MODEL_VOLUME_PATH = "/models"
ARTIFACT_VOLUME_NAME = "xenon-data"
MODAL_ARTIFACT_ROOT = "/data/artifacts/assistant_axis_llama33_70b_trait_trace_score"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "assistant_axis_llama33_70b_trait_trace_score"
LAYER = 40
TRAITS = (
    "concise",
    "calm",
    "supportive",
    "technical",
    "analytical",
    "confident",
    "verbose",
    "sycophantic",
    "hostile",
    "condescending",
)


def _fake_trace() -> str:
    return (
        "Human: The deploy failed after the migration step. Can you help me debug it?\n\n"
        "Assistant: Start by separating the failure into migration, application, and environment checks. "
        "First confirm the migration command, database URL, and revision id used in the failing run. "
        "Then compare the production error with a local dry run against a copied schema, and only rerun "
        "the migration after you know whether it is failing before or after the first write."
    )


def _assistant_span(trace: str) -> dict[str, int]:
    marker = "Assistant:"
    start = trace.index(marker) + len(marker)
    while start < len(trace) and trace[start].isspace():
        start += 1
    return {"char_start": start, "char_end": len(trace)}


def build_dataset() -> Dataset:
    trace = _fake_trace()
    assistant_span = _assistant_span(trace)
    return Dataset.from_examples(
        [
            Example(
                key="fake_trace_0",
                prompt=trace,
                labels={"surface": "fake_trace", "model_id": MODEL_ID},
                metadata={
                    "token_sections": {"assistant_response": assistant_span},
                    "section_records": [
                        {
                            "name": "assistant_response",
                            "unit": "turn",
                            "role": "assistant",
                            "index": 0,
                            **assistant_span,
                        }
                    ],
                },
            )
        ],
        name=WORKFLOW_NAME,
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
        add_generation_prompt=False,
    )


def summarize_trait_scores(*, scores: Any) -> TransformResult:
    payload = scores.result() if hasattr(scores, "result") else scores
    rows = payload.get("rows", []) if isinstance(payload, Mapping) else []
    compact: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        coordinate = str(row.get("coordinate") or "")
        trait = coordinate.removeprefix("assistant_axis_trait__")
        compact.append(
            {
                "trait": trait,
                "coordinate": coordinate,
                "score": float(row.get("score", 0.0)),
                "layer": int(row.get("layer", LAYER)),
                "slice": str(row.get("slice_name") or ""),
                "tokens": int(row.get("slice_token_count", 0)),
            }
        )
    compact.sort(key=lambda item: item["score"], reverse=True)
    return TransformResult(
        payload={
            "kind": "assistant_axis_trait_trace_score_summary",
            "model_id": MODEL_ID,
            "layer": LAYER,
            "trace": _fake_trace(),
            "scores": compact,
            "summary": {
                "score_count": len(compact),
                "top_positive": compact[:3],
                "top_negative": list(reversed(compact[-3:])),
            },
        },
        example_keys=["fake_trace_0"] if compact else [],
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
                cpu=8,
                memory_mb=96 * 1024,
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
    trait_steps = tuple(
        WorkflowStep(
            name=f"trait_{trait}",
            runner="analysis_cpu",
            spec=AssistantAxisTraitCoordinateSpec(
                model_id=MODEL_ID,
                trait=trait,
                token_env_var="HF_TOKEN",
            ),
        )
        for trait in TRAITS
    )
    return WorkflowSpec(
        name=WORKFLOW_NAME,
        steps=(
            WorkflowStep(
                name="capture_trace",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=_engine(),
                    dataset=dataset,
                    sites=(
                        ResidualSite(
                            name="residual_full",
                            site="resid_post",
                            layers=(LAYER,),
                            tokens=TokenSelector.full_sequence(),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                    ),
                ),
            ),
            WorkflowStep(
                name="assistant_axis",
                runner="analysis_cpu",
                spec=AssistantAxisPrecomputedCoordinateSpec(
                    model_id=MODEL_ID,
                    token_env_var="HF_TOKEN",
                ),
            ),
            *trait_steps,
            WorkflowStep(
                name="score_trace",
                runner="analysis_cpu",
                spec=ProjectionSpec(
                    feature=StepRef("capture_trace").feature("residual_full"),
                    coordinates=(
                        StepRef("assistant_axis"),
                        *(StepRef(f"trait_{trait}") for trait in TRAITS),
                    ),
                    slices=SectionSelector.named("assistant_response"),
                    layers=(LAYER,),
                    summaries=("mean",),
                    emit_labels=True,
                ),
            ),
            WorkflowStep(
                name="score_summary",
                runner="analysis_cpu",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_trait_scores,
                        local_python_sources=("scripts",),
                    ),
                    inputs={"scores": StepRef("score_trace")},
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=(StepRef("score_trace"), StepRef("score_summary")),
                    template="assistant_axis_llama33_70b_trait_trace_score",
                    output_dir="scripts/reports/assistant_axis_llama33_70b_trait_trace_score",
                ),
            ),
        ),
    )
