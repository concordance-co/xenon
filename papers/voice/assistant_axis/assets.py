"""Packaged Assistant Axis model assets and scoring helpers."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines_v2.api import (
    AssistantAxisPrecomputedCoordinateSpec,
    AssistantAxisTraitCoordinateSpec,
    CaptureSpec,
    Dataset,
    ProjectionSpec,
    ReportSpec,
    ResidualSite,
    SectionSelector,
    StepRef,
    TensorStorage,
    TokenSelector,
    WorkflowSpec,
    WorkflowStep,
)

from papers.voice.assistant_axis.runtime import trace_dataset_from_records, vllm_engine


DEFAULT_LLAMA33_70B_MANIFEST = (
    Path(__file__).parents[1]
    / "model_assets"
    / "vectors"
    / "assistant-axis"
    / "llama-3.3-70b"
    / "released"
    / "v1"
    / "manifest.toml"
)
DEFAULT_SCORE_WORKFLOW_NAME = "papers_voice_assistant_axis_llama33_70b_asset_score"


def load_asset_manifest(path: str | Path = DEFAULT_LLAMA33_70B_MANIFEST) -> dict[str, Any]:
    """Load a source-controlled vector asset manifest."""

    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def default_traits(manifest: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Return the manifest's default demo/scoring traits."""

    payload = manifest or load_asset_manifest()
    usage = _mapping(payload.get("usage"))
    return tuple(str(item) for item in usage.get("default_traits", ()) if str(item).strip())


def assistant_axis_coordinate_spec(
    manifest: Mapping[str, Any] | None = None,
    *,
    token_env_var: str | None = None,
) -> AssistantAxisPrecomputedCoordinateSpec:
    """Build the released Assistant Axis coordinate spec from the manifest."""

    payload = manifest or load_asset_manifest()
    model = _mapping(payload.get("model"))
    source = _mapping(payload.get("source"))
    usage = _mapping(payload.get("usage"))
    revision = str(source.get("revision") or "") or None
    return AssistantAxisPrecomputedCoordinateSpec(
        model_id=str(model["model_id"]),
        repo_id=str(source["repo_id"]),
        revision=revision,
        filename=str(source["assistant_axis_file"]),
        select_layer=int(model["target_layer"]),
        token_env_var=token_env_var if token_env_var is not None else str(usage.get("token_env_var") or "HF_TOKEN"),
        metadata=_coordinate_metadata(payload),
    )


def trait_coordinate_spec(
    trait: str,
    manifest: Mapping[str, Any] | None = None,
    *,
    token_env_var: str | None = None,
) -> AssistantAxisTraitCoordinateSpec:
    """Build one released trait coordinate spec from the manifest."""

    payload = manifest or load_asset_manifest()
    model = _mapping(payload.get("model"))
    source = _mapping(payload.get("source"))
    usage = _mapping(payload.get("usage"))
    revision = str(source.get("revision") or "") or None
    normalized_trait = normalize_trait(trait)
    return AssistantAxisTraitCoordinateSpec(
        model_id=str(model["model_id"]),
        trait=normalized_trait,
        repo_id=str(source["repo_id"]),
        revision=revision,
        filename=f"{source['trait_vector_prefix']}/{normalized_trait}.pt",
        select_layer=int(model["target_layer"]),
        token_env_var=token_env_var if token_env_var is not None else str(usage.get("token_env_var") or "HF_TOKEN"),
        metadata={**_coordinate_metadata(payload), "trait": normalized_trait},
    )


def coordinate_specs(
    traits: Sequence[str] | None = None,
    manifest: Mapping[str, Any] | None = None,
    *,
    include_assistant_axis: bool = True,
    token_env_var: str | None = None,
) -> tuple[AssistantAxisPrecomputedCoordinateSpec | AssistantAxisTraitCoordinateSpec, ...]:
    """Return coordinate specs for the assistant axis plus selected traits."""

    payload = manifest or load_asset_manifest()
    selected_traits = tuple(traits) if traits is not None else default_traits(payload)
    specs: list[AssistantAxisPrecomputedCoordinateSpec | AssistantAxisTraitCoordinateSpec] = []
    if include_assistant_axis:
        specs.append(assistant_axis_coordinate_spec(payload, token_env_var=token_env_var))
    specs.extend(trait_coordinate_spec(trait, payload, token_env_var=token_env_var) for trait in selected_traits)
    return tuple(specs)


def trace_scoring_dataset(records: Sequence[Mapping[str, Any]], *, name: str = "assistant_axis_asset_traces") -> Dataset:
    """Build a trace dataset with assistant-response sections from records."""

    return trace_dataset_from_records(records, name=name)


def build_trace_scoring_workflow(
    *,
    dataset: Dataset,
    traits: Sequence[str] | None = None,
    manifest: Mapping[str, Any] | None = None,
    workflow_name: str = DEFAULT_SCORE_WORKFLOW_NAME,
    include_assistant_axis: bool = True,
) -> WorkflowSpec:
    """Build a Llama 70B trace-scoring workflow from the released asset manifest."""

    payload = manifest or load_asset_manifest()
    model = _mapping(payload.get("model"))
    layer = int(model["target_layer"])
    model_key = str(model["model_key"])
    selected_traits = tuple(traits) if traits is not None else default_traits(payload)
    coordinate_steps: list[WorkflowStep] = []
    coordinate_refs: list[StepRef] = []
    if include_assistant_axis:
        coordinate_steps.append(
            WorkflowStep(
                name="assistant_axis",
                runner="analysis_cpu",
                spec=assistant_axis_coordinate_spec(payload),
            )
        )
        coordinate_refs.append(StepRef("assistant_axis"))
    for trait in selected_traits:
        step_name = trait_step_name(trait)
        coordinate_steps.append(
            WorkflowStep(
                name=step_name,
                runner="analysis_cpu",
                spec=trait_coordinate_spec(trait, payload),
            )
        )
        coordinate_refs.append(StepRef(step_name))

    return WorkflowSpec(
        name=workflow_name,
        steps=(
            WorkflowStep(
                name="capture_trace",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=vllm_engine(model_key=model_key, add_generation_prompt=False),
                    dataset=dataset,
                    sites=(
                        ResidualSite(
                            name="response_residual",
                            site="resid_post",
                            layers=(layer,),
                            tokens=TokenSelector.section("assistant_response"),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                    ),
                ),
            ),
            *coordinate_steps,
            WorkflowStep(
                name="score_trace",
                runner="analysis_cpu",
                spec=ProjectionSpec(
                    feature=StepRef("capture_trace").feature("response_residual"),
                    coordinates=tuple(coordinate_refs),
                    slices=SectionSelector.named("assistant_response"),
                    layers=(layer,),
                    summaries=("mean", "min", "max", "trend"),
                    emit_labels=True,
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=(StepRef("score_trace"),),
                    template="voice_assistant_axis_byot",
                    output_dir=f"papers/voice/assistant_axis/reports/{workflow_name}",
                ),
            ),
        ),
    )


def trait_step_name(trait: str) -> str:
    return "trait_" + normalize_trait(trait)


def normalize_trait(trait: str) -> str:
    return str(trait).strip().lower().replace(" ", "_").replace("-", "_")


def _coordinate_metadata(manifest: Mapping[str, Any]) -> dict[str, Any]:
    asset = _mapping(manifest.get("asset"))
    model = _mapping(manifest.get("model"))
    source = _mapping(manifest.get("source"))
    method = _mapping(manifest.get("method"))
    storage = _mapping(manifest.get("storage"))
    return {
        "asset_id": asset.get("id"),
        "asset_status": asset.get("status"),
        "model_key": model.get("model_key"),
        "model_volume": model.get("model_volume"),
        "model_volume_status": model.get("model_volume_status"),
        "source_repo_id": source.get("repo_id"),
        "paper": method.get("paper"),
        "asset_root": storage.get("asset_root"),
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
