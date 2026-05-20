from __future__ import annotations

"""Steer Llama 3.3 70B generation with a released Assistant Axis trait vector."""

import os
from pathlib import Path
from typing import Any, Mapping

from pipelines_v2.api import (
    AddDirectionPatch,
    AssistantAxisTraitCoordinateSpec,
    Dataset,
    Example,
    GenerationRunSpec,
    GenerationSpec,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    PatchApplication,
    PatchedGenerationSpec,
    ReportSpec,
    ResidualInterventionSite,
    StepRef,
    TokenSelector,
    TransformBuilder,
    TransformResult,
    TransformSpec,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)


DEFAULT_PROMPT = (
    "The production deploy failed during a database migration and the team is tense. "
    "Write a short assistant response that helps me decide what to do next."
)
WORKFLOW_NAME = os.getenv(
    "ASSISTANT_AXIS_STEERING_WORKFLOW",
    "assistant_axis_llama33_70b_precomputed_steering",
)
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_VOLUME_NAME = "yora-models"
MODEL_VOLUME_PATH = "/models"
ARTIFACT_VOLUME_NAME = "xenon-data"
MODAL_ARTIFACT_ROOT = f"/data/artifacts/{WORKFLOW_NAME}"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / WORKFLOW_NAME
LAYER = 40
TRAIT = os.getenv("ASSISTANT_AXIS_STEERING_TRAIT", "calm")
STRENGTH = float(os.getenv("ASSISTANT_AXIS_STEERING_STRENGTH", "2.0"))
GENERATION_MAX_TOKENS = int(os.getenv("ASSISTANT_AXIS_STEERING_MAX_TOKENS", "96"))
PROMPT = os.getenv("ASSISTANT_AXIS_STEERING_PROMPT", DEFAULT_PROMPT)


def build_dataset() -> Dataset:
    return Dataset.from_examples(
        [
            Example(
                key="deploy_debug_probe",
                prompt=[{"role": "user", "content": PROMPT}],
                labels={"surface": "steering_probe", "trait": TRAIT, "strength": STRENGTH},
            )
        ],
        name=WORKFLOW_NAME,
    )


def _engine(*, patched: bool = False) -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        model_path_root=MODEL_VOLUME_PATH,
        max_model_len=1024 if patched else 2048,
        tensor_parallel_size=4 if patched else 2,
        gpu_memory_utilization=0.88,
        enforce_eager=bool(patched),
        max_num_seqs=1,
        enable_prefix_caching=not patched,
        add_generation_prompt=True,
    )


def coordinate_to_unit_direction(*, coordinate: Any, name: str) -> TransformResult:
    payload = coordinate.result() if hasattr(coordinate, "result") else coordinate
    if not isinstance(payload, Mapping):
        raise TypeError(f"coordinate must resolve to a mapping, got {type(payload).__name__}")
    layers = payload.get("layers")
    if not isinstance(layers, Mapping) or not layers:
        raise ValueError("coordinate payload must contain non-empty layers")

    direction_layers: dict[str, dict[str, Any]] = {}
    for layer, raw_layer_payload in layers.items():
        if not isinstance(raw_layer_payload, Mapping):
            continue
        vector = raw_layer_payload.get("vector")
        if vector is None:
            raise ValueError(f"coordinate layer {layer!r} is missing normalized vector")
        direction_layers[str(layer)] = {
            **dict(raw_layer_payload),
            "vector": list(vector),
            "raw_vector": list(vector),
            "norm": 1.0,
        }
    return TransformResult(
        payload={
            "kind": "direction_result",
            "feature": payload.get("feature"),
            "name": str(name),
            "layers": direction_layers,
            "metadata": {
                **(dict(payload.get("metadata")) if isinstance(payload.get("metadata"), Mapping) else {}),
                "source": "released_assistant_axis_trait_unit_direction",
                "source_coordinate_kind": payload.get("kind"),
                "steering_units": "raw_vector is set to the released normalized vector; AddDirectionPatch strength is in unit-vector multiples.",
            },
            "summary": {
                "layer_count": len(direction_layers),
                "source_coordinate": payload.get("name"),
                "unit_raw_vector": True,
            },
        },
        example_keys=[],
    )


def summarize_generations(*, baseline: Any, steered: Any) -> TransformResult:
    baseline_payload = baseline.result() if hasattr(baseline, "result") else baseline
    steered_payload = steered.result() if hasattr(steered, "result") else steered
    trait = _first_label(steered_payload, "trait") or TRAIT
    strength = _first_label(steered_payload, "strength")
    if strength is None:
        strength = _first_patch_strength(steered_payload)
    if strength is None:
        strength = STRENGTH
    return TransformResult(
        payload={
            "kind": "assistant_axis_precomputed_steering_summary",
            "model_id": MODEL_ID,
            "layer": LAYER,
            "trait": trait,
            "strength": float(strength),
            "baseline_text": _first_generation_text(baseline_payload),
            "steered_text": _first_generation_text(steered_payload),
            "summary": {
                "model_id": MODEL_ID,
                "layer": LAYER,
                "trait": trait,
                "strength": float(strength),
            },
        },
        example_keys=["deploy_debug_probe"],
    )


def _first_generation_text(payload: Any) -> str:
    rows = payload.get("rows", []) if isinstance(payload, Mapping) else []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        for key in ("generated_text", "text", "completion", "output"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _first_label(payload: Any, label: str) -> Any | None:
    rows = payload.get("rows", []) if isinstance(payload, Mapping) else []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        example = row.get("example")
        labels = example.get("labels") if isinstance(example, Mapping) else None
        if isinstance(labels, Mapping) and label in labels:
            return labels[label]
    return None


def _first_patch_strength(payload: Any) -> float | None:
    rows = payload.get("rows", []) if isinstance(payload, Mapping) else []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        patch_stats = row.get("patch_stats")
        if not isinstance(patch_stats, Mapping):
            continue
        for layer_stats in patch_stats.values():
            if not isinstance(layer_stats, Mapping):
                continue
            chunks = layer_stats.get("chunk_stats")
            if not isinstance(chunks, list):
                continue
            for chunk in chunks:
                if not isinstance(chunk, Mapping):
                    continue
                strength = chunk.get("strength")
                if isinstance(strength, (int, float)):
                    return float(strength)
    return None


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
        "XENON_ACTIVATION_PATCH_MAX_TOKENS": "1",
    }
    return {
        "generation_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H100:4",
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
    return WorkflowSpec(
        name=WORKFLOW_NAME,
        steps=(
            WorkflowStep(
                name="load_calm_trait",
                runner="analysis_cpu",
                spec=AssistantAxisTraitCoordinateSpec(
                    model_id=MODEL_ID,
                    trait=TRAIT,
                    token_env_var="HF_TOKEN",
                ),
            ),
            WorkflowStep(
                name="calm_unit_direction",
                runner="analysis_cpu",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        coordinate_to_unit_direction,
                        local_python_sources=("scripts",),
                    ),
                    inputs={"coordinate": StepRef("load_calm_trait"), "name": f"assistant_axis_trait__{TRAIT}"},
                ),
            ),
            WorkflowStep(
                name="baseline_generation",
                runner="generation_gpu",
                spec=GenerationRunSpec(
                    engine=_engine(patched=False),
                    dataset=dataset,
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=GENERATION_MAX_TOKENS,
                        temperature=0.7,
                        top_p=0.95,
                    ),
                ),
            ),
            WorkflowStep(
                name="calm_steered_generation",
                runner="generation_gpu",
                depends_on=("baseline_generation", "calm_unit_direction"),
                spec=PatchedGenerationSpec(
                    engine=_engine(patched=True),
                    dataset=dataset,
                    patch=AddDirectionPatch(
                        direction=StepRef("calm_unit_direction"),
                        write_site=ResidualInterventionSite(site="resid_post", layers=(LAYER,)),
                        target_tokens=TokenSelector.last(),
                        application=PatchApplication.every_token(include_prompt=True, include_decode=True),
                        strength=STRENGTH,
                    ),
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=GENERATION_MAX_TOKENS,
                        temperature=0.7,
                        top_p=0.95,
                    ),
                ),
            ),
            WorkflowStep(
                name="steering_summary",
                runner="analysis_cpu",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_generations,
                        local_python_sources=("scripts",),
                    ),
                    inputs={
                        "baseline": StepRef("baseline_generation"),
                        "steered": StepRef("calm_steered_generation"),
                    },
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=(
                        StepRef("load_calm_trait"),
                        StepRef("calm_unit_direction"),
                        StepRef("baseline_generation"),
                        StepRef("calm_steered_generation"),
                        StepRef("steering_summary"),
                    ),
                    template="assistant_axis_llama33_70b_precomputed_steering",
                    output_dir=f"scripts/reports/{WORKFLOW_NAME}",
                ),
            ),
        ),
    )
