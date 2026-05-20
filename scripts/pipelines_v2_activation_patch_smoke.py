from __future__ import annotations

"""Minimal Modal smoke workflow for pipelines_v2 activation patching."""

import re
from pathlib import Path
from typing import Any

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    Example,
    GenerationRunSpec,
    GenerationSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalVolumeMount,
    ModalVolumeStore,
    PatchComparisonSpec,
    PatchApplication,
    PatchedGenerationSpec,
    PromptMetadataBuilder,
    ProjectOutPatch,
    ResidualInterventionSite,
    ResidualSite,
    StepRef,
    SubspaceSpec,
    TokenPooling,
    TokenSelector,
    TransformBuilder,
    TransformSpec,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)


MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"
ARTIFACT_ROOT = Path("/data/artifacts/pipelines_v2_activation_patch_smoke")


def build_prompt_metadata(rendered_prompt: str) -> dict[str, object]:
    strategy_marker = "STRATEGY\n"
    settings_marker = "\n\nSETTINGS\n"
    strategy_start = rendered_prompt.index(strategy_marker) + len(strategy_marker)
    strategy_end = rendered_prompt.index(settings_marker, strategy_start)
    settings_start = strategy_end + len(settings_marker)
    return {
        "token_sections": {
            "STRATEGY": {"char_start": strategy_start, "char_end": strategy_end},
            "SETTINGS": {"char_start": settings_start, "char_end": len(rendered_prompt)},
        }
    }


def evaluate_patch_row(
    *,
    example: dict[str, Any],
    baseline: dict[str, Any],
    variants: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    patched = dict(variants or {}).get("main", {})
    baseline_word = _extract_choice_word(str(baseline.get("generated_text") or ""))
    patched_word = _extract_choice_word(str(patched.get("generated_text") or ""))
    return {
        "metrics": {
            "baseline_nonempty": bool(baseline_word),
            "changed": baseline_word != patched_word,
            "patched_nonempty": bool(patched_word),
        },
        "evaluation": {
            "example_key": str(example.get("key") or ""),
            "baseline_word": baseline_word,
            "patched_word": patched_word,
            "baseline_text": str(baseline.get("generated_text") or ""),
            "patched_text": str(patched.get("generated_text") or ""),
        },
    }


def validate_compiled_patch_smoke(*, patched: Any) -> dict[str, Any]:
    payload = patched.result() if hasattr(patched, "result") else dict(patched)
    rows = list(payload.get("rows") or [])
    ok_rows = [dict(row) for row in rows if str(row.get("status") or "ok") == "ok"]
    failures: list[str] = []
    operator_counts: dict[str, int] = {}
    dispatch_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    checked_layers = 0
    missing_runtime_stats_count = 0

    if not rows:
        failures.append("patched generation returned no rows")
    if not ok_rows:
        failures.append("patched generation returned no ok rows")

    for row in ok_rows:
        row_key = str(row.get("example_key") or row.get("case_key") or "<unknown>")
        patch_stats = row.get("patch_stats")
        if not isinstance(patch_stats, dict) or not patch_stats:
            failures.append(f"{row_key}: missing patch_stats")
            continue
        for layer, raw_stats in patch_stats.items():
            if not isinstance(raw_stats, dict):
                failures.append(f"{row_key}: layer {layer} stats are not a mapping")
                continue
            checked_layers += 1
            status = str(raw_stats.get("status") or "")
            operator = str(raw_stats.get("operator") or "")
            dispatch = str(raw_stats.get("dispatch") or "")
            status_counts[status] = status_counts.get(status, 0) + 1
            operator_counts[operator] = operator_counts.get(operator, 0) + 1
            dispatch_counts[dispatch] = dispatch_counts.get(dispatch, 0) + 1
            if status == "missing_runtime_stats":
                missing_runtime_stats_count += 1
            if status != "ok":
                failures.append(f"{row_key}: layer {layer} status={status!r}")
            if operator != "project_out":
                failures.append(f"{row_key}: layer {layer} operator={operator!r}")
            if dispatch != "compiled_custom_op":
                failures.append(f"{row_key}: layer {layer} dispatch={dispatch!r}")
            if int(raw_stats.get("token_count") or 0) <= 0:
                failures.append(f"{row_key}: layer {layer} token_count did not show an applied patch")

    summary = {
        "row_count": len(rows),
        "patched_count": len(ok_rows),
        "checked_patch_stat_layers": checked_layers,
        "missing_runtime_stats_count": missing_runtime_stats_count,
        "operator_counts": operator_counts,
        "dispatch_counts": dispatch_counts,
        "status_counts": status_counts,
    }
    if failures:
        raise AssertionError(f"compiled patch smoke validation failed: {failures}; summary={summary}")
    return {
        "payload": {
            "kind": "transform_result",
            "summary": summary,
        },
    }


def build_dataset() -> Dataset:
    return Dataset.from_examples(
        (
            Example(
                key="pair1_target_buy",
                prompt=_decision_prompt("BUY"),
                labels={
                    "patch_role": "target",
                    "strategy_word": "BUY",
                },
                case_key="pair_1",
            ),
            Example(
                key="pair1_donor_sell",
                prompt=_decision_prompt("SELL"),
                labels={
                    "patch_role": "donor",
                    "strategy_word": "SELL",
                },
                case_key="pair_1",
            ),
            Example(
                key="pair2_target_sell",
                prompt=_decision_prompt("SELL"),
                labels={
                    "patch_role": "target",
                    "strategy_word": "SELL",
                },
                case_key="pair_2",
            ),
            Example(
                key="pair2_donor_buy",
                prompt=_decision_prompt("BUY"),
                labels={
                    "patch_role": "donor",
                    "strategy_word": "BUY",
                },
                case_key="pair_2",
            ),
        ),
        name="pipelines_v2_activation_patch_smoke",
    )


def build_runner_specs() -> dict[str, object]:
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="A100-80GB",
                timeout_seconds=3600,
                volumes=(ModalVolumeMount(name=MODEL_VOLUME_NAME, mount_path=MODEL_VOLUME_PATH),),
            ),
            artifacts=ModalVolumeStore(
                name="xenon-data",
                root=str(ARTIFACT_ROOT),
            ),
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=8,
                memory_mb=24 * 1024,
                timeout_seconds=3600,
                volumes=(ModalVolumeMount(name="xenon-data", mount_path="/data"),),
            ),
            artifacts=ModalVolumeStore(
                name="xenon-data",
                root=str(ARTIFACT_ROOT),
            ),
        ),
    }


def build_engine() -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        max_model_len=4096,
        enforce_eager=False,
        max_num_seqs=32,
        max_num_batched_tokens=32768,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        async_scheduling=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    prompt_metadata = PromptMetadataBuilder.from_function(
        build_prompt_metadata,
        local_python_sources=("scripts",),
    )
    row_evaluator = TransformBuilder.from_function(
        evaluate_patch_row,
        local_python_sources=("scripts",),
    )
    smoke_validator = TransformBuilder.from_function(
        validate_compiled_patch_smoke,
        local_python_sources=("scripts",),
    )
    return WorkflowSpec(
        name="pipelines_v2_activation_patch_smoke",
        steps=(
            WorkflowStep(
                name="capture_prompt_residual",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=build_engine(),
                    dataset=dataset,
                    sites=(
                        ResidualSite(
                            name="prompt_residual",
                            site="resid_post",
                            layers=(24,),
                            tokens=TokenSelector.full_sequence(),
                        ),
                    ),
                    prompt_metadata_builder=prompt_metadata,
                ),
            ),
            WorkflowStep(
                name="learn_strategy_subspace",
                runner="analysis_cpu",
                depends_on=("capture_prompt_residual",),
                spec=SubspaceSpec(
                    feature=StepRef("capture_prompt_residual").feature("prompt_residual"),
                    layers=(24,),
                    components=2,
                    tokens=TokenSelector.section("STRATEGY"),
                    pooling=TokenPooling.mean(),
                ),
            ),
            WorkflowStep(
                name="baseline_targets",
                runner="capture_gpu",
                depends_on=("learn_strategy_subspace",),
                spec=GenerationRunSpec(
                    engine=build_engine(),
                    dataset=dataset,
                    select_when=dataset.labels("patch_role").equals("target"),
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=256,
                        temperature=0.0,
                        capture_reasoning=False,
                    ),
                ),
            ),
            WorkflowStep(
                name="lesion_generated_tokens",
                runner="capture_gpu",
                depends_on=("baseline_targets",),
                spec=PatchedGenerationSpec(
                    engine=build_engine(),
                    dataset=dataset,
                    patch=ProjectOutPatch(
                        subspace=StepRef("learn_strategy_subspace"),
                        write_site=ResidualInterventionSite(site="resid_post", layers=(24,)),
                        target_tokens=TokenSelector.section("STRATEGY"),
                        application=PatchApplication.every_token(
                            include_prompt=False,
                            include_decode=True,
                        ),
                        component_indices_by_layer={24: (0, 1)},
                        strength=1.0,
                    ),
                    select_when=dataset.labels("patch_role").equals("target"),
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=256,
                        temperature=0.0,
                        capture_reasoning=False,
                    ),
                    prompt_metadata_builder=prompt_metadata,
                ),
            ),
            WorkflowStep(
                name="validate_compiled_patch_stats",
                runner="analysis_cpu",
                depends_on=("lesion_generated_tokens",),
                spec=TransformSpec(
                    builder=smoke_validator,
                    inputs={"patched": StepRef("lesion_generated_tokens")},
                ),
            ),
            WorkflowStep(
                name="compare_patch_runs",
                runner="analysis_cpu",
                depends_on=("validate_compiled_patch_stats",),
                spec=PatchComparisonSpec(
                    baseline=StepRef("baseline_targets"),
                    variants={"main": StepRef("lesion_generated_tokens")},
                    row_evaluator=row_evaluator,
                ),
            ),
        ),
    )


def _decision_prompt(strategy_word: str) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "SYSTEM\nRespond with exactly one word: BUY or SELL.\n\n"
                f"STRATEGY\n{strategy_word}\n\n"
                "SETTINGS\nRepeat the strategy word only.\n"
            ),
        }
    ]


def _extract_choice_word(text: str) -> str:
    match = re.search(r"\b(BUY|SELL)\b", text.upper())
    return match.group(1) if match is not None else ""
