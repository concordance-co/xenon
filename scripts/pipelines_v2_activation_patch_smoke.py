from __future__ import annotations

"""Minimal Modal smoke workflow for pipelines_v2 activation patching."""

import re
from pathlib import Path
from typing import Any

from pipelines_v2.api import (
    ActivationPatchSpec,
    CaptureSpec,
    Dataset,
    Example,
    GenerationSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalVolumeMount,
    ModalVolumeStore,
    PromptMetadataBuilder,
    ResidualInterventionSite,
    ResidualSite,
    StepRef,
    TokenSelector,
    TransformBuilder,
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
    patched: dict[str, Any],
    controls: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    labels = dict(example.get("labels") or {})
    expected_baseline = str(labels.get("expected_baseline_word") or "").upper()
    expected_donor = str(labels.get("expected_donor_word") or "").upper()
    baseline_word = _extract_choice_word(str(baseline.get("generated_text") or ""))
    patched_word = _extract_choice_word(str(patched.get("generated_text") or ""))
    return {
        "metrics": {
            "baseline_matches_expected": baseline_word == expected_baseline,
            "patched_matches_donor": patched_word == expected_donor,
            "changed": baseline_word != patched_word,
            "patched_nonempty": bool(patched_word),
            "control_count": float(len(dict(controls or {}))),
        },
        "evaluation": {
            "example_key": str(example.get("key") or ""),
            "baseline_word": baseline_word,
            "patched_word": patched_word,
            "expected_baseline_word": expected_baseline,
            "expected_donor_word": expected_donor,
            "baseline_text": str(baseline.get("generated_text") or ""),
            "patched_text": str(patched.get("generated_text") or ""),
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
                    "expected_baseline_word": "BUY",
                    "expected_donor_word": "SELL",
                },
                case_key="pair_1",
            ),
            Example(
                key="pair1_donor_sell",
                prompt=_decision_prompt("SELL"),
                labels={
                    "patch_role": "donor",
                    "expected_baseline_word": "SELL",
                    "expected_donor_word": "BUY",
                },
                case_key="pair_1",
            ),
            Example(
                key="pair2_target_sell",
                prompt=_decision_prompt("SELL"),
                labels={
                    "patch_role": "target",
                    "expected_baseline_word": "SELL",
                    "expected_donor_word": "BUY",
                },
                case_key="pair_2",
            ),
            Example(
                key="pair2_donor_buy",
                prompt=_decision_prompt("BUY"),
                labels={
                    "patch_role": "donor",
                    "expected_baseline_word": "BUY",
                    "expected_donor_word": "SELL",
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
    }


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
    return WorkflowSpec(
        name="pipelines_v2_activation_patch_smoke",
        steps=(
            WorkflowStep(
                name="capture_prompt_residual",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=VLLMEngine(
                        model_id=MODEL_ID,
                        max_model_len=4096,
                        enforce_eager=True,
                        max_num_seqs=4,
                        enable_prefix_caching=False,
                        add_generation_prompt=True,
                        enable_thinking=True,
                    ),
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
                name="patch_strategy",
                runner="capture_gpu",
                spec=ActivationPatchSpec(
                    engine=VLLMEngine(
                        model_id=MODEL_ID,
                        max_model_len=4096,
                        enforce_eager=True,
                        max_num_seqs=4,
                        enable_prefix_caching=False,
                        add_generation_prompt=True,
                        enable_thinking=True,
                    ),
                    dataset=dataset,
                    source_feature=StepRef("capture_prompt_residual").feature("prompt_residual"),
                    pair_by=dataset.cases("case_key"),
                    target_when=dataset.labels("patch_role").equals("target"),
                    donor_when=dataset.labels("patch_role").equals("donor"),
                    write_site=ResidualInterventionSite(site="resid_post", layers=(24,)),
                    target_tokens=TokenSelector.section("STRATEGY"),
                    donor_tokens=TokenSelector.section("STRATEGY"),
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=256,
                        temperature=0.0,
                        capture_reasoning=True,
                    ),
                    prompt_metadata_builder=prompt_metadata,
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
