"""Phase 04 deont/util add-direction steering workflow."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from pipelines_v2.api import (
    AddDirectionPatch,
    Dataset,
    Example,
    FileCatalog,
    GenerationRunSpec,
    GenerationSpec,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    PatchedGenerationSpec,
    PostgresCatalog,
    PostgresSource,
    ReportSpec,
    ResidualInterventionSite,
    StepRef,
    TokenSelector,
    TransformBuilder,
    TransformResult,
    TransformSpec,
    WorkflowSpec,
    WorkflowStep,
)

from projects.MOREBENCH.phase_03.specs import experiment_02_workflow as base
from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import analyze_generated_slices as slices
from projects.MOREBENCH.theory_persona_vectors.phase_03.scripts import analyze_natural_prompt_paired as paired
from projects.MOREBENCH.theory_persona_vectors.phase_03.specs import (
    all_theories_natural_prompt_workflow as phase03,
)
from projects.MOREBENCH.theory_persona_vectors.phase_04.specs import (
    conflict_baseline_workflow as baseline,
)
from projects.MOREBENCH.theory_persona_vectors.phase_04.specs import (
    conflict_stability_workflow as stability,
)


PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_04")
STEERING_MANIFEST = PHASE_ROOT / "outputs" / "steering_trial_manifest.json"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "morebench_theory_persona_vectors_phase04"
LOCAL_CATALOG_ROOT = Path("artifacts") / "morebench_theory_persona_vectors_phase04_catalog"
MODAL_ARTIFACT_ROOT = "/data/artifacts/morebench_theory_persona_vectors_phase04"
REPORT_OUTPUT_ROOT = PHASE_ROOT / "reports" / "deont_util_steering"

DB_ENV_VAR = baseline.DB_ENV_VAR
GENERATION_MAX_TOKENS = 384
GENERATION_TEMPERATURE = 0.7
GENERATION_TOP_P = 0.95
CAPTURE_ID = "capture_1_1d7271d73617"
PHASE03_REPORT_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_03/reports/all_theories_brief_recommendation_report")

DIRECTION_CONDITIONS = {
    "deont": ("P_deont_01", "N_neutral_01"),
    "util": ("P_util_01", "N_neutral_01"),
    "generic": ("N_generic_moral_01", "N_neutral_01"),
}
STEERING_STRENGTHS = (1.0,)


def _write_layer() -> int:
    return int(os.getenv("MOREBENCH_STEERING_WRITE_LAYER", "16"))


def _workflow_name() -> str:
    return f"morebench_theory_persona_vectors_phase04_deont_util_steering_L{_write_layer()}"


def _report_output_dir() -> Path:
    return REPORT_OUTPUT_ROOT / f"L{_write_layer()}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _variant_suffix(variant_id: str) -> str:
    for variant in stability.QUESTION_VARIANTS:
        if str(variant["variant_id"]) == variant_id:
            return str(variant["question_suffix"])
    raise KeyError(f"unknown question variant {variant_id!r}")


def _prompt(*, dilemma: str, variant_id: str) -> list[dict[str, str]]:
    case_text = phase03._strip_embedded_question(dilemma)
    user = "\n\n".join([f"Dilemma: {case_text}", _variant_suffix(variant_id)])
    return [
        {"role": "system", "content": phase03.SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_dataset() -> Dataset:
    manifest = _read_json(STEERING_MANIFEST)
    source_by_group = baseline._source_dilemmas()
    primary_trials = [trial for trial in manifest["trials"] if trial["tier"] == "primary"]
    examples: list[Example] = []
    for trial in primary_trials:
        group_id = str(trial["group_id"])
        variant_id = str(trial["variant_id"])
        source = source_by_group.get(group_id)
        if source is None:
            raise KeyError(f"missing source dilemma for {group_id}")
        examples.append(
            Example(
                key=f"{group_id}__{variant_id}__neutral_target",
                prompt=_prompt(dilemma=str(source["dilemma"]), variant_id=variant_id),
                labels={
                    "group_id": group_id,
                    "variant_id": variant_id,
                    "tier": "primary",
                    "condition_id": "N_neutral_01",
                    "expected_deont_action": trial["expected_deont_action"],
                    "expected_util_action": trial["expected_util_action"],
                    "neutral_majority_action": trial["neutral_majority_action"],
                    "baseline_neutral_action": trial["baseline_neutral_action"],
                    "prompt_regime": "phase04_steering_primary_neutral",
                },
                metadata={
                    "dilemma_text": source["dilemma"],
                    "question_suffix": _variant_suffix(variant_id),
                },
                cases={"group_id": group_id, "variant_id": variant_id},
                case_key=f"{group_id}__{variant_id}",
            )
        )
    return Dataset.from_examples(examples, name=_workflow_name())


def _engine():
    return replace(base._engine(), enable_prefix_caching=False)


def export_direction(*, direction_name: str, layer: int) -> TransformResult:
    generation_rows = paired._latest_generation_rows_path(PHASE03_REPORT_ROOT)
    row_index = paired._row_index(paired._rows(generation_rows))
    capture = paired._load_capture(CAPTURE_ID)
    feats = slices._feature_slice_map(capture, site="generated_sequence_residual", layer=int(layer), slice_name="first_16")

    if direction_name == "random":
        deont, _ = slices._paired_direction(
            row_index=row_index,
            feats=feats,
            pos_condition="P_deont_01",
            neg_condition="N_neutral_01",
        )
        rng = np.random.default_rng(1701 + int(layer))
        raw = rng.normal(size=deont.shape).astype(np.float32)
        raw_norm = float(np.linalg.norm(raw))
        target_norm = float(np.linalg.norm(deont))
        if raw_norm > 1e-12:
            raw = raw * (target_norm / raw_norm)
        positive_count = negative_count = 0
    else:
        if direction_name not in DIRECTION_CONDITIONS:
            raise KeyError(f"unknown direction {direction_name!r}")
        pos_condition, neg_condition = DIRECTION_CONDITIONS[direction_name]
        raw, deltas = slices._paired_direction(
            row_index=row_index,
            feats=feats,
            pos_condition=pos_condition,
            neg_condition=neg_condition,
        )
        positive_count = negative_count = int(deltas.shape[0])

    norm = float(np.linalg.norm(raw))
    unit = (raw / norm).astype(np.float32) if norm > 1e-12 else raw.astype(np.float32)
    payload = {
        "kind": "direction_result",
        "feature": f"phase03_generated_first16_L{int(layer)}",
        "name": f"{direction_name}_minus_neutral_L{int(layer)}",
        "layers": {
            str(int(layer)): {
                "vector": unit.astype(float).tolist(),
                "raw_vector": raw.astype(np.float32).astype(float).tolist(),
                "norm": norm,
                "direction_name": direction_name,
                "source_capture_id": CAPTURE_ID,
                "source_slice": "generated_sequence_residual:first_16",
                "positive_count": positive_count,
                "negative_count": negative_count,
            }
        },
        "metadata": {
            "source": "phase04_deont_util_steering_workflow.export_direction",
            "generation_rows": str(generation_rows),
            "write_layer": int(layer),
            "steering_units": "AddDirectionPatch strength multiplies raw_vector.",
        },
        "summary": {
            "layer_count": 1,
            "direction_name": direction_name,
            "layer": int(layer),
            "norm": norm,
        },
    }
    return TransformResult(payload=payload)


def summarize_steering(
    *,
    baseline_generation: Any,
    deont_1_0: Any,
    util_1_0: Any,
    generic_1_0: Any,
    random_1_0: Any,
) -> TransformResult:
    def _rows(artifact: Any) -> list[dict[str, Any]]:
        payload = artifact.result() if hasattr(artifact, "result") else {}
        rows = payload.get("rows") if isinstance(payload, Mapping) else []
        return [dict(row) for row in rows if isinstance(row, Mapping)]

    variants = {
        "baseline": _rows(baseline_generation),
        "deont_1_0": _rows(deont_1_0),
        "util_1_0": _rows(util_1_0),
        "generic_1_0": _rows(generic_1_0),
        "random_1_0": _rows(random_1_0),
    }
    finish_reasons = {
        name: dict(Counter(str(row.get("finish_reason") or row.get("status") or "") for row in rows))
        for name, rows in variants.items()
    }
    token_summary: dict[str, dict[str, float]] = {}
    for name, rows in variants.items():
        counts = []
        for row in rows:
            token_ids = row.get("generated_token_ids")
            counts.append(len(token_ids) if isinstance(token_ids, list) else len(str(row.get("generated_text") or "").split()))
        token_summary[name] = {
            "n": len(counts),
            "mean": mean(counts) if counts else 0,
            "median": median(counts) if counts else 0,
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
        }
    samples: list[dict[str, Any]] = []
    for name, rows in variants.items():
        for row in rows[:3]:
            example = row.get("example") if isinstance(row.get("example"), Mapping) else {}
            labels = example.get("labels") if isinstance(example.get("labels"), Mapping) else {}
            samples.append(
                {
                    "variant": name,
                    "group_id": labels.get("group_id"),
                    "prompt_variant": labels.get("variant_id"),
                    "preview": str(row.get("generated_text") or "")[:500],
                }
            )
    return TransformResult(
        payload={
            "workflow": _workflow_name(),
            "write_layer": _write_layer(),
            "row_counts": {name: len(rows) for name, rows in variants.items()},
            "finish_reasons": finish_reasons,
            "token_summary": token_summary,
            "sample_rows": samples,
            "note": "Semantic action-shift scoring is performed by a follow-up script over the report results.",
        }
    )


def build_runner_specs() -> dict[str, object]:
    if os.getenv(DB_ENV_VAR):
        db = PostgresSource.from_env(DB_ENV_VAR)
        catalog = PostgresCatalog(source=db)
        modal_secrets = (ModalSecret.from_env_var(DB_ENV_VAR, secret_name="xenon-neon"),)
    else:
        catalog = FileCatalog(root=LOCAL_CATALOG_ROOT)
        modal_secrets = ()

    modal_store = ModalVolumeStore(name="xenon-data", root=MODAL_ARTIFACT_ROOT)
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H200",
                timeout_seconds=60 * 60 * 3,
                shard_count=base.GPU_SHARD_COUNT,
                secrets=modal_secrets,
                volumes=(ModalVolumeMount(name=base.MODEL_VOLUME_NAME, mount_path=base.MODEL_VOLUME_PATH),),
            ),
            artifacts=modal_store,
            catalog=catalog,
        ),
        "patch_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H200",
                timeout_seconds=60 * 60 * 3,
                secrets=modal_secrets,
                volumes=(ModalVolumeMount(name=base.MODEL_VOLUME_NAME, mount_path=base.MODEL_VOLUME_PATH),),
            ),
            artifacts=modal_store,
            catalog=catalog,
        ),
        "analysis_cpu": LocalRunnerSpec(
            artifacts=LocalArtifactStore(LOCAL_ARTIFACT_ROOT),
            catalog=catalog,
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(LOCAL_ARTIFACT_ROOT),
            catalog=catalog,
        ),
    }


def _direction_step(name: str, layer: int) -> WorkflowStep:
    return WorkflowStep(
        name=f"export_{name}_direction",
        runner="analysis_cpu",
        spec=TransformSpec(
            builder=TransformBuilder.from_function(
                export_direction,
                local_python_sources=("projects/MOREBENCH", "pipelines_v2"),
            ),
            inputs={"direction_name": name, "layer": layer},
            inline=True,
        ),
    )


def _patch_step(*, direction: str, strength: float, layer: int, extra_depends: tuple[str, ...] = ()) -> WorkflowStep:
    step_name = f"steer_{direction}_{str(strength).replace('.', '_')}"
    return WorkflowStep(
        name=step_name,
        runner="patch_gpu",
        depends_on=(f"export_{direction}_direction", "baseline_generation", *extra_depends),
        spec=PatchedGenerationSpec(
            engine=_engine(),
            dataset=build_dataset(),
            patch=AddDirectionPatch(
                direction=StepRef(f"export_{direction}_direction"),
                write_site=ResidualInterventionSite(site="resid_post", layers=(layer,)),
                target_tokens=TokenSelector.last(),
                strength=float(strength),
            ),
            generation=GenerationSpec(
                enabled=True,
                max_tokens=GENERATION_MAX_TOKENS,
                temperature=GENERATION_TEMPERATURE,
                top_p=GENERATION_TOP_P,
                capture_reasoning=False,
            ),
        ),
    )


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    layer = _write_layer()
    steps: list[WorkflowStep] = [
        _direction_step("deont", layer),
        _direction_step("util", layer),
        _direction_step("generic", layer),
        _direction_step("random", layer),
        WorkflowStep(
            name="baseline_generation",
            runner="capture_gpu",
            spec=GenerationRunSpec(
                engine=_engine(),
                dataset=dataset,
                generation=GenerationSpec(
                    enabled=True,
                    max_tokens=GENERATION_MAX_TOKENS,
                    temperature=GENERATION_TEMPERATURE,
                    top_p=GENERATION_TOP_P,
                    capture_reasoning=False,
                ),
            ),
        ),
    ]
    prior_patch_steps: list[str] = []
    for direction in ("deont", "util", "generic", "random"):
        for strength in STEERING_STRENGTHS:
            step = _patch_step(
                direction=direction,
                strength=strength,
                layer=layer,
                extra_depends=tuple(prior_patch_steps) if os.getenv("MOREBENCH_STEERING_SEQUENTIAL_PATCHES") == "1" else (),
            )
            steps.append(step)
            prior_patch_steps.append(step.name)
    steps.extend(
        [
            WorkflowStep(
                name="summarize_steering",
                runner="analysis_cpu",
                depends_on=("baseline_generation", "steer_deont_1_0", "steer_util_1_0", "steer_generic_1_0", "steer_random_1_0"),
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_steering,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={
                        "baseline_generation": StepRef("baseline_generation"),
                        "deont_1_0": StepRef("steer_deont_1_0"),
                        "util_1_0": StepRef("steer_util_1_0"),
                        "generic_1_0": StepRef("steer_generic_1_0"),
                        "random_1_0": StepRef("steer_random_1_0"),
                    },
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                depends_on=("summarize_steering",),
                spec=ReportSpec(
                    inputs=(
                        StepRef("baseline_generation"),
                        StepRef("steer_deont_1_0"),
                        StepRef("steer_util_1_0"),
                        StepRef("steer_generic_1_0"),
                        StepRef("steer_random_1_0"),
                        StepRef("summarize_steering"),
                    ),
                    template="default",
                    output_dir=str(_report_output_dir()),
                ),
            ),
        ]
    )
    return WorkflowSpec(name=_workflow_name(), steps=tuple(steps))
