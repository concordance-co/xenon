"""Activation-steering smoke on the controlled deontology prompt-isolation setup.

This workflow reuses the controlled phase-03 deontology capture artifact and
tests whether generated-space directions can steer neutral/generic target
prompts toward the corresponding deontological responses.
"""

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
from projects.MOREBENCH.theory_persona_vectors.phase_03.specs import all_theories_natural_prompt_workflow as natural
from projects.MOREBENCH.theory_persona_vectors.phase_03.specs import deont_prompt_isolation_workflow as prompt_iso


WORKFLOW_NAME = "morebench_theory_persona_vectors_phase03_deont_prompt_isolation_steering"
PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_03")
SOURCE_REPORT_ROOT = PHASE_ROOT / "reports" / "deont_prompt_isolation_report"
REPORT_OUTPUT_ROOT = PHASE_ROOT / "reports" / "deont_prompt_isolation_steering"
CAPTURE_ID = "capture_1_c479d296a725"
SOURCE_SITE = "generated_sequence_residual"
SOURCE_SLICE = "first_16"
GENERATION_MAX_TOKENS = 128
WRITE_STRENGTH = 1.0

TARGET_CONDITIONS = {"N_neutral_iso_01", "N_generic_moral_iso_01"}

DIRECTION_PAIRS = {
    "deont01_neutral": ("P_deont_iso_01", "N_neutral_iso_01"),
    "deont02_neutral": ("P_deont_iso_02", "N_neutral_iso_01"),
    "generic_neutral": ("N_generic_moral_iso_01", "N_neutral_iso_01"),
    "deont01_generic": ("P_deont_iso_01", "N_generic_moral_iso_01"),
    "deont02_generic": ("P_deont_iso_02", "N_generic_moral_iso_01"),
}

PATCH_VARIANTS = (
    ("steer_deont01_on_neutral", "deont01_neutral", "N_neutral_iso_01"),
    ("steer_deont02_on_neutral", "deont02_neutral", "N_neutral_iso_01"),
    ("steer_generic_on_neutral", "generic_neutral", "N_neutral_iso_01"),
    ("steer_random_on_neutral", "random_neutral", "N_neutral_iso_01"),
    ("steer_deont01_on_generic", "deont01_generic", "N_generic_moral_iso_01"),
    ("steer_deont02_on_generic", "deont02_generic", "N_generic_moral_iso_01"),
    ("steer_random_on_generic", "random_generic", "N_generic_moral_iso_01"),
)


def _write_layer() -> int:
    return int(os.getenv("MOREBENCH_DEONT_STEERING_WRITE_LAYER", "32"))


def _workflow_name() -> str:
    return f"{WORKFLOW_NAME}_L{_write_layer()}"


def _report_output_dir() -> Path:
    return REPORT_OUTPUT_ROOT / f"L{_write_layer()}"


def _engine():
    return replace(base._engine(), enable_prefix_caching=False)


def _source_rows_path() -> Path:
    candidates = sorted(
        SOURCE_REPORT_ROOT.glob("report_*/results/generate_natural_responses_results.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no generation rows found under {SOURCE_REPORT_ROOT}")
    return candidates[0]


def build_dataset() -> Dataset:
    source = prompt_iso.build_dataset()
    examples = [
        example
        for example in source.examples
        if str(example.labels.get("condition_id") or "") in TARGET_CONDITIONS
    ]
    return Dataset.from_examples(examples, name=f"{_workflow_name()}_targets")


def _condition_rows(dataset: Dataset, condition_id: str):
    return dataset.labels("condition_id").equals(condition_id)


def export_direction(*, direction_name: str, layer: int) -> TransformResult:
    if direction_name.startswith("random_"):
        anchor = direction_name.removeprefix("random_")
        if anchor == "neutral":
            norm_source = "deont01_neutral"
        elif anchor == "generic":
            norm_source = "deont01_generic"
        else:
            raise KeyError(f"unknown random anchor {anchor!r}")
        pos_condition, neg_condition = DIRECTION_PAIRS[norm_source]
        seed = 1701 + int(layer) + (11 if anchor == "generic" else 0)
    else:
        if direction_name not in DIRECTION_PAIRS:
            raise KeyError(f"unknown direction {direction_name!r}")
        pos_condition, neg_condition = DIRECTION_PAIRS[direction_name]
        seed = None

    row_index = paired._row_index(paired._rows(_source_rows_path()))
    capture = paired._load_capture(CAPTURE_ID)
    feats = slices._feature_slice_map(capture, site=SOURCE_SITE, layer=int(layer), slice_name=SOURCE_SLICE)

    raw, deltas = slices._paired_direction(
        row_index=row_index,
        feats=feats,
        pos_condition=pos_condition,
        neg_condition=neg_condition,
    )
    norm = float(np.linalg.norm(raw))
    if direction_name.startswith("random_"):
        rng = np.random.default_rng(seed)
        rand = rng.normal(size=raw.shape).astype(np.float32)
        rand_norm = float(np.linalg.norm(rand))
        if rand_norm > 1e-12 and norm > 1e-12:
            raw = rand * (norm / rand_norm)
        else:
            raw = rand
    unit = (raw / norm).astype(np.float32) if norm > 1e-12 else raw.astype(np.float32)
    payload = {
        "kind": "direction_result",
        "feature": f"{SOURCE_SITE}:{SOURCE_SLICE}:L{int(layer)}",
        "name": f"{direction_name}_L{int(layer)}",
        "layers": {
            str(int(layer)): {
                "vector": unit.astype(float).tolist(),
                "raw_vector": raw.astype(np.float32).astype(float).tolist(),
                "norm": float(np.linalg.norm(raw)),
                "direction_name": direction_name,
                "source_capture_id": CAPTURE_ID,
                "source_slice": f"{SOURCE_SITE}:{SOURCE_SLICE}",
                "positive_condition": pos_condition,
                "negative_condition": neg_condition,
                "positive_count": int(deltas.shape[0]),
                "negative_count": int(deltas.shape[0]),
            }
        },
        "metadata": {
            "source": "phase_03/deont_prompt_isolation_steering_workflow.export_direction",
            "source_rows_path": str(_source_rows_path()),
            "write_layer": int(layer),
            "steering_units": "AddDirectionPatch strength multiplies raw_vector.",
        },
        "summary": {
            "direction_name": direction_name,
            "layer": int(layer),
            "norm": float(np.linalg.norm(raw)),
        },
    }
    return TransformResult(payload=payload)


def summarize_runs(**artifacts: Any) -> TransformResult:
    def _rows(artifact: Any) -> list[dict[str, Any]]:
        payload = artifact.result() if hasattr(artifact, "result") else {}
        rows = payload.get("rows") if isinstance(payload, Mapping) else []
        return [dict(row) for row in rows if isinstance(row, Mapping)]

    rows_by_name = {name: _rows(artifact) for name, artifact in artifacts.items()}
    finish_reasons = {
        name: dict(Counter(str(row.get("finish_reason") or row.get("status") or "") for row in rows))
        for name, rows in rows_by_name.items()
    }
    token_summary: dict[str, dict[str, float]] = {}
    for name, rows in rows_by_name.items():
        counts: list[int] = []
        for row in rows:
            token_ids = row.get("generated_token_ids")
            if isinstance(token_ids, list):
                counts.append(len(token_ids))
            else:
                counts.append(len(str(row.get("generated_text") or "").split()))
        token_summary[name] = {
            "n": len(counts),
            "mean": mean(counts) if counts else 0.0,
            "median": median(counts) if counts else 0.0,
            "min": min(counts) if counts else 0.0,
            "max": max(counts) if counts else 0.0,
        }
    samples: list[dict[str, Any]] = []
    for name, rows in rows_by_name.items():
        for row in rows[:2]:
            example = row.get("example") if isinstance(row.get("example"), Mapping) else {}
            labels = example.get("labels") if isinstance(example.get("labels"), Mapping) else {}
            samples.append(
                {
                    "variant": name,
                    "dilemma_id": labels.get("dilemma_id"),
                    "condition_id": labels.get("condition_id"),
                    "preview": str(row.get("generated_text") or "")[:400],
                }
            )
    return TransformResult(
        payload={
            "workflow": _workflow_name(),
            "write_layer": _write_layer(),
            "source_site": SOURCE_SITE,
            "source_slice": SOURCE_SLICE,
            "row_counts": {name: len(rows) for name, rows in rows_by_name.items()},
            "finish_reasons": finish_reasons,
            "token_summary": token_summary,
            "sample_rows": samples,
        }
    )


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


def _patch_step(*, step_name: str, direction_name: str, target_condition: str, dataset: Dataset, layer: int) -> WorkflowStep:
    return WorkflowStep(
        name=step_name,
        runner="patch_gpu",
        depends_on=("baseline_generation", f"export_{direction_name}_direction"),
        spec=PatchedGenerationSpec(
            engine=_engine(),
            dataset=dataset,
            patch=AddDirectionPatch(
                direction=StepRef(f"export_{direction_name}_direction"),
                write_site=ResidualInterventionSite(site="resid_post", layers=(layer,)),
                target_tokens=TokenSelector.last(),
                strength=WRITE_STRENGTH,
            ),
            select_when=_condition_rows(dataset, target_condition),
            generation=GenerationSpec(
                enabled=True,
                max_tokens=GENERATION_MAX_TOKENS,
                temperature=0.0,
                top_p=1.0,
                capture_reasoning=False,
            ),
        ),
    )


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    layer = _write_layer()
    direction_names = (
        "deont01_neutral",
        "deont02_neutral",
        "generic_neutral",
        "random_neutral",
        "deont01_generic",
        "deont02_generic",
        "random_generic",
    )
    steps: list[WorkflowStep] = [
        *(_direction_step(name, layer) for name in direction_names),
        WorkflowStep(
            name="baseline_generation",
            runner="capture_gpu",
            spec=GenerationRunSpec(
                engine=_engine(),
                dataset=dataset,
                generation=GenerationSpec(
                    enabled=True,
                    max_tokens=GENERATION_MAX_TOKENS,
                    temperature=0.0,
                    top_p=1.0,
                    capture_reasoning=False,
                ),
            ),
        ),
        *(
            _patch_step(
                step_name=step_name,
                direction_name=direction_name,
                target_condition=target_condition,
                dataset=dataset,
                layer=layer,
            )
            for step_name, direction_name, target_condition in PATCH_VARIANTS
        ),
        WorkflowStep(
            name="summarize_steering",
            runner="analysis_cpu",
            depends_on=("baseline_generation", *(name for name, _, _ in PATCH_VARIANTS)),
            spec=TransformSpec(
                builder=TransformBuilder.from_function(
                    summarize_runs,
                    local_python_sources=("projects/MOREBENCH",),
                ),
                inputs={
                    "baseline_generation": StepRef("baseline_generation"),
                    **{name: StepRef(name) for name, _, _ in PATCH_VARIANTS},
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
                    *(StepRef(name) for name, _, _ in PATCH_VARIANTS),
                    StepRef("summarize_steering"),
                ),
                template="default",
                output_dir=str(_report_output_dir()),
            ),
        ),
    ]
    return WorkflowSpec(name=_workflow_name(), steps=tuple(steps))


def build_runner_specs() -> dict[str, object]:
    if os.getenv(natural.DB_ENV_VAR):
        db = PostgresSource.from_env(natural.DB_ENV_VAR)
        catalog = PostgresCatalog(source=db)
        modal_secrets = (ModalSecret.from_env_var(natural.DB_ENV_VAR, secret_name="xenon-neon"),)
    else:
        catalog = FileCatalog(root=natural.LOCAL_CATALOG_ROOT)
        modal_secrets = ()

    modal_store = ModalVolumeStore(name="xenon-data", root=natural.MODAL_ARTIFACT_ROOT)
    model_volume = ModalVolumeMount(name=base.MODEL_VOLUME_NAME, mount_path=base.MODEL_VOLUME_PATH)
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H200",
                timeout_seconds=60 * 60 * 4,
                shard_count=base.GPU_SHARD_COUNT,
                secrets=modal_secrets,
                volumes=(model_volume,),
            ),
            artifacts=modal_store,
            catalog=catalog,
        ),
        "patch_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H200",
                timeout_seconds=60 * 60 * 4,
                secrets=modal_secrets,
                volumes=(model_volume,),
            ),
            artifacts=modal_store,
            catalog=catalog,
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=4,
                memory_mb=12 * 1024,
                timeout_seconds=60 * 60,
                secrets=modal_secrets,
            ),
            artifacts=modal_store,
            catalog=catalog,
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(natural.LOCAL_ARTIFACT_ROOT),
            catalog=catalog,
        ),
    }
