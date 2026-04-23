from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    Example,
    GenerationSpec,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalVolumeMount,
    ModalVolumeStore,
    ProbeSpec,
    PromptMetadataBuilder,
    ReportSpec,
    ResidualSite,
    StepRef,
    TensorStorage,
    TextBaselineSpec,
    TokenPooling,
    TokenSelector,
    TransferProbeSpec,
    TransformBuilder,
    TransformResult,
    TransformSpec,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)


PHASE_ROOT = Path("projects/MOREBENCH/phase_03")
DATASET_PATH = PHASE_ROOT / "outputs" / "experiment_01_prompt_dataset.jsonl"
REPORT_OUTPUT_DIR = PHASE_ROOT / "reports" / "experiment_01_report"

MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"
MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
MODAL_ARTIFACT_ROOT = "/data/artifacts/mech_interp_morebench_phase03_experiment01"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "mech_interp_morebench_phase03_experiment01"

CAPTURED_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44)
THEORY_NAMES = (
    "Act Utilitarianism",
    "Aristotelian Virtue Ethics",
    "Gauthierian Contractarianism",
    "Kantian Deontology",
    "Scanlonian Contractualism",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def build_dataset() -> Dataset:
    records = _load_jsonl(DATASET_PATH)
    examples: list[Example] = []
    for record in records:
        examples.append(
            Example(
                key=str(record["example_id"]),
                prompt=str(record["prompt"]),
                labels={
                    "theory_identity": str(record["theory_identity"]),
                    "theory_or_control": str(record["theory_or_control"]),
                    "prompt_family": str(record["prompt_family"]),
                    "split": str(record["split"]),
                    "source_family": str(record["source_family"]),
                    "dilemma_type": str(record["dilemma_type"]),
                    "context": str(record["context"]),
                    "role_domain": str(record["role_domain"]),
                    "is_theory_bearing": bool(record["is_theory_bearing"]),
                    "is_named_theory": bool(record["is_named_theory"]),
                    "has_theory_name": bool(record["has_theory_name"]),
                    "anchor_text": str(record["anchor_text"]),
                },
                cases={"group_id": str(record["group_id"])},
                case_key=str(record["group_id"]),
            )
        )
    return Dataset.from_examples(examples, name="morebench_phase03_experiment01_theory_identity")


def _prompt_sections(rendered_prompt: str) -> dict[str, object]:
    marker = "\n\nDILEMMA:\n"
    dilemma_header_start = rendered_prompt.index(marker)
    intro_end = dilemma_header_start
    intro = rendered_prompt[:intro_end]
    dilemma_start = dilemma_header_start + len(marker)

    token_sections: dict[str, dict[str, int]] = {
        "INTRO": {"char_start": 0, "char_end": intro_end},
        "DILEMMA": {"char_start": dilemma_start, "char_end": len(rendered_prompt)},
    }

    sentence_matches = list(re.finditer(r"[^.]+\.", intro))
    if sentence_matches:
        first = sentence_matches[0]
        first_text = first.group(0)
        if any(name in first_text for name in THEORY_NAMES):
            token_sections["THEORY_CLAUSE"] = {"char_start": first.start(), "char_end": first.end()}
            if len(sentence_matches) > 1:
                second = sentence_matches[1]
                token_sections["ANCHOR_CLAUSE"] = {"char_start": second.start(), "char_end": second.end()}
        elif len(sentence_matches) > 1:
            second = sentence_matches[1]
            token_sections["ANCHOR_CLAUSE"] = {"char_start": second.start(), "char_end": second.end()}

    return {"token_sections": token_sections}


def _engine(*, max_num_seqs: int = 8) -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        max_model_len=8192,
        enforce_eager=False,
        enable_prefix_caching=False,
        max_num_seqs=max_num_seqs,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def build_runner_specs() -> dict[str, object]:
    modal_store = ModalVolumeStore(name="xenon-data", root=MODAL_ARTIFACT_ROOT)
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H100",
                timeout_seconds=60 * 60 * 4,
                volumes=(ModalVolumeMount(name=MODEL_VOLUME_NAME, mount_path=MODEL_VOLUME_PATH),),
            ),
            artifacts=modal_store,
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=8,
                memory_mb=24 * 1024,
                timeout_seconds=60 * 60 * 2,
            ),
            artifacts=modal_store,
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(LOCAL_ARTIFACT_ROOT),
        ),
    }


def summarize_experiment_01(
    overall_probe: Any,
    theory_clause_probe: Any,
    anchor_clause_probe: Any,
    eos_transfer: Any,
    text_baseline: Any,
) -> TransformResult:
    def _best(result: Any, metric: str) -> dict[str, Any]:
        layers = list(result.payload().get("layers", []))
        return max(layers, key=lambda item: float(item.get(metric, 0.0))) if layers else {}

    def _transfer_best(result: Any, metric: str) -> dict[str, Any]:
        best: dict[str, Any] | None = None
        for layer in result.payload().get("layers", []):
            for comparison in layer.get("comparisons", []):
                score = float(comparison.get(metric, 0.0))
                if best is None or score > float(best.get(metric, 0.0)):
                    best = {
                        "layer": layer.get("layer"),
                        "train_cohort": comparison.get("train_cohort"),
                        "test_cohort": comparison.get("test_cohort"),
                        metric: score,
                    }
        return best or {}

    overall_best = _best(overall_probe, "balanced_accuracy")
    theory_clause_best = _best(theory_clause_probe, "balanced_accuracy")
    anchor_clause_best = _best(anchor_clause_probe, "balanced_accuracy")
    transfer_best = _transfer_best(eos_transfer, "balanced_accuracy")
    text_summary = dict(text_baseline.payload().get("summary", {}))

    payload = {
        "benchmark": "morebench",
        "phase": "03",
        "experiment": "experiment_01_theory_identity_prompt_readout",
        "overall_prompt_eos_best": overall_best,
        "theory_clause_best": theory_clause_best,
        "anchor_clause_best": anchor_clause_best,
        "transfer_best": transfer_best,
        "text_baseline_summary": text_summary,
    }
    return TransformResult(payload=payload)


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    theory_rows = dataset.labels("is_theory_bearing").equals(True)
    named_rows = dataset.labels("is_named_theory").equals(True)
    theory_keys = [example.key for example in dataset.examples if bool(example.labels["is_theory_bearing"])]
    named_keys = [example.key for example in dataset.examples if bool(example.labels["is_named_theory"])]
    theory_dataset = dataset.select(keys=theory_keys)
    named_dataset = dataset.select(keys=named_keys)

    return WorkflowSpec(
        name="morebench_phase03_experiment01_theory_identity",
        steps=(
            WorkflowStep(
                name="anchor_text_baseline",
                runner="analysis_cpu",
                spec=TextBaselineSpec(
                    text=dataset.labels("anchor_text"),
                    rows=theory_rows,
                    labels=dataset.labels("theory_identity"),
                    group_by=dataset.cases("group_id"),
                    split_by={"split": dataset.labels("split")},
                    train_values=("train",),
                    test_values=("test",),
                    model="countvectorizer_logreg",
                    metrics=("accuracy", "balanced_accuracy"),
                ),
            ),
            WorkflowStep(
                name="capture_prompt_eos",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=_engine(),
                    dataset=dataset,
                    prompt_metadata_builder=PromptMetadataBuilder.from_function(
                        _prompt_sections,
                        local_python_sources=("projects/MOREBENCH/phase_03/specs",),
                    ),
                    sites=[
                        ResidualSite(
                            name="prompt_eos_residual",
                            site="resid_post",
                            layers=list(CAPTURED_LAYERS),
                            tokens=TokenSelector.last(),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        )
                    ],
                    generation=GenerationSpec(enabled=False),
                ),
            ),
            WorkflowStep(
                name="capture_theory_clause",
                runner="capture_gpu",
                depends_on=("capture_prompt_eos",),
                spec=CaptureSpec(
                    engine=_engine(max_num_seqs=2),
                    dataset=named_dataset,
                    prompt_metadata_builder=PromptMetadataBuilder.from_function(
                        _prompt_sections,
                        local_python_sources=("projects/MOREBENCH/phase_03/specs",),
                    ),
                    sites=[
                        ResidualSite(
                            name="theory_clause_residual",
                            site="resid_post",
                            layers=list(CAPTURED_LAYERS),
                            tokens=TokenSelector.section("THEORY_CLAUSE"),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        )
                    ],
                    generation=GenerationSpec(enabled=False),
                ),
            ),
            WorkflowStep(
                name="capture_anchor_clause",
                runner="capture_gpu",
                depends_on=("capture_theory_clause",),
                spec=CaptureSpec(
                    engine=_engine(max_num_seqs=2),
                    dataset=theory_dataset,
                    prompt_metadata_builder=PromptMetadataBuilder.from_function(
                        _prompt_sections,
                        local_python_sources=("projects/MOREBENCH/phase_03/specs",),
                    ),
                    sites=[
                        ResidualSite(
                            name="anchor_clause_residual",
                            site="resid_post",
                            layers=list(CAPTURED_LAYERS),
                            tokens=TokenSelector.section("ANCHOR_CLAUSE"),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        )
                    ],
                    generation=GenerationSpec(enabled=False),
                ),
            ),
            WorkflowStep(
                name="probe_prompt_eos_theory_or_control",
                runner="analysis_cpu",
                spec=ProbeSpec(
                    feature=StepRef("capture_prompt_eos").feature("prompt_eos_residual"),
                    labels=dataset.labels("theory_or_control"),
                    group_by=dataset.cases("group_id"),
                    split=dataset.labels("split"),
                    train_values=("train",),
                    test_values=("test",),
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.last(),
                    metrics=("accuracy", "balanced_accuracy", "selectivity"),
                    baselines=("majority", "shuffled_label"),
                ),
            ),
            WorkflowStep(
                name="probe_theory_clause_named_rows",
                runner="analysis_cpu",
                spec=ProbeSpec(
                    feature=StepRef("capture_theory_clause").feature("theory_clause_residual"),
                    rows=named_rows,
                    labels=dataset.labels("theory_identity"),
                    group_by=dataset.cases("group_id"),
                    split=dataset.labels("split"),
                    train_values=("train",),
                    test_values=("test",),
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.mean(),
                    metrics=("accuracy", "balanced_accuracy", "selectivity"),
                    baselines=("majority", "shuffled_label"),
                ),
            ),
            WorkflowStep(
                name="probe_anchor_clause_theory_rows",
                runner="analysis_cpu",
                spec=ProbeSpec(
                    feature=StepRef("capture_anchor_clause").feature("anchor_clause_residual"),
                    rows=theory_rows,
                    labels=dataset.labels("theory_identity"),
                    group_by=dataset.cases("group_id"),
                    split=dataset.labels("split"),
                    train_values=("train",),
                    test_values=("test",),
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.mean(),
                    metrics=("accuracy", "balanced_accuracy", "selectivity"),
                    baselines=("majority", "shuffled_label"),
                ),
            ),
            WorkflowStep(
                name="transfer_prompt_eos_across_families",
                runner="analysis_cpu",
                spec=TransferProbeSpec(
                    feature=StepRef("capture_prompt_eos").feature("prompt_eos_residual"),
                    rows=theory_rows,
                    labels=dataset.labels("theory_identity"),
                    group_by=dataset.cases("group_id"),
                    cohort_by=dataset.labels("prompt_family"),
                    cohort_values=("theory_direct", "theory_wording_variant", "anchor_only"),
                    split_by={"split": dataset.labels("split")},
                    train_values=("train",),
                    test_values=("test",),
                    metrics=("accuracy", "balanced_accuracy"),
                    baselines=("majority",),
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.last(),
                ),
            ),
            WorkflowStep(
                name="summarize_experiment_01",
                runner="analysis_cpu",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(summarize_experiment_01),
                    inputs={
                        "overall_probe": StepRef("probe_prompt_eos_theory_or_control"),
                        "theory_clause_probe": StepRef("probe_theory_clause_named_rows"),
                        "anchor_clause_probe": StepRef("probe_anchor_clause_theory_rows"),
                        "eos_transfer": StepRef("transfer_prompt_eos_across_families"),
                        "text_baseline": StepRef("anchor_text_baseline"),
                    },
                    inline=False,
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=(
                        StepRef("anchor_text_baseline"),
                        StepRef("probe_prompt_eos_theory_or_control"),
                        StepRef("probe_theory_clause_named_rows"),
                        StepRef("probe_anchor_clause_theory_rows"),
                        StepRef("transfer_prompt_eos_across_families"),
                        StepRef("summarize_experiment_01"),
                    ),
                    template="default",
                    output_dir=REPORT_OUTPUT_DIR,
                ),
            ),
        ),
    )
