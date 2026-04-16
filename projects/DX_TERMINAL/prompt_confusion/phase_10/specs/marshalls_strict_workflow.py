from __future__ import annotations

"""Strict both-axes holdout battery for prompt_confusion phase_10."""

import json
import os
from pathlib import Path

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalVolumeMount,
    ModalVolumeStore,
    NullCatalog,
    ProbeSpec,
    ReportSpec,
    ResidualSite,
    StepRef,
    TensorStorage,
    TextBaselineSpec,
    TokenPooling,
    TokenSelector,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)


PHASE_ROOT = Path("projects/DX_TERMINAL/prompt_confusion/phase_10")
DEFAULT_DATASET_PATH = Path(
    os.environ.get(
        "PHASE_10_DATASET_PATH",
        str(PHASE_ROOT / "outputs" / "phase_10_dataset" / "phase_10_dataset.jsonl"),
    )
)
BATTERY_REPORT_DIR = str(PHASE_ROOT / "reports" / "marshalls_strict")
MODAL_ARTIFACT_ROOT = "/data/artifacts/prompt_confusion_phase_10_marshalls_strict"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "prompt_confusion_phase_10_marshalls_strict"

MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"
MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
ARTIFACTS_VOLUME = "xenon-data"

CAPTURED_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44)


def _load_dataset_records() -> list[dict[str, object]]:
    if not DEFAULT_DATASET_PATH.exists():
        raise FileNotFoundError(f"Phase 10 dataset not found: {DEFAULT_DATASET_PATH}")
    rows: list[dict[str, object]] = []
    with DEFAULT_DATASET_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    if not rows:
        raise ValueError(f"Phase 10 dataset is empty: {DEFAULT_DATASET_PATH}")
    return rows


def build_dataset() -> Dataset:
    return Dataset.from_records(
        _load_dataset_records(),
        prompt_column="prompt_messages_json",
        example_key_column="example_id",
        label_columns=[
            "user_text",
            "target_dimension",
            "strategy_direction",
            "setting_value",
            "setting_implied_direction",
            "conflict_present",
            "edge_conflict",
            "conflict_band",
            "lexical_split",
            "strategy_lexical_split",
            "settings_lexical_split",
            "strict_combined_split",
        ],
        case_columns=["matched_group_id", "matched_pair_id"],
        case_key_column="matched_group_id",
        name="prompt_confusion_phase_10_marshalls_strict",
    )


def _engine() -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        max_model_len=8192,
        enforce_eager=False,
        enable_prefix_caching=True,
        max_num_seqs=16,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_runner_specs() -> dict[str, object]:
    artifact_store = ModalVolumeStore(name=ARTIFACTS_VOLUME, root=MODAL_ARTIFACT_ROOT)
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H100",
                timeout_seconds=60 * 60 * 2,
                volumes=(ModalVolumeMount(name=MODEL_VOLUME_NAME, mount_path=MODEL_VOLUME_PATH),),
            ),
            artifacts=artifact_store,
            catalog=NullCatalog(),
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(cpu=6, memory_mb=24 * 1024),
            artifacts=artifact_store,
            catalog=NullCatalog(),
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(LOCAL_ARTIFACT_ROOT),
            catalog=NullCatalog(),
        ),
    }


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    canonical = dataset.labels("edge_conflict").equals(False)

    return WorkflowSpec(
        name="phase_10_marshalls_strict",
        steps=(
            WorkflowStep(
                name="text_baseline_conflict_strict_combined",
                runner="analysis_cpu",
                spec=TextBaselineSpec(
                    text=dataset.labels("user_text"),
                    rows=canonical,
                    labels=dataset.labels("conflict_present"),
                    group_by=dataset.cases("matched_group_id"),
                    split_by={"combined": dataset.labels("strict_combined_split")},
                    train_values=("strict_train",),
                    test_values=("strict_test",),
                    model="countvectorizer_logreg",
                    metrics=("balanced_accuracy", "auroc"),
                ),
            ),
            WorkflowStep(
                name="capture_residual",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=_engine(),
                    dataset=dataset,
                    sites=[
                        ResidualSite(
                            name="residual_prompt_eos",
                            site="resid_post",
                            layers=list(CAPTURED_LAYERS),
                            tokens=TokenSelector.last(),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        )
                    ],
                ),
            ),
            WorkflowStep(
                name="probe_conflict_strict_combined_holdout",
                runner="analysis_cpu",
                spec=ProbeSpec(
                    feature=StepRef("capture_residual").feature("residual_prompt_eos"),
                    rows=canonical,
                    labels=dataset.labels("conflict_present"),
                    group_by=dataset.cases("matched_group_id"),
                    split=dataset.labels("strict_combined_split"),
                    train_values=("strict_train",),
                    test_values=("strict_test",),
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.last(),
                    metrics=("balanced_accuracy", "auroc", "selectivity"),
                    baselines=("majority", "shuffled_label"),
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    template="default",
                    output_dir=BATTERY_REPORT_DIR,
                    inputs=(
                        StepRef("text_baseline_conflict_strict_combined"),
                        StepRef("probe_conflict_strict_combined_holdout"),
                    ),
                ),
            ),
        ),
    )
