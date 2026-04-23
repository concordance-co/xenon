from __future__ import annotations

"""Initial pipelines_v2 workflow for prompt_confusion phase_11."""

import json
import os
from pathlib import Path

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    DirectionSpec,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalVolumeMount,
    ModalVolumeStore,
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
from projects.DX_TERMINAL.prompt_confusion.catalogs import build_prompt_confusion_catalog


MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"
DEFAULT_DATASET_PATH = Path(
    os.environ.get(
        "PHASE_11_DATASET_PATH",
        "projects/DX_TERMINAL/prompt_confusion/phase_11/outputs/phase_11_dataset/phase_11_dataset.jsonl",
    )
)
DEFAULT_REPORT_DIR = "projects/DX_TERMINAL/prompt_confusion/phase_11/reports/pipelines_v2"
CAPTURED_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44)


def _load_dataset_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Phase 11 dataset not found: {path}")
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    if not rows:
        raise ValueError(f"Phase 11 dataset is empty: {path}")
    return rows


def build_dataset() -> Dataset:
    return Dataset.from_records(
        _load_dataset_records(DEFAULT_DATASET_PATH),
        prompt_column="prompt_messages_json",
        example_key_column="example_id",
        label_columns=[
            "user_text",
            "target_dimension",
            "strategy_size_direction",
            "strategy_risk_direction",
            "size_setting_value",
            "risk_setting_value",
            "size_setting_implied_direction",
            "risk_setting_implied_direction",
            "size_conflict_present",
            "risk_conflict_present",
            "any_conflict_present",
            "double_conflict_present",
            "conflict_count",
            "edge_conflict",
            "conflict_band",
            "lexical_split",
            "strategy_lexical_split",
            "settings_lexical_split",
            "context_variant_id",
        ],
        case_columns=["matched_group_id", "matched_pair_id"],
        case_key_column="matched_group_id",
        name="prompt_confusion_phase_11_multi_conflict",
    )


def _default_residual_engine() -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        max_model_len=8192,
        enforce_eager=False,
        max_num_seqs=16,
    )


def build_runner_specs() -> dict[str, object]:
    artifact_store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts/prompt_confusion_phase_11",
    )
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H100",
                volumes=(ModalVolumeMount(name=MODEL_VOLUME_NAME, mount_path=MODEL_VOLUME_PATH),),
            ),
            artifacts=artifact_store,
            catalog=build_prompt_confusion_catalog(__file__),
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(cpu=6, memory_mb=24 * 1024),
            artifacts=artifact_store,
            catalog=build_prompt_confusion_catalog(__file__),
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(Path("artifacts") / "prompt_confusion_phase_11"),
            catalog=build_prompt_confusion_catalog(__file__),
        ),
    }


def _probe_step(*, name: str, dataset: Dataset, label_name: str) -> WorkflowStep:
    return WorkflowStep(
        name=name,
        runner="analysis_cpu",
        spec=ProbeSpec(
            feature=StepRef("capture_prompt_eos_residual").feature("residual_prompt_eos"),
            rows=dataset.labels("edge_conflict").equals(False),
            labels=dataset.labels(label_name),
            group_by=dataset.cases("matched_group_id"),
            split=dataset.labels("lexical_split"),
            train_values=("train",),
            test_values=("test",),
            tokens=TokenSelector.full_sequence(),
            pooling=TokenPooling.last(),
            metrics=("accuracy", "balanced_accuracy", "auroc", "selectivity"),
            baselines=("majority", "shuffled_label"),
        ),
    )


def _direction_step(*, name: str, dataset: Dataset, label_name: str) -> WorkflowStep:
    return WorkflowStep(
        name=name,
        runner="analysis_cpu",
        spec=DirectionSpec(
            feature=StepRef("capture_prompt_eos_residual").feature("residual_prompt_eos"),
            positive=dataset.labels(label_name).equals(True),
            negative=dataset.labels(label_name).equals(False),
            layers=list(CAPTURED_LAYERS),
            tokens=TokenSelector.full_sequence(),
            pooling=TokenPooling.last(),
        ),
    )


def build_workflow(
    dataset: Dataset | None = None,
    *,
    residual_engine: object | None = None,
    report_output_dir: str = DEFAULT_REPORT_DIR,
) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    residual_engine = residual_engine or _default_residual_engine()
    rows = dataset.labels("edge_conflict").equals(False)

    return WorkflowSpec(
        name="prompt_confusion_phase_11_pipelines_v2",
        steps=(
            WorkflowStep(
                name="text_size_conflict_gate",
                runner="analysis_cpu",
                spec=TextBaselineSpec(
                    text=dataset.labels("user_text"),
                    rows=rows,
                    labels=dataset.labels("size_conflict_present"),
                    group_by=dataset.cases("matched_group_id"),
                    split_by={"lexical_split": dataset.labels("lexical_split")},
                    train_values=("train",),
                    test_values=("test",),
                    model="countvectorizer_logreg",
                    metrics=("accuracy", "balanced_accuracy", "auroc"),
                ),
            ),
            WorkflowStep(
                name="text_risk_conflict_gate",
                runner="analysis_cpu",
                spec=TextBaselineSpec(
                    text=dataset.labels("user_text"),
                    rows=rows,
                    labels=dataset.labels("risk_conflict_present"),
                    group_by=dataset.cases("matched_group_id"),
                    split_by={"lexical_split": dataset.labels("lexical_split")},
                    train_values=("train",),
                    test_values=("test",),
                    model="countvectorizer_logreg",
                    metrics=("accuracy", "balanced_accuracy", "auroc"),
                ),
            ),
            WorkflowStep(
                name="capture_prompt_eos_residual",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=residual_engine,
                    dataset=dataset,
                    sites=[
                        ResidualSite(
                            name="residual_prompt_eos",
                            site="resid_post",
                            layers=list(CAPTURED_LAYERS),
                            tokens=TokenSelector.last(),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                    ],
                ),
            ),
            _probe_step(name="size_conflict_probe", dataset=dataset, label_name="size_conflict_present"),
            _probe_step(name="risk_conflict_probe", dataset=dataset, label_name="risk_conflict_present"),
            _probe_step(name="any_conflict_probe", dataset=dataset, label_name="any_conflict_present"),
            _probe_step(name="double_conflict_probe", dataset=dataset, label_name="double_conflict_present"),
            _direction_step(name="size_conflict_direction", dataset=dataset, label_name="size_conflict_present"),
            _direction_step(name="risk_conflict_direction", dataset=dataset, label_name="risk_conflict_present"),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=(
                        StepRef("text_size_conflict_gate"),
                        StepRef("text_risk_conflict_gate"),
                        StepRef("size_conflict_probe"),
                        StepRef("risk_conflict_probe"),
                        StepRef("any_conflict_probe"),
                        StepRef("double_conflict_probe"),
                        StepRef("size_conflict_direction"),
                        StepRef("risk_conflict_direction"),
                    ),
                    template="default",
                    output_dir=report_output_dir,
                ),
            ),
        ),
    )
