from __future__ import annotations

"""pipelines_v2 workflow scaffold for prompt_confusion phase_12."""

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
        "PHASE_12_DATASET_PATH",
        "projects/DX_TERMINAL/prompt_confusion/phase_12/outputs/phase_12_dataset/phase_12_dataset.jsonl",
    )
)
DEFAULT_REPORT_DIR = "projects/DX_TERMINAL/prompt_confusion/phase_12/reports/pipelines_v2"
CAPTURED_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44)


def _dataset_limit_from_env() -> int | None:
    raw = os.environ.get("PHASE_12_DATASET_LIMIT")
    if raw is None or not raw.strip():
        return None
    limit = int(raw)
    if limit <= 0:
        raise ValueError("PHASE_12_DATASET_LIMIT must be a positive integer")
    return limit


def _load_dataset_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Phase 12 dataset not found: {path}")
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    if not rows:
        raise ValueError(f"Phase 12 dataset is empty: {path}")
    return rows


def build_dataset(*, limit: int | None = None) -> Dataset:
    dataset = Dataset.from_records(
        _load_dataset_records(DEFAULT_DATASET_PATH),
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
            "main_benchmark_row",
            "stress_test_slice",
            "lexical_split",
            "strategy_lexical_split",
            "settings_lexical_split",
            "strict_combined_split",
            "context_variant_id",
        ],
        case_columns=["matched_group_id", "matched_pair_id"],
        case_key_column="matched_group_id",
        name="prompt_confusion_phase_12_diversification_preference",
    )
    final_limit = limit if limit is not None else _dataset_limit_from_env()
    return dataset.select(limit=final_limit) if final_limit is not None else dataset


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
        root="/data/artifacts/prompt_confusion_phase_12",
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
            artifacts=LocalArtifactStore(Path("artifacts") / "prompt_confusion_phase_12"),
            catalog=build_prompt_confusion_catalog(__file__),
        ),
    }


def _probe_step(*, name: str, dataset: Dataset, split_label: str) -> WorkflowStep:
    return WorkflowStep(
        name=name,
        runner="analysis_cpu",
        spec=ProbeSpec(
            feature=StepRef("capture_prompt_eos_residual").feature("residual_prompt_eos"),
            rows=dataset.labels("edge_conflict").equals(False),
            labels=dataset.labels("conflict_present"),
            group_by=dataset.cases("matched_group_id"),
            split=dataset.labels(split_label),
            train_values=("train",),
            test_values=("test",),
            tokens=TokenSelector.full_sequence(),
            pooling=TokenPooling.last(),
            metrics=("accuracy", "balanced_accuracy", "auroc", "selectivity"),
            baselines=("majority", "shuffled_label"),
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
        name="prompt_confusion_phase_12_pipelines_v2",
        steps=(
            WorkflowStep(
                name="text_conflict_gate",
                runner="analysis_cpu",
                spec=TextBaselineSpec(
                    text=dataset.labels("user_text"),
                    rows=rows,
                    labels=dataset.labels("conflict_present"),
                    group_by=dataset.cases("matched_group_id"),
                    split_by={
                        "strategy_lexical_split": dataset.labels("strategy_lexical_split"),
                        "settings_lexical_split": dataset.labels("settings_lexical_split"),
                    },
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
            _probe_step(name="conflict_probe_xor_split", dataset=dataset, split_label="lexical_split"),
            _probe_step(name="conflict_probe_strategy_holdout", dataset=dataset, split_label="strategy_lexical_split"),
            _probe_step(name="conflict_probe_settings_holdout", dataset=dataset, split_label="settings_lexical_split"),
            WorkflowStep(
                name="conflict_direction",
                runner="analysis_cpu",
                spec=DirectionSpec(
                    feature=StepRef("capture_prompt_eos_residual").feature("residual_prompt_eos"),
                    positive=dataset.labels("conflict_present").equals(True),
                    negative=dataset.labels("conflict_present").equals(False),
                    layers=list(CAPTURED_LAYERS),
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.last(),
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=(
                        StepRef("text_conflict_gate"),
                        StepRef("conflict_probe_xor_split"),
                        StepRef("conflict_probe_strategy_holdout"),
                        StepRef("conflict_probe_settings_holdout"),
                        StepRef("conflict_direction"),
                    ),
                    template="default",
                    output_dir=report_output_dir,
                ),
            ),
        ),
    )
