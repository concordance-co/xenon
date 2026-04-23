from __future__ import annotations

"""Cross-dimension similarity workflow for settled trade_size vs risk_preference."""

import json
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
    PairDeltaSpec,
    ReportSpec,
    ResidualSite,
    StepRef,
    TensorStorage,
    TokenPooling,
    TokenSelector,
    TransferProbeSpec,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)
from projects.DX_TERMINAL.prompt_confusion.catalogs import build_prompt_confusion_catalog


MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"
PHASE_09_DATASET_PATH = Path(
    "projects/DX_TERMINAL/prompt_confusion/phase_09/outputs/phase_09_dataset/phase_09_dataset.jsonl"
)
PHASE_10_DATASET_PATH = Path(
    "projects/DX_TERMINAL/prompt_confusion/phase_10/outputs/phase_10_dataset/phase_10_dataset.jsonl"
)
DEFAULT_REPORT_DIR = "projects/DX_TERMINAL/prompt_confusion/phase_10/reports/trade_size_risk_similarity"
CAPTURED_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def _combined_records() -> list[dict[str, object]]:
    phase_09_rows = [
        row
        for row in _load_jsonl(PHASE_09_DATASET_PATH)
        if row.get("target_dimension") == "trade_size"
        and bool(row.get("main_benchmark_row", True))
        and not bool(row.get("edge_conflict", False))
    ]
    phase_10_rows = [
        row
        for row in _load_jsonl(PHASE_10_DATASET_PATH)
        if row.get("target_dimension") == "risk_preference"
        and bool(row.get("main_benchmark_row", True))
        and not bool(row.get("edge_conflict", False))
    ]
    if not phase_09_rows:
        raise ValueError("No settled trade_size rows found")
    if not phase_10_rows:
        raise ValueError("No settled risk_preference rows found")
    return phase_09_rows + phase_10_rows


def build_dataset() -> Dataset:
    return Dataset.from_records(
        _combined_records(),
        prompt_column="prompt_messages_json",
        example_key_column="example_id",
        label_columns=[
            "target_dimension",
            "strategy_direction",
            "setting_value",
            "setting_implied_direction",
            "conflict_present",
            "lexical_split",
            "strategy_lexical_split",
            "settings_lexical_split",
            "matched_pair_id",
            "context_variant_id",
        ],
        case_columns=["matched_group_id", "matched_pair_id"],
        case_key_column="matched_group_id",
        name="prompt_confusion_trade_size_vs_risk_preference",
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
        root="/data/artifacts/prompt_confusion_trade_size_vs_risk_preference",
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
            artifacts=LocalArtifactStore(Path("artifacts") / "prompt_confusion_trade_size_vs_risk_preference"),
            catalog=build_prompt_confusion_catalog(__file__),
        ),
    }


def build_workflow(
    dataset: Dataset | None = None,
    *,
    residual_engine: object | None = None,
    report_output_dir: str = DEFAULT_REPORT_DIR,
) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    residual_engine = residual_engine or _default_residual_engine()

    return WorkflowSpec(
        name="prompt_confusion_trade_size_vs_risk_preference",
        steps=(
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
            WorkflowStep(
                name="cross_dimension_similarity",
                runner="analysis_cpu",
                spec=TransferProbeSpec(
                    feature=StepRef("capture_prompt_eos_residual").feature("residual_prompt_eos"),
                    labels=dataset.labels("conflict_present"),
                    group_by=dataset.cases("matched_group_id"),
                    cohort_by=dataset.labels("target_dimension"),
                    cohort_values=("trade_size", "risk_preference"),
                    metrics=("accuracy", "balanced_accuracy", "auroc"),
                    compare_within_baseline=True,
                    compare_direction_similarity=True,
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.last(),
                ),
            ),
            WorkflowStep(
                name="pair_delta_conflict",
                runner="analysis_cpu",
                spec=PairDeltaSpec(
                    feature=StepRef("capture_prompt_eos_residual").feature("residual_prompt_eos"),
                    case=dataset.labels("matched_pair_id"),
                    positive=dataset.labels("conflict_present").equals(True),
                    negative=dataset.labels("conflict_present").equals(False),
                    layers=list(CAPTURED_LAYERS),
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.last(),
                    output_feature_name="conflict_pair_delta",
                    labels={
                        "target_dimension": dataset.labels("target_dimension"),
                        "strategy_direction": dataset.labels("strategy_direction"),
                        "context_variant_id": dataset.labels("context_variant_id"),
                    },
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=(
                        StepRef("cross_dimension_similarity"),
                        StepRef("pair_delta_conflict"),
                    ),
                    template="default",
                    output_dir=report_output_dir,
                ),
            ),
        ),
    )
