from __future__ import annotations

"""Wave 1 follow-up workflow for prompt_confusion phase_09."""

import json
import os
from pathlib import Path

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    GenerationSpec,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalVolumeMount,
    ModalVolumeStore,
    NullCatalog,
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


MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"
DEFAULT_DATASET_PATH = Path(
    os.environ.get(
        "PHASE_09_DATASET_PATH",
        "projects/DX_TERMINAL/prompt_confusion/phase_09/outputs/phase_09_dataset/phase_09_dataset.jsonl",
    )
)
DEFAULT_REPORT_DIR = "projects/DX_TERMINAL/prompt_confusion/phase_09/reports/pipelines_v2_wave1"
CAPTURED_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44)


def _load_dataset_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Phase 09 dataset not found: {path}")
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    if not rows:
        raise ValueError(f"Phase 09 dataset is empty: {path}")
    return rows


def _full_dataset_records() -> list[dict[str, object]]:
    return _load_dataset_records(DEFAULT_DATASET_PATH)


def _boundary_generation_records() -> list[dict[str, object]]:
    rows = _full_dataset_records()
    filtered = [
        row
        for row in rows
        if row.get("target_dimension") == "trading_activity"
        and int(row.get("setting_value", -1)) == 1
        and any(tag in str(row.get("context_variant_id", "")) for tag in ("solid", "exceptional"))
    ]
    if not filtered:
        raise ValueError("Boundary generation slice is empty")
    return filtered


def build_dataset() -> Dataset:
    return Dataset.from_records(
        _full_dataset_records(),
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
            "matched_pair_id",
            "context_variant_id",
        ],
        case_columns=["matched_group_id", "matched_pair_id"],
        case_key_column="matched_group_id",
        name="prompt_confusion_phase_09_wave1",
    )


def build_boundary_dataset() -> Dataset:
    return Dataset.from_records(
        _boundary_generation_records(),
        prompt_column="prompt_messages_json",
        example_key_column="example_id",
        label_columns=[
            "target_dimension",
            "strategy_direction",
            "setting_value",
            "setting_implied_direction",
            "conflict_present",
            "conflict_band",
            "context_variant_id",
            "expected_output_json",
        ],
        case_columns=["matched_group_id", "matched_pair_id"],
        case_key_column="matched_group_id",
        name="prompt_confusion_phase_09_wave1_boundary",
    )


def _default_residual_engine() -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        max_model_len=8192,
        enforce_eager=False,
        max_num_seqs=16,
    )


def _boundary_generation_engine() -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        max_model_len=8192,
        enforce_eager=False,
        max_num_seqs=16,
        add_generation_prompt=True,
        reasoning_parser="",
        extra={"chat_template_kwargs": {"enable_thinking": False}},
    )


def build_runner_specs() -> dict[str, object]:
    artifact_store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts/prompt_confusion_phase_09_wave1",
    )
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="A100-80GB",
                volumes=(ModalVolumeMount(name=MODEL_VOLUME_NAME, mount_path=MODEL_VOLUME_PATH),),
            ),
            artifacts=artifact_store,
            catalog=NullCatalog(),
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=6,
                memory_mb=24 * 1024,
            ),
            artifacts=artifact_store,
            catalog=NullCatalog(),
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(Path("artifacts") / "prompt_confusion_phase_09_wave1"),
            catalog=NullCatalog(),
        ),
    }


def build_workflow(
    dataset: Dataset | None = None,
    *,
    residual_engine: object | None = None,
    report_output_dir: str = DEFAULT_REPORT_DIR,
) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    boundary_dataset = build_boundary_dataset()
    residual_engine = residual_engine or _default_residual_engine()
    boundary_engine = _boundary_generation_engine()

    canonical_rows = dataset.labels("edge_conflict").equals(False)
    paired_rows = dataset.labels("matched_pair_id").equals("")

    return WorkflowSpec(
        name="prompt_confusion_phase_09_wave1",
        steps=(
            WorkflowStep(
                name="capture_prompt_eos_residual_wave1",
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
                    generation=GenerationSpec(enabled=False),
                ),
            ),
            WorkflowStep(
                name="cross_dimension_similarity",
                runner="analysis_cpu",
                spec=TransferProbeSpec(
                    feature=StepRef("capture_prompt_eos_residual_wave1").feature("residual_prompt_eos"),
                    rows=canonical_rows,
                    labels=dataset.labels("conflict_present"),
                    group_by=dataset.cases("matched_group_id"),
                    cohort_by=dataset.labels("target_dimension"),
                    cohort_values=("trade_size", "trading_activity"),
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
                    feature=StepRef("capture_prompt_eos_residual_wave1").feature("residual_prompt_eos"),
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
                name="boundary_generation",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=boundary_engine,
                    dataset=boundary_dataset,
                    sites=[],
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=256,
                        temperature=0.0,
                        capture_reasoning=False,
                    ),
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=(StepRef("cross_dimension_similarity"),),
                    template="default",
                    output_dir=report_output_dir,
                ),
            ),
        ),
    )
