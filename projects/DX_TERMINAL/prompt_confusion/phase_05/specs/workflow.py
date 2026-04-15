from __future__ import annotations

"""pipelines_v2 workflow for prompt_confusion phase_05."""

import json
import os
from pathlib import Path
from typing import Any

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    GeometrySpec,
    GenerationSpec,
    LabelMapSpec,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    MoERoutingSite,
    PostgresCatalog,
    PostgresSource,
    ProbeSpec,
    ResidualizedProbeSpec,
    ReportSpec,
    ResidualSite,
    RoutingRecord,
    StepRef,
    TensorStorage,
    TextBaselineSpec,
    TokenPooling,
    TokenSelector,
    TransferProbeSpec,
    VLLMEngine,
    WorkflowOrchestrator,
    WorkflowSpec,
    WorkflowStep,
)


DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"

BASE_RELATION = "workflow_dataset_conflict_probe_v3_v1"
ARBITRATION_RELATION = "workflow_dataset_conflict_probe_v3_conflict_readout_side_v1"

CAPTURED_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44)
GEOMETRY_LAYERS = (20, 24, 28, 36)
REG_SWEEP_C = (0.01, 0.1, 1.0, 10.0, 100.0)

FAMILY_GROUP_MAPPING = {
    "trade_size_force_large": "size",
    "trade_size_force_small": "size",
    "activity_force_trade": "activity",
    "activity_force_observe": "activity",
}

DEFAULT_REPORT_DIR = "projects/DX_TERMINAL/prompt_confusion/phase_05/reports/pipelines_v2"


def _dataset_limit_from_env() -> int | None:
    raw = os.environ.get("PHASE_05_DATASET_LIMIT")
    if raw is None or not raw.strip():
        return None
    limit = int(raw)
    if limit <= 0:
        raise ValueError("PHASE_05_DATASET_LIMIT must be a positive integer")
    return limit


def build_phase_05_base_dataset(*, limit: int | None = None) -> Dataset:
    db = PostgresSource.from_env(DB_ENV_VAR)
    dataset = Dataset.from_postgres(
        source=db,
        table=BASE_RELATION,
        prompt_column="prompt_messages_json",
        example_key_column="log_id",
        label_columns=[
            "example_id",
            "user_text",
            "strategy_family",
            "conflict_present",
            "strategy_lexical_split",
            "setting_lexical_split",
        ],
        case_columns=["matched_pair_id"],
        case_key_column="matched_pair_id",
        name="prompt_confusion_phase_05_base",
    )
    return dataset.select(limit=limit) if limit is not None else dataset


def build_phase_05_arbitration_dataset() -> Dataset:
    db = PostgresSource.from_env(DB_ENV_VAR)
    return Dataset.from_postgres(
        source=db,
        table=ARBITRATION_RELATION,
        prompt_column="prompt_messages_json",
        example_key_column="log_id",
        label_columns=[
            "example_id",
            "strategy_family",
            "workflow_label",
        ],
        case_columns=[
            "arbitration_group_id",
            "matched_pair_id",
        ],
        case_key_column="arbitration_group_id",
        name="prompt_confusion_phase_05_arbitration",
    )


def _default_residual_engine() -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        max_model_len=8192,
        enforce_eager=False,
        max_num_seqs=16,
    )


def _default_router_engine() -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        max_model_len=8192,
        enforce_eager=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
    )


def build_runner_specs() -> dict[str, object]:
    db = PostgresSource.from_env(DB_ENV_VAR)
    artifact_store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts/prompt_confusion_phase_05",
    )
    db_secret = ModalSecret.from_env_var(DB_ENV_VAR, secret_name="xenon-neon")
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="A100-80GB",
                secrets=(db_secret,),
                volumes=(ModalVolumeMount(name=MODEL_VOLUME_NAME, mount_path=MODEL_VOLUME_PATH),),
            ),
            artifacts=artifact_store,
            catalog=PostgresCatalog(source=db),
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=6,
                memory_mb=24 * 1024,
                secrets=(db_secret,),
            ),
            artifacts=artifact_store,
            catalog=PostgresCatalog(source=db),
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(Path("artifacts") / "prompt_confusion_phase_05"),
        ),
    }


def build_runners() -> dict[str, object]:
    return {
        name: spec.to_runner()
        for name, spec in build_runner_specs().items()
    }


def build_phase_05_missing_pieces() -> list[str]:
    return [
        "Report outputs are still thin. Phase 05 still wants richer rendered summaries and geometry plotting, not just structured artifact payloads.",
    ]


def build_workflow(
    dataset: Dataset | None = None,
    *,
    arbitration_dataset: Dataset | None = None,
    residual_engine: object | None = None,
    router_engine: object | None = None,
    report_output_dir: str = DEFAULT_REPORT_DIR,
) -> WorkflowSpec:
    dataset = dataset or build_phase_05_base_dataset()
    arbitration_dataset = arbitration_dataset or build_phase_05_arbitration_dataset()
    residual_engine = residual_engine or _default_residual_engine()
    router_engine = router_engine or _default_router_engine()

    family_group_label = StepRef("derive_family_group").label("family_group")

    return WorkflowSpec(
        name="prompt_confusion_phase_05_pipelines_v2",
        steps=(
            WorkflowStep(
                name="derive_family_group",
                runner="analysis_cpu",
                spec=LabelMapSpec(
                    source=dataset.labels("strategy_family"),
                    output_name="family_group",
                    mapping=FAMILY_GROUP_MAPPING,
                    strict=True,
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
                    generation=GenerationSpec(enabled=False),
                ),
            ),
            WorkflowStep(
                name="capture_prompt_eos_router",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=router_engine,
                    dataset=dataset,
                    sites=[
                        MoERoutingSite(
                            name="router_prompt_eos",
                            layers=list(CAPTURED_LAYERS),
                            tokens=TokenSelector.last(),
                            record=[
                                RoutingRecord.gate_logits(dtype="float16"),
                                RoutingRecord.topk_from_gate(k=8, include_weights=True),
                                RoutingRecord.expert_load(source="topk_from_gate"),
                            ],
                        ),
                    ],
                    generation=GenerationSpec(enabled=False),
                ),
            ),
            WorkflowStep(
                name="family_identity_residual",
                runner="analysis_cpu",
                spec=ProbeSpec(
                    feature=StepRef("capture_prompt_eos_residual").feature("residual_prompt_eos"),
                    labels=dataset.labels("strategy_family"),
                    group_by=dataset.cases("matched_pair_id"),
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.last(),
                    metrics=("accuracy", "balanced_accuracy"),
                    baselines=("majority",),
                ),
            ),
            WorkflowStep(
                name="family_identity_router",
                runner="analysis_cpu",
                spec=ProbeSpec(
                    feature=StepRef("capture_prompt_eos_router").feature("router_prompt_eos"),
                    labels=dataset.labels("strategy_family"),
                    group_by=dataset.cases("matched_pair_id"),
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.last(),
                    metrics=("accuracy", "balanced_accuracy"),
                    baselines=("majority",),
                ),
            ),
            WorkflowStep(
                name="detection_transfer_residual",
                runner="analysis_cpu",
                spec=TransferProbeSpec(
                    feature=StepRef("capture_prompt_eos_residual").feature("residual_prompt_eos"),
                    labels=dataset.labels("conflict_present"),
                    group_by=dataset.cases("matched_pair_id"),
                    cohort_by=family_group_label,
                    cohort_values=("size", "activity"),
                    metrics=("balanced_accuracy", "auroc"),
                    compare_within_baseline=True,
                    compare_direction_similarity=True,
                ),
            ),
            WorkflowStep(
                name="detection_transfer_router",
                runner="analysis_cpu",
                spec=TransferProbeSpec(
                    feature=StepRef("capture_prompt_eos_router").feature("router_prompt_eos"),
                    labels=dataset.labels("conflict_present"),
                    group_by=dataset.cases("matched_pair_id"),
                    cohort_by=family_group_label,
                    cohort_values=("size", "activity"),
                    metrics=("balanced_accuracy", "auroc"),
                    compare_within_baseline=True,
                    compare_direction_similarity=True,
                ),
            ),
            WorkflowStep(
                name="arbitration_transfer_residual",
                runner="analysis_cpu",
                spec=TransferProbeSpec(
                    feature=StepRef("capture_prompt_eos_residual").feature("residual_prompt_eos"),
                    rows=arbitration_dataset,
                    labels=arbitration_dataset.labels("workflow_label"),
                    group_by=arbitration_dataset.cases("arbitration_group_id"),
                    cohort_by=family_group_label,
                    cohort_values=("size", "activity"),
                    metrics=("balanced_accuracy",),
                    compare_within_baseline=True,
                    compare_direction_similarity=True,
                ),
            ),
            WorkflowStep(
                name="arbitration_transfer_router",
                runner="analysis_cpu",
                spec=TransferProbeSpec(
                    feature=StepRef("capture_prompt_eos_router").feature("router_prompt_eos"),
                    rows=arbitration_dataset,
                    labels=arbitration_dataset.labels("workflow_label"),
                    group_by=arbitration_dataset.cases("arbitration_group_id"),
                    cohort_by=family_group_label,
                    cohort_values=("size", "activity"),
                    metrics=("balanced_accuracy",),
                    compare_within_baseline=True,
                    compare_direction_similarity=True,
                ),
            ),
            WorkflowStep(
                name="lexical_family_identity",
                runner="analysis_cpu",
                spec=TextBaselineSpec(
                    text=dataset.labels("user_text"),
                    labels=dataset.labels("strategy_family"),
                    group_by=dataset.cases("matched_pair_id"),
                    model="countvectorizer_logreg",
                    metrics=("balanced_accuracy", "auroc"),
                ),
            ),
            WorkflowStep(
                name="lexical_cross_family_detection_transfer",
                runner="analysis_cpu",
                spec=TextBaselineSpec(
                    text=dataset.labels("user_text"),
                    labels=dataset.labels("conflict_present"),
                    cohort_by=family_group_label,
                    cohort_values=("size", "activity"),
                    model="countvectorizer_logreg",
                    metrics=("balanced_accuracy", "auroc"),
                ),
            ),
            WorkflowStep(
                name="lexical_holdout_detection_residual",
                runner="analysis_cpu",
                spec=TransferProbeSpec(
                    feature=StepRef("capture_prompt_eos_residual").feature("residual_prompt_eos"),
                    labels=dataset.labels("conflict_present"),
                    cohort_by=family_group_label,
                    cohort_values=("size", "activity"),
                    split_by={
                        "strategy_lexical_split": dataset.labels("strategy_lexical_split"),
                        "setting_lexical_split": dataset.labels("setting_lexical_split"),
                    },
                    train_values=("train",),
                    test_values=("test",),
                    metrics=("balanced_accuracy", "auroc"),
                ),
            ),
            WorkflowStep(
                name="lexical_holdout_detection_text",
                runner="analysis_cpu",
                spec=TextBaselineSpec(
                    text=dataset.labels("user_text"),
                    labels=dataset.labels("conflict_present"),
                    cohort_by=family_group_label,
                    cohort_values=("size", "activity"),
                    split_by={
                        "strategy_lexical_split": dataset.labels("strategy_lexical_split"),
                        "setting_lexical_split": dataset.labels("setting_lexical_split"),
                    },
                    train_values=("train",),
                    test_values=("test",),
                    model="countvectorizer_logreg",
                    metrics=("balanced_accuracy", "auroc"),
                ),
            ),
            WorkflowStep(
                name="family_residualized_conflict_residual",
                runner="analysis_cpu",
                spec=ResidualizedProbeSpec(
                    feature=StepRef("capture_prompt_eos_residual").feature("residual_prompt_eos"),
                    labels=dataset.labels("conflict_present"),
                    residualize_against=dataset.labels("strategy_family"),
                    group_by=dataset.cases("matched_pair_id"),
                    metrics=("balanced_accuracy", "auroc"),
                ),
            ),
            WorkflowStep(
                name="family_residualized_conflict_router",
                runner="analysis_cpu",
                spec=ResidualizedProbeSpec(
                    feature=StepRef("capture_prompt_eos_router").feature("router_prompt_eos"),
                    labels=dataset.labels("conflict_present"),
                    residualize_against=dataset.labels("strategy_family"),
                    group_by=dataset.cases("matched_pair_id"),
                    metrics=("balanced_accuracy", "auroc"),
                ),
            ),
            WorkflowStep(
                name="detection_transfer_regularization_sweep",
                runner="analysis_cpu",
                spec=TransferProbeSpec(
                    feature=StepRef("capture_prompt_eos_residual").feature("residual_prompt_eos"),
                    labels=dataset.labels("conflict_present"),
                    group_by=dataset.cases("matched_pair_id"),
                    cohort_by=family_group_label,
                    cohort_values=("size", "activity"),
                    regularization=REG_SWEEP_C,
                    metrics=("balanced_accuracy", "auroc"),
                    compare_within_baseline=False,
                    compare_direction_similarity=False,
                ),
            ),
            WorkflowStep(
                name="family_geometry_pca_full",
                runner="analysis_cpu",
                spec=GeometrySpec(
                    feature=StepRef("capture_prompt_eos_residual").feature("residual_prompt_eos"),
                    method="pca",
                    layers=GEOMETRY_LAYERS,
                    normalize="rms_per_row",
                    color_by={
                        "family": dataset.labels("strategy_family"),
                        "alignment": dataset.labels("conflict_present"),
                    },
                    components=2,
                ),
            ),
            WorkflowStep(
                name="family_geometry_pca_conflict_only",
                runner="analysis_cpu",
                spec=GeometrySpec(
                    feature=StepRef("capture_prompt_eos_residual").feature("residual_prompt_eos"),
                    method="pca",
                    layers=GEOMETRY_LAYERS,
                    label=dataset.labels("strategy_family"),
                    subset=dataset.labels("conflict_present").equals(True),
                    normalize="rms_per_row",
                    components=2,
                ),
            ),
            WorkflowStep(
                name="family_geometry_lda_conflict_only",
                runner="analysis_cpu",
                spec=GeometrySpec(
                    feature=StepRef("capture_prompt_eos_residual").feature("residual_prompt_eos"),
                    method="lda",
                    layers=GEOMETRY_LAYERS,
                    label=dataset.labels("strategy_family"),
                    subset=dataset.labels("conflict_present").equals(True),
                    normalize="rms_per_row",
                    components=2,
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    output_dir=report_output_dir,
                    inputs=[
                        StepRef("capture_prompt_eos_residual"),
                        StepRef("capture_prompt_eos_router"),
                        StepRef("family_identity_residual"),
                        StepRef("family_identity_router"),
                        StepRef("detection_transfer_residual"),
                        StepRef("detection_transfer_router"),
                        StepRef("arbitration_transfer_residual"),
                        StepRef("arbitration_transfer_router"),
                        StepRef("lexical_family_identity"),
                        StepRef("lexical_cross_family_detection_transfer"),
                        StepRef("lexical_holdout_detection_residual"),
                        StepRef("lexical_holdout_detection_text"),
                        StepRef("family_residualized_conflict_residual"),
                        StepRef("family_residualized_conflict_router"),
                        StepRef("detection_transfer_regularization_sweep"),
                        StepRef("family_geometry_pca_full"),
                        StepRef("family_geometry_pca_conflict_only"),
                        StepRef("family_geometry_lda_conflict_only"),
                        {"missing_pieces": build_phase_05_missing_pieces()},
                    ],
                ),
            ),
        ),
    )


def build_dataset() -> Dataset:
    return build_phase_05_base_dataset(limit=_dataset_limit_from_env())


def build_orchestrator() -> WorkflowOrchestrator:
    return WorkflowOrchestrator(runners=build_runners())


def build_phase_05_target_payload() -> dict[str, Any]:
    base_dataset = build_phase_05_base_dataset()
    arbitration_dataset = build_phase_05_arbitration_dataset()
    workflow = build_workflow(base_dataset, arbitration_dataset=arbitration_dataset)
    return {
        "kind": "phase_05_pipelines_v2_target",
        "schema_version": 1,
        "mode": "library_backed",
        "project_id": "DX_TERMINAL",
        "subproject": "prompt_confusion",
        "phase": "phase_05",
        "goal": (
            "Represent the real Phase 05 capture and analysis stack directly in "
            "pipelines_v2 so the remaining gaps are operational, not just missing API surface."
        ),
        "base_dataset": base_dataset.to_dict(),
        "arbitration_dataset": arbitration_dataset.to_dict(),
        "runners": {
            name: spec.to_dict()
            for name, spec in build_runner_specs().items()
        },
        "workflow": workflow.to_dict(),
        "missing_pieces": build_phase_05_missing_pieces(),
    }


def load_workflow_json(
    path: str | Path = Path(__file__).with_name("workflow.json"),
) -> dict[str, Any]:
    """Load the checked-in aspirational workflow snapshot as raw JSON."""

    return json.loads(Path(path).read_text())
