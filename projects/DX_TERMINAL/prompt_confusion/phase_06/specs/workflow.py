"""Phase 06 combined workflow: QA baselines + capture + conflict detection.

Size-axis-only dataset (`conflict_probe_examples_v4`). Residual-only
capture on Modal with CUDA graphs + batch size 16 (no MoE router
capture in this phase). Analyses run on a Modal CPU runner; reporting
is materialized locally.
"""
from __future__ import annotations

from pathlib import Path

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    GenerationSpec,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    PostgresSource,
    ProbeSpec,
    ReportSpec,
    ResidualSite,
    StepRef,
    TensorStorage,
    TextBaselineSpec,
    TokenSelector,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)


PHASE_ROOT = Path("projects/DX_TERMINAL/prompt_confusion/phase_06")
LOCAL_ARTIFACT_ROOT = PHASE_ROOT / "outputs" / "artifacts"
REPORT_DIR = PHASE_ROOT / "outputs" / "report"

MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"
MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
ARTIFACTS_VOLUME = "xenon-data"
ARTIFACTS_ROOT = "/data/artifacts/prompt_confusion/phase_06"

CAPTURE_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]

DATASET_SQL = """
WITH ranked AS (
    SELECT
        row_number() OVER (ORDER BY example_id) AS log_id,
        src.*,
        CASE
            WHEN strategy_lexical_split = 'train' AND setting_lexical_split = 'train' THEN 'strict_train'
            WHEN strategy_lexical_split = 'test' AND setting_lexical_split = 'test' THEN 'strict_test'
            ELSE 'mixed'
        END AS combined_lexical_split
    FROM conflict_probe_examples_v4 src
)
SELECT * FROM ranked
"""


def build_dataset() -> Dataset:
    return Dataset.from_postgres(
        source=PostgresSource.from_env("XENON_NEON_DATABASE_URL"),
        sql=DATASET_SQL,
        prompt_column="prompt_messages_json",
        example_key_column="example_id",
        case_key_column="matched_pair_id",
        label_columns=[
            "user_text",
            "conflict_present",
            "strategy_family",
            "strategy_variant_id",
            "strategy_lexical_split",
            "setting_lexical_family_id",
            "setting_variant_id",
            "setting_lexical_split",
            "combined_lexical_split",
            "section_order",
            "environment_pressure_bucket",
            "pair_member",
        ],
    )


def build_runner_specs() -> dict[str, object]:
    neon_secret = ModalSecret.from_env_var(
        "XENON_NEON_DATABASE_URL", secret_name="xenon-neon"
    )
    artifact_store = ModalVolumeStore(name=ARTIFACTS_VOLUME, root=ARTIFACTS_ROOT)
    gpu_resources = ModalResources(
        gpu="A100-80GB",
        timeout_seconds=60 * 60 * 4,
        secrets=(neon_secret,),
        volumes=(ModalVolumeMount(name=MODEL_VOLUME_NAME, mount_path=MODEL_VOLUME_PATH),),
    )
    cpu_resources = ModalResources(
        cpu=8,
        memory_mb=32 * 1024,
        timeout_seconds=60 * 60 * 2,
        secrets=(neon_secret,),
    )
    return {
        "capture_gpu": ModalRunnerSpec(resources=gpu_resources, artifacts=artifact_store),
        "analysis_cpu": ModalRunnerSpec(resources=cpu_resources, artifacts=artifact_store),
        "report_local": LocalRunnerSpec(artifacts=LocalArtifactStore(LOCAL_ARTIFACT_ROOT)),
    }


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()

    text_baseline_strategy = WorkflowStep(
        name="text_baseline_conflict_strategy_holdout",
        runner="analysis_cpu",
        spec=TextBaselineSpec(
            text=dataset.labels("user_text"),
            labels=dataset.labels("conflict_present"),
            group_by=dataset.cases("matched_pair_id"),
            split_by={"strategy": dataset.labels("strategy_lexical_split")},
            train_values=("train",),
            test_values=("test",),
            metrics=("balanced_accuracy", "auroc"),
        ),
    )
    text_baseline_settings = WorkflowStep(
        name="text_baseline_conflict_settings_holdout",
        runner="analysis_cpu",
        spec=TextBaselineSpec(
            text=dataset.labels("user_text"),
            labels=dataset.labels("conflict_present"),
            group_by=dataset.cases("matched_pair_id"),
            split_by={"settings": dataset.labels("setting_lexical_split")},
            train_values=("train",),
            test_values=("test",),
            metrics=("balanced_accuracy", "auroc"),
        ),
    )
    text_baseline_grouped = WorkflowStep(
        name="text_baseline_conflict_grouped_cv",
        runner="analysis_cpu",
        spec=TextBaselineSpec(
            text=dataset.labels("user_text"),
            labels=dataset.labels("conflict_present"),
            group_by=dataset.cases("matched_pair_id"),
            metrics=("balanced_accuracy", "auroc"),
        ),
    )
    text_baseline_combined = WorkflowStep(
        name="text_baseline_conflict_combined_holdout",
        runner="analysis_cpu",
        spec=TextBaselineSpec(
            text=dataset.labels("user_text"),
            labels=dataset.labels("conflict_present"),
            group_by=dataset.cases("matched_pair_id"),
            split_by={"combined": dataset.labels("combined_lexical_split")},
            train_values=("strict_train",),
            test_values=("strict_test",),
            metrics=("balanced_accuracy", "auroc"),
        ),
    )

    capture = WorkflowStep(
        name="capture_resid",
        runner="capture_gpu",
        spec=CaptureSpec(
            engine=VLLMEngine(
                model_id=MODEL_ID,
                max_model_len=8192,
                enforce_eager=False,
                enable_prefix_caching=True,
                max_num_seqs=16,
                add_generation_prompt=True,
                enable_thinking=False,
            ),
            dataset=dataset,
            sites=[
                ResidualSite(
                    name="resid_post_last",
                    site="resid_post",
                    layers=CAPTURE_LAYERS,
                    tokens=TokenSelector.last(),
                    storage=TensorStorage(dtype="float16"),
                ),
            ],
            generation=GenerationSpec(
                enabled=True,
                max_tokens=64,
                temperature=0.0,
                capture_reasoning=False,
            ),
        ),
    )

    probe_strategy = WorkflowStep(
        name="probe_conflict_strategy_holdout",
        runner="analysis_cpu",
        spec=ProbeSpec(
            feature=StepRef("capture_resid").feature("resid_post_last"),
            labels=dataset.labels("conflict_present"),
            group_by=dataset.cases("matched_pair_id"),
            split=dataset.labels("strategy_lexical_split"),
            train_values=("train",),
            test_values=("test",),
            metrics=("balanced_accuracy", "auroc"),
        ),
    )
    probe_settings = WorkflowStep(
        name="probe_conflict_settings_holdout",
        runner="analysis_cpu",
        spec=ProbeSpec(
            feature=StepRef("capture_resid").feature("resid_post_last"),
            labels=dataset.labels("conflict_present"),
            group_by=dataset.cases("matched_pair_id"),
            split=dataset.labels("setting_lexical_split"),
            train_values=("train",),
            test_values=("test",),
            metrics=("balanced_accuracy", "auroc"),
        ),
    )
    probe_grouped = WorkflowStep(
        name="probe_conflict_grouped_cv",
        runner="analysis_cpu",
        spec=ProbeSpec(
            feature=StepRef("capture_resid").feature("resid_post_last"),
            labels=dataset.labels("conflict_present"),
            group_by=dataset.cases("matched_pair_id"),
            folds=5,
            metrics=("balanced_accuracy", "auroc"),
        ),
    )
    probe_combined = WorkflowStep(
        name="probe_conflict_combined_holdout",
        runner="analysis_cpu",
        spec=ProbeSpec(
            feature=StepRef("capture_resid").feature("resid_post_last"),
            labels=dataset.labels("conflict_present"),
            group_by=dataset.cases("matched_pair_id"),
            split=dataset.labels("combined_lexical_split"),
            train_values=("strict_train",),
            test_values=("strict_test",),
            metrics=("balanced_accuracy", "auroc"),
        ),
    )

    report = WorkflowStep(
        name="report",
        runner="report_local",
        spec=ReportSpec(
            template="phase_06",
            output_dir=str(REPORT_DIR),
            inputs=[
                StepRef("text_baseline_conflict_strategy_holdout"),
                StepRef("text_baseline_conflict_settings_holdout"),
                StepRef("text_baseline_conflict_grouped_cv"),
                StepRef("text_baseline_conflict_combined_holdout"),
                StepRef("probe_conflict_strategy_holdout"),
                StepRef("probe_conflict_settings_holdout"),
                StepRef("probe_conflict_grouped_cv"),
                StepRef("probe_conflict_combined_holdout"),
            ],
        ),
    )

    return WorkflowSpec(
        name="phase_06",
        steps=(
            text_baseline_strategy,
            text_baseline_settings,
            text_baseline_grouped,
            text_baseline_combined,
            capture,
            probe_strategy,
            probe_settings,
            probe_grouped,
            probe_combined,
            report,
        ),
    )
