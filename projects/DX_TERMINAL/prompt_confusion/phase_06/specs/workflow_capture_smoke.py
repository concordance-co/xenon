"""Phase 06 capture smoke workflow.

Same engine + sites config as workflow_capture.py, but restricted to a
small representative slice (2 families x 2 orders x 2 pair_members, with
one strategy variant and one setting phrase family each) so we can
validate Modal wiring before burning the full run.
"""
from __future__ import annotations

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    GenerationSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    MoERoutingSite,
    PostgresSource,
    ResidualSite,
    RoutingRecord,
    TensorStorage,
    TokenSelector,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)


MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"
MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
ARTIFACTS_VOLUME = "xenon-data"
ARTIFACTS_ROOT = "/data/artifacts/prompt_confusion/phase_06/capture_smoke"

CAPTURE_LAYERS = [0, 24, 44]

DATASET_SQL = """
WITH ranked AS (
    SELECT row_number() OVER (ORDER BY example_id) AS log_id, src.*
    FROM conflict_probe_examples_v4 src
    WHERE strategy_variant_id IN ('size_large_v0', 'size_small_v0')
      AND setting_lexical_family_id = 'size_setting_phrase_v0'
      AND environment_pressure_bucket = 'balanced'
      AND context_variant_id = 'size_balanced_v0'
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
            "section_order",
            "pair_member",
        ],
    )


def build_runner_specs() -> dict[str, object]:
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="A100-80GB",
                timeout_seconds=60 * 30,
                secrets=(
                    ModalSecret.from_env_var("XENON_NEON_DATABASE_URL", secret_name="xenon-neon"),
                ),
                volumes=(ModalVolumeMount(name=MODEL_VOLUME_NAME, mount_path=MODEL_VOLUME_PATH),),
            ),
            artifacts=ModalVolumeStore(name=ARTIFACTS_VOLUME, root=ARTIFACTS_ROOT),
        ),
    }


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    return WorkflowSpec(
        name="phase_06_capture_smoke",
        steps=(
            WorkflowStep(
                name="capture_smoke",
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
            ),
        ),
    )
