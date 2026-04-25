from __future__ import annotations

"""Dashboard-only inspection workflow for the strict Stage 1b bridge table.

This uses ToyEngine so the workflow is cheap to run locally. The purpose is to
make the Neon dataset visible through the pipelines_v2 dashboard Dataset,
Labels, and Prompt tabs before launching real model capture.
"""

from pathlib import Path

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    LocalArtifactStore,
    LocalRunnerSpec,
    PostgresSource,
    ResidualSite,
    TensorStorage,
    TokenSelector,
    ToyEngine,
    WorkflowSpec,
    WorkflowStep,
)
from projects.DX_TERMINAL.prompt_confusion.catalogs import build_prompt_confusion_catalog


DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
TABLE_NAME = "dx_terminal_trade_size_stage1b_adapter_strict_v1"


def build_dataset(*, limit: int | None = None) -> Dataset:
    dataset = Dataset.from_postgres(
        source=PostgresSource.from_env(DB_ENV_VAR),
        table=TABLE_NAME,
        prompt_column="prompt_messages_json",
        example_key_column="example_id",
        label_columns=[
            "trace_id",
            "source_example_id",
            "label",
            "fault",
            "root_cause",
            "complaint_type",
            "has_strategy",
            "slider_ta",
            "slider_arp",
            "slider_ts",
            "slider_hs",
            "slider_div",
            "size_relevant_complaint",
            "activity_relevant_complaint",
            "config_conflict_like",
            "system_fault",
            "transfer_stage",
            "transfer_family",
            "transfer_format",
            "adapter_alignment_label",
            "strategy_size_preference",
            "slider_size_bucket",
            "extracted_portfolio_present",
            "extracted_market_present",
        ],
        case_columns=["trace_id"],
        case_key_column="trace_id",
        name="dx_terminal_trade_size_stage1b_adapter_strict_v1",
    )
    return dataset.select(limit=limit) if limit is not None else dataset


def build_runner_specs() -> dict[str, object]:
    return {
        "inspect_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(
                Path("artifacts") / "dx_terminal_stage1b_strict_dashboard_inspect"
            ),
            catalog=build_prompt_confusion_catalog(__file__),
        )
    }


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    return WorkflowSpec(
        name="dx_terminal_stage1b_strict_dashboard_inspect",
        steps=(
            WorkflowStep(
                name="inspect_stage1b_strict_prompts",
                runner="inspect_local",
                spec=CaptureSpec(
                    engine=ToyEngine(),
                    dataset=dataset,
                    sites=[
                        ResidualSite(
                            name="toy_prompt_last",
                            site="resid_post",
                            layers=[0],
                            tokens=TokenSelector.last(),
                            storage=TensorStorage(dtype="float32", format="safetensors"),
                        )
                    ],
                ),
            ),
        ),
    )

