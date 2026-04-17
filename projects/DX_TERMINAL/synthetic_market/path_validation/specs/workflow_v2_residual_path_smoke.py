from __future__ import annotations

import os

from pipelines_v2.api import (
    ActivationBankSpec,
    CaptureSpec,
    Dataset,
    ExplicitPathEdge,
    ExplicitPathMaskSpec,
    GenerationRunSpec,
    PatchComparisonSpec,
    PatchedGenerationSpec,
    PostgresSource,
    PromptMetadataBuilder,
    ResidualInterventionSite,
    ResidualPathPatch,
    ResidualSite,
    StepRef,
    TokenSelector,
    TransformBuilder,
    WorkflowSpec,
    WorkflowStep,
)

from projects.DX_TERMINAL.synthetic_market.path_validation.specs._workflow_v2_common import (
    DB_ENV_VAR,
    build_engine,
    build_generation_spec,
    build_prompt_metadata,
    build_runner_specs as build_common_runner_specs,
    evaluate_patch_row,
    generation_max_tokens,
)


PHASE_NAME = "phase15_market_basis_discovery_v1"
CONTEXT_VARIANT = "market_only"
FAMILY = "market_basis_coupled"
DEFAULT_TARGET_VARIANT = "pct_5m__net_flow_5m"
DEFAULT_DONOR_VARIANT = "unique_traders_5m__top20_holder_pct"
DEFAULT_LIMIT = 16
PATCH_LAYER = 4
ARTIFACT_ROOT = "/data/artifacts/dx_terminal_synthetic_market/path_validation_v2_residual_path_smoke"
DEFAULT_TARGET_SECTION = "instruction"
DEFAULT_READ_SECTION = "instruction"


def _dataset_limit() -> int:
    raw = os.environ.get("SYNTHETIC_MARKET_V2_PATH_SMOKE_LIMIT")
    if raw is None or not raw.strip():
        return DEFAULT_LIMIT
    value = int(raw)
    if value <= 0:
        raise ValueError("SYNTHETIC_MARKET_V2_PATH_SMOKE_LIMIT must be a positive integer")
    return value


def _target_variant() -> str:
    raw = os.environ.get("SYNTHETIC_MARKET_V2_PATH_TARGET_VARIANT")
    return DEFAULT_TARGET_VARIANT if raw is None or not raw.strip() else str(raw).strip()


def _donor_variant() -> str:
    raw = os.environ.get("SYNTHETIC_MARKET_V2_PATH_DONOR_VARIANT")
    return DEFAULT_DONOR_VARIANT if raw is None or not raw.strip() else str(raw).strip()


def _transport() -> str:
    raw = os.environ.get("SYNTHETIC_MARKET_V2_PATH_TRANSPORT")
    value = "delta" if raw is None or not raw.strip() else str(raw).strip().lower()
    if value not in {"replace", "delta"}:
        raise ValueError("SYNTHETIC_MARKET_V2_PATH_TRANSPORT must be one of {'replace', 'delta'}")
    return value


def _target_section() -> str:
    raw = os.environ.get("SYNTHETIC_MARKET_V2_PATH_TARGET_SECTION")
    return DEFAULT_TARGET_SECTION if raw is None or not raw.strip() else str(raw).strip()


def _read_section() -> str:
    raw = os.environ.get("SYNTHETIC_MARKET_V2_PATH_READ_SECTION")
    return DEFAULT_READ_SECTION if raw is None or not raw.strip() else str(raw).strip()


def _paired_market_sql(*, target_variant: str, donor_variant: str) -> str:
    return f"""
WITH base AS (
    SELECT
        log_id,
        phase_name,
        example_id,
        family,
        family_variant,
        context_variant,
        prompt_messages_json,
        regexp_replace(example_id, '^.*(_r[0-9]+_x[0-9]+_y[0-9]+)$', '\\1') AS pair_key
    FROM synthetic_market_examples_v0
    WHERE phase_name = '{PHASE_NAME}'
      AND context_variant = '{CONTEXT_VARIANT}'
      AND family = '{FAMILY}'
      AND family_variant IN ('{target_variant}', '{donor_variant}')
),
ranked AS (
    SELECT
        *,
        dense_rank() OVER (ORDER BY pair_key) AS pair_rank
    FROM base
)
SELECT *
FROM ranked
ORDER BY pair_rank, family_variant, log_id
"""


def build_dataset(
    *,
    limit: int | None = None,
    target_variant: str | None = None,
    donor_variant: str | None = None,
) -> Dataset:
    actual_limit = _dataset_limit() if limit is None else int(limit)
    actual_target = _target_variant() if target_variant is None else str(target_variant)
    actual_donor = _donor_variant() if donor_variant is None else str(donor_variant)
    dataset = Dataset.from_postgres(
        source=PostgresSource.from_env(DB_ENV_VAR),
        sql=_paired_market_sql(target_variant=actual_target, donor_variant=actual_donor),
        prompt_column="prompt_messages_json",
        example_key_column="log_id",
        label_columns=[
            "phase_name",
            "example_id",
            "family",
            "family_variant",
            "context_variant",
            "pair_key",
        ],
        case_columns=["pair_key"],
        case_key_column="pair_key",
        name="dx_terminal_synthetic_market_path_validation_v2_residual_path_smoke",
    )
    return dataset.select(limit=max(1, int(actual_limit)) * 2)


def build_workflow(
    dataset: Dataset | None = None,
    *,
    target_variant: str | None = None,
    donor_variant: str | None = None,
    transport: str | None = None,
    target_section: str | None = None,
    read_section: str | None = None,
) -> WorkflowSpec:
    selected_target = _target_variant() if target_variant is None else str(target_variant).strip()
    selected_donor = _donor_variant() if donor_variant is None else str(donor_variant).strip()
    selected_transport = _transport() if transport is None else str(transport).strip().lower()
    selected_target_section = _target_section() if target_section is None else str(target_section).strip()
    selected_read_section = _read_section() if read_section is None else str(read_section).strip()
    dataset = dataset or build_dataset(target_variant=selected_target, donor_variant=selected_donor)

    prompt_metadata = PromptMetadataBuilder.from_function(
        build_prompt_metadata,
        local_python_sources=("projects",),
    )
    row_evaluator = TransformBuilder.from_function(
        evaluate_patch_row,
        local_python_sources=("projects",),
    )
    engine = build_engine(batch_size=_dataset_limit())
    generation = build_generation_spec(max_tokens=generation_max_tokens())
    target_rows = dataset.labels("family_variant").equals(selected_target)
    donor_rows = dataset.labels("family_variant").equals(selected_donor)

    return WorkflowSpec(
        name="dx_terminal_synthetic_market_path_validation_v2_residual_path_smoke",
        steps=(
            WorkflowStep(
                name="capture_market_residual",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=engine,
                    dataset=dataset,
                    sites=(
                        ResidualSite(
                            name="market_residual",
                            site="resid_post",
                            layers=(PATCH_LAYER,),
                            tokens=TokenSelector.full_sequence(),
                        ),
                    ),
                    prompt_metadata_builder=prompt_metadata,
                ),
            ),
            WorkflowStep(
                name="build_activation_bank",
                runner="analysis_cpu",
                depends_on=("capture_market_residual",),
                spec=ActivationBankSpec(
                    feature=StepRef("capture_market_residual").feature("market_residual"),
                    layers=(PATCH_LAYER,),
                ),
            ),
            WorkflowStep(
                name="build_path_mask",
                runner="analysis_cpu",
                spec=ExplicitPathMaskSpec(
                    edges=(
                        ExplicitPathEdge(source_layer=PATCH_LAYER, write_layer=PATCH_LAYER, weight=1.0),
                    ),
                ),
            ),
            WorkflowStep(
                name="baseline_generation",
                runner="capture_gpu",
                spec=GenerationRunSpec(
                    engine=engine,
                    dataset=dataset,
                    select_when=target_rows,
                    generation=generation,
                ),
            ),
            WorkflowStep(
                name="patch_market",
                runner="capture_gpu",
                depends_on=("capture_market_residual", "build_activation_bank", "build_path_mask", "baseline_generation"),
                spec=PatchedGenerationSpec(
                    engine=engine,
                    dataset=dataset,
                    patch=ResidualPathPatch(
                        activation_bank=StepRef("build_activation_bank"),
                        path_mask=StepRef("build_path_mask"),
                        write_site=ResidualInterventionSite(site="resid_post", layers=(PATCH_LAYER,)),
                        target_tokens=TokenSelector.section(selected_target_section),
                        read_tokens=TokenSelector.section(selected_read_section),
                        transport=selected_transport,
                        strength=1.0,
                    ),
                    pair_by=dataset.cases("pair_key"),
                    target_when=target_rows,
                    donor_when=donor_rows,
                    generation=generation,
                    prompt_metadata_builder=prompt_metadata,
                ),
            ),
            WorkflowStep(
                name="compare_patch",
                runner="analysis_cpu",
                depends_on=("patch_market",),
                spec=PatchComparisonSpec(
                    baseline=StepRef("baseline_generation"),
                    variants={"patch": StepRef("patch_market")},
                    row_evaluator=row_evaluator,
                ),
            ),
        ),
    )


def build_runner_specs() -> dict[str, object]:
    return build_common_runner_specs(artifact_root=ARTIFACT_ROOT)
