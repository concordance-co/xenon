from __future__ import annotations

import os

from pipelines_v2.api import (
    AddDirectionPatch,
    CaptureSpec,
    CentroidSpec,
    Dataset,
    DirectionSpec,
    GenerationRunSpec,
    PatchComparisonSpec,
    PatchedGenerationSpec,
    PostgresSource,
    PromptMetadataBuilder,
    RandomControlPatch,
    ResidualInterventionSite,
    ResidualSite,
    StepRef,
    SubspaceSpec,
    SwapComponentsPatch,
    SwapMeanPatch,
    TokenPooling,
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
PATCH_COMPONENTS = 4
ARTIFACT_ROOT = "/data/artifacts/dx_terminal_synthetic_market/path_validation_v2_source_operator_smoke"


def _dataset_limit() -> int:
    raw = os.environ.get("SYNTHETIC_MARKET_V2_SOURCE_OPERATOR_SMOKE_LIMIT")
    if raw is None or not raw.strip():
        return DEFAULT_LIMIT
    value = int(raw)
    if value <= 0:
        raise ValueError("SYNTHETIC_MARKET_V2_SOURCE_OPERATOR_SMOKE_LIMIT must be a positive integer")
    return value


def _patch_operator() -> str:
    raw = os.environ.get("SYNTHETIC_MARKET_V2_SOURCE_OPERATOR")
    value = "add_direction" if raw is None or not raw.strip() else str(raw).strip().lower()
    allowed = {"add_direction", "random_control", "swap_mean", "swap_components"}
    if value not in allowed:
        raise ValueError(
            f"SYNTHETIC_MARKET_V2_SOURCE_OPERATOR must be one of {sorted(allowed)}"
        )
    return value


def _target_variant() -> str:
    raw = os.environ.get("SYNTHETIC_MARKET_V2_SOURCE_TARGET_VARIANT")
    return DEFAULT_TARGET_VARIANT if raw is None or not raw.strip() else str(raw).strip()


def _donor_variant() -> str:
    raw = os.environ.get("SYNTHETIC_MARKET_V2_SOURCE_DONOR_VARIANT")
    return DEFAULT_DONOR_VARIANT if raw is None or not raw.strip() else str(raw).strip()


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
        name="dx_terminal_synthetic_market_path_validation_v2_source_operator_smoke",
    )
    return dataset.select(limit=max(1, int(actual_limit)) * 2)


def build_workflow(
    dataset: Dataset | None = None,
    *,
    patch_operator: str | None = None,
    target_variant: str | None = None,
    donor_variant: str | None = None,
) -> WorkflowSpec:
    selected_operator = _patch_operator() if patch_operator is None else str(patch_operator).strip().lower()
    selected_target = _target_variant() if target_variant is None else str(target_variant).strip()
    selected_donor = _donor_variant() if donor_variant is None else str(donor_variant).strip()
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

    steps: list[WorkflowStep] = [
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
    ]

    capture_feature = StepRef("capture_market_residual").feature("market_residual")
    patch_depends_on = ["capture_market_residual", "baseline_generation"]

    if selected_operator in {"add_direction", "random_control", "swap_components"}:
        steps.append(
            WorkflowStep(
                name="learn_market_subspace",
                runner="analysis_cpu",
                depends_on=("capture_market_residual",),
                spec=SubspaceSpec(
                    feature=capture_feature,
                    layers=(PATCH_LAYER,),
                    components=PATCH_COMPONENTS,
                    tokens=TokenSelector.section("market"),
                    pooling=TokenPooling.mean(),
                ),
            )
        )
        patch_depends_on.append("learn_market_subspace")

    if selected_operator == "add_direction":
        steps.append(
            WorkflowStep(
                name="learn_market_direction",
                runner="analysis_cpu",
                depends_on=("capture_market_residual", "learn_market_subspace"),
                spec=DirectionSpec(
                    feature=capture_feature,
                    positive=donor_rows,
                    negative=target_rows,
                    subspace=StepRef("learn_market_subspace"),
                    layers=(PATCH_LAYER,),
                    tokens=TokenSelector.section("market"),
                    pooling=TokenPooling.mean(),
                ),
            )
        )
        patch_depends_on.append("learn_market_direction")

    if selected_operator in {"swap_mean", "swap_components"}:
        centroid_deps = ("capture_market_residual",)
        centroid_subspace = None
        if selected_operator == "swap_components":
            centroid_deps = ("capture_market_residual", "learn_market_subspace")
            centroid_subspace = StepRef("learn_market_subspace")
        steps.append(
            WorkflowStep(
                name="learn_market_centroids",
                runner="analysis_cpu",
                depends_on=centroid_deps,
                spec=CentroidSpec(
                    feature=capture_feature,
                    by=dataset.labels("family_variant"),
                    layers=(PATCH_LAYER,),
                    tokens=TokenSelector.section("market"),
                    pooling=TokenPooling.mean(),
                    subspace=centroid_subspace,
                ),
            )
        )
        patch_depends_on.append("learn_market_centroids")

    steps.append(
        WorkflowStep(
            name="baseline_generation",
            runner="capture_gpu",
            spec=GenerationRunSpec(
                engine=engine,
                dataset=dataset,
                select_when=target_rows,
                generation=generation,
            ),
        )
    )

    if selected_operator == "add_direction":
        patch = AddDirectionPatch(
            direction=StepRef("learn_market_direction"),
            subspace=StepRef("learn_market_subspace"),
            write_site=ResidualInterventionSite(site="resid_post", layers=(PATCH_LAYER,)),
            target_tokens=TokenSelector.section("market"),
            strength=1.0,
        )
    elif selected_operator == "random_control":
        patch = RandomControlPatch(
            subspace=StepRef("learn_market_subspace"),
            write_site=ResidualInterventionSite(site="resid_post", layers=(PATCH_LAYER,)),
            target_tokens=TokenSelector.section("market"),
            component_indices_by_layer={PATCH_LAYER: tuple(range(PATCH_COMPONENTS))},
            strength=1.0,
            random_seed=17,
            match_projected_norm=True,
        )
    elif selected_operator == "swap_mean":
        patch = SwapMeanPatch(
            centroids=StepRef("learn_market_centroids"),
            centroid_name=selected_donor,
            write_site=ResidualInterventionSite(site="resid_post", layers=(PATCH_LAYER,)),
            target_tokens=TokenSelector.section("market"),
            strength=1.0,
        )
    elif selected_operator == "swap_components":
        patch = SwapComponentsPatch(
            subspace=StepRef("learn_market_subspace"),
            centroids=StepRef("learn_market_centroids"),
            centroid_name=selected_donor,
            write_site=ResidualInterventionSite(site="resid_post", layers=(PATCH_LAYER,)),
            target_tokens=TokenSelector.section("market"),
            component_indices_by_layer={PATCH_LAYER: tuple(range(PATCH_COMPONENTS))},
            strength=1.0,
        )
    else:
        raise ValueError(f"Unsupported source-operator smoke patch operator: {selected_operator!r}")

    steps.extend(
        [
            WorkflowStep(
                name="patch_market",
                runner="capture_gpu",
                depends_on=tuple(patch_depends_on),
                spec=PatchedGenerationSpec(
                    engine=engine,
                    dataset=dataset,
                    patch=patch,
                    select_when=target_rows,
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
        ]
    )
    return WorkflowSpec(
        name="dx_terminal_synthetic_market_path_validation_v2_source_operator_smoke",
        steps=tuple(steps),
    )


def build_runner_specs() -> dict[str, object]:
    return build_common_runner_specs(artifact_root=ARTIFACT_ROOT)
