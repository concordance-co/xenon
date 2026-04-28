"""L32 donor-interchange test for forced-choice ethical-vs-exploit prompts."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from pipelines_v2.api import (
    ActivationBankSpec,
    CaptureSpec,
    Dataset,
    Example,
    FileCatalog,
    GenerationRunSpec,
    GenerationSpec,
    InterchangePatch,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    PatchComparisonSpec,
    PatchedGenerationSpec,
    PostgresCatalog,
    PostgresSource,
    PromptMetadataBuilder,
    ReportSpec,
    ResidualInterventionSite,
    ResidualSite,
    StepRef,
    TensorStorage,
    TokenSelector,
    TransformBuilder,
    TransformSpec,
    WorkflowSpec,
    WorkflowStep,
)

from projects.MOREBENCH.ethical_advantage_vectors.phase_02.scripts.donor_patch_eval import (
    build_patch_prompt_metadata,
    evaluate_donor_patch_row,
    summarize_patch_comparison,
)
from projects.MOREBENCH.ethical_advantage_vectors.phase_02.specs import forced_choice_workflow as fc
from projects.MOREBENCH.phase_03.specs import experiment_02_workflow as base


WORKFLOW_NAME = "morebench_ethical_advantage_vectors_phase02_forced_choice_donor_patch"
PHASE_ROOT = Path("projects/MOREBENCH/ethical_advantage_vectors/phase_02")
REPORT_OUTPUT_DIR = PHASE_ROOT / "reports" / "forced_choice_donor_patch"

LOCAL_ARTIFACT_ROOT = Path("artifacts") / "morebench_ethical_advantage_vectors_phase02_donor_patch"
LOCAL_CATALOG_ROOT = Path("artifacts") / "morebench_ethical_advantage_vectors_phase02_donor_patch_catalog"
MODAL_ARTIFACT_ROOT = "/data/artifacts/morebench_ethical_advantage_vectors_phase02_donor_patch"
DB_ENV_VAR = "XENON_NEON_DATABASE_URL"

WRITE_LAYER = int(os.getenv("MOREBENCH_ETH_ADV_PATCH_LAYER", "32"))
PATCH_SECTION = os.getenv("MOREBENCH_ETH_ADV_PATCH_SECTION", "options_end").strip() or "options_end"
GENERATION_MAX_TOKENS = 4
GENERATION_TEMPERATURE = 0.0
GENERATION_TOP_P = 1.0

ETHICAL_CONDITION = "P_ethical_choice_01"
EXPLOIT_CONDITION = "P_exploit_choice_01"


def _engine():
    return replace(
        base._engine(max_num_seqs=8),
        enable_prefix_caching=False,
    )


def build_dataset() -> Dataset:
    """Reuse the forced-choice substrate, but pair rows by dilemma + option order."""

    source = fc.build_dataset()
    examples: list[Example] = []
    for example in source.examples:
        condition_id = str(example.labels.get("condition_id") or "")
        if condition_id not in {ETHICAL_CONDITION, EXPLOIT_CONDITION}:
            continue
        dilemma_id = str(example.labels.get("dilemma_id") or "")
        order_index = int(example.labels.get("option_order_index") or 0)
        patch_pair_id = f"{dilemma_id}__order_{order_index:02d}"
        examples.append(
            replace(
                example,
                labels={
                    **dict(example.labels),
                    "patch_pair_id": patch_pair_id,
                    "patch_condition": "ethical" if condition_id == ETHICAL_CONDITION else "exploit",
                },
                cases={
                    **{key: value for key, value in dict(example.cases).items() if key != "case_key"},
                    "patch_pair_id": patch_pair_id,
                },
                case_key=patch_pair_id,
            )
        )
    return Dataset.from_examples(examples, name=f"{WORKFLOW_NAME}_dataset")


def build_runner_specs() -> dict[str, object]:
    if os.getenv(DB_ENV_VAR):
        db = PostgresSource.from_env(DB_ENV_VAR)
        catalog = PostgresCatalog(source=db)
        modal_secrets = (ModalSecret.from_env_var(DB_ENV_VAR, secret_name="xenon-neon"),)
    else:
        catalog = FileCatalog(root=LOCAL_CATALOG_ROOT)
        modal_secrets = ()

    modal_store = ModalVolumeStore(name="xenon-data", root=MODAL_ARTIFACT_ROOT)
    return {
        "gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H200",
                timeout_seconds=60 * 60 * 4,
                shard_count=1,
                secrets=modal_secrets,
                volumes=(ModalVolumeMount(name=base.MODEL_VOLUME_NAME, mount_path=base.MODEL_VOLUME_PATH),),
            ),
            artifacts=modal_store,
            catalog=catalog,
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=4,
                memory_mb=12 * 1024,
                timeout_seconds=60 * 60,
                secrets=modal_secrets,
            ),
            artifacts=modal_store,
            catalog=catalog,
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(LOCAL_ARTIFACT_ROOT),
            catalog=catalog,
        ),
    }


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    engine = _engine()
    metadata_builder = PromptMetadataBuilder.from_function(
        build_patch_prompt_metadata,
        local_python_sources=("projects/MOREBENCH/ethical_advantage_vectors/phase_02/scripts",),
    )
    ethical_rows = dataset.labels("condition_id").equals(ETHICAL_CONDITION)
    exploit_rows = dataset.labels("condition_id").equals(EXPLOIT_CONDITION)
    row_evaluator = TransformBuilder.from_function(
        evaluate_donor_patch_row,
        local_python_sources=("projects/MOREBENCH",),
    )

    ethical_to_exploit_name = f"ethical_to_exploit_l{WRITE_LAYER}_{PATCH_SECTION}"
    exploit_to_ethical_name = f"exploit_to_ethical_l{WRITE_LAYER}_{PATCH_SECTION}"

    steps: list[WorkflowStep] = [
        WorkflowStep(
            name="capture_patch_site_residual",
            runner="gpu",
            description=f"Capture L{WRITE_LAYER} {PATCH_SECTION} residuals for ethical/exploit donor banks.",
            spec=CaptureSpec(
                engine=engine,
                dataset=dataset,
                prompt_metadata_builder=metadata_builder,
                sites=[
                    ResidualSite(
                        name=f"{PATCH_SECTION}_residual",
                        site="resid_post",
                        layers=(WRITE_LAYER,),
                        tokens=TokenSelector.section(PATCH_SECTION),
                        storage=TensorStorage(dtype="float16", format="safetensors"),
                    ),
                ],
                generation=GenerationSpec(enabled=False),
            ),
        ),
        WorkflowStep(
            name="ethical_activation_bank",
            runner="analysis_cpu",
            depends_on=("capture_patch_site_residual",),
            spec=ActivationBankSpec(
                feature=StepRef("capture_patch_site_residual").feature(f"{PATCH_SECTION}_residual"),
                layers=(WRITE_LAYER,),
                rows=ethical_rows,
            ),
        ),
        WorkflowStep(
            name="exploit_activation_bank",
            runner="analysis_cpu",
            depends_on=("capture_patch_site_residual",),
            spec=ActivationBankSpec(
                feature=StepRef("capture_patch_site_residual").feature(f"{PATCH_SECTION}_residual"),
                layers=(WRITE_LAYER,),
                rows=exploit_rows,
            ),
        ),
        WorkflowStep(
            name="baseline_exploit_targets",
            runner="gpu",
            depends_on=("ethical_activation_bank", "exploit_activation_bank"),
            spec=GenerationRunSpec(
                engine=engine,
                dataset=dataset,
                select_when=exploit_rows,
                generation=GenerationSpec(
                    enabled=True,
                    max_tokens=GENERATION_MAX_TOKENS,
                    temperature=GENERATION_TEMPERATURE,
                    top_p=GENERATION_TOP_P,
                    capture_reasoning=False,
                ),
            ),
        ),
        WorkflowStep(
            name="baseline_ethical_targets",
            runner="gpu",
            depends_on=("baseline_exploit_targets",),
            spec=GenerationRunSpec(
                engine=engine,
                dataset=dataset,
                select_when=ethical_rows,
                generation=GenerationSpec(
                    enabled=True,
                    max_tokens=GENERATION_MAX_TOKENS,
                    temperature=GENERATION_TEMPERATURE,
                    top_p=GENERATION_TOP_P,
                    capture_reasoning=False,
                ),
            ),
        ),
        WorkflowStep(
            name="patch_ethical_into_exploit",
            runner="gpu",
            depends_on=("baseline_ethical_targets",),
            spec=PatchedGenerationSpec(
                engine=engine,
                dataset=dataset,
                prompt_metadata_builder=metadata_builder,
                patch=InterchangePatch(
                    activation_bank=StepRef("ethical_activation_bank"),
                    write_site=ResidualInterventionSite(site="resid_post", layers=(WRITE_LAYER,)),
                    target_tokens=TokenSelector.section(PATCH_SECTION),
                    donor_tokens=TokenSelector.section(PATCH_SECTION),
                ),
                pair_by=dataset.cases("patch_pair_id"),
                target_when=exploit_rows,
                donor_when=ethical_rows,
                generation=GenerationSpec(
                    enabled=True,
                    max_tokens=GENERATION_MAX_TOKENS,
                    temperature=GENERATION_TEMPERATURE,
                    top_p=GENERATION_TOP_P,
                    capture_reasoning=False,
                ),
            ),
        ),
        WorkflowStep(
            name="patch_exploit_into_ethical",
            runner="gpu",
            depends_on=("patch_ethical_into_exploit",),
            spec=PatchedGenerationSpec(
                engine=engine,
                dataset=dataset,
                prompt_metadata_builder=metadata_builder,
                patch=InterchangePatch(
                    activation_bank=StepRef("exploit_activation_bank"),
                    write_site=ResidualInterventionSite(site="resid_post", layers=(WRITE_LAYER,)),
                    target_tokens=TokenSelector.section(PATCH_SECTION),
                    donor_tokens=TokenSelector.section(PATCH_SECTION),
                ),
                pair_by=dataset.cases("patch_pair_id"),
                target_when=ethical_rows,
                donor_when=exploit_rows,
                generation=GenerationSpec(
                    enabled=True,
                    max_tokens=GENERATION_MAX_TOKENS,
                    temperature=GENERATION_TEMPERATURE,
                    top_p=GENERATION_TOP_P,
                    capture_reasoning=False,
                ),
            ),
        ),
        WorkflowStep(
            name="compare_ethical_to_exploit",
            runner="analysis_cpu",
            depends_on=("patch_exploit_into_ethical",),
            spec=PatchComparisonSpec(
                baseline=StepRef("baseline_exploit_targets"),
                variants={ethical_to_exploit_name: StepRef("patch_ethical_into_exploit")},
                row_evaluator=row_evaluator,
            ),
        ),
        WorkflowStep(
            name="compare_exploit_to_ethical",
            runner="analysis_cpu",
            depends_on=("compare_ethical_to_exploit",),
            spec=PatchComparisonSpec(
                baseline=StepRef("baseline_ethical_targets"),
                variants={exploit_to_ethical_name: StepRef("patch_exploit_into_ethical")},
                row_evaluator=row_evaluator,
            ),
        ),
        WorkflowStep(
            name="summarize_ethical_to_exploit",
            runner="analysis_cpu",
            depends_on=("compare_exploit_to_ethical",),
            spec=TransformSpec(
                builder=TransformBuilder.from_function(
                    summarize_patch_comparison,
                    local_python_sources=("projects/MOREBENCH/ethical_advantage_vectors/phase_02/scripts",),
                ),
                inputs={
                    "comparison": StepRef("compare_ethical_to_exploit"),
                    "direction": "ethical_to_exploit",
                    "write_layer": WRITE_LAYER,
                    "patch_section": PATCH_SECTION,
                },
            ),
        ),
        WorkflowStep(
            name="summarize_exploit_to_ethical",
            runner="analysis_cpu",
            depends_on=("summarize_ethical_to_exploit",),
            spec=TransformSpec(
                builder=TransformBuilder.from_function(
                    summarize_patch_comparison,
                    local_python_sources=("projects/MOREBENCH/ethical_advantage_vectors/phase_02/scripts",),
                ),
                inputs={
                    "comparison": StepRef("compare_exploit_to_ethical"),
                    "direction": "exploit_to_ethical",
                    "write_layer": WRITE_LAYER,
                    "patch_section": PATCH_SECTION,
                },
            ),
        ),
        WorkflowStep(
            name="report",
            runner="report_local",
            depends_on=("summarize_exploit_to_ethical",),
            spec=ReportSpec(
                inputs=(
                    StepRef("baseline_exploit_targets"),
                    StepRef("baseline_ethical_targets"),
                    StepRef("patch_ethical_into_exploit"),
                    StepRef("patch_exploit_into_ethical"),
                    StepRef("compare_ethical_to_exploit"),
                    StepRef("compare_exploit_to_ethical"),
                    StepRef("summarize_ethical_to_exploit"),
                    StepRef("summarize_exploit_to_ethical"),
                ),
                template="default",
                output_dir=str(REPORT_OUTPUT_DIR),
            ),
        ),
    ]

    return WorkflowSpec(name=WORKFLOW_NAME, steps=tuple(steps))


workflow = build_workflow()
runner_specs = build_runner_specs()
