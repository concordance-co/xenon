"""All-theories persona-vector pole pilot workflow (phase 02).

Re-runs the phase_01 deontology pole pilot across four moral theories
(deontology, utilitarian, virtue_ethics, contractualism) to test whether the
phase_01 null result was deontology-specific (because Qwen3-30B-A3B's default
mode is approximately deontology-aligned) or whether the persona-vectors
primed-vs-default recipe simply does not extract clean directions for any
theory on this model.

Conditions (14 total): for each theory a primary positive, a positive variant,
and a theory-specific anti pole. Plus two shared neutral poles. Each
positive's "alt theory" diagnostic is a different theory's positive (no
extra conditions needed).

Reuses the phase_01 dilemmas (30 compact MoReBench-shaped dilemmas).

Phase: theory_persona_vectors / phase_02.
"""

from __future__ import annotations

from pathlib import Path

from pipelines_v2.api import (
    ArtifactDatasetSource,
    CaptureSpec,
    Dataset,
    FileCatalog,
    GenerationRunSpec,
    GenerationSpec,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    PostgresCatalog,
    PostgresSource,
    ReportSpec,
    ResidualSite,
    StepRef,
    TensorStorage,
    TokenSelector,
    TransformBuilder,
    TransformSpec,
    WorkflowSpec,
    WorkflowStep,
)

from projects.MOREBENCH.phase_03.specs import experiment_02_workflow as base
from projects.MOREBENCH.theory_persona_vectors.phase_01.specs import (
    deontology_pole_pilot_workflow as p1,
)


WORKFLOW_NAME = "morebench_theory_persona_vectors_phase02_all_theories_pole_pilot"
PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_02")
DILEMMAS_PATH = PHASE_ROOT / "outputs" / "all_theories_pole_pilot_synth_dilemmas.jsonl"
CONDITIONS_PATH = PHASE_ROOT / "specs" / "all_theories_pole_pilot_prompt_conditions.json"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "morebench_theory_persona_vectors_phase02"
LOCAL_CATALOG_ROOT = Path("artifacts") / "morebench_theory_persona_vectors_phase02_catalog"
MODAL_ARTIFACT_ROOT = "/data/artifacts/morebench_theory_persona_vectors_phase02"
REPORT_OUTPUT_DIR = PHASE_ROOT / "reports" / "all_theories_pole_pilot_report"

# vLLM dedupes identical prompts within a batch, so generations-per-prompt
# collapses to 1 regardless of the dataset-level multiplier. We set
# SAMPLES_PER_CONDITION=1 explicitly here (vs phase_01 which asked for 3 and
# silently got 1) to make the artifact count honest.
SAMPLES_PER_CONDITION = 1
GENERATION_MAX_TOKENS = p1.GENERATION_MAX_TOKENS
GENERATION_TEMPERATURE = p1.GENERATION_TEMPERATURE
GENERATION_TOP_P = p1.GENERATION_TOP_P
CAPTURED_LAYERS = p1.CAPTURED_LAYERS
SYSTEM_PROMPT = p1.SYSTEM_PROMPT
DB_ENV_VAR = p1.DB_ENV_VAR


def build_dataset() -> Dataset:
    """Build 30 dilemmas x 14 conditions = 420 examples."""
    # Reuse phase_01's builder by temporarily redirecting its module-level paths.
    original_dilemmas = p1.DILEMMAS_PATH
    original_conditions = p1.CONDITIONS_PATH
    original_samples = p1.SAMPLES_PER_CONDITION
    try:
        p1.DILEMMAS_PATH = DILEMMAS_PATH
        p1.CONDITIONS_PATH = CONDITIONS_PATH
        p1.SAMPLES_PER_CONDITION = SAMPLES_PER_CONDITION
        return p1.build_dataset()
    finally:
        p1.DILEMMAS_PATH = original_dilemmas
        p1.CONDITIONS_PATH = original_conditions
        p1.SAMPLES_PER_CONDITION = original_samples


def _artifact_capture_dataset() -> Dataset:
    return Dataset.from_source(
        source=ArtifactDatasetSource(),
        artifact=StepRef("build_capture_dataset"),
        result_key="dataset",
        provides_token_sections=True,
        name=f"{WORKFLOW_NAME}_capture_dataset",
    )


def build_runner_specs() -> dict[str, object]:
    import os

    if os.getenv(DB_ENV_VAR):
        db = PostgresSource.from_env(DB_ENV_VAR)
        catalog = PostgresCatalog(source=db)
        modal_secrets = (ModalSecret.from_env_var(DB_ENV_VAR, secret_name="xenon-neon"),)
    else:
        catalog = FileCatalog(root=LOCAL_CATALOG_ROOT)
        modal_secrets = ()

    modal_store = ModalVolumeStore(name="xenon-data", root=MODAL_ARTIFACT_ROOT)

    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H200",
                timeout_seconds=60 * 60 * 4,
                shard_count=base.GPU_SHARD_COUNT,
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

    return WorkflowSpec(
        name=WORKFLOW_NAME,
        steps=(
            WorkflowStep(
                name="generate_terse_responses",
                runner="capture_gpu",
                description=(
                    "Generate terse recommendation-only responses for the 14-condition all-theories "
                    "pole pilot. One sample per (dilemma, condition) at temperature 0.7."
                ),
                spec=GenerationRunSpec(
                    engine=base._engine(),
                    dataset=dataset,
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=GENERATION_MAX_TOKENS,
                        temperature=GENERATION_TEMPERATURE,
                        top_p=GENERATION_TOP_P,
                    ),
                ),
            ),
            WorkflowStep(
                name="build_capture_dataset",
                runner="analysis_cpu",
                description=(
                    "Filter the terse-generation batch into a capture dataset, dropping empty and "
                    "length-finished rows. Records prompt-end and generated section spans."
                ),
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        p1.build_capture_dataset,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={"generation": StepRef("generate_terse_responses")},
                ),
            ),
            WorkflowStep(
                name="capture_residuals",
                runner="capture_gpu",
                description=(
                    "Capture prompt-end and generated-sequence residuals on the filtered terse-response dataset. "
                    "Layers: 0, 4, 16, 24, 32, 40 (pilot subset; primary L32)."
                ),
                spec=CaptureSpec(
                    engine=base._engine(max_num_seqs=8),
                    dataset=_artifact_capture_dataset(),
                    sites=[
                        ResidualSite(
                            name="prompt_end_residual",
                            site="resid_post",
                            layers=list(CAPTURED_LAYERS),
                            tokens=TokenSelector.section("prompt_end"),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                        ResidualSite(
                            name="generated_sequence_residual",
                            site="resid_post",
                            layers=list(CAPTURED_LAYERS),
                            tokens=TokenSelector.section("generated"),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                    ],
                    generation=GenerationSpec(enabled=False),
                ),
            ),
            WorkflowStep(
                name="summarize_pilot",
                runner="analysis_cpu",
                description="Compact post-capture summary of generation+capture status for the all-theories pole pilot.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        p1.summarize_pilot,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={
                        "capture_result": StepRef("capture_residuals"),
                        "capture_dataset": StepRef("build_capture_dataset"),
                    },
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                description="Package pilot generation+capture artifacts for local browsing.",
                spec=ReportSpec(
                    inputs=(
                        StepRef("generate_terse_responses"),
                        StepRef("build_capture_dataset"),
                        StepRef("capture_residuals"),
                        StepRef("summarize_pilot"),
                    ),
                    template="default",
                    output_dir=str(REPORT_OUTPUT_DIR),
                ),
            ),
        ),
    )
