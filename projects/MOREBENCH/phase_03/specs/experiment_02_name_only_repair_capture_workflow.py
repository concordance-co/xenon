from __future__ import annotations

from pipelines_v2.api import (
    ArtifactDatasetSource,
    CaptureSpec,
    Dataset,
    GenerationSpec,
    ResidualSite,
    StepRef,
    TensorStorage,
    TokenSelector,
    TransformBuilder,
    TransformSpec,
    WorkflowSpec,
    WorkflowStep,
)
from pipelines_v2.storage.artifacts import OperationArtifact, artifact_from_manifest

from projects.MOREBENCH.phase_03.specs import experiment_02_workflow as base


NAME_ONLY_GENERATION_ARTIFACT_ID = "generation_run_1_b7d85acbea54"


def _load_generation_artifact() -> OperationArtifact:
    manifest = base._catalog().load_artifact(NAME_ONLY_GENERATION_ARTIFACT_ID)
    if manifest is None:
        raise RuntimeError(f"Could not load generation artifact {NAME_ONLY_GENERATION_ARTIFACT_ID!r}")
    artifact = artifact_from_manifest(manifest, store=base.build_runner_specs()["analysis_cpu"].artifacts)
    if not isinstance(artifact, OperationArtifact):
        raise TypeError(f"Artifact {NAME_ONLY_GENERATION_ARTIFACT_ID!r} is not an operation artifact")
    return artifact


def build_repaired_name_only_capture_dataset():
    generation = _load_generation_artifact()
    return base.build_theory_persistence_capture_dataset(generation=generation)


def _artifact_capture_dataset() -> Dataset:
    return Dataset.from_source(
        source=ArtifactDatasetSource(),
        artifact=StepRef("build_repaired_name_only_capture_dataset"),
        result_key="dataset",
        provides_token_sections=True,
        name="morebench_phase03_experiment02_name_only_generation_capture_repaired",
    )


def build_dataset():
    return base.build_dataset(
        dataset_path=base.PHASE_ROOT / "outputs" / "experiment_02_name_only_generation_dataset.jsonl",
        dataset_name="morebench_phase03_experiment02_name_only_generation_batch",
    )


def build_runner_specs():
    return base.build_runner_specs()


def build_workflow():
    return WorkflowSpec(
        name="morebench_phase03_experiment02_name_only_repair_capture",
        steps=(
            WorkflowStep(
                name="build_repaired_name_only_capture_dataset",
                runner="analysis_cpu",
                description="Rebuild the strict name_only capture dataset from the existing generation artifact under the updated copy filter.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        build_repaired_name_only_capture_dataset,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={},
                ),
            ),
            WorkflowStep(
                name="capture_repaired_name_only_generated_sequence_residual",
                runner="capture_gpu",
                description="Capture generated-token residuals for the repaired name_only strict dataset.",
                spec=CaptureSpec(
                    engine=base._engine(max_num_seqs=8),
                    dataset=_artifact_capture_dataset(),
                    sites=[
                        ResidualSite(
                            name="generated_sequence_residual",
                            site="resid_post",
                            layers=list(base.CAPTURED_LAYERS),
                            tokens=TokenSelector.section("generated"),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        )
                    ],
                    generation=GenerationSpec(enabled=False),
                ),
            ),
        ),
    )
