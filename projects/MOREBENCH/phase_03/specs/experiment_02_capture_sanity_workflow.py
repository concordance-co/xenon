from __future__ import annotations

from pipelines_v2.api import (
    ArtifactDatasetSource,
    CaptureSpec,
    Dataset,
    GenerationRunSpec,
    GenerationSpec,
    ProbeSpec,
    ResidualSite,
    StepRef,
    TensorStorage,
    TextBaselineSpec,
    TokenPooling,
    TokenSelector,
    TransformBuilder,
    TransformSpec,
    WorkflowSpec,
    WorkflowStep,
)

from projects.MOREBENCH.phase_03.specs import experiment_02_smoke_workflows as smoke
from projects.MOREBENCH.phase_03.specs import experiment_02_workflow as base


def build_dataset() -> Dataset:
    return smoke.build_capture_readout_smoke_dataset()


def build_runner_specs() -> dict[str, object]:
    return smoke.build_runner_specs()


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    capture_rows = StepRef("build_theory_persistence_capture_dataset_lenient").label("group_id")
    capture_labels = StepRef("build_theory_persistence_capture_dataset_lenient").label("prime_condition")
    capture_split = StepRef("build_theory_persistence_capture_dataset_lenient").label("split")
    return WorkflowSpec(
        name="morebench_phase03_experiment02_capture_sanity",
        steps=(
            WorkflowStep(
                name="generate_theory_primed_responses",
                runner="capture_gpu",
                description="Small corrected-prompt generation batch for capture sanity.",
                spec=GenerationRunSpec(
                    engine=base._engine(max_num_seqs=8),
                    dataset=dataset,
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=base.GENERATION_MAX_TOKENS,
                        temperature=0.0,
                        top_p=1.0,
                    ),
                ),
            ),
            WorkflowStep(
                name="build_theory_persistence_capture_dataset_lenient",
                runner="analysis_cpu",
                description="Lenient capture dataset that preserves rows so pooled capture/probe plumbing can be checked.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        smoke.build_theory_persistence_capture_dataset_lenient_smoke,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={"generation": StepRef("generate_theory_primed_responses")},
                ),
            ),
            WorkflowStep(
                name="capture_generated_sequence_residual",
                runner="capture_gpu",
                description="Replay the sanity slice and capture the full generated-token residual sequence.",
                spec=CaptureSpec(
                    engine=base._engine(max_num_seqs=4),
                    dataset=Dataset.from_source(
                        source=ArtifactDatasetSource(),
                        artifact=StepRef("build_theory_persistence_capture_dataset_lenient"),
                        result_key="dataset",
                        provides_token_sections=True,
                        name="morebench_phase03_experiment02_capture_sanity_dataset",
                    ),
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
            WorkflowStep(
                name="text_baseline_generation_prime_condition",
                runner="analysis_cpu",
                description="Generated-text baseline on the capture sanity slice.",
                spec=TextBaselineSpec(
                    text=StepRef("build_theory_persistence_capture_dataset_lenient").label("generated_text"),
                    rows=capture_rows,
                    labels=capture_labels,
                    group_by=capture_rows,
                    split_by={"split": capture_split},
                    train_values=("train",),
                    test_values=("test",),
                    model="countvectorizer_logreg",
                    metrics=("accuracy", "balanced_accuracy"),
                ),
            ),
            WorkflowStep(
                name="probe_generation_prime_condition_residual",
                runner="analysis_cpu",
                description="Mean-pooled first-pass generation-time readout on the capture sanity slice.",
                spec=ProbeSpec(
                    feature=StepRef("capture_generated_sequence_residual").feature("generated_sequence_residual"),
                    rows=capture_rows,
                    labels=capture_labels,
                    group_by=capture_rows,
                    split=capture_split,
                    train_values=("train",),
                    test_values=("test",),
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.mean(),
                    metrics=("accuracy", "balanced_accuracy", "selectivity"),
                    baselines=("majority", "shuffled_label"),
                ),
            ),
            WorkflowStep(
                name="summarize_capture_sanity",
                runner="analysis_cpu",
                description="Collect generation, capture, baseline, and pooled-probe sanity outputs.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        base.summarize_experiment_02,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={
                        "generation": StepRef("generate_theory_primed_responses"),
                        "capture_dataset": StepRef("build_theory_persistence_capture_dataset_lenient"),
                        "capture_result": StepRef("capture_generated_sequence_residual"),
                        "text_baseline": StepRef("text_baseline_generation_prime_condition"),
                        "probe_result": StepRef("probe_generation_prime_condition_residual"),
                    },
                ),
            ),
        ),
    )
