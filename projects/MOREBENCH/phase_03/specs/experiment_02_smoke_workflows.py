from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    Example,
    FileCatalog,
    GenerationRunSpec,
    GenerationSpec,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalVolumeMount,
    ModalVolumeStore,
    ProbeSpec,
    ReportSpec,
    ResidualSite,
    StepRef,
    TensorStorage,
    TextBaselineSpec,
    TokenPooling,
    TokenSelector,
    TransformBuilder,
    TransformResult,
    TransformSpec,
    WorkflowSpec,
    WorkflowStep,
)

from projects.MOREBENCH.phase_03.specs import experiment_02_workflow as base


GENERATION_ONLY_GROUPS = (
    "theory_group_001",
    "theory_group_002",
)

CAPTURE_SMOKE_GROUPS = (
    "theory_group_003",
    "theory_group_006",
    "theory_group_017",
    "theory_group_018",
    "theory_group_027",
)

CAPTURE_READOUT_SMOKE_GROUPS = (
    "theory_group_003",
    "theory_group_018",
)


def build_runner_specs() -> dict[str, object]:
    catalog = FileCatalog(root=Path("artifacts") / "morebench_phase03_experiment02_smoke_catalog")
    modal_store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts/morebench_phase_03_experiment02_smoke",
    )
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H200",
                timeout_seconds=60 * 60 * 3,
                volumes=(
                    ModalVolumeMount(
                        name=base.MODEL_VOLUME_NAME,
                        mount_path=base.MODEL_VOLUME_PATH,
                    ),
                ),
            ),
            artifacts=modal_store,
            catalog=catalog,
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=8,
                memory_mb=24 * 1024,
                timeout_seconds=60 * 60,
            ),
            artifacts=modal_store,
            catalog=catalog,
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(Path("artifacts") / "morebench_phase03_experiment02_smoke_reports"),
            catalog=catalog,
        ),
    }


def _subset_dataset(group_ids: Sequence[str], *, name: str) -> Dataset:
    group_set = set(group_ids)
    source = base.build_dataset()
    examples: list[Example] = [
        example
        for example in source.examples
        if str(example.labels.get("group_id", "")) in group_set
    ]
    return Dataset.from_examples(examples, name=name)


def build_generation_only_smoke_dataset() -> Dataset:
    return _subset_dataset(
        GENERATION_ONLY_GROUPS,
        name="morebench_phase03_experiment02_generation_only_smoke",
    )


def build_capture_smoke_dataset() -> Dataset:
    return _subset_dataset(
        CAPTURE_SMOKE_GROUPS,
        name="morebench_phase03_experiment02_capture_smoke",
    )


def build_capture_readout_smoke_dataset() -> Dataset:
    return _subset_dataset(
        CAPTURE_READOUT_SMOKE_GROUPS,
        name="morebench_phase03_experiment02_capture_readout_smoke",
    )


def build_theory_persistence_capture_dataset_lenient_smoke(*, generation: Any) -> dict[str, Any]:
    if not hasattr(generation, "result"):
        raise TypeError("lenient smoke capture dataset expects a generation artifact")

    payload = generation.result()
    if not isinstance(payload, Mapping):
        raise TypeError("generation artifact result must be a mapping")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise TypeError("generation artifact result must contain a rows list")

    examples: list[Example] = []
    generated_text_labels: dict[str, str] = {}
    group_id_labels: dict[str, str] = {}
    split_labels: dict[str, str] = {}
    prime_condition_labels: dict[str, str] = {}
    finish_reason_labels: dict[str, str] = {}
    token_count_labels: dict[str, int] = {}
    source_family_labels: dict[str, str] = {}
    response_length_labels: dict[str, int] = {}
    theory_name_copy_labels: dict[str, str] = {}
    cue_overlap_copy_labels: dict[str, str] = {}
    direct_copy_labels: dict[str, str] = {}

    length_rows = 0
    copy_flag_rows = 0

    for row in raw_rows:
        if not isinstance(row, Mapping):
            continue
        source_example = base._mapping(row.get("example"))
        key = str(row.get("example_key") or source_example.get("key") or "").strip()
        if not key:
            continue

        prompt_labels = dict(base._mapping(source_example.get("labels")))
        if not bool(prompt_labels.get("capture_enabled")):
            continue

        generated_text = str(row.get("generated_text") or row.get("text") or "")
        source_prompt = base._render_prompt_text(source_example.get("prompt") or "")
        if not generated_text.strip() or not source_prompt.strip():
            continue

        finish_reason = str(row.get("finish_reason") or "")
        token_ids = row.get("generated_token_ids")
        split = str(prompt_labels.get("split") or "train")
        prime_condition = str(prompt_labels.get("prime_condition") or "")
        group_id = str(prompt_labels.get("group_id") or key)
        source_family = str(prompt_labels.get("source_family") or "")
        cue_text = str(prompt_labels.get("cue_text") or "")
        theory_name = str(prompt_labels.get("theory_name") or "")
        is_theory_prime = bool(prompt_labels.get("is_theory_prime"))

        theory_name_metrics = (
            base._theory_name_copy_metrics(theory_name=theory_name, generated_text=generated_text)
            if is_theory_prime and bool(theory_name)
            else {"theory_name_mention_count": 0, "repeated_theory_name_copy": False}
        )
        theory_name_copy = bool(theory_name_metrics["repeated_theory_name_copy"])
        cue_overlap, cue_metrics = base._near_verbatim_cue_copy(
            cue_text=cue_text,
            generated_text=generated_text,
        ) if is_theory_prime else (False, {"cue_overlap_count": 0, "cue_overlap_fraction": 0.0, "cue_longest_run": 0})
        direct_copy = theory_name_copy or cue_overlap

        if finish_reason == "length":
            length_rows += 1
        if direct_copy:
            copy_flag_rows += 1

        combined_prompt, token_sections = base._combined_prompt_and_sections(
            source_prompt=source_prompt,
            generated_text=generated_text,
        )
        labels = {
            **prompt_labels,
            "generated_text": generated_text,
            "generation_finish_reason": finish_reason,
            "generated_token_count": len(token_ids) if isinstance(token_ids, list) else 0,
            "response_char_length": len(generated_text),
            "theory_name_copy_flag": "yes" if theory_name_copy else "no",
            "theory_name_mention_count": int(theory_name_metrics["theory_name_mention_count"]),
            "cue_overlap_copy_flag": "yes" if cue_overlap else "no",
            "cue_overlap_count": int(cue_metrics["cue_overlap_count"]),
            "cue_overlap_fraction": round(float(cue_metrics["cue_overlap_fraction"]), 4),
            "cue_longest_run": int(cue_metrics["cue_longest_run"]),
            "direct_theory_copy_flag": "yes" if direct_copy else "no",
        }
        metadata = {
            **base._mapping(source_example.get("metadata")),
            "source_generation_artifact_id": getattr(generation, "id", ""),
            "token_sections": token_sections,
        }
        examples.append(
            Example(
                key=key,
                prompt=combined_prompt,
                labels=labels,
                metadata=metadata,
                cases={
                    "group_id": group_id,
                    "base_dilemma_id": str(prompt_labels.get("base_dilemma_id") or group_id),
                },
                case_key=group_id,
            )
        )
        generated_text_labels[key] = generated_text
        group_id_labels[key] = group_id
        split_labels[key] = split
        prime_condition_labels[key] = prime_condition
        finish_reason_labels[key] = finish_reason
        token_count_labels[key] = len(token_ids) if isinstance(token_ids, list) else 0
        source_family_labels[key] = source_family
        response_length_labels[key] = len(generated_text)
        theory_name_copy_labels[key] = "yes" if theory_name_copy else "no"
        cue_overlap_copy_labels[key] = "yes" if cue_overlap else "no"
        direct_copy_labels[key] = "yes" if direct_copy else "no"

    dataset = Dataset.from_examples(
        examples,
        name="morebench_phase03_experiment02_generation_capture_readout_smoke",
    )
    return {
        "payload": {
            "kind": "morebench_theory_generation_persistence_capture_dataset_lenient_smoke",
            "dataset": dataset.to_dict(),
            "summary": {
                "source_generation_artifact_id": getattr(generation, "id", ""),
                "kept_capture_example_count": len(examples),
                "length_row_count": length_rows,
                "copy_flag_row_count": copy_flag_rows,
                "prime_condition_counts": dict(sorted(Counter(prime_condition_labels.values()).items())),
            },
        },
        "labels": {
            "generated_text": generated_text_labels,
            "group_id": group_id_labels,
            "split": split_labels,
            "prime_condition": prime_condition_labels,
            "generation_finish_reason": finish_reason_labels,
            "generated_token_count": token_count_labels,
            "source_family": source_family_labels,
            "response_char_length": response_length_labels,
            "theory_name_copy_flag": theory_name_copy_labels,
            "cue_overlap_copy_flag": cue_overlap_copy_labels,
            "direct_theory_copy_flag": direct_copy_labels,
        },
        "metadata": {
            "source": "GenerationRunSpec result rows",
            "unit": "group_id x prime_condition",
            "status": "lenient smoke dataset keeps length-finished and copy-flagged rows to verify downstream readouts",
        },
        "example_keys": sorted(generated_text_labels),
    }


def summarize_generation_only_smoke(*, generation: Any) -> TransformResult:
    payload = generation.result() if hasattr(generation, "result") else {}
    rows = payload.get("rows", []) if isinstance(payload, Mapping) else []
    finish_reason_counts = Counter(
        str(row.get("finish_reason") or "")
        for row in rows
        if isinstance(row, Mapping)
    )
    nonempty_count = sum(
        1
        for row in rows
        if isinstance(row, Mapping) and str(row.get("generated_text") or row.get("text") or "").strip()
    )
    return TransformResult(
        payload={
            "workflow": "morebench_phase03_experiment02_generation_only_smoke",
            "row_count": len(rows),
            "nonempty_count": nonempty_count,
            "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
            "sample_keys": [
                str(row.get("example_key") or "")
                for row in rows[:5]
                if isinstance(row, Mapping)
            ],
        }
    )


def build_generation_only_smoke_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_generation_only_smoke_dataset()
    return WorkflowSpec(
        name="morebench_phase03_experiment02_generation_only_smoke",
        steps=(
            WorkflowStep(
                name="generate_theory_primed_responses",
                runner="capture_gpu",
                description="Tiny uncaptured generation-only smoke for Experiment 2.",
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
                name="summarize_generation_only_smoke",
                runner="analysis_cpu",
                description="Check that generation rows, finish reasons, and nonempty outputs materialize.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_generation_only_smoke,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={"generation": StepRef("generate_theory_primed_responses")},
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                description="Write a local report artifact so dashboard previews can inspect the smoke outputs.",
                spec=ReportSpec(
                    inputs=(
                        StepRef("generate_theory_primed_responses"),
                        StepRef("summarize_generation_only_smoke"),
                    ),
                    template="default",
                    output_dir="projects/MOREBENCH/phase_03/reports/experiment_02_generation_only_smoke_report",
                ),
            ),
        ),
    )


def build_capture_smoke_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_capture_smoke_dataset()
    capture_rows = StepRef("build_theory_persistence_capture_dataset").label("group_id")
    capture_labels = StepRef("build_theory_persistence_capture_dataset").label("prime_condition")
    capture_split = StepRef("build_theory_persistence_capture_dataset").label("split")
    return WorkflowSpec(
        name="morebench_phase03_experiment02_capture_smoke",
        steps=(
            WorkflowStep(
                name="generate_theory_primed_responses",
                runner="capture_gpu",
                description="Tiny full-sequence capture smoke for Experiment 2.",
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
                name="build_theory_persistence_capture_dataset",
                runner="analysis_cpu",
                description="Build the filtered generation-time capture dataset for the smoke slice.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        base.build_theory_persistence_capture_dataset,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={"generation": StepRef("generate_theory_primed_responses")},
                ),
            ),
            WorkflowStep(
                name="capture_generated_sequence_residual",
                runner="capture_gpu",
                description="Replay the smoke slice and capture the full generated-token residual sequence.",
                spec=CaptureSpec(
                    engine=base._engine(max_num_seqs=4),
                    dataset=base._artifact_capture_dataset(),
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
                description="Cheap lexical baseline on generated text for the smoke slice.",
                spec=TextBaselineSpec(
                    text=StepRef("build_theory_persistence_capture_dataset").label("generated_text"),
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
                description="Mean-pooled readout over the full generated-token sequence on the smoke slice.",
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
                name="summarize_capture_smoke",
                runner="analysis_cpu",
                description="Collect the generation, capture, baseline, and probe smoke readouts.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        base.summarize_experiment_02,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={
                        "generation": StepRef("generate_theory_primed_responses"),
                        "capture_dataset": StepRef("build_theory_persistence_capture_dataset"),
                        "capture_result": StepRef("capture_generated_sequence_residual"),
                        "text_baseline": StepRef("text_baseline_generation_prime_condition"),
                        "probe_result": StepRef("probe_generation_prime_condition_residual"),
                    },
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                description="Write a local report artifact for the strict capture smoke.",
                spec=ReportSpec(
                    inputs=(
                        StepRef("generate_theory_primed_responses"),
                        StepRef("build_theory_persistence_capture_dataset"),
                        StepRef("capture_generated_sequence_residual"),
                        StepRef("text_baseline_generation_prime_condition"),
                        StepRef("probe_generation_prime_condition_residual"),
                        StepRef("summarize_capture_smoke"),
                    ),
                    template="default",
                    output_dir="projects/MOREBENCH/phase_03/reports/experiment_02_capture_smoke_report",
                ),
            ),
        ),
    )


def build_capture_readout_smoke_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_capture_readout_smoke_dataset()
    capture_rows = StepRef("build_theory_persistence_capture_dataset_lenient_smoke").label("group_id")
    capture_labels = StepRef("build_theory_persistence_capture_dataset_lenient_smoke").label("prime_condition")
    capture_split = StepRef("build_theory_persistence_capture_dataset_lenient_smoke").label("split")
    return WorkflowSpec(
        name="morebench_phase03_experiment02_capture_readout_smoke",
        steps=(
            WorkflowStep(
                name="generate_theory_primed_responses",
                runner="capture_gpu",
                description="Tiny readout-oriented full-sequence capture smoke for Experiment 2.",
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
                name="build_theory_persistence_capture_dataset_lenient_smoke",
                runner="analysis_cpu",
                description="Smoke-only capture dataset that preserves rows so the readout path can be exercised.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        build_theory_persistence_capture_dataset_lenient_smoke,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={"generation": StepRef("generate_theory_primed_responses")},
                ),
            ),
            WorkflowStep(
                name="capture_generated_sequence_residual",
                runner="capture_gpu",
                description="Replay the readout-smoke slice and capture the full generated-token residual sequence.",
                spec=CaptureSpec(
                    engine=base._engine(max_num_seqs=4),
                    dataset=Dataset.from_source(
                        source=base.ArtifactDatasetSource(),
                        artifact=StepRef("build_theory_persistence_capture_dataset_lenient_smoke"),
                        result_key="dataset",
                        provides_token_sections=True,
                        name="morebench_phase03_experiment02_generation_capture_readout_smoke",
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
                description="Generated-text lexical baseline for the readout-smoke slice.",
                spec=TextBaselineSpec(
                    text=StepRef("build_theory_persistence_capture_dataset_lenient_smoke").label("generated_text"),
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
                description="Mean-pooled generation-time readout for the readout-smoke slice.",
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
                name="report",
                runner="report_local",
                description="Write a local report artifact for the readout smoke.",
                spec=ReportSpec(
                    inputs=(
                        StepRef("generate_theory_primed_responses"),
                        StepRef("build_theory_persistence_capture_dataset_lenient_smoke"),
                        StepRef("capture_generated_sequence_residual"),
                        StepRef("text_baseline_generation_prime_condition"),
                        StepRef("probe_generation_prime_condition_residual"),
                    ),
                    template="default",
                    output_dir="projects/MOREBENCH/phase_03/reports/experiment_02_capture_readout_smoke_report",
                ),
            ),
        ),
    )
