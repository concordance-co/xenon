from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

from pipelines_v2.api import (
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
    StepRef,
    TransformBuilder,
    TransformResult,
    TransformSpec,
    WorkflowSpec,
    WorkflowStep,
)

from projects.MOREBENCH.phase_03.specs import experiment_02_workflow as base


SANITY_GROUPS = (
    "theory_group_001",
    "theory_group_002",
    "theory_group_003",
    "theory_group_004",
    "theory_group_005",
)


def build_runner_specs() -> dict[str, object]:
    catalog = FileCatalog(root=Path("artifacts") / "morebench_phase03_experiment02_behavior_sanity_catalog")
    modal_store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts/morebench_phase_03_experiment02_behavior_sanity",
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
        "analysis_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(Path("artifacts") / "morebench_phase03_experiment02_behavior_sanity"),
            catalog=catalog,
        ),
    }


def build_dataset() -> Dataset:
    group_set = set(SANITY_GROUPS)
    source = base.build_dataset()
    examples: list[Example] = [
        example
        for example in source.examples
        if str(example.labels.get("group_id", "")) in group_set
    ]
    return Dataset.from_examples(examples, name="morebench_phase03_experiment02_generation_behavior_sanity")


def summarize_generation_behavior(*, generation: Any) -> TransformResult:
    payload = generation.result() if hasattr(generation, "result") else {}
    rows = payload.get("rows", []) if isinstance(payload, Mapping) else []

    finish_reason_counts: Counter[str] = Counter()
    token_counts: list[int] = []
    char_lengths: list[int] = []
    theory_name_copy_count = 0
    cue_overlap_ge3_count = 0
    direct_copy_count = 0
    nonempty_count = 0
    distinct_responses_by_group: dict[str, set[str]] = defaultdict(set)
    sample_rows: list[dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, Mapping):
            continue
        finish_reason = str(row.get("finish_reason") or "")
        finish_reason_counts[finish_reason] += 1

        generated_text = str(row.get("generated_text") or row.get("text") or "")
        if generated_text.strip():
            nonempty_count += 1
        char_lengths.append(len(generated_text))

        token_ids = row.get("generated_token_ids")
        token_count = len(token_ids) if isinstance(token_ids, list) else 0
        token_counts.append(token_count)

        source_example = base._mapping(row.get("example"))
        labels = dict(base._mapping(source_example.get("labels")))
        group_id = str(labels.get("group_id") or "")
        prime_condition = str(labels.get("prime_condition") or "")
        cue_text = str(labels.get("cue_text") or "")
        theory_name = str(labels.get("theory_name") or "")
        is_theory_prime = bool(labels.get("is_theory_prime"))

        normalized_generated = " ".join(generated_text.strip().lower().split())
        theory_name_copy = is_theory_prime and bool(theory_name) and theory_name.lower() in normalized_generated
        cue_tokens = set(base._content_tokens(cue_text)) if is_theory_prime else set()
        generated_tokens = set(base._content_tokens(generated_text)) if is_theory_prime else set()
        cue_overlap_ge3 = len(cue_tokens & generated_tokens) >= 3
        direct_copy = theory_name_copy or cue_overlap_ge3

        if theory_name_copy:
            theory_name_copy_count += 1
        if cue_overlap_ge3:
            cue_overlap_ge3_count += 1
        if direct_copy:
            direct_copy_count += 1

        if group_id and generated_text.strip():
            distinct_responses_by_group[group_id].add(" ".join(generated_text.split()))

        if len(sample_rows) < 12:
            sample_rows.append(
                {
                    "example_key": str(row.get("example_key") or ""),
                    "group_id": group_id,
                    "prime_condition": prime_condition,
                    "finish_reason": finish_reason,
                    "generated_token_count": token_count,
                    "preview": generated_text[:400],
                }
            )

    def _summary_stats(values: Sequence[int]) -> dict[str, float | int]:
        if not values:
            return {"count": 0, "min": 0, "median": 0, "max": 0}
        return {
            "count": len(values),
            "min": min(values),
            "median": int(median(values)),
            "max": max(values),
        }

    distinct_response_counts = {
        group_id: len(responses)
        for group_id, responses in sorted(distinct_responses_by_group.items())
    }

    return TransformResult(
        payload={
            "workflow": "morebench_phase03_experiment02_generation_behavior_sanity",
            "group_ids": list(SANITY_GROUPS),
            "row_count": len(rows),
            "nonempty_count": nonempty_count,
            "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
            "generated_token_count_summary": _summary_stats(token_counts),
            "response_char_length_summary": _summary_stats(char_lengths),
            "theory_name_copy_count": theory_name_copy_count,
            "cue_overlap_ge3_count": cue_overlap_ge3_count,
            "direct_copy_count": direct_copy_count,
            "distinct_response_counts_by_group": distinct_response_counts,
            "groups_with_at_least_3_distinct_responses": sum(
                1 for count in distinct_response_counts.values() if count >= 3
            ),
            "sample_rows": sample_rows,
        }
    )


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    return WorkflowSpec(
        name="morebench_phase03_experiment02_generation_behavior_sanity",
        steps=(
            WorkflowStep(
                name="generate_theory_primed_responses",
                runner="capture_gpu",
                description="Generation-only behavioral sanity pass on 30 theory-primed examples.",
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
                name="summarize_generation_behavior",
                runner="analysis_local",
                description="Summarize stopping behavior, copy heuristics, and cross-prime divergence.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_generation_behavior,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={"generation": StepRef("generate_theory_primed_responses")},
                ),
            ),
        ),
    )
