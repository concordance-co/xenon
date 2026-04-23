"""Raw rubric-dimension and generation-text labels for MoReBench."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pipelines_v2.data.datasets import Dataset, Example
from projects.MOREBENCH.shared.rubric_validation import DIMENSIONS


DIMENSION_SLUGS = {
    "identifying": "identifying",
    "clear process": "clear_process",
    "logical process": "logical_process",
    "helpful outcome": "helpful_outcome",
    "harmless outcome": "harmless_outcome",
}

RUNNABLE_DOMINANT_DIMENSION_SLUGS = {
    dimension: slug
    for dimension, slug in DIMENSION_SLUGS.items()
    if dimension != "harmless outcome"
}


@dataclass(frozen=True, slots=True)
class RawRubricDimensionTarget:
    label: str
    step_slug: str
    feature_hypothesis: str
    positive_condition: str
    probe_question: str
    text_baseline_question: str
    target_kind: str = "binary"


RAW_RUBRIC_DIMENSION_TARGETS: tuple[RawRubricDimensionTarget, ...] = (
    RawRubricDimensionTarget(
        label="dominant_dimension_by_count",
        step_slug="dominant_dimension_by_count",
        feature_hypothesis="raw rubric-dimension family state",
        positive_condition="multiclass label: rubric dimension with the largest raw criterion count",
        probe_question=(
            "Can activations classify which raw MoReBench rubric dimension is most represented, ignoring weights?"
        ),
        text_baseline_question=(
            "Can text alone classify the dominant raw rubric dimension, before using activations?"
        ),
        target_kind="multiclass",
    ),
    *(
        RawRubricDimensionTarget(
            label=f"dominant_{slug}_by_count",
            step_slug=f"dominant_{slug}_by_count",
            feature_hypothesis=f"{dimension} rubric-emphasis state",
            positive_condition=(
                f"the largest raw criterion-count dimension is {dimension!r}, with criterion weights ignored"
            ),
            probe_question=(
                f"Is {dimension!r} rubric emphasis decodable from the model state when weights are ignored?"
            ),
            text_baseline_question=(
                f"Can text alone predict whether {dimension!r} is the dominant raw rubric dimension?"
            ),
        )
        for dimension, slug in RUNNABLE_DOMINANT_DIMENSION_SLUGS.items()
    ),
)


def build_raw_rubric_dimension_labels(*, criteria: Any) -> dict[str, Any]:
    """Collapse criterion rows into raw dimension-count labels per dilemma."""
    dataset = criteria.resolve() if getattr(criteria, "is_deferred", False) else criteria

    grouped: dict[str, list[Any]] = defaultdict(list)
    for example in dataset.examples:
        base_id = str(example.cases["base_dilemma_id"])
        grouped[base_id].append(example)

    labels: dict[str, dict[str, Any]] = {
        "dilemma_text": {},
        "base_dilemma_id": {},
        "dominant_dimension_by_count": {},
    }
    for dimension, slug in DIMENSION_SLUGS.items():
        labels[f"dominant_{slug}_by_count"] = {}
        labels[f"{slug}_criterion_count"] = {}
        labels[f"has_{slug}_criterion"] = {}

    profiles: list[dict[str, Any]] = []
    for base_id, examples in sorted(grouped.items()):
        counts = {dimension: 0 for dimension in DIMENSIONS}
        signed_counts = {
            dimension: {"positive": 0, "negative": 0, "zero": 0}
            for dimension in DIMENSIONS
        }

        for example in examples:
            item_labels = dict(example.labels)
            dimension = _dimension(item_labels.get("rubric_dimension"))
            if dimension not in counts:
                continue
            weight = _weight(item_labels.get("criterion_weight"))
            counts[dimension] += 1
            if weight > 0:
                signed_counts[dimension]["positive"] += 1
            elif weight < 0:
                signed_counts[dimension]["negative"] += 1
            else:
                signed_counts[dimension]["zero"] += 1

        dominant = _dominant_dimension_by_count(counts)
        dilemma = str(examples[0].labels.get("DILEMMA", examples[0].prompt))
        labels["dilemma_text"][base_id] = dilemma
        labels["base_dilemma_id"][base_id] = base_id
        labels["dominant_dimension_by_count"][base_id] = dominant

        for dimension, slug in DIMENSION_SLUGS.items():
            count = int(counts[dimension])
            labels[f"dominant_{slug}_by_count"][base_id] = "yes" if dominant == dimension else "no"
            labels[f"{slug}_criterion_count"][base_id] = count
            labels[f"has_{slug}_criterion"][base_id] = "yes" if count > 0 else "no"

        profiles.append(
            {
                "base_dilemma_id": base_id,
                "criterion_count": int(sum(counts.values())),
                "dominant_dimension_by_count": dominant,
                "dimension_counts": dict(counts),
                "signed_dimension_counts": {
                    dimension: dict(values) for dimension, values in signed_counts.items()
                },
            }
        )

    target_class_counts = {
        target.label: _class_counts(labels[target.label])
        for target in RAW_RUBRIC_DIMENSION_TARGETS
    }
    return {
        "payload": {
            "kind": "morebench_raw_rubric_dimension_labels",
            "summary": _profile_summary(profiles),
            "target_class_counts": target_class_counts,
            "targets": [
                {
                    "label": target.label,
                    "target_kind": target.target_kind,
                    "feature_hypothesis": target.feature_hypothesis,
                    "positive_condition": target.positive_condition,
                    "probe_question": target.probe_question,
                    "text_baseline_question": target.text_baseline_question,
                }
                for target in RAW_RUBRIC_DIMENSION_TARGETS
            ],
            "metric_interpretation": {
                "prompt_end_probe": (
                    "Prompt-end probes ask whether the model can read the dilemma and infer which raw "
                    "rubric dimension family will matter before producing an answer."
                ),
                "generation_probe": (
                    "Generation probes replay the prompt plus generated answer after the generation step "
                    "and read hidden states at the generated-answer endpoint."
                ),
                "text_baseline": (
                    "High text-baseline metrics mean the label is visible from surface text. Prompt text "
                    "baselines use dilemma text; generation text baselines use the generated answer."
                ),
            },
        },
        "labels": labels,
        "metadata": {
            "source": "MoReBench public rubric criteria",
            "unit": "base_dilemma_id",
            "status": "raw dimension-count labels; criterion weights ignored for target construction",
        },
        "example_keys": sorted(grouped),
    }


def build_successful_generation_capture_dataset(*, generation: Any) -> dict[str, Any]:
    """Build prompt+generated capture examples from successful generation rows."""
    if not hasattr(generation, "result"):
        raise TypeError("build_successful_generation_capture_dataset expects a generation artifact")

    payload = generation.result()
    if not isinstance(payload, Mapping):
        raise TypeError("Generation artifact result must be a mapping")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise TypeError("Generation artifact result must contain a rows list")

    examples: list[Example] = []
    skipped_length: list[str] = []
    skipped_empty: list[str] = []
    generated_text: dict[str, str] = {}
    finish_reason: dict[str, str] = {}
    generated_token_count: dict[str, int] = {}
    base_dilemma_id: dict[str, str] = {}
    dilemma_text: dict[str, str] = {}

    for row in raw_rows:
        if not isinstance(row, Mapping):
            continue
        key = str(row.get("example_key") or "").strip()
        if not key:
            continue
        reason = str(row.get("finish_reason") or "")
        if reason == "length":
            skipped_length.append(key)
            continue
        text = str(row.get("generated_text") or row.get("text") or "")
        if not text.strip():
            skipped_empty.append(key)
            continue
        source_example = _mapping(row.get("example"))
        source_prompt = str(source_example.get("prompt") or "")
        if not source_prompt.strip():
            skipped_empty.append(key)
            continue
        token_ids = row.get("generated_token_ids")
        cases = {str(name): value for name, value in _mapping(source_example.get("cases")).items()}
        labels = dict(_mapping(source_example.get("labels")))
        metadata = dict(_mapping(source_example.get("metadata")))
        case_key = str(source_example.get("case_key") or cases.get("base_dilemma_id") or key)
        base_id = str(cases.get("base_dilemma_id") or labels.get("base_dilemma_id") or key)
        combined_prompt, token_sections = _combined_prompt_and_sections(
            source_prompt=source_prompt,
            generated_text=text,
        )
        labels.update(
            {
                "base_dilemma_id": base_id,
                "dilemma_text": source_prompt,
                "generated_text": text,
                "generation_finish_reason": reason,
                "generated_token_count": len(token_ids) if isinstance(token_ids, list) else 0,
            }
        )
        metadata.update(
            {
                "source_generation_artifact_id": getattr(generation, "id", ""),
                "source_prompt_hash": source_example.get("prompt_hash"),
                "generation_finish_reason": reason,
                "generated_token_count": len(token_ids) if isinstance(token_ids, list) else 0,
                "token_sections": token_sections,
            }
        )
        examples.append(
            Example(
                key=key,
                prompt=combined_prompt,
                labels=labels,
                metadata=metadata,
                cases={**cases, "base_dilemma_id": base_id},
                case_key=case_key,
            )
        )
        generated_text[key] = text
        finish_reason[key] = reason
        generated_token_count[key] = len(token_ids) if isinstance(token_ids, list) else 0
        base_dilemma_id[key] = base_id
        dilemma_text[key] = source_prompt

    dataset = Dataset.from_examples(
        examples,
        name="morebench_phase_02_successful_prompt_generated_contexts",
    )

    return {
        "payload": {
            "kind": "morebench_successful_generation_capture_dataset",
            "dataset": dataset.to_dict(),
            "summary": {
                "source_generation_artifact_id": getattr(generation, "id", ""),
                "source_row_count": len(raw_rows),
                "kept_example_count": len(examples),
                "skipped_length_count": len(skipped_length),
                "skipped_empty_count": len(skipped_empty),
                "finish_reason_counts": dict(sorted(Counter(finish_reason.values()).items())),
                "skipped_length_example_keys": sorted(skipped_length),
                "skipped_empty_example_keys": sorted(skipped_empty),
            },
        },
        "labels": {
            "base_dilemma_id": base_dilemma_id,
            "dilemma_text": dilemma_text,
            "generated_text": generated_text,
            "generation_finish_reason": finish_reason,
            "generated_token_count": generated_token_count,
        },
        "metadata": {
            "source": "GenerationRunSpec result rows",
            "unit": "base_dilemma_id",
            "status": "length-finished generations dropped before activation capture",
        },
        "example_keys": sorted(generated_text),
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _combined_prompt_and_sections(*, source_prompt: str, generated_text: str) -> tuple[str, dict[str, Any]]:
    separator = "\n\nAssistant response:\n"
    prompt_start = 0
    prompt_end = len(source_prompt)
    generated_start = prompt_end + len(separator)
    combined = f"{source_prompt}{separator}{generated_text}"
    generated_end = len(combined)
    return combined, {
        "prompt": {"char_start": prompt_start, "char_end": prompt_end},
        "generated": {"char_start": generated_start, "char_end": generated_end},
        "full": {"char_start": 0, "char_end": generated_end},
        "prompt_end": _last_non_whitespace_span(combined, prompt_start, prompt_end),
        "generated_end": _last_non_whitespace_span(combined, generated_start, generated_end),
        "full_end": _last_non_whitespace_span(combined, 0, generated_end),
    }


def _last_non_whitespace_span(text: str, start: int, end: int) -> dict[str, int]:
    index = int(end) - 1
    while index >= int(start) and text[index].isspace():
        index -= 1
    if index < int(start):
        index = max(int(start), int(end) - 1)
    return {"char_start": index, "char_end": index + 1}


def _dimension(value: Any) -> str:
    return str(value or "").strip().lower()


def _weight(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _dominant_dimension_by_count(counts: Mapping[str, int]) -> str:
    order = {dimension: index for index, dimension in enumerate(DIMENSIONS)}
    return max(DIMENSIONS, key=lambda dimension: (int(counts.get(dimension, 0)), -order[dimension]))


def _class_counts(values: Mapping[str, Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values.values()).items()))


def _profile_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "profile_count": 0,
            "dominant_dimension_counts": {},
            "mean_criterion_count": None,
            "mean_dimension_counts": {},
        }
    dominant_counts = Counter(str(row["dominant_dimension_by_count"]) for row in rows)
    mean_dimension_counts = {
        dimension: round(
            sum(int(_mapping(row.get("dimension_counts")).get(dimension, 0)) for row in rows)
            / max(len(rows), 1),
            4,
        )
        for dimension in DIMENSIONS
    }
    return {
        "profile_count": len(rows),
        "dominant_dimension_counts": dict(sorted(dominant_counts.items())),
        "mean_criterion_count": round(
            sum(int(row["criterion_count"]) for row in rows) / max(len(rows), 1),
            4,
        ),
        "mean_dimension_counts": mean_dimension_counts,
    }
