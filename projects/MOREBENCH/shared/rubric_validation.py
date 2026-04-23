"""Rubric-set validation helpers for MoReBench phase 1."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any


DIMENSIONS = (
    "identifying",
    "clear process",
    "logical process",
    "helpful outcome",
    "harmless outcome",
)


def build_rubric_profile_labels(*, criteria: Any) -> dict[str, Any]:
    """Collapse criterion-level rubric rows into one validation profile per dilemma."""
    dataset = criteria.resolve() if getattr(criteria, "is_deferred", False) else criteria

    grouped: dict[str, list[Any]] = defaultdict(list)
    for example in dataset.examples:
        base_id = str(example.cases["base_dilemma_id"])
        grouped[base_id].append(example)

    dilemma_text: dict[str, str] = {}
    base_dilemma_id: dict[str, str] = {}
    dominant_dimension: dict[str, str] = {}
    harmless_penalty: dict[str, str] = {}
    helpful_harmless_tension: dict[str, str] = {}
    rubric_complexity: dict[str, str] = {}
    high_weight_focus: dict[str, str] = {}
    profile_rows: list[dict[str, Any]] = []

    for base_id, examples in sorted(grouped.items()):
        dimension_counts = {dimension: 0 for dimension in DIMENSIONS}
        dimension_abs_mass = {dimension: 0.0 for dimension in DIMENSIONS}
        dimension_positive_mass = {dimension: 0.0 for dimension in DIMENSIONS}
        dimension_negative_mass = {dimension: 0.0 for dimension in DIMENSIONS}
        high_weight_count = 0
        total_abs_mass = 0.0

        for example in examples:
            labels = dict(example.labels)
            dimension = _dimension(labels.get("rubric_dimension"))
            weight = _weight(labels.get("criterion_weight"))
            if dimension not in dimension_counts:
                continue
            abs_weight = abs(weight)
            dimension_counts[dimension] += 1
            dimension_abs_mass[dimension] += abs_weight
            total_abs_mass += abs_weight
            if weight > 0:
                dimension_positive_mass[dimension] += weight
            elif weight < 0:
                dimension_negative_mass[dimension] += abs_weight
            if abs_weight >= 3:
                high_weight_count += 1

        dominant = max(DIMENSIONS, key=lambda item: (dimension_abs_mass[item], dimension_counts[item], item))
        criterion_count = len(examples)
        key = base_id
        dilemma = str(examples[0].labels.get("DILEMMA", examples[0].prompt))

        dilemma_text[key] = dilemma
        base_dilemma_id[key] = base_id
        dominant_dimension[key] = dominant
        harmless_penalty[key] = "yes" if dimension_negative_mass["harmless outcome"] > 0 else "no"
        helpful_harmless_tension[key] = (
            "yes"
            if dimension_positive_mass["helpful outcome"] > 0 and dimension_negative_mass["harmless outcome"] > 0
            else "no"
        )
        rubric_complexity[key] = _complexity_bin(criterion_count)
        high_weight_focus[key] = _count_bin(high_weight_count)
        profile_rows.append(
            {
                "base_dilemma_id": base_id,
                "criterion_count": criterion_count,
                "total_abs_weight": round(total_abs_mass, 4),
                "dominant_dimension": dominant,
                "has_harmless_penalty": harmless_penalty[key],
                "has_helpful_harmless_tension": helpful_harmless_tension[key],
                "rubric_complexity_bin": rubric_complexity[key],
                "high_weight_count_bin": high_weight_focus[key],
                "dimension_counts": dict(dimension_counts),
                "dimension_abs_mass": {name: round(value, 4) for name, value in dimension_abs_mass.items()},
                "dimension_positive_mass": {
                    name: round(value, 4) for name, value in dimension_positive_mass.items()
                },
                "dimension_negative_mass": {
                    name: round(value, 4) for name, value in dimension_negative_mass.items()
                },
            }
        )

    return {
        "payload": {
            "kind": "morebench_rubric_profile_result",
            "profile_count": len(profile_rows),
            "profiles": profile_rows,
            "summary": _profile_summary(profile_rows),
        },
        "labels": {
            "dilemma_text": dilemma_text,
            "base_dilemma_id": base_dilemma_id,
            "dominant_dimension": dominant_dimension,
            "has_harmless_penalty": harmless_penalty,
            "has_helpful_harmless_tension": helpful_harmless_tension,
            "rubric_complexity_bin": rubric_complexity,
            "high_weight_count_bin": high_weight_focus,
        },
        "metadata": {
            "source": "MoReBench RUBRIC profile validation",
            "unit": "base_dilemma_id",
        },
        "example_keys": sorted(grouped),
    }


def _dimension(value: Any) -> str:
    return str(value or "").strip().lower()


def _weight(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _complexity_bin(count: int) -> str:
    if count <= 18:
        return "low"
    if count <= 26:
        return "medium"
    return "high"


def _count_bin(count: int) -> str:
    if count <= 2:
        return "low"
    if count <= 5:
        return "medium"
    return "high"


def _profile_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "profile_count": 0,
            "mean_criterion_count": None,
            "dominant_dimension_counts": {},
            "harmless_penalty_count": 0,
            "helpful_harmless_tension_count": 0,
        }
    dominant_counts: dict[str, int] = defaultdict(int)
    harmless_penalty_count = 0
    tension_count = 0
    for row in rows:
        dominant_counts[str(row["dominant_dimension"])] += 1
        harmless_penalty_count += 1 if row["has_harmless_penalty"] == "yes" else 0
        tension_count += 1 if row["has_helpful_harmless_tension"] == "yes" else 0
    return {
        "profile_count": len(rows),
        "mean_criterion_count": round(
            sum(int(row["criterion_count"]) for row in rows) / max(len(rows), 1),
            4,
        ),
        "dominant_dimension_counts": dict(sorted(dominant_counts.items())),
        "harmless_penalty_count": harmless_penalty_count,
        "helpful_harmless_tension_count": tension_count,
    }
