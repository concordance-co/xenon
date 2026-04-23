"""Rubric-derived signal labels for MoReBench prompt activation probes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from projects.MOREBENCH.shared.rubric_validation import build_rubric_profile_labels


HIGH_LOGICAL_PROCESS_MASS = 16.0
HIGH_HELPFUL_OUTCOME_MASS = 10.0
HIGH_HARMLESS_PENALTY_MASS = 6.0
HIGH_TOTAL_RUBRIC_MASS = 57.0


@dataclass(frozen=True, slots=True)
class SignalProbeTarget:
    label: str
    step_slug: str
    feature_hypothesis: str
    positive_condition: str
    probe_question: str
    text_baseline_question: str


SIGNAL_PROBE_TARGETS: tuple[SignalProbeTarget, ...] = (
    SignalProbeTarget(
        label="has_helpful_harmless_tension",
        step_slug="helpful_harmless_tension",
        feature_hypothesis="helpful-vs-harmless tradeoff state",
        positive_condition="rubric has positive helpful-outcome mass and negative harmless-outcome mass",
        probe_question=(
            "Is the helpful-vs-harmless tradeoff linearly decodable from the final prompt-token residual stream?"
        ),
        text_baseline_question="How much of this tradeoff is predictable from dilemma wording alone?",
    ),
    SignalProbeTarget(
        label="has_harmless_penalty",
        step_slug="harmless_penalty",
        feature_hypothesis="harm-avoidance / penalty-awareness state",
        positive_condition="rubric contains at least one negative harmless-outcome criterion",
        probe_question=(
            "Is the presence of harmlessness penalties linearly decodable before the model generates an answer?"
        ),
        text_baseline_question="How much does raw dilemma text reveal whether harmless penalties are present?",
    ),
    SignalProbeTarget(
        label="high_harmless_penalty_burden",
        step_slug="high_harmless_penalty_burden",
        feature_hypothesis="high safety-penalty burden state",
        positive_condition=f"negative harmless-outcome absolute mass is at least {HIGH_HARMLESS_PENALTY_MASS:g}",
        probe_question="Can residual activations identify dilemmas with unusually heavy harmful-outcome penalties?",
        text_baseline_question="Can bag-of-words text identify high harmless-penalty burden by itself?",
    ),
    SignalProbeTarget(
        label="high_helpful_outcome_demand",
        step_slug="high_helpful_outcome_demand",
        feature_hypothesis="helpful-resolution demand state",
        positive_condition=f"positive helpful-outcome mass is at least {HIGH_HELPFUL_OUTCOME_MASS:g}",
        probe_question="Can residual activations identify dilemmas whose rubrics strongly require a helpful conclusion?",
        text_baseline_question="Can raw dilemma wording identify high helpful-outcome demand?",
    ),
    SignalProbeTarget(
        label="high_logical_process_demand",
        step_slug="high_logical_process_demand",
        feature_hypothesis="procedural/logical reasoning demand state",
        positive_condition=f"logical-process absolute mass is at least {HIGH_LOGICAL_PROCESS_MASS:g}",
        probe_question="Can residual activations identify dilemmas whose rubrics heavily reward logical process?",
        text_baseline_question="Can raw dilemma wording identify high logical-process demand?",
    ),
    SignalProbeTarget(
        label="high_rubric_weight_burden",
        step_slug="high_rubric_weight_burden",
        feature_hypothesis="overall rubric burden / evaluation complexity state",
        positive_condition=f"total absolute rubric weight is at least {HIGH_TOTAL_RUBRIC_MASS:g}",
        probe_question="Can residual activations identify dilemmas with unusually high total weighted rubric burden?",
        text_baseline_question="Can bag-of-words text identify high total rubric burden?",
    ),
)


def build_rubric_signal_probe_labels(*, criteria: Any) -> dict[str, Any]:
    """Collapse rubric criteria into probe-ready dilemma-level signal labels."""
    profile_result = build_rubric_profile_labels(criteria=criteria)
    profiles = list(profile_result["payload"]["profiles"])
    base_labels = dict(profile_result["labels"])

    high_logical_process_demand: dict[str, str] = {}
    high_helpful_outcome_demand: dict[str, str] = {}
    high_harmless_penalty_burden: dict[str, str] = {}
    high_rubric_weight_burden: dict[str, str] = {}
    logical_process_mass_bin: dict[str, str] = {}
    helpful_outcome_mass_bin: dict[str, str] = {}
    harmless_penalty_mass_bin: dict[str, str] = {}
    rubric_weight_burden_bin: dict[str, str] = {}

    for row in profiles:
        base_id = str(row["base_dilemma_id"])
        dimension_abs_mass = _mapping(row.get("dimension_abs_mass"))
        dimension_positive_mass = _mapping(row.get("dimension_positive_mass"))
        dimension_negative_mass = _mapping(row.get("dimension_negative_mass"))

        logical_mass = _number(dimension_abs_mass.get("logical process"))
        helpful_mass = _number(dimension_positive_mass.get("helpful outcome"))
        harmless_penalty_mass = _number(dimension_negative_mass.get("harmless outcome"))
        total_mass = _number(row.get("total_abs_weight"))

        high_logical_process_demand[base_id] = _yes_no(logical_mass >= HIGH_LOGICAL_PROCESS_MASS)
        high_helpful_outcome_demand[base_id] = _yes_no(helpful_mass >= HIGH_HELPFUL_OUTCOME_MASS)
        high_harmless_penalty_burden[base_id] = _yes_no(harmless_penalty_mass >= HIGH_HARMLESS_PENALTY_MASS)
        high_rubric_weight_burden[base_id] = _yes_no(total_mass >= HIGH_TOTAL_RUBRIC_MASS)
        logical_process_mass_bin[base_id] = _mass_bin(logical_mass, low=8.0, high=16.0)
        helpful_outcome_mass_bin[base_id] = _mass_bin(helpful_mass, low=5.0, high=10.0)
        harmless_penalty_mass_bin[base_id] = _none_low_high(harmless_penalty_mass, high=6.0)
        rubric_weight_burden_bin[base_id] = _mass_bin(total_mass, low=46.0, high=57.0)

    labels = {
        **base_labels,
        "high_logical_process_demand": high_logical_process_demand,
        "high_helpful_outcome_demand": high_helpful_outcome_demand,
        "high_harmless_penalty_burden": high_harmless_penalty_burden,
        "high_rubric_weight_burden": high_rubric_weight_burden,
        "logical_process_mass_bin": logical_process_mass_bin,
        "helpful_outcome_mass_bin": helpful_outcome_mass_bin,
        "harmless_penalty_mass_bin": harmless_penalty_mass_bin,
        "rubric_weight_burden_bin": rubric_weight_burden_bin,
    }

    target_class_counts = {
        target.label: _class_counts(labels[target.label])
        for target in SIGNAL_PROBE_TARGETS
    }
    return {
        "payload": {
            "kind": "morebench_rubric_signal_probe_labels",
            "profile_summary": profile_result["payload"]["summary"],
            "thresholds": {
                "high_logical_process_demand": HIGH_LOGICAL_PROCESS_MASS,
                "high_helpful_outcome_demand": HIGH_HELPFUL_OUTCOME_MASS,
                "high_harmless_penalty_burden": HIGH_HARMLESS_PENALTY_MASS,
                "high_rubric_weight_burden": HIGH_TOTAL_RUBRIC_MASS,
            },
            "target_class_counts": target_class_counts,
            "targets": [
                {
                    "label": target.label,
                    "feature_hypothesis": target.feature_hypothesis,
                    "positive_condition": target.positive_condition,
                    "probe_question": target.probe_question,
                    "text_baseline_question": target.text_baseline_question,
                }
                for target in SIGNAL_PROBE_TARGETS
            ],
            "metric_interpretation": {
                "residual_probe": (
                    "Balanced accuracy or AUROC near 0.5 means the target is not reliably decodable. "
                    "Values above the text baseline and shuffled-control/selectivity checks mean the "
                    "prompt residual stream carries rubric-signal information beyond a cheap lexical baseline."
                ),
                "text_baseline": (
                    "High text-baseline balanced accuracy or AUROC means the target is mostly visible in "
                    "dilemma wording; low text baseline with stronger residual probes is the cleaner signal-probe result."
                ),
                "direction": (
                    "A direction artifact is a mean-difference readout for later steering or patching. "
                    "It is not causal evidence until an intervention changes behavior in the predicted direction."
                ),
            },
        },
        "labels": labels,
        "metadata": {
            "source": "MoReBench public rubric criteria",
            "unit": "base_dilemma_id",
            "status": "rubric-derived prompt-probe targets; no response judging required",
        },
        "example_keys": list(profile_result["example_keys"]),
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _mass_bin(value: float, *, low: float, high: float) -> str:
    if value >= high:
        return "high"
    if value >= low:
        return "medium"
    return "low"


def _none_low_high(value: float, *, high: float) -> str:
    if value <= 0:
        return "none"
    if value >= high:
        return "high"
    return "low"


def _class_counts(values: Mapping[str, Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values.values()).items()))
