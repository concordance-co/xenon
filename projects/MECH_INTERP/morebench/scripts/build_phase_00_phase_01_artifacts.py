from __future__ import annotations

import ast
import csv
import io
import json
import re
import statistics
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE_00_ROOT = ROOT / "phase_00"
PHASE_01_ROOT = ROOT / "phase_01"
CANONICAL_ROOT = ROOT.parents[2] / "docs" / "mech-interp" / "benchmarks" / "morebench"

DATASET_URLS = {
    "morebench_public": "https://huggingface.co/datasets/morebench/morebench/resolve/main/morebench_public.csv",
    "morebench_theory": "https://huggingface.co/datasets/morebench/morebench/resolve/main/morebench_theory.csv",
}

LEGACY_OUTPUTS = [
    PHASE_00_ROOT / "outputs" / "benchmark_snapshot.json",
    PHASE_00_ROOT / "outputs" / "confound_audit.json",
    PHASE_01_ROOT / "outputs" / "mechanistic_questions.json",
    PHASE_01_ROOT / "outputs" / "latent_label_spec.json",
    PHASE_01_ROOT / "outputs" / "first_pass_ontology.json",
]


def fetch_rows(url: str) -> list[dict[str, str]]:
    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def parse_rubric_items(row: dict[str, str]) -> list[dict[str, object]]:
    return ast.literal_eval(row["RUBRIC"])


def median_or_zero(values: list[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def count_by(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    counts = Counter(row[field] for row in rows)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def pair_counts(rows: list[dict[str, str]], left: str, right: str) -> dict[str, int]:
    counts = Counter((row[left], row[right]) for row in rows)
    return {
        f"{left}={left_value} | {right}={right_value}": count
        for (left_value, right_value), count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    }


def top_contexts_by_source(rows: list[dict[str, str]], top_n: int = 6) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for source in sorted({row["DILEMMA_SOURCE"] for row in rows}):
        counter = Counter(row["CONTEXT"] for row in rows if row["DILEMMA_SOURCE"] == source)
        result[source] = [{"context": context, "count": count} for context, count in counter.most_common(top_n)]
    return result


def lexical_prefixes(rows: list[dict[str, str]], dimension: str, top_n: int = 12) -> list[dict[str, object]]:
    counter: Counter[str] = Counter()
    for row in rows:
        for item in parse_rubric_items(row):
            if item.get("annotations", {}).get("rubric_dimension") != dimension:
                continue
            tokens = re.findall(r"[A-Za-z']+", str(item.get("title", "")).lower())[:4]
            counter[" ".join(tokens)] += 1
    return [{"prefix": prefix, "count": count} for prefix, count in counter.most_common(top_n)]


def sample_titles(rows: list[dict[str, str]], dimension: str, sample_n: int = 4) -> list[str]:
    titles: list[str] = []
    for row in rows:
        for item in parse_rubric_items(row):
            if item.get("annotations", {}).get("rubric_dimension") != dimension:
                continue
            title = str(item.get("title", "")).replace("\n", " ").strip()
            if title not in titles:
                titles.append(title)
            if len(titles) >= sample_n:
                return titles
    return titles


def frontmatter(phase: str, input_artifacts: list[str]) -> str:
    lines = [
        "---",
        "benchmark: morebench",
        f"phase: {phase}",
        "version: v1",
        "frozen_date: 2026-04-22",
        "input_artifacts:",
    ]
    lines.extend([f"  - {artifact}" for artifact in input_artifacts])
    lines.append("---")
    return "\n".join(lines)


def analyze_config(name: str, rows: list[dict[str, str]]) -> dict[str, object]:
    dilemma_lengths = [len(row["DILEMMA"]) for row in rows]
    rubric_items_per_row: list[int] = []
    rubric_dimension_counts: Counter[str] = Counter()
    weight_counts: Counter[int] = Counter()
    dimension_weight_counts: Counter[tuple[str, int]] = Counter()

    for row in rows:
        items = parse_rubric_items(row)
        rubric_items_per_row.append(len(items))
        for item in items:
            dimension = str(item.get("annotations", {}).get("rubric_dimension", "<missing>"))
            weight = int(item.get("weight", 0))
            rubric_dimension_counts[dimension] += 1
            weight_counts[weight] += 1
            dimension_weight_counts[(dimension, weight)] += 1

    return {
        "config_name": name,
        "row_count": len(rows),
        "fields": list(rows[0].keys()),
        "value_distributions": {
            "DILEMMA_SOURCE": count_by(rows, "DILEMMA_SOURCE"),
            "DILEMMA_TYPE": count_by(rows, "DILEMMA_TYPE"),
            "THEORY": count_by(rows, "THEORY"),
            "ROLE_DOMAIN": count_by(rows, "ROLE_DOMAIN"),
            "CONTEXT": count_by(rows, "CONTEXT"),
        },
        "pair_distributions": {
            "source_x_role": pair_counts(rows, "DILEMMA_SOURCE", "ROLE_DOMAIN"),
            "source_x_type": pair_counts(rows, "DILEMMA_SOURCE", "DILEMMA_TYPE"),
            "role_x_type": pair_counts(rows, "ROLE_DOMAIN", "DILEMMA_TYPE"),
        },
        "dilemma_char_length": {
            "min": min(dilemma_lengths),
            "median": median_or_zero(dilemma_lengths),
            "max": max(dilemma_lengths),
        },
        "rubric_items_per_row": {
            "min": min(rubric_items_per_row),
            "median": median_or_zero(rubric_items_per_row),
            "max": max(rubric_items_per_row),
        },
        "rubric_dimension_counts": dict(
            sorted(rubric_dimension_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "weight_counts": dict(sorted(weight_counts.items(), key=lambda item: (-item[1], item[0]))),
        "dimension_weight_counts_top": [
            {"rubric_dimension": dimension, "weight": weight, "count": count}
            for (dimension, weight), count in dimension_weight_counts.most_common(16)
        ],
        "sample_rubric_titles": {
            dimension: sample_titles(rows, dimension)
            for dimension in [
                "identifying",
                "clear process",
                "logical process",
                "helpful outcome",
                "harmless outcome",
                "other",
            ]
        },
        "lexical_prefixes": {
            dimension: lexical_prefixes(rows, dimension)
            for dimension in [
                "identifying",
                "clear process",
                "logical process",
                "helpful outcome",
                "harmless outcome",
            ]
        },
    }


def build_native_label_inventory() -> list[dict[str, object]]:
    return [
        {
            "label_name": "DILEMMA",
            "where_it_lives": "prompt field",
            "designed_to_do": "present the case and competing action alternatives",
            "assignment_method": "benchmark authoring",
            "granularity": "example-level",
            "value_type": "free text",
            "label_type": "prompt-side structure",
            "mech_interp_match": "derived",
            "recommended_use": "derive benchmark-specific prompt-side labels",
        },
        {
            "label_name": "DILEMMA_SOURCE",
            "where_it_lives": "metadata",
            "designed_to_do": "track source/template family",
            "assignment_method": "benchmark metadata",
            "granularity": "example-level",
            "value_type": "multiclass",
            "label_type": "metadata / nuisance variable",
            "mech_interp_match": "nuisance-only",
            "recommended_use": "stratify, hold out, or augment against source-family leakage",
        },
        {
            "label_name": "DILEMMA_TYPE",
            "where_it_lives": "metadata",
            "designed_to_do": "track case format family",
            "assignment_method": "benchmark metadata",
            "granularity": "example-level",
            "value_type": "multiclass",
            "label_type": "metadata / nuisance variable",
            "mech_interp_match": "nuisance-only",
            "recommended_use": "track length and format imbalance",
        },
        {
            "label_name": "THEORY",
            "where_it_lives": "metadata",
            "designed_to_do": "track neutral versus theory-conditioned rubric overlays",
            "assignment_method": "benchmark metadata",
            "granularity": "example-level",
            "value_type": "multiclass",
            "label_type": "outcome / rubric context",
            "mech_interp_match": "derived",
            "recommended_use": "defer prompt-side claims until theory exposure is confirmed or augmented cleanly",
        },
        {
            "label_name": "ROLE_DOMAIN",
            "where_it_lives": "metadata",
            "designed_to_do": "distinguish advisor-like from agent-like deployment framing",
            "assignment_method": "benchmark metadata",
            "granularity": "example-level",
            "value_type": "multiclass",
            "label_type": "prompt-side structure",
            "mech_interp_match": "derived",
            "recommended_use": "treat as a hypothesis seed, not a current clean probe target on the public split",
        },
        {
            "label_name": "CONTEXT",
            "where_it_lives": "metadata",
            "designed_to_do": "track broad scenario domain",
            "assignment_method": "benchmark metadata",
            "granularity": "example-level",
            "value_type": "multiclass",
            "label_type": "metadata / nuisance variable",
            "mech_interp_match": "nuisance-only",
            "recommended_use": "track domain/topic shortcuts",
        },
        {
            "label_name": "RUBRIC.identifying",
            "where_it_lives": "criterion list inside RUBRIC",
            "designed_to_do": "reward dilemma recognition and coverage of live considerations",
            "assignment_method": "expert-authored criterion",
            "granularity": "criterion-level",
            "value_type": "weighted signed criterion",
            "label_type": "outcome / rubric score",
            "mech_interp_match": "validation-only",
            "recommended_use": "use for validation of coverage, not direct probing",
        },
        {
            "label_name": "RUBRIC.clear process",
            "where_it_lives": "criterion list inside RUBRIC",
            "designed_to_do": "reward clear structured reasoning presentation",
            "assignment_method": "expert-authored criterion",
            "granularity": "criterion-level",
            "value_type": "weighted signed criterion",
            "label_type": "outcome / rubric score",
            "mech_interp_match": "validation-only",
            "recommended_use": "use for response-structure validation only",
        },
        {
            "label_name": "RUBRIC.logical process",
            "where_it_lives": "criterion list inside RUBRIC",
            "designed_to_do": "reward coherent argument and consequence chaining",
            "assignment_method": "expert-authored criterion",
            "granularity": "criterion-level",
            "value_type": "weighted signed criterion",
            "label_type": "outcome / rubric score",
            "mech_interp_match": "validation-only",
            "recommended_use": "use as evaluation scaffold rather than direct latent label",
        },
        {
            "label_name": "RUBRIC.helpful outcome",
            "where_it_lives": "criterion list inside RUBRIC",
            "designed_to_do": "reward actionable and supported recommendation quality",
            "assignment_method": "expert-authored criterion",
            "granularity": "criterion-level",
            "value_type": "weighted signed criterion",
            "label_type": "outcome / rubric score",
            "mech_interp_match": "validation-only",
            "recommended_use": "evaluate helpfulness-oriented policy without collapsing it into safety",
        },
        {
            "label_name": "RUBRIC.harmless outcome",
            "where_it_lives": "criterion list inside RUBRIC",
            "designed_to_do": "penalize harmful or reckless recommendation failures",
            "assignment_method": "expert-authored criterion",
            "granularity": "criterion-level",
            "value_type": "weighted signed criterion",
            "label_type": "outcome / rubric score",
            "mech_interp_match": "validation-only",
            "recommended_use": "evaluate harm avoidance as a separate axis from helpfulness",
        },
    ]


def build_theory_pairing_audit(theory_rows: list[dict[str, str]], public_rows: list[dict[str, str]]) -> dict[str, object]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in theory_rows:
        groups[normalize_text(row["DILEMMA"])].append(row)

    group_sizes = Counter(len(group) for group in groups.values())
    public_dilemmas = {normalize_text(row["DILEMMA"]) for row in public_rows}
    overlap_count = sum(1 for dilemma in groups if dilemma in public_dilemmas)

    return {
        "unique_theory_dilemmas": len(groups),
        "group_size_distribution": dict(sorted(group_sizes.items())),
        "all_groups_are_five_way_theory_sets": all(size == 5 for size in group_sizes),
        "exact_dilemma_overlap_with_public": {
            "overlap_count": overlap_count,
            "theory_unique_count": len(groups),
        },
        "judgment": {
            "status": "caution_now_but_good_augmentation_candidate",
            "reason": (
                "The theory split is cleanly paired at the evaluator level, but the repeated rows share "
                "the same DILEMMA text. Treat theory as a high-priority augmentation target rather than "
                "a current prompt-side variable unless the runtime prompt injects THEORY."
            ),
        },
        "augmentation_priority": {
            "priority": "high",
            "reason": (
                "Creating matched prompt sets that expose theory cleanly is likely one of the cheapest "
                "and highest-value augmentation steps available."
            ),
        },
    }


def build_action_locus_probeability_audit(public_rows: list[dict[str, str]]) -> dict[str, object]:
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_source_type: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    by_source_type_context: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)

    for row in public_rows:
        role = row["ROLE_DOMAIN"]
        by_source[row["DILEMMA_SOURCE"]][role] += 1
        by_source_type[(row["DILEMMA_SOURCE"], row["DILEMMA_TYPE"])][role] += 1
        by_source_type_context[(row["DILEMMA_SOURCE"], row["DILEMMA_TYPE"], row["CONTEXT"])][role] += 1

    source_cells = [
        {"source": source, "role_counts": dict(counter), "has_both_roles": len(counter) > 1}
        for source, counter in sorted(by_source.items())
    ]
    source_type_cells = [
        {
            "source": source,
            "dilemma_type": dilemma_type,
            "role_counts": dict(counter),
            "has_both_roles": len(counter) > 1,
        }
        for (source, dilemma_type), counter in sorted(by_source_type.items())
    ]
    source_type_context_cells = [
        {
            "source": source,
            "dilemma_type": dilemma_type,
            "context": context,
            "role_counts": dict(counter),
            "has_both_roles": len(counter) > 1,
        }
        for (source, dilemma_type, context), counter in sorted(by_source_type_context.items())
    ]

    return {
        "target_label": "action_locus",
        "scope": "morebench_public",
        "summary": {
            "total_rows": len(public_rows),
            "source_cells_with_both_roles": sum(1 for cell in source_cells if cell["has_both_roles"]),
            "source_type_cells_with_both_roles": sum(1 for cell in source_type_cells if cell["has_both_roles"]),
            "source_type_context_cells_with_both_roles": sum(
                1 for cell in source_type_context_cells if cell["has_both_roles"]
            ),
        },
        "source_role_cells": source_cells,
        "source_type_role_cells": source_type_cells,
        "source_type_context_role_cells": source_type_context_cells,
        "judgment": {
            "status": "not_probeable_without_augmentation",
            "reason": (
                "Post-stratification usable N is effectively nonexistent. Under source control there are zero "
                "cells containing both advisor and agent examples, and that remains true after adding type or "
                "context controls."
            ),
        },
        "recommended_fix": [
            "Create matched advisor/agent rewrites within the same source family.",
            "Use augmentation rather than regularization to repair the design.",
            "Do not treat current public ROLE_DOMAIN as a standalone probe target.",
        ],
    }


def build_prompt_side_labels() -> list[dict[str, object]]:
    return [
        {
            "label": "action_locus",
            "status": "deferred_pending_augmentation",
            "source_fields": ["ROLE_DOMAIN", "DILEMMA"],
            "definition": "Whether the system is framed as advising a user or acting as the responsible decider.",
            "signal_location": "prompt-side representation",
            "why_it_matters": "Directly tied to benchmark-native advisor versus agent framing.",
            "readiness": "not_probeable_on_current_public_split",
            "labeling_function": "direct metadata readout with augmentation-gated use",
        },
        {
            "label": "stakeholder_tradeoff_density",
            "status": "derived_from_prompt",
            "source_fields": ["DILEMMA"],
            "definition": "How many stakeholder or consequence clusters are simultaneously live in the scenario.",
            "signal_location": "prompt-side representation",
            "why_it_matters": "Most direct prompt-side expression of the benchmark's multi-consideration design.",
            "readiness": "candidate_first_pass_label",
            "labeling_function": "human or LLM-assisted count on a validated gold slice",
        },
        {
            "label": "dilemma_structure",
            "status": "derived_from_prompt_plus_metadata",
            "source_fields": ["DILEMMA", "DILEMMA_TYPE"],
            "definition": "Case-format structure such as long-case, short-case, or expert-case presentation.",
            "signal_location": "prompt-side representation",
            "why_it_matters": "Useful auxiliary variable for prompt parsing differences.",
            "readiness": "nuisance_or_auxiliary",
            "labeling_function": "direct metadata readout",
        },
        {
            "label": "domain_topic",
            "status": "direct_from_metadata",
            "source_fields": ["CONTEXT"],
            "definition": "Broad scenario domain such as healthcare, interpersonal, or science and technology.",
            "signal_location": "prompt-side representation",
            "why_it_matters": "Necessary nuisance control for topical shortcuts.",
            "readiness": "nuisance_primary",
            "labeling_function": "direct metadata readout",
        },
        {
            "label": "theory_identity",
            "status": "deferred_but_prioritized_for_augmentation",
            "source_fields": ["THEORY"],
            "definition": "Which explicit ethical framework is surfaced to the model.",
            "signal_location": "prompt-side representation and later justification policy",
            "why_it_matters": "Natural route to clean prompt-side control comparisons if augmented correctly.",
            "readiness": "blocked_on_prompt_exposure_or_matched_theory_rewrites",
            "labeling_function": "direct metadata readout only after prompt exposure is explicit",
        },
    ]


def build_response_side_labels() -> list[dict[str, object]]:
    return [
        {
            "label": "tradeoff_engagement",
            "status": "derived_from_new_generations",
            "source_fields": ["generated_response"],
            "definition": "Whether the response keeps multiple live considerations active before converging.",
            "signal_location": "generation-time deliberation",
            "readiness": "high_after_generation",
            "labeling_function": "human or LLM rubric-aligned annotation on fresh generations",
        },
        {
            "label": "commitment_style",
            "status": "derived_from_new_generations",
            "source_fields": ["generated_response"],
            "definition": "Whether the model defers, recommends, refuses, or commits directly after deliberation.",
            "signal_location": "late generation and final readout",
            "readiness": "high_after_generation",
            "labeling_function": "rule-based or annotation-based conclusion classification",
        },
        {
            "label": "refuses_or_hedges",
            "status": "derived_from_new_generations",
            "source_fields": ["generated_response"],
            "definition": "Whether the response declines commitment or leans on heavy hedging rather than recommending.",
            "signal_location": "late generation and final readout",
            "readiness": "high_after_generation",
            "labeling_function": "annotation over generated conclusions",
        },
        {
            "label": "helpfulness_invoked",
            "status": "derived_from_new_generations_with_rubric_validation",
            "source_fields": ["generated_response", "RUBRIC"],
            "definition": "Whether the response explicitly optimizes for actionable assistance and practical usefulness.",
            "signal_location": "generation-time objective orientation and late readout",
            "readiness": "high_after_generation",
            "labeling_function": "annotation validated against helpful outcome criteria",
        },
        {
            "label": "harm_avoidance_invoked",
            "status": "derived_from_new_generations_with_rubric_validation",
            "source_fields": ["generated_response", "RUBRIC"],
            "definition": "Whether the response explicitly optimizes for avoiding harm, recklessness, or unsafe overreach.",
            "signal_location": "generation-time objective orientation and late readout",
            "readiness": "high_after_generation",
            "labeling_function": "annotation validated against harmless outcome criteria",
        },
        {
            "label": "uncertainty_and_scope_calibration",
            "status": "derived_from_new_generations",
            "source_fields": ["generated_response"],
            "definition": "Whether the response marks uncertainty, limits, or role-appropriate scope boundaries.",
            "signal_location": "generation-time deliberation and conclusion framing",
            "readiness": "medium_after_generation",
            "labeling_function": "annotation over uncertainty and scope markers in fresh generations",
        },
    ]


def build_validation_signals() -> list[dict[str, str]]:
    return [
        {"signal": "identifying", "use": "validate dilemma recognition and coverage of live considerations"},
        {"signal": "clear process", "use": "validate structured reasoning presentation"},
        {"signal": "logical process", "use": "validate argument and consequence chaining quality"},
        {"signal": "helpful outcome", "use": "validate helpfulness-oriented response quality"},
        {"signal": "harmless outcome", "use": "validate harm avoidance and reckless recommendation avoidance"},
    ]


def build_nuisance_variables() -> list[dict[str, str]]:
    return [
        {"variable": "DILEMMA_SOURCE", "reason": "strong source/template aliasing with role, style, and domain"},
        {"variable": "DILEMMA_TYPE", "reason": "co-moves with source and changes prompt structure"},
        {"variable": "prompt length", "reason": "substantial variation in dilemma length across case families"},
        {"variable": "CONTEXT", "reason": "topic/domain concentration differs sharply by source"},
        {"variable": "lexical template family", "reason": "prompt wording may leak source identity"},
        {"variable": "theory metadata when not exposed to the prompt", "reason": "evaluator-side field, not automatically model-side input"},
    ]


def build_candidate_mechanistic_questions() -> list[dict[str, object]]:
    return [
        {
            "question_id": "mq_001_multi_consideration_representation",
            "mechanistic_question": "Does the model keep multiple live considerations active before recommending?",
            "signal_location": "prompt-side representation and generation-time deliberation",
            "readiness": "high",
            "benchmark_basis": "criterion-dense emphasis on dilemma coverage before conclusion",
        },
        {
            "question_id": "mq_002_commitment_transition",
            "mechanistic_question": "When does the model shift from exploration to concrete recommendation?",
            "signal_location": "generation-time deliberation and late commitment state",
            "readiness": "high",
            "benchmark_basis": "process criteria and helpful-outcome criteria pull apart deliberation and conclusion",
        },
        {
            "question_id": "mq_003_helpfulness_harm_avoidance_separability",
            "mechanistic_question": "Are helpfulness-oriented and harm-avoidance-oriented objectives separable in the response policy?",
            "signal_location": "generation-time objective orientation and late readout",
            "readiness": "high",
            "benchmark_basis": "distinct helpful outcome and harmless outcome rubric families",
        },
        {
            "question_id": "mq_004_action_locus_control_state",
            "mechanistic_question": "Does advisor versus agent framing induce a distinct control state?",
            "signal_location": "prompt-side representation with downstream policy effects",
            "readiness": "blocked_on_augmentation",
            "benchmark_basis": "explicit ROLE_DOMAIN framing, but current public support is confounded",
        },
        {
            "question_id": "mq_005_theory_conditioned_reasoning_mode",
            "mechanistic_question": "When theory is explicitly exposed, does it alter early representation, later justification policy, or both?",
            "signal_location": "prompt-side representation and generation-time justification",
            "readiness": "blocked_on_prompt_exposure_or_theory_augmentation",
            "benchmark_basis": "clean five-way theory pairing in the theory split suggests a good augmentation path",
        },
    ]


def build_follow_on_data_plan() -> list[dict[str, object]]:
    return [
        {
            "priority": 1,
            "task": "theory-matched augmentation",
            "why": "Expose theory cleanly for the existing 30 x 5 paired dilemmas.",
            "artifact_goal": "matched prompt sets with explicit theory exposure and same-label controls",
        },
        {
            "priority": 2,
            "task": "advisor-agent matched rewrites",
            "why": "Required to make action_locus scientifically probeable rather than merely suggestive.",
            "artifact_goal": "source-balanced advisor/agent pairs within shared scenario templates",
        },
        {
            "priority": 3,
            "task": "fresh generation capture",
            "why": "Required for all response-side labels including tradeoff engagement and the separated objective labels.",
            "artifact_goal": "response-labeled generation dataset on a stratified prompt slice",
        },
        {
            "priority": 4,
            "task": "label validation set",
            "why": "Needed to validate stakeholder_tradeoff_density and the response-side labeling functions.",
            "artifact_goal": "small hand-checked gold slice with disagreement notes",
        },
    ]


def build_recommended_first_experiments() -> list[dict[str, object]]:
    return [
        {
            "experiment": "behavioral smoke on stratified public slice",
            "purpose": "Verify parseability and task sanity before probing response-side labels.",
        },
        {
            "experiment": "prompt-side readout for stakeholder_tradeoff_density",
            "purpose": "Test the cleanest benchmark-native prompt-side distinction.",
        },
        {
            "experiment": "generation-time labeling pilot for tradeoff_engagement and commitment_style",
            "purpose": "Create the first usable response-side labels from fresh generations.",
        },
        {
            "experiment": "separate helpfulness_invoked and harm_avoidance_invoked labels on the same responses",
            "purpose": "Test separability directly rather than baking it into one balance scalar.",
        },
    ]


def build_frozen_label_set_placeholder(
    prompt_side_labels: list[dict[str, object]],
    response_side_labels: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "status": "partial",
        "reason": (
            "Phase 01 can freeze the ontology families and available prompt-side labels, but not the full "
            "response-side operational label set because fresh generations and augmentation are still required."
        ),
        "frozen_now": {
            "prompt_side_candidates": [label["label"] for label in prompt_side_labels],
            "response_side_candidates": [label["label"] for label in response_side_labels],
            "validation_signals": [
                "identifying",
                "clear process",
                "logical process",
                "helpful outcome",
                "harmless outcome",
            ],
            "nuisance_variables": [
                "DILEMMA_SOURCE",
                "DILEMMA_TYPE",
                "prompt length",
                "CONTEXT",
                "lexical template family",
            ],
        },
        "blocked_items": [
            "operational thresholds and labeler rules for response-side labels",
            "gold-slice validation for stakeholder_tradeoff_density",
            "clean action_locus target set",
            "clean theory_identity target set",
        ],
    }


def build_benchmark_framing(snapshot: dict[str, object]) -> dict[str, object]:
    public_stats = snapshot["configs"]["morebench_public"]
    theory_stats = snapshot["configs"]["morebench_theory"]
    return {
        "benchmark_name": "MoReBench",
        "artifact_freeze_date": "2026-04-22",
        "source_urls": DATASET_URLS,
        "scope": (
            "Claims here are grounded in direct inspection of the morebench_public and morebench_theory CSVs "
            "downloaded from Hugging Face."
        ),
        "public_split": {
            "row_count": public_stats["row_count"],
            "fields": public_stats["fields"],
            "rubric_dimension_counts": public_stats["rubric_dimension_counts"],
        },
        "theory_split": {
            "row_count": theory_stats["row_count"],
            "fields": theory_stats["fields"],
            "rubric_dimension_counts": theory_stats["rubric_dimension_counts"],
        },
        "initial_readiness_judgment": {
            "status": "ready_with_restrictions",
            "ready_now": [
                "benchmark-to-latent-label extraction",
                "behavioral smoke setup on a stratified prompt slice",
            ],
            "not_ready_yet": [
                "advisor-vs-agent mechanistic claims on the public split without augmentation",
                "theory-conditioned mechanistic claims without prompt exposure confirmation or augmentation",
            ],
        },
    }


def build_confound_analysis(
    public_rows: list[dict[str, str]],
    public_analysis: dict[str, object],
    theory_analysis: dict[str, object],
    theory_pairing: dict[str, object],
    action_locus_probeability: dict[str, object],
) -> dict[str, object]:
    return {
        "headline_confounds": [
            {
                "name": "source_role_aliasing_public",
                "severity": "high",
                "evidence": public_analysis["pair_distributions"]["source_x_role"],
                "impact": "Advisor-vs-agent is entangled with source family and style in the public split.",
            },
            {
                "name": "source_type_aliasing_public",
                "severity": "high",
                "evidence": public_analysis["pair_distributions"]["source_x_type"],
                "impact": "Case format differences can masquerade as mechanistic differences.",
            },
            {
                "name": "domain_topic_imbalance",
                "severity": "high",
                "evidence": top_contexts_by_source(public_rows),
                "impact": "Domain/topic concentration creates topical shortcuts.",
            },
            {
                "name": "action_locus_not_probeable_without_augmentation",
                "severity": "high",
                "evidence": action_locus_probeability["summary"],
                "impact": "There are zero source-controlled or source-type-controlled mixed-role cells.",
            },
            {
                "name": "theory_not_automatically_prompt_side",
                "severity": "high",
                "evidence": theory_pairing,
                "impact": "Clean evaluator-side pairing does not by itself create a model-side theory variable.",
            },
            {
                "name": "rubric_instruction_compositeness",
                "severity": "medium",
                "evidence": public_analysis["sample_rubric_titles"],
                "impact": "Many rubric criteria are grader instructions or case-specific desiderata, not latent names.",
            },
            {
                "name": "length_and_format_variation",
                "severity": "medium",
                "evidence": {
                    "public_dilemma_char_length": public_analysis["dilemma_char_length"],
                    "theory_dilemma_char_length": theory_analysis["dilemma_char_length"],
                },
                "impact": "Prompt length and format must be tracked for prompt-side work.",
            },
        ],
        "required_controls": [
            "Treat DILEMMA_SOURCE as a mandatory nuisance control.",
            "Track DILEMMA_TYPE and prompt length in all first-pass prompt-side studies.",
            "Keep helpfulness and harm avoidance as separate response-side labels.",
            "Do not treat ROLE_DOMAIN as probeable on the current public split.",
            "Use theory only after confirming prompt exposure or after theory-focused augmentation.",
        ],
    }


def build_frozen_label_rows(
    rows_by_config: dict[str, list[dict[str, str]]],
) -> list[dict[str, str]]:
    frozen_rows: list[dict[str, str]] = []
    for config_name, rows in rows_by_config.items():
        for row_idx, row in enumerate(rows):
            frozen_rows.append(
                {
                    "benchmark": "morebench",
                    "config_name": config_name,
                    "row_id": f"{config_name}__{row_idx:04d}",
                    "row_idx": str(row_idx),
                    "dilemma_source": row["DILEMMA_SOURCE"],
                    "dilemma_type": row["DILEMMA_TYPE"],
                    "role_domain": row["ROLE_DOMAIN"],
                    "context": row["CONTEXT"],
                    "theory": row["THEORY"],
                    "action_locus_label": row["ROLE_DOMAIN"],
                    "action_locus_status": "blocked_on_augmentation",
                    "theory_identity_label": row["THEORY"],
                    "theory_identity_status": "metadata_only_until_prompt_exposed",
                    "dilemma_structure_label": row["DILEMMA_TYPE"],
                    "domain_topic_label": row["CONTEXT"],
                    "stakeholder_tradeoff_density_label": "",
                    "stakeholder_tradeoff_density_status": "pending_gold_slice_validation",
                    "response_side_labels_status": "requires_fresh_generations",
                    "freeze_version": "v1",
                    "freeze_date": "2026-04-22",
                }
            )
    return frozen_rows


def build_phase_00_report(
    benchmark_framing: dict[str, object],
    confound_analysis: dict[str, object],
    theory_pairing: dict[str, object],
    action_locus_probeability: dict[str, object],
) -> str:
    public = benchmark_framing["public_split"]
    theory = benchmark_framing["theory_split"]
    return f"""# MoReBench Phase 00 Benchmark Validation

## Bottom Line

Phase 00 succeeded.
`MoReBench` is a strong benchmark-first substrate, but the current public split already tells us two things we should treat as hard contract facts before phase 02:

- `action_locus` is effectively not probeable on the current public split without augmentation
- `theory_identity` is promising, but it is not yet a clean prompt-side variable unless the runtime prompt exposes `THEORY`

## Key Counts

- public rows: `{public["row_count"]}`
- theory rows: `{theory["row_count"]}`
- source-controlled mixed-role cells for `action_locus`: `{action_locus_probeability["summary"]["source_cells_with_both_roles"]}`
- source-and-type-controlled mixed-role cells for `action_locus`: `{action_locus_probeability["summary"]["source_type_cells_with_both_roles"]}`
- exact theory/public dilemma overlap: `{theory_pairing["exact_dilemma_overlap_with_public"]["overlap_count"]}` of `{theory_pairing["exact_dilemma_overlap_with_public"]["theory_unique_count"]}`

## High-Priority Confounds

- `source_role_aliasing_public`
- `source_type_aliasing_public`
- `domain_topic_imbalance`
- `action_locus_not_probeable_without_augmentation`
- `theory_not_automatically_prompt_side`

## Recommendation

Proceed to phase 01, but treat theory and action-locus as likely augmentation-bound from the start.
"""


def build_phase_01_report(
    candidate_mechanistic_questions: list[dict[str, object]],
    prompt_side_labels: list[dict[str, object]],
    response_side_labels: list[dict[str, object]],
    follow_on_data_plan: list[dict[str, object]],
) -> str:
    surviving_questions = [question["question_id"] for question in candidate_mechanistic_questions]
    prompt_labels = [label["label"] for label in prompt_side_labels]
    response_labels = [label["label"] for label in response_side_labels]
    return f"""# MoReBench Phase 01 Benchmark-To-Latent-Labels

## Bottom Line

Phase 01 succeeded as a label-formation phase, but not all candidate labels are operational today.

- prompt-side vs response-side separation is preserved
- helpfulness and harm avoidance remain separate
- rubric criteria are treated as validation surfaces
- `action_locus` and `theory_identity` are explicitly augmentation-gated

## Mechanistic Questions That Survived

- `{surviving_questions[0]}`
- `{surviving_questions[1]}`
- `{surviving_questions[2]}`
- `{surviving_questions[3]}`
- `{surviving_questions[4]}`

## Prompt-Side Label Families

`{", ".join(prompt_labels)}`

## Response-Side Label Families

`{", ".join(response_labels)}`

## Next Action

Proceed to latent-label-data-augmentation rather than to analysis, because the benchmark still needs paired theory exposure, action-locus rewrites, and fresh generations.

## Follow-On Data Plan

{json.dumps(follow_on_data_plan, indent=2)}
"""


def build_validation_notes_markdown(snapshot: dict[str, object]) -> str:
    return (
        frontmatter("00", ["projects/MECH_INTERP/morebench/phase_00/outputs/benchmark_snapshot_detail.json"])
        + "\n\n# MoReBench 00 Validation Notes\n\n"
        + "These notes capture the main operational hazards discovered during raw CSV inspection.\n\n"
        + "- The public and theory splits are legible and directly downloadable.\n"
        + "- Theory rows form clean five-way dilemma groups.\n"
        + "- Public source-role aliasing is severe enough to collapse action-locus cells under required controls.\n"
        + f"- Public split size: {snapshot['configs']['morebench_public']['row_count']}\n"
        + f"- Theory split size: {snapshot['configs']['morebench_theory']['row_count']}\n"
    )


def build_latent_label_spec_markdown(
    prompt_side_labels: list[dict[str, object]],
    response_side_labels: list[dict[str, object]],
    candidate_mechanistic_questions: list[dict[str, object]],
    validation_signals: list[dict[str, str]],
    nuisance_variables: list[dict[str, str]],
) -> str:
    return (
        frontmatter(
            "01",
            [
                "projects/MECH_INTERP/morebench/phase_00/outputs/benchmark_framing.json",
                "projects/MECH_INTERP/morebench/phase_00/outputs/confound_analysis.json",
            ],
        )
        + "\n\n# MoReBench 01 Latent Label Spec\n\n"
        + "## Required Inputs At Phase Start\n\n"
        + "- probe-target model(s): not yet frozen\n"
        + "- generation protocol: not yet frozen for response-side work\n"
        + "- activation capture regime: prompt-side and generation-time both relevant, but not yet frozen\n"
        + "- research mode: benchmark-first correlational readout first, causal follow-up later\n"
        + "- seeds and sampling parameters: not yet frozen because fresh generations have not started\n\n"
        + "## Benchmark-First Mechanistic Questions\n\n"
        + json.dumps(candidate_mechanistic_questions, indent=2)
        + "\n\n## Prompt-Side Labels\n\n"
        + json.dumps(prompt_side_labels, indent=2)
        + "\n\n## Response-Side Labels\n\n"
        + json.dumps(response_side_labels, indent=2)
        + "\n\n## Validation Signals\n\n"
        + json.dumps(validation_signals, indent=2)
        + "\n\n## Nuisance Variables\n\n"
        + json.dumps(nuisance_variables, indent=2)
        + "\n"
    )


def build_labeling_functions_markdown(
    prompt_side_labels: list[dict[str, object]], response_side_labels: list[dict[str, object]]
) -> str:
    return (
        frontmatter(
            "01",
            [
                "projects/MECH_INTERP/morebench/phase_01/outputs/prompt_side_labels.json",
                "projects/MECH_INTERP/morebench/phase_01/outputs/response_side_labels.json",
            ],
        )
        + "\n\n# MoReBench 01 Labeling Functions\n\n"
        + "## Prompt-Side Labels\n\n"
        + json.dumps(prompt_side_labels, indent=2)
        + "\n\n## Response-Side Labels\n\n"
        + json.dumps(response_side_labels, indent=2)
        + "\n\n## Validation Note\n\n"
        + "Prompt-side direct metadata labels are frozen now. Response-side labels remain blocked on fresh generations.\n"
    )


def build_confound_audit_markdown(
    confound_analysis: dict[str, object],
    action_locus_probeability: dict[str, object],
) -> str:
    return (
        frontmatter(
            "01",
            [
                "projects/MECH_INTERP/morebench/phase_00/outputs/confound_analysis.json",
                "projects/MECH_INTERP/morebench/phase_00/outputs/action_locus_probeability_audit.json",
            ],
        )
        + "\n\n# MoReBench 01 Confound Audit\n\n"
        + "## Headline Target-vs-Nuisance Risks\n\n"
        + json.dumps(confound_analysis["headline_confounds"], indent=2)
        + "\n\n## Planned Controls\n\n"
        + json.dumps(confound_analysis["required_controls"], indent=2)
        + "\n\n## Probeability Gate\n\n"
        + json.dumps(action_locus_probeability["judgment"], indent=2)
        + "\n"
    )


def build_gap_list_markdown(
    frozen_placeholder: dict[str, object], follow_on_data_plan: list[dict[str, object]]
) -> str:
    return (
        frontmatter(
            "01",
            [
                "projects/MECH_INTERP/morebench/phase_00/outputs/action_locus_probeability_audit.json",
                "projects/MECH_INTERP/morebench/phase_00/outputs/theory_pairing_audit.json",
            ],
        )
        + "\n\n# MoReBench 01 Gap List\n\n"
        + "## Highest-Priority Gaps\n\n"
        + "1. `theory_identity` is not yet a clean prompt-side variable.\n"
        + "2. `action_locus` is not currently probeable on the public split.\n"
        + "3. Response-side labels still require fresh generations.\n"
        + "4. `stakeholder_tradeoff_density` still needs a gold-slice validation pass.\n\n"
        + "## Partial Freeze Status\n\n"
        + json.dumps(frozen_placeholder, indent=2)
        + "\n\n## Suggested Next Repairs\n\n"
        + json.dumps(follow_on_data_plan, indent=2)
        + "\n"
    )


def build_derivability_report_markdown() -> str:
    return (
        frontmatter(
            "01",
            [
                "projects/MECH_INTERP/morebench/phase_01/outputs/prompt_side_labels.json",
                "projects/MECH_INTERP/morebench/phase_01/outputs/response_side_labels.json",
            ],
        )
        + "\n\n# MoReBench 01 Derivability Report\n\n"
        + "## Labels Derivable Now\n\n"
        + "- `action_locus` as metadata surface, but not as a clean probe target\n"
        + "- `dilemma_structure`\n"
        + "- `domain_topic`\n"
        + "- `theory_identity` as metadata only\n\n"
        + "## Labels Not Yet Derivable Reliably\n\n"
        + "- `stakeholder_tradeoff_density`: needs a validated counting policy and gold slice\n"
        + "- all response-side labels: need fresh generations under the intended protocol\n"
    )


def build_phase_00_summary_json(benchmark_framing: dict[str, object]) -> dict[str, object]:
    return {
        "benchmark": "morebench",
        "phase": "00",
        "status": "proceed_to_benchmark_to_latent_labels",
        "public_row_count": benchmark_framing["public_split"]["row_count"],
        "theory_row_count": benchmark_framing["theory_split"]["row_count"],
        "top_findings": [
            "action_locus not probeable on current public split",
            "theory structurally promising but not automatically prompt-side",
            "dense rubric structure supports richer mechanistic questions",
        ],
    }


def build_phase_01_summary_json(
    prompt_side_labels: list[dict[str, object]],
    response_side_labels: list[dict[str, object]],
    frozen_placeholder: dict[str, object],
) -> dict[str, object]:
    return {
        "benchmark": "morebench",
        "phase": "01",
        "status": "proceed_to_latent_label_data_augmentation",
        "prompt_side_labels": [label["label"] for label in prompt_side_labels],
        "response_side_labels": [label["label"] for label in response_side_labels],
        "freeze_status": frozen_placeholder["status"],
        "blocked_items": frozen_placeholder["blocked_items"],
        "next_action": "augmentation",
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cleanup_legacy_outputs() -> None:
    for path in LEGACY_OUTPUTS:
        if path.exists():
            path.unlink()


def main() -> None:
    rows = {name: fetch_rows(url) for name, url in DATASET_URLS.items()}
    public_rows = rows["morebench_public"]
    theory_rows = rows["morebench_theory"]

    snapshot = {
        "benchmark": "MoReBench",
        "artifact_freeze_date": "2026-04-22",
        "source_urls": DATASET_URLS,
        "configs": {
            "morebench_public": analyze_config("morebench_public", public_rows),
            "morebench_theory": analyze_config("morebench_theory", theory_rows),
        },
    }

    benchmark_framing = build_benchmark_framing(snapshot)
    native_label_inventory = build_native_label_inventory()
    theory_pairing = build_theory_pairing_audit(theory_rows, public_rows)
    action_locus_probeability = build_action_locus_probeability_audit(public_rows)
    confound_analysis = build_confound_analysis(
        public_rows,
        snapshot["configs"]["morebench_public"],
        snapshot["configs"]["morebench_theory"],
        theory_pairing,
        action_locus_probeability,
    )

    candidate_mechanistic_questions = build_candidate_mechanistic_questions()
    prompt_side_labels = build_prompt_side_labels()
    response_side_labels = build_response_side_labels()
    validation_signals = build_validation_signals()
    nuisance_variables = build_nuisance_variables()
    follow_on_data_plan = build_follow_on_data_plan()
    recommended_first_experiments = build_recommended_first_experiments()
    frozen_label_set_placeholder = build_frozen_label_set_placeholder(prompt_side_labels, response_side_labels)
    frozen_label_rows = build_frozen_label_rows(rows)

    cleanup_legacy_outputs()

    write_json(PHASE_00_ROOT / "outputs" / "benchmark_framing.json", benchmark_framing)
    write_json(PHASE_00_ROOT / "outputs" / "benchmark_snapshot_detail.json", snapshot)
    write_json(PHASE_00_ROOT / "outputs" / "native_label_inventory.json", native_label_inventory)
    write_json(PHASE_00_ROOT / "outputs" / "theory_pairing_audit.json", theory_pairing)
    write_json(PHASE_00_ROOT / "outputs" / "confound_analysis.json", confound_analysis)
    write_json(PHASE_00_ROOT / "outputs" / "action_locus_probeability_audit.json", action_locus_probeability)
    write_text(
        PHASE_00_ROOT / "reports" / "phase_00_benchmark_validation.md",
        build_phase_00_report(benchmark_framing, confound_analysis, theory_pairing, action_locus_probeability),
    )

    write_json(PHASE_01_ROOT / "outputs" / "candidate_mechanistic_questions.json", candidate_mechanistic_questions)
    write_json(PHASE_01_ROOT / "outputs" / "prompt_side_labels.json", prompt_side_labels)
    write_json(PHASE_01_ROOT / "outputs" / "response_side_labels.json", response_side_labels)
    write_json(PHASE_01_ROOT / "outputs" / "validation_signals.json", validation_signals)
    write_json(PHASE_01_ROOT / "outputs" / "nuisance_variables.json", nuisance_variables)
    write_json(PHASE_01_ROOT / "outputs" / "follow_on_data_plan.json", follow_on_data_plan)
    write_json(PHASE_01_ROOT / "outputs" / "recommended_first_experiments.json", recommended_first_experiments)
    write_json(PHASE_01_ROOT / "outputs" / "frozen_label_set_placeholder.json", frozen_label_set_placeholder)
    write_text(
        PHASE_01_ROOT / "reports" / "phase_01_benchmark_to_latent_labels.md",
        build_phase_01_report(
            candidate_mechanistic_questions, prompt_side_labels, response_side_labels, follow_on_data_plan
        ),
    )
    write_text(
        PHASE_01_ROOT / "specs" / "labeling-functions.md",
        build_labeling_functions_markdown(prompt_side_labels, response_side_labels),
    )
    write_text(
        PHASE_01_ROOT / "reports" / "gap-list.md",
        build_gap_list_markdown(frozen_label_set_placeholder, follow_on_data_plan),
    )

    write_text(
        CANONICAL_ROOT / "00-validation-memo.md",
        frontmatter("00", ["projects/MECH_INTERP/morebench/phase_00/outputs/benchmark_framing.json"])
        + "\n\n"
        + build_phase_00_report(benchmark_framing, confound_analysis, theory_pairing, action_locus_probeability)
        + "\n",
    )
    write_json(CANONICAL_ROOT / "00-validation-summary.json", build_phase_00_summary_json(benchmark_framing))
    write_text(CANONICAL_ROOT / "00-validation-notes.md", build_validation_notes_markdown(snapshot))

    write_text(
        CANONICAL_ROOT / "01-latent-label-spec.md",
        build_latent_label_spec_markdown(
            prompt_side_labels,
            response_side_labels,
            candidate_mechanistic_questions,
            validation_signals,
            nuisance_variables,
        ),
    )
    write_csv(CANONICAL_ROOT / "01-label-inventory.csv", native_label_inventory)
    write_text(
        CANONICAL_ROOT / "01-labeling-functions.md",
        build_labeling_functions_markdown(prompt_side_labels, response_side_labels),
    )
    write_text(
        CANONICAL_ROOT / "01-confound-audit.md",
        build_confound_audit_markdown(confound_analysis, action_locus_probeability),
    )
    write_csv(CANONICAL_ROOT / "01-frozen-label-set.csv", frozen_label_rows)
    write_text(
        CANONICAL_ROOT / "01-gap-list.md",
        build_gap_list_markdown(frozen_label_set_placeholder, follow_on_data_plan),
    )
    write_json(
        CANONICAL_ROOT / "01-latent-label-summary.json",
        build_phase_01_summary_json(prompt_side_labels, response_side_labels, frozen_label_set_placeholder),
    )
    write_text(CANONICAL_ROOT / "01-derivability-report.md", build_derivability_report_markdown())


if __name__ == "__main__":
    main()
