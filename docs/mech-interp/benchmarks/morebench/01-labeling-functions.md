---
benchmark: morebench
phase: 01
version: v1
frozen_date: 2026-04-22
input_artifacts:
  - projects/MECH_INTERP/morebench/phase_01/outputs/prompt_side_labels.json
  - projects/MECH_INTERP/morebench/phase_01/outputs/response_side_labels.json
---

# MoReBench 01 Labeling Functions

## Prompt-Side Labels

[
  {
    "label": "action_locus",
    "status": "deferred_pending_augmentation",
    "source_fields": [
      "ROLE_DOMAIN",
      "DILEMMA"
    ],
    "definition": "Whether the system is framed as advising a user or acting as the responsible decider.",
    "signal_location": "prompt-side representation",
    "why_it_matters": "Directly tied to benchmark-native advisor versus agent framing.",
    "readiness": "not_probeable_on_current_public_split",
    "labeling_function": "direct metadata readout with augmentation-gated use"
  },
  {
    "label": "stakeholder_tradeoff_density",
    "status": "derived_from_prompt",
    "source_fields": [
      "DILEMMA"
    ],
    "definition": "How many stakeholder or consequence clusters are simultaneously live in the scenario.",
    "signal_location": "prompt-side representation",
    "why_it_matters": "Most direct prompt-side expression of the benchmark's multi-consideration design.",
    "readiness": "candidate_first_pass_label",
    "labeling_function": "human or LLM-assisted count on a validated gold slice"
  },
  {
    "label": "dilemma_structure",
    "status": "derived_from_prompt_plus_metadata",
    "source_fields": [
      "DILEMMA",
      "DILEMMA_TYPE"
    ],
    "definition": "Case-format structure such as long-case, short-case, or expert-case presentation.",
    "signal_location": "prompt-side representation",
    "why_it_matters": "Useful auxiliary variable for prompt parsing differences.",
    "readiness": "nuisance_or_auxiliary",
    "labeling_function": "direct metadata readout"
  },
  {
    "label": "domain_topic",
    "status": "direct_from_metadata",
    "source_fields": [
      "CONTEXT"
    ],
    "definition": "Broad scenario domain such as healthcare, interpersonal, or science and technology.",
    "signal_location": "prompt-side representation",
    "why_it_matters": "Necessary nuisance control for topical shortcuts.",
    "readiness": "nuisance_primary",
    "labeling_function": "direct metadata readout"
  },
  {
    "label": "theory_identity",
    "status": "deferred_but_prioritized_for_augmentation",
    "source_fields": [
      "THEORY"
    ],
    "definition": "Which explicit ethical framework is surfaced to the model.",
    "signal_location": "prompt-side representation and later justification policy",
    "why_it_matters": "Natural route to clean prompt-side control comparisons if augmented correctly.",
    "readiness": "blocked_on_prompt_exposure_or_matched_theory_rewrites",
    "labeling_function": "direct metadata readout only after prompt exposure is explicit"
  }
]

## Response-Side Labels

[
  {
    "label": "tradeoff_engagement",
    "status": "derived_from_new_generations",
    "source_fields": [
      "generated_response"
    ],
    "definition": "Whether the response keeps multiple live considerations active before converging.",
    "signal_location": "generation-time deliberation",
    "readiness": "high_after_generation",
    "labeling_function": "human or LLM rubric-aligned annotation on fresh generations"
  },
  {
    "label": "commitment_style",
    "status": "derived_from_new_generations",
    "source_fields": [
      "generated_response"
    ],
    "definition": "Whether the model defers, recommends, refuses, or commits directly after deliberation.",
    "signal_location": "late generation and final readout",
    "readiness": "high_after_generation",
    "labeling_function": "rule-based or annotation-based conclusion classification"
  },
  {
    "label": "refuses_or_hedges",
    "status": "derived_from_new_generations",
    "source_fields": [
      "generated_response"
    ],
    "definition": "Whether the response declines commitment or leans on heavy hedging rather than recommending.",
    "signal_location": "late generation and final readout",
    "readiness": "high_after_generation",
    "labeling_function": "annotation over generated conclusions"
  },
  {
    "label": "helpfulness_invoked",
    "status": "derived_from_new_generations_with_rubric_validation",
    "source_fields": [
      "generated_response",
      "RUBRIC"
    ],
    "definition": "Whether the response explicitly optimizes for actionable assistance and practical usefulness.",
    "signal_location": "generation-time objective orientation and late readout",
    "readiness": "high_after_generation",
    "labeling_function": "annotation validated against helpful outcome criteria"
  },
  {
    "label": "harm_avoidance_invoked",
    "status": "derived_from_new_generations_with_rubric_validation",
    "source_fields": [
      "generated_response",
      "RUBRIC"
    ],
    "definition": "Whether the response explicitly optimizes for avoiding harm, recklessness, or unsafe overreach.",
    "signal_location": "generation-time objective orientation and late readout",
    "readiness": "high_after_generation",
    "labeling_function": "annotation validated against harmless outcome criteria"
  },
  {
    "label": "uncertainty_and_scope_calibration",
    "status": "derived_from_new_generations",
    "source_fields": [
      "generated_response"
    ],
    "definition": "Whether the response marks uncertainty, limits, or role-appropriate scope boundaries.",
    "signal_location": "generation-time deliberation and conclusion framing",
    "readiness": "medium_after_generation",
    "labeling_function": "annotation over uncertainty and scope markers in fresh generations"
  }
]

## Validation Note

Prompt-side direct metadata labels are frozen now. Response-side labels remain blocked on fresh generations.
