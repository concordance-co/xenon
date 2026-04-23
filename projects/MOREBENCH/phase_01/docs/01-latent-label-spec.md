---
benchmark: morebench
phase: 01
version: v1
frozen_date: 2026-04-22
input_artifacts:
  - projects/MOREBENCH/phase_00/outputs/benchmark_framing.json
  - projects/MOREBENCH/phase_00/outputs/confound_analysis.json
---

# MoReBench 01 Latent Label Spec

## Required Inputs At Phase Start

- probe-target model(s): not yet frozen
- generation protocol: not yet frozen for response-side work
- activation capture regime: prompt-side and generation-time both relevant, but not yet frozen
- research mode: benchmark-first correlational readout first, causal follow-up later
- seeds and sampling parameters: not yet frozen because fresh generations have not started

## Benchmark-First Mechanistic Questions

[
  {
    "question_id": "mq_001_multi_consideration_representation",
    "mechanistic_question": "Does the model keep multiple live considerations active before recommending?",
    "signal_location": "prompt-side representation and generation-time deliberation",
    "readiness": "high",
    "benchmark_basis": "criterion-dense emphasis on dilemma coverage before conclusion"
  },
  {
    "question_id": "mq_002_commitment_transition",
    "mechanistic_question": "When does the model shift from exploration to concrete recommendation?",
    "signal_location": "generation-time deliberation and late commitment state",
    "readiness": "high",
    "benchmark_basis": "process criteria and helpful-outcome criteria pull apart deliberation and conclusion"
  },
  {
    "question_id": "mq_003_helpfulness_harm_avoidance_separability",
    "mechanistic_question": "Are helpfulness-oriented and harm-avoidance-oriented objectives separable in the response policy?",
    "signal_location": "generation-time objective orientation and late readout",
    "readiness": "high",
    "benchmark_basis": "distinct helpful outcome and harmless outcome rubric families"
  },
  {
    "question_id": "mq_004_action_locus_control_state",
    "mechanistic_question": "Does advisor versus agent framing induce a distinct control state?",
    "signal_location": "prompt-side representation with downstream policy effects",
    "readiness": "blocked_on_augmentation",
    "benchmark_basis": "explicit ROLE_DOMAIN framing, but current public support is confounded"
  },
  {
    "question_id": "mq_005_theory_conditioned_reasoning_mode",
    "mechanistic_question": "When theory is explicitly exposed, does it alter early representation, later justification policy, or both?",
    "signal_location": "prompt-side representation and generation-time justification",
    "readiness": "blocked_on_prompt_exposure_or_theory_augmentation",
    "benchmark_basis": "clean five-way theory pairing in the theory split suggests a good augmentation path"
  }
]

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

## Validation Signals

[
  {
    "signal": "identifying",
    "use": "validate dilemma recognition and coverage of live considerations"
  },
  {
    "signal": "clear process",
    "use": "validate structured reasoning presentation"
  },
  {
    "signal": "logical process",
    "use": "validate argument and consequence chaining quality"
  },
  {
    "signal": "helpful outcome",
    "use": "validate helpfulness-oriented response quality"
  },
  {
    "signal": "harmless outcome",
    "use": "validate harm avoidance and reckless recommendation avoidance"
  }
]

## Nuisance Variables

[
  {
    "variable": "DILEMMA_SOURCE",
    "reason": "strong source/template aliasing with role, style, and domain"
  },
  {
    "variable": "DILEMMA_TYPE",
    "reason": "co-moves with source and changes prompt structure"
  },
  {
    "variable": "prompt length",
    "reason": "substantial variation in dilemma length across case families"
  },
  {
    "variable": "CONTEXT",
    "reason": "topic/domain concentration differs sharply by source"
  },
  {
    "variable": "lexical template family",
    "reason": "prompt wording may leak source identity"
  },
  {
    "variable": "theory metadata when not exposed to the prompt",
    "reason": "evaluator-side field, not automatically model-side input"
  }
]
