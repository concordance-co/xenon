---
benchmark: morebench
phase: 01
version: v1
frozen_date: 2026-04-22
input_artifacts:
  - projects/MOREBENCH/phase_00/outputs/action_locus_probeability_audit.json
  - projects/MOREBENCH/phase_00/outputs/theory_pairing_audit.json
---

# MoReBench 01 Gap List

## Highest-Priority Gaps

1. `theory_identity` is not yet a clean prompt-side variable.
2. `action_locus` is not currently probeable on the public split.
3. Response-side labels still require fresh generations.
4. `stakeholder_tradeoff_density` still needs a gold-slice validation pass.

## Partial Freeze Status

{
  "status": "partial",
  "reason": "Phase 01 can freeze the ontology families and available prompt-side labels, but not the full response-side operational label set because fresh generations and augmentation are still required.",
  "frozen_now": {
    "prompt_side_candidates": [
      "action_locus",
      "stakeholder_tradeoff_density",
      "dilemma_structure",
      "domain_topic",
      "theory_identity"
    ],
    "response_side_candidates": [
      "tradeoff_engagement",
      "commitment_style",
      "refuses_or_hedges",
      "helpfulness_invoked",
      "harm_avoidance_invoked",
      "uncertainty_and_scope_calibration"
    ],
    "validation_signals": [
      "identifying",
      "clear process",
      "logical process",
      "helpful outcome",
      "harmless outcome"
    ],
    "nuisance_variables": [
      "DILEMMA_SOURCE",
      "DILEMMA_TYPE",
      "prompt length",
      "CONTEXT",
      "lexical template family"
    ]
  },
  "blocked_items": [
    "operational thresholds and labeler rules for response-side labels",
    "gold-slice validation for stakeholder_tradeoff_density",
    "clean action_locus target set",
    "clean theory_identity target set"
  ]
}

## Suggested Next Repairs

[
  {
    "priority": 1,
    "task": "theory-matched augmentation",
    "why": "Expose theory cleanly for the existing 30 x 5 paired dilemmas.",
    "artifact_goal": "matched prompt sets with explicit theory exposure and same-label controls"
  },
  {
    "priority": 2,
    "task": "advisor-agent matched rewrites",
    "why": "Required to make action_locus scientifically probeable rather than merely suggestive.",
    "artifact_goal": "source-balanced advisor/agent pairs within shared scenario templates"
  },
  {
    "priority": 3,
    "task": "fresh generation capture",
    "why": "Required for all response-side labels including tradeoff engagement and the separated objective labels.",
    "artifact_goal": "response-labeled generation dataset on a stratified prompt slice"
  },
  {
    "priority": 4,
    "task": "label validation set",
    "why": "Needed to validate stakeholder_tradeoff_density and the response-side labeling functions.",
    "artifact_goal": "small hand-checked gold slice with disagreement notes"
  }
]
