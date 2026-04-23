# MoReBench Phase 01 Benchmark-To-Latent-Labels

## Bottom Line

Phase 01 succeeded as a label-formation phase, but not all candidate labels are operational today.

- prompt-side vs response-side separation is preserved
- helpfulness and harm avoidance remain separate
- rubric criteria are treated as validation surfaces
- `action_locus` and `theory_identity` are explicitly augmentation-gated

## Mechanistic Questions That Survived

- `mq_001_multi_consideration_representation`
- `mq_002_commitment_transition`
- `mq_003_helpfulness_harm_avoidance_separability`
- `mq_004_action_locus_control_state`
- `mq_005_theory_conditioned_reasoning_mode`

## Prompt-Side Label Families

`action_locus, stakeholder_tradeoff_density, dilemma_structure, domain_topic, theory_identity`

## Response-Side Label Families

`tradeoff_engagement, commitment_style, refuses_or_hedges, helpfulness_invoked, harm_avoidance_invoked, uncertainty_and_scope_calibration`

## Next Action

Proceed to latent-label-data-augmentation rather than to analysis, because the benchmark still needs paired theory exposure, action-locus rewrites, and fresh generations.

## Follow-On Data Plan

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
