# Phase 01 Seed Dataset Plan

## Why A New Dataset

The named-theory substrate mostly produced shared moral-response structure and weak
decision-posture signals. For this polar test we need cases where the actor's own
welfare is directly at stake. Many existing MoReBench dilemmas are third-person
institutional tradeoffs, so they are not sharp enough for an ethical-vs-self-advantage
direction.

## Candidate Inclusion Rule

A seed dilemma is usable only if:

- the decision-maker is `you`;
- the self-advantage option improves or protects your own welfare;
- the ethical option imposes a personal cost, risk, lost opportunity, or lost status;
- the two actions can be labeled without reading subtle theory language;
- the case is not merely "obey the law" versus "break the law."

## Seed Dataset

The initial builder materializes 40 candidates across domains such as workplace,
medicine, finance, education, research, journalism, housing, software, and personal
relationships. Each row includes:

- `ethical_action`
- `self_advantage_action`
- `personal_cost_type`
- `personal_benefit_type`
- `harm_if_self_advantage`
- `expected_conflict_strength`

These fields are intentionally explicit so action labels and filters can be derived
without interpreting the whole response from scratch.

## Behavior Smoke

Run generation before capture:

- 40 candidate dilemmas x 7 conditions x 2 samples = 560 generations.
- workflow: `specs/behavior_smoke_workflow.py`.
- system prompt: neutral `decision scenario` framing, not `moral decision`,
  so the ethical frame enters only through the ethical pole condition.
- sample stability uses two question-suffix paraphrases rather than duplicate
  identical prompts, preventing prompt-hash collapse in generation.
- v2 short-term exploit prompt set: `specs/ethical_advantage_prompt_conditions_v2_short_term_exploit.json`.
  Use this if the softer self-advantage prompt is domesticated into long-term
  reputation or compliance reasoning.
- Keep only dilemmas where `P_ethical_*` and `P_advantage_*` produce opposed action
  labels in at least 3 of 4 paired comparisons.
- Exclude dilemmas where neutral is unanimous with one pole across both samples unless
  the goal is to build an easy held-out steering set.

## Capture Plan After Smoke

Primary capture set:

- 24 to 30 filtered dilemmas.
- conditions: `P_ethical_01`, `P_ethical_02`, `P_advantage_01`,
  `P_advantage_02`, `N_neutral_01`, `N_practical_01`, `N_compliance_01`.
- sites: prompt-end and generated sequence.
- slices: prompt-end, first_16 generated, full generated.
- layers: L16, L24, L32, L40 initially.

Primary readout:

- paired ethical-minus-advantage direction by dilemma;
- split-half cosine and sign-flip null;
- layer and slice sweep;
- lexical controls using prefix-only text for first_16 and full-response text for
  full generated.

Stop if:

- fewer than 20 dilemmas show behavioral pole separation;
- a text baseline predicts the direction as well as activations;
- random matched directions steer action at more than half the ethical/self-advantage
  direction's effect.
