---
benchmark: morebench
phase: 03
version: v1
frozen_date: 2026-04-23
input_artifacts:
  - projects/MOREBENCH/phase_01/docs/01-latent-label-spec.md
  - projects/MOREBENCH/phase_02/docs/02-augmented-data-manifest.json
  - projects/MOREBENCH/phase_02/docs/02-behavioral-smoke-report.md
---

# MoReBench 03 Feature Hypotheses

## `theory_identity`

- candidate representation:
  prompt-side diagnostic state marking which alias-based theory cue family is present
- readable version:
  prompt-side residual distinction that survives held-out alias banks rather than one fixed canonical cue
- causally useful version:
  limited diagnostic use only; not the main causal target
- why this is plausible:
  the reopened phase-02 repair loop now treats alias-based theory cues as the best prompt-side diagnostic family while description-only moves toward generation-time use

## `theory_conditioned_generation_persistence`

- candidate representation:
  generation-time state reflecting which moral framework is currently shaping the model's reasoning on a held-constant dilemma
- readable version:
  generated-token activations distinguish theory-primed responses from generic-ethics or differently primed responses on the same dilemma
- causally useful version:
  a localizable generation-time state whose perturbation changes stakeholder emphasis, tradeoff framing, or recommendation style under different theory primes
- why this is plausible:
  MoReBench's interesting theory question is whether a framework cue changes generated reasoning, not merely whether the cue text is recoverable from prompt activations

## `stakeholder_tradeoff_density`

- candidate representation:
  multi-consideration load state that tracks how many stakeholder/consequence clusters are active simultaneously
- readable version:
  prompt-end representation that scales with validated density labels
- causally useful version:
  a state whose perturbation changes whether the response collapses to a single-axis answer
- why this is plausible:
  the benchmark design strongly emphasizes dilemma recognition and balancing multiple live considerations before conclusion

## `tradeoff_engagement`

- candidate representation:
  generation-time maintenance of multiple live considerations rather than early collapse
- readable version:
  response-token states predictive of continued multi-sided reasoning
- causally useful version:
  a state whose disruption increases one-sided recommendations or shallow answers
- why this is plausible:
  MoReBench cases are explicitly built to pressure tradeoff handling rather than single-objective optimization

## `commitment_style`

- candidate representation:
  transition from exploratory/deliberative reasoning into commitment or refusal
- readable version:
  late-generation state that predicts direct recommendation vs hedging vs refusal
- causally useful version:
  a localizable transition state that changes commitment timing without broadly destabilizing reasoning
- why this is plausible:
  the benchmark’s response quality axes implicitly separate process quality from recommendation quality

## `helpfulness_invoked`

- candidate representation:
  actionable-assistance orientation emphasizing practical recommendation quality
- readable version:
  late-generation state associated with direct guidance, practical framing, and actionability
- causally useful version:
  an objective-like state whose modulation alters helpful recommendation framing while preserving task understanding
- why this is plausible:
  `helpful outcome` is a distinct benchmark-native validation surface

## `harm_avoidance_invoked`

- candidate representation:
  caution / non-recklessness orientation emphasizing avoidance of harmful overreach
- readable version:
  late-generation state associated with risk marking, caution, and avoidance of unsafe commitments
- causally useful version:
  an objective-like state that can be modulated separately from helpfulness
- why this is plausible:
  `harmless outcome` is explicitly separate and includes signed penalties for reckless recommendations

## `helpfulness_invoked` vs `harm_avoidance_invoked`

- candidate relation:
  partially separable but correlated response-policy states
- readable version:
  two probes that each retain signal after controlling for the other
- causally useful version:
  one direction/intervention shifts helpfulness-associated behavior without symmetrically shifting harm avoidance, or vice versa
- why this is plausible:
  benchmark design and prior review both argue against collapsing them into one scalar too early

## `action_locus`

- candidate representation:
  advisor-vs-agent control state that changes recommendation framing and responsibility posture
- readable version:
  prompt-side difference on matched rewrite pairs
- causally useful version:
  a localized control state that shifts role-conditioned recommendation behavior
- why this is still cautious:
  only a starter rewrite batch exists, so this remains a pilot target rather than a central phase-03 claim
