---
benchmark: morebench
phase: 03
version: v3
frozen_date: 2026-04-23
input_artifacts:
  - projects/MOREBENCH/phase_01/docs/01-latent-label-spec.md
  - projects/MOREBENCH/phase_01/docs/01-confound-audit.md
  - projects/MOREBENCH/phase_01/docs/01-gap-list.md
  - projects/MOREBENCH/phase_02/docs/02-augmentation-plan.md
  - projects/MOREBENCH/phase_02/docs/02-augmented-data-manifest.json
  - projects/MOREBENCH/phase_02/docs/02-behavioral-smoke-report.md
  - projects/MOREBENCH/phase_03/docs/03-execution-targets.md
---

# MoReBench 03 Analysis Plan

## Behavioral-Sanity Gate

Behavioral sanity is satisfied for the current phase-03 plan and pilot execution design on the existing gate model.

- canonical gate artifact: `projects/MOREBENCH/phase_02/docs/02-behavioral-smoke-report.md`
- gate model: `/models/Qwen/Qwen3-30B-A3B`
- smoke scope: `20` augmented prompts across `theory_direct`, `theory_wording_variant`, `neutral_control`, and `action_locus_rewrite`
- tripwire result: `pass`
- substantive labelability result: `pass`
- interpretation: the current augmented slice yields responses that are genuinely annotatable for the planned response-side labels

Important scope rule:

- this gate is strong enough to support phase-03 planning and response-label pilot design without rerunning smoke
- if a new execution model is introduced later, it should inherit the same content-inspection standard before its generations are used for response-side probing

## Phase Goal

Turn the validated MoReBench latent-label set plus repaired phase-02 prompt families into a concrete evidence ladder:

1. establish benchmark-real representational readouts
2. localize where signals emerge in prompt and generation
3. identify which comparisons are worth causal follow-up
4. shape claims so they match the actual evidence we can earn

## Chosen Label Families

### Tier 1: begin immediately

- `objective_pressure_profile`
  reason: benchmark-native, prompt-side, uses the full `500`-row public test split, and avoids the response-side lexical trap that dominated Experiment 2
- `theory_conditioned_generation_persistence`
  reason: keep as a secondary follow-up only if the benchmark returns to theory work later; not the next mainline execution target
- `tradeoff_engagement`
  reason: still scientifically interesting, but no longer the immediate next capture target
- `commitment_style`
  reason: same as above
- `helpfulness_invoked` and `harm_avoidance_invoked`
  reason: still relevant for later response-side work, but the next step is to test the prompt-side objective profile first

### Tier 2: begin after one more data-quality gate

- `theory_identity`
  reason: keep as a prompt-side diagnostic track only if an alias-based family clears the prompt-side retry gate against stronger held-out text baselines
- `stakeholder_tradeoff_density`
  reason: likely the cleanest prompt-side benchmark-native feature, but still needs a validated gold slice

### Tier 3: defer or narrow

- `action_locus`
  reason: scientifically important, but only partially repaired; current 10-pair rewrite set is suitable for pilot readout, not broad claims

## Methodology By Label Family

### `theory_identity`

- first-pass readout:
  linear probes and difference-in-means directions on alias-only prompt states
- stronger follow-up:
  held-out alias-bank transfer plus comparison against the strongest held-out alias text baseline
- localization:
  token/section sweeps centered on the alias clause and late prompt summary states
- strongest realistic claim if successful:
  prompt-side diagnostic evidence that framework-conditioned aliases survive beyond one fixed surface form; not a standalone mechanistic target

### `objective_pressure_profile`

- first-pass readout:
  prompt-final linear probes and difference-in-means directions on the full `morebench_public/test` split
- stronger follow-up:
  source-family-aware and context-aware holdouts plus comparison against the lexical gate baselines on the exact same split
- localization:
  final prompt token first, then prompt span comparison only if the prompt-final readout survives the lexical gate
- strongest realistic claim if successful:
  prompt-side representational claim that the model encodes a rubric-derived helpfulness-vs-harm pressure profile before generation

### `theory_conditioned_generation_persistence`

- first-pass readout:
  paired-generation comparisons plus generation-time probes on generated tokens only
- stronger follow-up:
  same-dilemma cross-prime comparisons using description-only versus generic-ethics controls, plus generated-text baselines and explicit theory-copying checks
- localization:
  generated reasoning tokens, with explicit exclusion or flagging of direct theory-name copying when present
- strongest realistic claim if successful:
  localized representational claim that theory conditioning persists into generated reasoning beyond the prompt cue itself

### `stakeholder_tradeoff_density`

- first-pass readout:
  linear probes and geometry checks on prompt-end states
- stronger follow-up:
  transfer across source families and dilemma structures after the gold slice is validated
- localization:
  section-aware sweeps over dilemma spans containing stakeholder and consequence clusters
- strongest realistic claim if successful:
  representational claim that the model tracks multi-consideration load, potentially localized to stakeholder-bearing prompt spans

### `tradeoff_engagement`

- first-pass readout:
  response-label pilot plus generation-time probes over response tokens
- stronger follow-up:
  position-conditioned readout over early, middle, and conclusion segments with conclusion-span baselines explicitly beaten
- localization:
  token-position sweeps over generated reasoning rather than prompt-only states
- strongest realistic claim if successful:
  representational or localized representational claim about maintained multi-consideration reasoning during generation

### `commitment_style`

- first-pass readout:
  response-label pilot plus generation-time probes on response tokens
- stronger follow-up:
  transition analysis from deliberative tokens to final recommendation tokens
- localization:
  late-generation token windows and response-section comparison
- strongest realistic claim if successful:
  localized representational claim about commitment transition timing, not yet a mechanism claim

### `helpfulness_invoked` and `harm_avoidance_invoked`

- first-pass readout:
  separate probes, not a single merged balance probe
- stronger follow-up:
  transfer matrix and residualized readout to test whether one survives after controlling for the other
- localization:
  late-generation tokens, especially recommendation and caution spans
- strongest realistic claim if successful:
  representational claim that helpfulness- and harm-avoidance-oriented policies are partly separable in the response state

## Evidence Ladder

### Level 1: Behavioral

- satisfied on the current gate model and augmented slice
- any additional model used for response-side work should pass the same labelability standard before use

### Level 2: Representational

Planned via:

- linear probes
- difference-in-means directions
- simple geometry checks

Target families:

- `theory_identity` (diagnostic only if the alias family clears the retry gate)
- `theory_conditioned_generation_persistence`
- `stakeholder_tradeoff_density`
- `tradeoff_engagement`
- `commitment_style`
- `helpfulness_invoked`
- `harm_avoidance_invoked`

### Level 3: Localized Representational

Planned via:

- prompt section sweeps
- token-position sweeps
- conclusion-vs-deliberation comparisons
- transfer across direct and wording-variant prompt families

### Level 4: Causal

Not the first move for this benchmark.

Only promote a family to causal follow-up when:

- the label is behaviorally sane
- the readout survives nuisance-aware controls
- localization points to a reasonably small prompt span, token regime, or layer band
- the phase-04 entry criteria artifact is satisfied explicitly

Likeliest candidates for later causal work:

- `theory_conditioned_generation_persistence`
- `commitment_style`
- `helpfulness_invoked` / `harm_avoidance_invoked`

### Level 5: Mechanistic

No phase-03 plan should claim this in advance.

The best realistic outcome of phase 03 is:

- one or two well-supported localized representational targets that deserve mechanistic-interventions follow-up

## Expected Artifacts From Execution

- prompt-side rubric-derived label freeze and lexical-gate report for `objective_pressure_profile`
- prompt-side probe tables and split-robustness readouts for the full-public-split `objective_pressure_profile` experiment
- prompt-side probe tables for `theory_identity` and later `stakeholder_tradeoff_density`
- paired-generation comparisons and generation-time probe tables for `theory_conditioned_generation_persistence`
- response-label pilot and freeze artifacts for `tradeoff_engagement`, `commitment_style`, `helpfulness_invoked`, `harm_avoidance_invoked`, `refuses_or_hedges`, and `uncertainty_and_scope_calibration`
- generation-time probe tables for `tradeoff_engagement`, `commitment_style`, `helpfulness_invoked`, and `harm_avoidance_invoked`
- per-layer and per-position localization maps
- split-robustness comparisons across prompt families
- residualized or transfer evidence for helpful-vs-harm-avoidance separability

## Key Risks

- the rubric-derived `objective_pressure_profile` could still be shortcut-satisfiable from the dilemma text under source-family or context leakage, which is why the lexical gate must run before any probe claim
- explicit cue recoverability remains the main threat to overclaiming in prompt-side `theory_identity`
- generation-time theory work could still collapse into explicit theory-name copying or obvious response-text lexical cues if the generation controls are weak
- generation-time theory work could also collapse into response-length or dilemma-topic effects if same-prime / different-dilemma and length baselines are not included
- `stakeholder_tradeoff_density` remains blocked until the gold slice is validated
- `action_locus` is still only a pilot-ready track
- source/type/length/person-grammar controls remain incomplete, so some families still support only cautious claims
- response-side work must not skip the generate -> annotate -> validate -> freeze gate

## Recommended Phase-03 Order

1. finalize execution targets and per-batch config in `03-execution-targets.md`
2. construct and freeze the full-public-split rubric-derived `objective_pressure_profile` label
3. run the lexical gate for that label on the exact intended split family:
   source-family-aware text baseline plus prompt-length baseline
4. only if the gate is passed, run the prompt-final readout and split-robustness sweep for `objective_pressure_profile`
5. keep prompt-side `theory_identity` in diagnostic mode until an alias family clears the retry gate against stronger held-out text baselines
6. defer fresh response-side generation work until the prompt-side full-public-split experiment is triaged
7. only then decide whether `stakeholder_tradeoff_density`, `action_locus`, prompt-side theory retry, or phase-04 causal follow-up is worth immediate effort
