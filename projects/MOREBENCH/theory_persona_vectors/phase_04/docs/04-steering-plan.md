# Phase 04 Deont/Util Steering Plan

## Purpose

Test whether Phase 03 generated-token theory directions are causally sufficient to shift neutral responses on model-level deontology/utilitarian conflict cases.

This phase is not a monitor/readout claim. It is a small causal sufficiency test:

> If we add a theory direction during generation, does the neutral response move toward that theory's stable endpoint action more often than controls?

## Denominator

Use the fixed trial manifest:

`projects/MOREBENCH/theory_persona_vectors/phase_04/outputs/steering_trial_manifest.json`

Primary denominator:

- `5` endpoint-stable conflict groups.
- `5` prompt variants per group.
- `25` paired trials.

Sensitivity denominator:

- Add `3` endpoint-stable-at-4/5 groups.
- `40` paired trials total.

The causal unit is exact `group_id + variant_id`, not just dilemma. Each trial compares:

- unsteered neutral baseline for that exact prompt variant
- deont-steered neutral generation for that exact prompt variant
- util-steered neutral generation for that exact prompt variant
- controls for that exact prompt variant

Neutral is allowed to vary across paraphrases. The question is whether steering moves the paired neutral baseline toward the intended endpoint action.

## Direction Source

Primary direction source:

- Phase 03 brief-recommendation synthetic capture.
- L32 `generated_sequence_residual`.
- `first_16` generated-token slice.
- Paired deltas: `P_deont_01 - N_neutral_01` and `P_util_01 - N_neutral_01`.

Rationale:

- L32 generated first-16 showed strong paired gaps.
- Prompt-final L32 was weak, so prompt-final should not be the primary write site for this steering attempt.
- First-16 is less contaminated by late-response lexical accumulation than full response mean.

Candidate write site:

- first pass: add direction at L32 over generated tokens during neutral generation.
- if weak but nonzero: sweep L24/L32/L40 as write layers.
- do not multi-layer patch until single-layer results are interpretable.

## Patch Operator

First pass:

- `AddDirectionPatch`
- directions: `deont_minus_neutral`, `util_minus_neutral`
- magnitudes: `0.5x`, `1.0x`, `2.0x`

Report magnitudes separately. Do not collapse into best-of-magnitude without showing all three.

## Controls

Mandatory controls:

- Unsteered neutral baseline for each `group_id + variant_id`.
- Random matched-norm direction control at the same magnitudes.
- Generic moral direction control if available: `N_generic_moral_01 - N_neutral_01`.

Recommended controls:

- Opposite-direction check: deont direction should not shift toward util endpoint as much as util direction does, and vice versa.
- Same-endpoint cases: if included later, steering should mostly shift vocabulary/style rather than action when deont and util endpoints already converge.

## Success Criteria

Primary headline should be count-based because `N=25` is still small.

Minimal pass:

- Intended action shifts in at least `8/25` primary trials for a theory direction at some pre-declared magnitude.
- Random control shifts in at most `2/25` primary trials at matched magnitude.
- Direction effect is stronger than generic moral control.
- Dose response is monotonic or at least not sharply non-monotonic across `0.5x`, `1.0x`, `2.0x`.

Strong pass:

- Intended action shifts in at least `10/25`.
- Random control at `<=2/25`.
- Sensitivity denominator gives the same qualitative verdict.

Failure / stop:

- Random control produces shifts within ~50% of theory direction shifts.
- Steering shifts theory vocabulary but not action.
- Only one fragile dilemma family accounts for nearly all flips.
- Effect appears only at `2.0x` with malformed or degraded generations.

## Reporting

Report, in this order:

1. Primary `25` trial result.
2. Random/generic controls.
3. Magnitude sweep.
4. Sensitivity `40` trial result.
5. Per-group flip table.
6. Qualitative examples.

Use cautious language:

- Good: "the direction is causally sufficient to shift a subset of neutral responses toward the deont/util endpoint."
- Bad: "the model has a moral theory module."

## Open Implementation Note

The patching API supports `PatchedGenerationSpec` and `AddDirectionPatch`, but the direction must be available as a `direction_result` artifact. The next implementation step is to export the Phase 03 L32 first-16 deont/util directions as a transform artifact, then feed that artifact into the steering workflow.
