# Public Conflict Extension Review Packet

This document is the handoff packet for reviewing the new public-dilemma extension added to the Experiment 2 behavioral run.

## What Changed

We are expanding the original theory-generation behavior set from:

- `30` benchmark theory dilemmas x `6` prime conditions = `180`

to:

- existing `30` grouped theory dilemmas x `6` prime conditions = `180`
- plus `60` manually screened public dilemmas x `6` prime conditions = `360`
- combined total = `540`

The purpose of the extension is to increase the share of dilemmas where theory-conditioned behavioral divergence is structurally plausible, without selecting cases using theory-coded language.

## Important Benchmark Clarification

The benchmark theory split does **not** contain `150` unique base dilemmas.

It contains:

- `150` rows total
- but only `30` unique dilemma texts
- each repeated across `5` native benchmark theory labels

That means:

- the old grouped phase-02 artifacts intentionally collapse to `30`
- if we want additional real dilemma diversity, we must add new public-split dilemmas

## Selection Protocol

The new `60` public dilemmas were selected from the public split using a **manual structural screen**, not a theory-coded filter.

Selection principles:

- binary or very small action space
- explicit stakes for multiple parties
- immediate or forced commitment
- real tradeoff with no clean "do both" escape
- no obvious moral-consensus answer
- de-emphasize pure factual-uncertainty cases

The selection explicitly tries to avoid baking the desired theory distinction into the data construction process.

## Files To Inspect First

### Core spec / workflow files

- `/Users/trentelmore/Projects/concordance/xenon/projects/MOREBENCH/phase_03/specs/experiment_02_behavior_broad_workflow.py`
- `/Users/trentelmore/Projects/concordance/xenon/projects/MOREBENCH/phase_03/specs/experiment_02_workflow.py`

### New extension assets

- `/Users/trentelmore/Projects/concordance/xenon/projects/MOREBENCH/phase_03/docs/03-public-conflict-extension-spec.md`
- `/Users/trentelmore/Projects/concordance/xenon/projects/MOREBENCH/phase_03/scripts/build_experiment_02_public_conflict_extension.py`
- `/Users/trentelmore/Projects/concordance/xenon/projects/MOREBENCH/phase_03/outputs/experiment_02_public_conflict_extension.jsonl`
- `/Users/trentelmore/Projects/concordance/xenon/projects/MOREBENCH/phase_03/outputs/experiment_02_public_conflict_extension_summary.json`

### Supporting shortlist / provenance

- `/Users/trentelmore/Projects/concordance/xenon/projects/MOREBENCH/phase_03/reports/public_conflict_candidate_review.md`

### Benchmark background / audits

- `/Users/trentelmore/Projects/concordance/xenon/projects/MOREBENCH/benchmark_context.md`
- `/Users/trentelmore/Projects/concordance/xenon/projects/MOREBENCH/phase_00/outputs/theory_pairing_audit.json`

## Current Extension Summary

From `experiment_02_public_conflict_extension_summary.json`:

- selection count: `60`
- non-overlap public pool size: `482`
- excluded exact theory-overlap count: `18`

Source balance:

- `ai_risk_dilemmas`: `30`
- `daily_dilemmas`: `11`
- `expert_written_ethic_bowl`: `9`
- `expert_written_ethic_unwrapped`: `7`
- `expert_written_literature`: `3`

Role balance:

- `ai_advisor`: `30`
- `ai_agent`: `30`

Type balance:

- `expert_case`: `19`
- `long_case`: `41`

Important current workflow check:

- combined dataset size now verifies to `540`
- `90` unique dilemma groups
- `6` prime conditions per dilemma
- no recurrence of the earlier prompt-duplication collapse

## Why This Was Added

Earlier behavior analysis over the original `30` theory dilemmas suggested:

- many dilemmas collapse to the same recommendation across primes
- some real recommendation differences exist
- but the sample is too small and too concentrated for stable behavioral contrast discovery

The new extension is meant to:

- increase dilemma diversity
- raise the rate of genuine action-level disagreement
- give PCA / behavioral clustering / contrast-set discovery a better substrate
- support later capture on the reduced set of actual conflict cases rather than capturing everything blindly

## What To Review

Please evaluate the extension as a benchmark-construction step, not as a result claim.

Main questions:

1. Is the structural selection protocol sound?
   - Does it avoid theory-coded leakage while still plausibly enriching for behavioral divergence?

2. Does the chosen `60` look confound-prone in any obvious way?
   - Too many AI-governance cases?
   - Too many whistleblowing / privacy / professional-integrity cases?
   - Too much residual factual-uncertainty?
   - Too many cases with socially obvious answers?

3. Is the source/type/role balance good enough?
   - Or does the extension over-index on one source family or one case shape?

4. Is `60` the right scale?
   - Is `30 + 60 = 90` dilemmas enough for the intended behavioral contrast discovery?

5. Is the new combined `540`-prompt workflow actually the right object now?
   - Existing `30` grouped theory dilemmas + `60` public extension
   - each crossed with `5` theory primes + `1` generic control

6. Are there any remaining construction artifacts worth checking before launch?
   - prompt duplication
   - group leakage
   - source overlap with theory split
   - hidden imbalance in role/source/type/context

## Review Style Requested

Please do a hard-nosed construction review.

Distinguish clearly between:

- good benchmark design
- plausible but still noisy enrichment
- selection-induced confound risk
- things that should be fixed before running

Do not treat this as a result.
Treat it as a review of whether this extension is a sound next experimental substrate.

## Bottom-Line Decision To Help With

We want to know whether this new `60`-dilemma public extension is good enough to proceed with the `540`-prompt behavioral run, or whether the extension should be revised before spending more generation/capture budget.
