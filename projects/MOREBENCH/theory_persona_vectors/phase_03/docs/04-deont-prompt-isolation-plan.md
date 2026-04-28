---
project: MOREBENCH
subproject: theory_persona_vectors
phase: 03
artifact: deont_prompt_isolation_plan
date: 2026-04-27
status: draft
---

# Deontology Prompt Isolation Retry

## Why This Exists

The first deontology transfer diagnostic was useful, but it did not isolate a
single confound cleanly enough for the strongest claim we wanted.

What worked:

- activation transfer from `P_deont_01` to `P_deont_02` was strong
- transfer survived cheap suppression of the primary deont lexical family

What did not work:

- response-text transfer also remained strong
- free-form brief recommendations let prompt wording spill into overall style,
  framing, and lexical choice

This loopback is a prompt-design repair, not a new theory claim.

## Design Repair

Use a deontology-only mini-regime with:

- `P_deont_iso_01`
- `P_deont_iso_02`
- `N_generic_moral_iso_01`
- `N_neutral_iso_01`
- `N_anti_deont_iso_01`

Shared controls:

- fixed three-line output schema:
  `Recommendation`, `Reason`, `Caveat`
- target total length of 45-60 words
- plain-language instruction
- explicit ban on both deont lexical families in the response text

The intended effect is:

- keep the prompt-level lexical difference between `01` and `02`
- reduce response-level lexical echo
- reduce style drift unrelated to the targeted cue family

## Success Criteria

Behavioral success:

- the model follows the three-line structure for most rows
- banned cue words rarely appear in outputs
- `P_deont_iso_01` and `P_deont_iso_02` remain distinct as prompt conditions

Confound-isolation success:

- direct text classification for `01 vs 02` drops materially from the free-form
  regime
- transfer-style text baselines from `01 -> 02` weaken relative to the current
  free-form setup
- activation transfer from `01 -> 02` remains clearly above chance

Suggested initial bar:

- `01 vs 02` text BA below roughly `0.65`, or at least far below the current
  free-form result
- activation transfer still robust on generated early spans

## Failure Condition

If the controlled regime still yields strong text transfer and strong direct
`01 vs 02` text separability, we should treat this strategy as hard to realize
for deontology in natural language and not keep pushing it in this line.

## Recommended Next Run

1. behavior-only smoke on 8 dilemmas
2. inspect format compliance and banned-word leakage
3. if behavior is clean, run full capture
4. rerun the same activation-transfer and text-transfer diagnostics on the new
   report
