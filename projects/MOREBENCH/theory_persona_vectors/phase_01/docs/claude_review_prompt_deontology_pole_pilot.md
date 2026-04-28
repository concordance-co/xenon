# Claude Review Prompt: Deontology Persona-Vector Pole Pilot

Please review the new MoReBench theory-persona pole pilot before we generate model responses or activations.

## Files To Inspect

- `/Users/trentelmore/Projects/concordance/xenon/projects/MOREBENCH/theory_persona_vectors/phase_01/docs/01-deontology-pole-pilot-plan.md`
- `/Users/trentelmore/Projects/concordance/xenon/projects/MOREBENCH/theory_persona_vectors/phase_01/specs/deontology_pole_pilot_prompt_conditions.json`
- `/Users/trentelmore/Projects/concordance/xenon/projects/MOREBENCH/theory_persona_vectors/phase_01/outputs/deontology_pole_pilot_synth_dilemmas.jsonl`
- `/Users/trentelmore/Projects/concordance/xenon/projects/MOREBENCH/theory_persona_vectors/phase_01/outputs/deontology_pole_pilot_manifest.json`

## Context

We are returning to MoReBench theory work using a persona-vector strategy inspired by arXiv:2507.21509.

The conceptual shift is:

- Do not learn theory vectors directly from MoReBench.
- Instead, make a separate synthetic corpus where the model acts from a moral-theory stance.
- Extract theory-persona directions from that synthetic corpus.
- Treat MoReBench as held-out real-data return.

This first phase is intentionally small. It focuses only on deontology and only on the pole-construction problem.

## What Was Created

We created:

- `30` new compact synthetic dilemmas, manually written.
- `6` prompt conditions:
  - `P_deont_01`: primary deontological framing.
  - `P_deont_02`: positive deontological variant.
  - `N_neutral_01`: default recommendation-only, no framework mention.
  - `N_neutral_02`: length-matched neutral recommendation prompt, no framework mention.
  - `N_anti_01`: explicit anti-deontology diagnostic.
  - `N_alt_util_01`: utilitarian theory-vs-theory diagnostic.

Planned first run:

- `30` dilemmas x `6` conditions x `3` samples = `540` generations.
- Primary capture locus: L32 prompt-final / assistant-colon residual.
- Diagnostic locus: L32 response-mean residual.

## Design Intent

The primary direction should be:

`deont_neutral_length_matched = mean(P_deont) - mean(N_neutral_02)`

This matches the persona-vector style: framework-primed behavior minus default behavior, with a length-matched neutral instruction to reduce prompt-length confounding.

The other directions are diagnostics:

- `deont_neutral_short = mean(P_deont) - mean(N_neutral_01)`
- `deont_anti = mean(P_deont) - mean(N_anti)`
- `deont_util = mean(P_deont) - mean(N_alt_util)`

We do not want to accidentally scale a negation vector or a pair-specific deont-vs-util vector if the neutral persona-vector construction fails.

## Please Review

Focus on the following questions.

1. Are the 30 dilemmas good synthetic extraction material?

Check whether they are:

- short enough
- morally contested enough
- not copied from MoReBench
- MoReBench-shaped enough for transfer
- not dominated by one obviously correct answer
- not accidentally filled with deontological anchor words
- not too topic-skewed toward one source-family style

2. Are any dilemmas too easy or too one-sided?

Please flag specific `dilemma_id`s that should be removed or rewritten.

For each flagged row, explain:

- why it is problematic
- whether it should be dropped or rewritten
- suggested replacement wording if rewrite is easy

3. Are any dilemmas too deontology-coded?

We tried to avoid explicit theory terms, but some scenarios naturally involve policy, prior agreements, truthfulness, confidentiality, or boundaries. Those may be legitimate dilemma content, but they can become lexical deontology cues.

Please identify rows where the dilemma itself may strongly cue the deontological answer before any theory prompt is added.

4. Are the domains balanced enough for a pole pilot?

This does not need to be a perfect benchmark. But it should not be so skewed that the direction becomes "institutional/professional rule conflict" rather than deontology.

Please suggest whether we should rebalance before generation.

5. Are the prompt conditions right?

Specifically:

- Is `N_neutral_01` too short relative to `P_deont_01`, or is that acceptable?
- Is `N_neutral_02` a good length-matched neutral pole, or does it introduce its own content confound?
- Is `N_anti_01` useful as a diagnostic, or too contaminated by negation?
- Is `N_alt_util_01` the right first alternative theory pole?
- Is `P_deont_02` useful, or does it introduce a different construct from `P_deont_01`?

6. Are the planned smoke criteria sufficient?

Review the proposed checks:

- split-half direction cosine
- positive-prompt variant cosine
- random-label null
- pole-construction cosine matrix
- short-neutral vs length-matched-neutral cosine
- transfer to existing MoReBench theory captures
- negation/generic-framework/pair-specificity confound checks

Please identify missing controls or thresholds that should be frozen before generation.

7. Should behavior divergence be a hard gate?

Current plan treats action divergence as diagnostic, not a hard blocker. The user is not very concerned with behavior differences at this stage because:

- we already have some diverging behavior to test on
- temperature can be raised later
- the first question is whether the representation transfers

Please say whether you agree, and if not, what minimum behavior sanity check should be required.

## Desired Output

Please return:

1. Overall verdict: ready to generate / needs small edits / needs major redesign.
2. Specific row-level edits for the synthetic dilemmas.
3. Prompt-condition edits, if any.
4. Any missing pre-generation checks.
5. Whether this is a good minimal pole-construction pilot.
