---
name: latent-label-data-augmentation
description: Use when a benchmark cannot support the desired latent labels cleanly and needs rewrites, matched pairs, counterbalancing, response generations, or synthetic augmentation. Covers benchmark repair for confounds, split construction, framing variants, and contrast-set design for mechanistic interpretability.
---

# Latent Label Data Augmentation

Use this skill when the user wants to:

- repair a benchmark so refined latent labels become usable
- generate matched pairs or minimal contrasts
- create counterbalanced rewrites
- collect target-model responses for response-side labels
- remove or reduce confounds in a benchmark slice
- expand a benchmark with synthetic or semi-synthetic support data

This skill is the **data augmentation and benchmark repair** phase of the benchmark-flow pipeline.

It is broader than "synthetic data generation."
Often the right move is:

- rewriting
- pairing
- counterbalancing
- reframing

not whole-cloth synthesis.

## Core rule

`Augment to break surface-label leakage so the experiment can become real.`

The job is to produce data that makes the latent-label question cleaner:

- fewer shortcuts
- clearer contrasts
- better controls
- better slice balance
- cleaner response-side supervision

Use the gap list to identify which leaks matter most.
Do not treat augmentation as a last-resort cleanup step after phase 03; imported benchmark labels should be assumed leaky until phase 02 has tried to break the most plausible shortcut channels.

## Start from the gap list

Do not begin augmentation from vague dissatisfaction.

This skill should usually consume a gap list from:

- [benchmark-to-latent-labels](../benchmark-to-latent-labels/SKILL.md)

Typical gaps:

- no matched pairs for causal questions
- response-side labels need new generations
- theory or framing overlays are unpaired
- target label is too correlated with source or length
- a good label exists only in a tiny subset

## Preferred repair moves

### 0. Shortcut stress test by default

Every target label should get at least one explicit anti-shortcut repair design.

Do not wait until a probe fails in phase 03 to ask how the label might leak through:

- explicit name tokens
- fixed anchor sentences
- stable descriptive clauses
- source-family wrappers
- label-specific keywords
- carrier or template artifacts

The default phase-02 question is:

- what is the cheapest plausible shortcut for this label?
- what repair family would break that shortcut while preserving the semantic content?

### 1. Rewrite before inventing

Prefer:

- prompt rewrites
- carrier rewrites
- matched framing variants
- length-normalized versions
- source-balanced or domain-balanced subsets

before:

- fully new synthetic dilemmas

Why:

- lower artifact risk
- closer to original benchmark semantics
- easier to justify in writeups

### 2. Build matched pairs when the question is causal

If the downstream question is causal, try to create:

- matched donor-target pairs
- same-label controls
- cross-label contrasts with nuisance dimensions held fixed

Matched pairs and controls must preserve everything except the target variable.

`Preserve` means at the level of:

- scenario content
- stakes
- action alternatives
- prompt skeleton when the repair is a wrapper or framing control

A prefix-only rewrite is not a pair.

### 3. Generate model responses when labels are response-side

If the label lives in:

- deliberative process
- commitment
- hedging
- explicit objective invocation

then benchmark text alone is not enough.
Generate responses under the intended protocol and label those.

### 4. Preserve the real decision bottleneck

Simplify toward the core computation, not toward convenience.

The augmented data should still force the intended reasoning problem.

### 5. Add shortcut-stress-test families for explicit cues

If the target label is exposed through:

- explicit names
- fixed anchor sentences
- lexically stable descriptions
- template-locked wrapper clauses

then phase 02 should materialize a `shortcut_stress_test` repair family before calling that track clean.

Useful repair patterns include:

- name-only vs description-only vs name-plus-description factorials
- held-out aliases
- paraphrase banks
- shared-vocabulary descriptions
- position-counterbalanced cue placement
- decoy or mismatch controls

The goal is not just "more variants."
The goal is to break one-to-one surface recoverability.

### Validate variant equivalence before treating a variant pair as a holdout

Constructing variants is not the same as having a working holdout. A variant pair earns holdout status only when a within-variant text classifier on the responses they produce lands near chance.

Required validation step:

- train a text classifier (e.g., char TF-IDF 3–5 + ridge) to distinguish the responses produced under variant A from the responses produced under variant B
- record balanced accuracy and AUROC
- a within-variant text classifier at ceiling (AUROC ≥ ~0.95 or BA near 1.0) means the variants are not lexically equivalent and are not functioning as a holdout — the pair is two different prompts with the same target label, and any downstream activation classifier on it remains confounded by the lexical signature the text classifier exploits

Repair if the validation fails:

- add output-schema constraints (e.g., fixed three-line structure)
- add length bounds
- add vocabulary bans on the canonical lexical family of each variant
- iterate until the within-variant text classifier lands near chance (BA ≤ ~0.65, AUROC ≤ ~0.75)

A track that ships variant pairs without running this validation has gestured at the technique rather than executing it. Mark such tracks as `shortcut_stress_unvalidated` in the manifest until validation passes.

This was empirically motivated by the MOREBENCH/theory_persona_vectors phase_03 deont-prompt-isolation experiment: natural-prompt within-variant text classifier landed at AUROC 0.998, and the variants were not functioning as holdouts despite being labeled as such. Detected only when the validation was finally run.

## Confound-focused design checklist

Always consider:

- lexical leakage
- template aliasing
- source-family confounds
- length imbalance
- role-token leakage
- domain/topic leakage
- annotator-family leakage
- theory/style leakage
- competence vs target-variable confounds

If possible, construct data that lets these be:

- balanced
- stratified
- held out
- explicitly contrasted

For targets vulnerable to lexical or semantic leakage, phase 02 should also run a prompt-side preflight against cheap baselines before handing the family to phase 03.

Typical preflight baselines:

- bag-of-words on the full prompt
- bag-of-words on the cue clause only
- name-token-only or alias-token-only baselines
- rare-word / keyword baselines
- clause-position-only baseline when cue placement varies

If these baselines still solve the target family cleanly, phase 02 is not done for that track.

## Output hygiene rules

Materialized artifacts in `outputs/` must be fully instantiated.

Do not place literal placeholder tokens such as:

- `<FRAMEWORK>`
- `<THEORY>`
- `<LABEL>`

inside materialized JSON, JSONL, CSV, or parquet outputs.

If templates with holes are useful, store them under `specs/` or `specs/templates/`.

## Required outputs

Produce:

- augmented dataset or benchmark slice
- description of what was rewritten, generated, or paired
- rationale for each repair move
- expected confounds reduced
- residual confounds
- mapping back to the original gap list
- current phase status:
  - `scaffold_only`
  - `partial_repair_materialized`
  - `repair_complete_for_track`
  - `phase_complete`

## Required artifacts

At minimum, leave behind:

- `projects/<BENCHMARK_PROJECT>/phase_02/docs/02-augmentation-plan.md`
  what is being repaired and why
- `projects/<BENCHMARK_PROJECT>/phase_02/docs/02-gap-list-resolution.md`
  direct mapping from each gap-list item to the chosen repair move
- `projects/<BENCHMARK_PROJECT>/phase_02/docs/02-augmented-data-manifest.json`
  where rewritten, paired, generated, or counterbalanced data now lives
- `projects/<BENCHMARK_PROJECT>/phase_02/docs/02-augmentation-report.md`
  what was changed, what improved, and what residual confounds remain
- `projects/<BENCHMARK_PROJECT>/phase_02/docs/02-behavioral-smoke-report.md`
  small-slice behavioral sanity check on the augmented data before phase 03

If a phase materializes prompts or responses, also leave behind:

- `projects/<BENCHMARK_PROJECT>/phase_02/docs/02-generation-protocol.md`
  prompt / rewrite / generation rules used for the materialized data

If new rows or slices were created, also leave behind:

- dataset location(s)
- pairing or rewrite rules used

If the smoke artifact is used to justify downstream response-side labeling or probing, it should also make explicit:

- which proposed downstream labels were checked for labelability
- whether the smoke result reflects only tripwire checks, substantive content inspection, or both
- at least a few concrete generated responses from the checked slice
- the actual failure categories used during inspection

Use simple frontmatter on markdown artifacts:

- `benchmark`
- `phase`
- `version`
- `frozen_date`
- `input_artifacts`

Someone inspecting this phase should be able to answer:

- what defect in the original benchmark was being repaired
- what data was added or changed
- whether the repair actually reduced the intended confound

The augmented-data manifest should make quality visible, not just existence.
At minimum, each materialized dataset entry should record:

- `row_count`
- `rows_with_all_placeholders_substituted`
- `controls_structurally_matched_to_target` when applicable
- `known_bugs`

If a materialized family is later shown to be shortcut-dominated, keep it in the manifest and record that explicitly in `known_bugs` rather than silently treating it as clean.

If a confound-repair matrix marks a track as `ready_now`, the phase should either materialize that track or explicitly downgrade it with a reason.

## Phase Done Criteria

This phase is done when:

- all `ready_now` repair tracks are materialized or explicitly downgraded with a reason
- materialized `outputs/` contain no unsubstituted placeholders
- matched pairs and controls satisfy the structural-match rule for the intended target variable
- a behavioral smoke report exists on a small augmented slice and is non-empty
- the materialized data beats the mandatory cheap-baseline preflight for the active target family, or the phase is explicitly marked not done for that family
- the manifest, augmentation report, and benchmark sidecar state the true phase status and residual confounds honestly
- any explicit-cue track either beats its cheap prompt-side preflight baselines or is explicitly marked shortcut-dominated and kept out of phase-03-ready status

The behavioral smoke report should distinguish:

- `tripwire checks`
  examples: nonempty output, parseability, no truncation, no refusal, recommendation cue present
- `substantive labelability checks`
  examples: whether the generated responses can actually support the proposed downstream labels

If the smoke is intended to support response-side label work, a pure tripwire pass is not enough by itself.
The report should include at least one explicit content-inspection section tied to the proposed labels.

## References

Primary references:

- [methodology/PRINCIPLES.md](../../../methodology/PRINCIPLES.md)
- [synthetic-data-generation](../synthetic-data-generation/SKILL.md)
- [benchmark-to-latent-labels](../benchmark-to-latent-labels/SKILL.md)
- [activation-patching-causal-evals](../activation-patching-causal-evals/SKILL.md)

Use the synthetic-data-generation skill as the main methodology library for:

- lexical controls
- benchmark repair
- split schemes
- prompt role placement
- behavior-validating smoke tests

## Handoff

Once the data is repaired enough to support the latent labels:

- hand back to [benchmark-to-latent-labels](../benchmark-to-latent-labels/SKILL.md) if the ontology needs refreezing
- or hand forward to [benchmark-mech-interp-analysis](../benchmark-mech-interp-analysis/SKILL.md) if the label spec is already stable
- update the benchmark sidecar with:
  - what augmentation was performed
  - which confounds were reduced
  - what residual gaps remain
