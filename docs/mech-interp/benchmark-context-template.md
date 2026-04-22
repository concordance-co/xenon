# Benchmark Context Template

**Version:** `v1`
**Freeze Date:** `YYYY-MM-DD`

Canonical shared rules live in:
- [mech-interp-principles.md](/Users/trentelmore/Projects/concordance/xenon/docs/mech-interp/mech-interp-principles.md)

Use this template for benchmark-specific sidecars such as `morebench-context.md`.

## Purpose

This file stores `benchmark-specific frozen knowledge`.

It is not a general technique skill.
It is not a cross-benchmark principles doc.

It should capture:

- what this benchmark is
- what latent labels we extracted
- what confounds we know about
- what methods look promising
- what failed
- what remains open

## Suggested sections

### 1. Benchmark Snapshot

- name
- links
- scale
- splits/configs
- raw prompt/response availability
- notes on access/runnability
- scope convention for factual claims
  for example: "full public split", "100-row stratified sample", etc.

### 2. Why It Matters

- product relevance
- likely mechanistic question richness
- benchmark-specific reason it is interesting

### 3. Native Label Surfaces

- native fields
- rubric dimensions
- metadata
- trajectory structure if any

### 4. Refined Latent Label Spec

- prompt-side labels
- response-side labels
- objective-orientation labels
- nuisance variables
- which are direct / derived / validation-only

### 5. Known Confounds

- source/template aliasing
- role-token leakage
- length imbalance
- domain imbalance
- benchmark-specific quirks

### 6. Behavioral Sanity Notes

- what has been checked
- what still needs checking
- known model-fit issues

### 6.1 Benchmark-Specific Gotchas

- README mismatches
- field-name mismatches
- loader path quirks
- viewer vs API differences
- operational hazards that are not exactly confounds

### 7. Strong Candidate Feature Hypotheses

- label family
- candidate internal feature
- why it seems plausible

### 8. Methods That Look Promising

- readout methods
- localization strategies
- intervention candidates

### 9. Methods Or Hypotheses To Be Careful About

- weakly supported labels
- known confounded comparisons
- methods likely to overclaim

### 10. Data Gap List

- missing matched pairs
- missing response generations
- missing counterbalanced subsets
- missing theory or framing variants

### 11. Open Questions

- what still seems unresolved
- what would justify revisiting deferred labels
