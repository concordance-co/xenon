---
frozen_date: 2026-04-24
benchmark: MoReBench
phase: phase_03
status: frozen_before_process_probe_runs
---

# MoReBench Process-Probe Precommitment

## Pivot

This experiment pivots from grader-designed outcome supervision to latent-oriented
process supervision. The target is not whether a response is simply helpful or
harmless, but whether model activations encode reasoning structure from the
rubric-rich benchmark: which considerations are covered, where commitment happens,
and whether the response sustains multiple live considerations before concluding.

This follows the methodology roster's rubric-rich benchmark recommendation:
multi-label probe families plus claim decomposition for long-form outputs.

## Label Source

Labels must be produced by non-Qwen semantic judges. Qwen/Qwen3-30B-A3B is the
target model and must not be used as the process-label judge.

The annotation pass uses Codex subagents as independent non-Qwen readers. The
target model's original generations are fixed at:

- generation artifact: `generation_run_1_d6e12a467208`
- capture artifact: `capture_1_f2a9e4531dec`
- captured layers: `0, 4, 8, 16, 28, 36, 40, 44`

## Registered Tracks

F.1 and F.2 are separate registered claims. Results must not be reported as
"the better of F.1/F.2"; both tracks are reported whether positive or negative.

- `F.1`: prompt-final residual -> criterion-family coverage vector.
- `F.2`: claim-span activation -> criterion-family label.
- `B`: commitment-span activation vs matched mid-reasoning activation.
- `C`: early-collapse vs sustained-multi-consideration, after train-fold linear
  residualization against length/count features.

## Labelability Gates

Probe runs are blocked for any track that fails its labelability gate on the
30-row reviewer audit.

- `F`: criterion-family coverage must achieve Cohen's kappa >= `0.60`.
- `B`: commitment span must achieve span-IoU >= `0.50` on at least `80%` of
  reviewed rows with a commitment.
- `C`: early-collapse/sustained label must achieve Cohen's kappa >= `0.60`.

## Criterion Family Freeze

Raw rubric criteria are too sparse to probe directly. Criterion families are the
unit of F probing. Families are frozen before row-level labels are merged and
before probes run.

Family construction rules:

- Families are semantic buckets with stable `family_id` values.
- Families must not be merged, split, renamed, dropped, or reweighted based on
  probe performance.
- Each raw criterion may map to one primary family.
- Families must record representative raw rubric titles and support counts.
- F probes only use families with at least `30` applicable examples and at least
  `30` positive and `30` negative labels after annotation.

## Controls And Thresholds

All positive claims require source-family leave-one-out generalization. Source
family LOO must stay within `0.05` BA/AUROC of random CV for the same metric.

F controls:

- char-TFIDF baseline on prompt text for F.1
- char-TFIDF baseline on claim text for F.2
- response-length baseline
- criterion-frequency majority baseline
- shuffled-label null with `50` permutations

F positive threshold:

- mean eligible-family AUROC >= `0.70`
- at least `30%` of eligible families above AUROC `0.75`
- mean lexical baseline AUROC <= `0.65`
- shuffled-label null p95 below the positive threshold
- per-family hits are exploratory unless they survive BH-FDR over shuffled-label
  p-values; the mean AUROC threshold is the primary family-level claim.

B controls:

- same-position noncommitment null
- char-TFIDF viewport baseline on the same token/claim window
- L0-L4 are diagnostics only

B positive threshold:

- primary layer is `16`
- commitment-vs-mid AUROC >= `0.75` at layer `16`
- same-position null AUROC <= `0.60` at layer `16`
- layers `8, 28, 36, 40, 44` are exploratory and cannot replace the primary
  layer claim.

B same-position null construction:

- For each response, use a noncommitment span sampled from the reasoning body.
- The reasoning body is text before the commitment span, excluding headings and
  bullet markers when possible.
- Match the commitment span's relative position within the response as closely
  as possible while staying outside the commitment span and any post-commitment
  recommendation text.

C controls:

- length-only baseline
- train-fold linear residualization of activations against response char count,
  generated token count, precommitment token count, and claim count

C positive threshold:

- post-residualization AUROC >= `0.70`
- if below `0.60`, mark the construct as length-confounded/null and stop.

## Exit Rule

Any track passing labelability, controls, thresholds, and source-family transfer
supports a Level 2 representational claim for that track.

If all tracks fail under controls, close MoReBench response-side with the
methodology-aligned negative: MoReBench's rubric-rich design did not produce
mechanistically probeable response-side process representations for this model
at this scale.
