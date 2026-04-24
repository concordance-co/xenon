---
benchmark: morebench
phase: 03
experiment: 02
frozen_date: 2026-04-24
status: internal
---

# Cross-Language Prompt Probe Controls

This note freezes the intended interpretation gates for the `deontology` vs `virtue_ethics` cross-language prompt-side result and records the immediate follow-up controls.

## Current Canonical Result

- canonical prompt-final capture artifact:
  `capture_1_2c011b403d39`
- canonical report:
  `projects/MOREBENCH/phase_03/reports/experiment_02_cross_language_prompt_probe_full/report.md`

## Frozen Claim Thresholds

For the full `30`-dilemma prompt-side cross-language run, use these tiers:

- strong representational reopening:
  - best-layer probe-minus-text delta `>= 0.25`
  - grouped bootstrap CI for the delta excludes `0`
  - emergence shape preserved:
    - `L0 <= 0.60`
    - monotonic rise through at least `L16`
  - `en->zh` and `zh->en` both `>= 0.80` at the best layer
- moderate / suggestive reopening:
  - best-layer delta `>= 0.15`
  - grouped bootstrap CI excludes `0`
  - best-layer AUROC `>= 0.75`
- not promotable:
  - delta `< 0.15`
  - or grouped bootstrap CI crosses `0`
  - or the emergence shape flattens early
  - or one of the cross-script ordered pairs drops below `0.70`

## Specificity Control Note

The original proposed next control was a neutral-tag ablation replacing framework names with `Framework A` / `Framework B`.

For the current prompt-side full-30 asset, that control is largely satisfied by construction:

- the prompt family is description-only rather than name-only
- prompt-text audit finds `0` occurrences of:
  - `deontology`
  - `Kantian`
  - `virtue ethics`
  - `Aristotelian`
  - target-language equivalents such as `康德`, `亚里士多德`, `义务论`, `德性伦理`

So the remaining specificity question is not "is the probe reading framework names?" but rather:

- is the prompt-final probe tracking framework-description content in a stance-specific way?
- how similar is that prompt-side direction to the old response-side readout directions?
- where exactly does the prompt-side direction become linearly clean?

## Immediate Follow-Up Controls

1. Prompt-text residue / name audit on the full-30 asset.
2. Cosine comparison between the prompt-side direction and old response-side `description_only` directions.
3. Dense prompt-side layer sweep around the `L24` dip (`L20`, `L22`, `L26`, `L28`) plus anchor layers.
4. Reproducibility note documenting that the canonical prompt-final full-30 result came from the recovered capture-only workflow because the original mixed local-transform / GPU-capture workflow failed at the capture handoff.

## Reproducibility Note

The original full workflow `morebench_phase03_experiment02_cross_language_prompt_probe_full` failed because the GPU capture step could not consume the local transform artifact directly.

The canonical capture for the defended prompt-side full-30 result is therefore:

- workflow:
  `morebench_phase03_experiment02_cross_language_prompt_probe_full_capture`
- capture artifact:
  `capture_1_2c011b403d39`

Any future reproduction should either:

- inline the serialized dataset for the GPU capture step, or
- ensure the artifact handoff path is GPU-readable before launching the run.
