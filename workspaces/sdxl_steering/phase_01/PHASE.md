## Premise

Test whether a paired warm-minus-cool activation direction shifts an image's
background-color proxy more than a norm-matched random direction. Eight discovery
pairs train one channel vector; eight different object/prompt families and two
seeds each form the held-out set. The frozen executable design is `../study.py`.

Use the conditional output of `mid_block.resnets.1` at denoising step 10. The
coarse 32×32 map is a plausible candidate for broad spatial color structure;
this is an exploratory site hypothesis, not a probe-derived localization claim.
Read discovery activations over the outer region; write to four disjoint outer
rectangles, leaving the central half-width/half-height rectangle untouched at
the edit point. Downstream changes can still spread to the foreground.

Normalize the paired mean direction to the discovery activation RMS. Freeze
strengths 0.5, 1 and 2, plus baseline, zero-strength and three norm-matched random
controls. Each prompt/seed reuses identical initial noise across all eight arms.
Strength 1 is the primary comparison; other strengths are descriptive, not a
post-hoc opportunity to select a winner.

The primary measure is mean sRGB R−B outside the central crop, a background-color
proxy without semantic segmentation. Report paired changes versus baseline and
versus the matched random control, averaging seeds within prompt family before
bootstrap uncertainty. Positive lower confidence bound at primary strength
supports this bounded proxy claim; null or reverse effects remain valid results.
Also report central RGB change, edge energy, clipping and near-black/white
fractions. These are collateral-change diagnostics, not perceptual quality or
prompt-adherence scores. No-op must match the existing numerical envelope.

No held-out retuning, substituted prompts or selected-gallery presentation.
Nonfinite/invalid images are failures, retained in the planned coverage table.
Missing human ratings remain missing; export a blinded review sheet.
Budget: 16 discovery images, 128 held-out images, one worker, at most two GPU-hours.

## What we ran

## Primary result

## Qualitative inspection

## Corrections

## Running hypothesis

## Claim boundary

## Artifacts

- `../study.py`: frozen prompts, seeds, site, arms and budget.
- `../metrics.py`: direction fit and primary/diagnostic measures.
- `../../../tests/pipelines_v2/integration/test_sdxl_steering_pack.py`: analytical metric controls (repository-relative path begins at `tests/`).

## Open threads

Execute discovery before held-out evaluation. A null channel-mean intervention
does not rule out spatially structured color representations. Segmentation and
blinded human evaluation would narrow the current proxy's blind spots.

## Active scope update — 2026-10-16

User steering reduced this work to an obvious engineering plumbing demonstration. The larger frozen discovery/held-out design above remains planned and unexecuted. See `../README.md` and the phase workflow for the active image budget; no research-quality effect claim is being pursued. Live ordinary-workflow results remain pending.
