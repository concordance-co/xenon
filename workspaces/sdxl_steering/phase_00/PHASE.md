## Premise

The first SDXL adapter demonstration generated three 1024² images using the
same initial noise: baseline, read-only capture, and spatial ablation. That
establishes a usable intervention surface, not a semantic steering direction.
The next question is whether a channel direction learned from paired warm/cool
backdrops shifts a held-out background-color proxy beyond random controls.

## What we ran

Adapter demonstration on one A100 80GB:
[Modal run](https://modal.com/apps/concordance/main/ap-AS3VNjwxaO9u69h0rtipQd).
SDXL revision `462165984030d82259a11f4367a4eed129e94a7b`, Euler, 20 steps,
guidance 5, seed 0. Conditional `mid_block.resnets.1` output at step 10.

## Primary result

| Comparison | Mean absolute pixel difference | Maximum |
|---|---:|---:|
| Baseline / capture | 0 | 0 |
| Baseline / center ablation | 0.00787918 | 0.55903733 |

## Qualitative inspection

Both bounded previews depict a red teapot beside a window. The coarse scene
remains recognizable. This visual inspection is not a blinded quality rating.

## Corrections

The first attempt generated a baseline but failed during artifact publication.
The corrected publication path completed all three images and their manifests.

## Running hypothesis

The mid-resolution UNet representation may support coarse color changes at this
denoising step. This is a candidate write site, not established color localization.

## Claim boundary

Internal ablation changes the generated image under paired noise. Read-only
capture did not change pixels in this run. Neither result identifies a semantic
feature or establishes background-specific control.

## Artifacts

- `../study.py`: frozen discovery and held-out design.
- `../metrics.py`: explicit color and collateral-change proxies.
- Local execution evidence: `.planning/evidence/T16/first-image-result.json`.
- Remote full images and tensors: `/data/t16-sdxl/7865c36392354398afdf6cb33f0c0918`
  in `xenon-data`.

## Open threads

Learn the direction on discovery pairs, then evaluate all preregistered held-out
arms. Human prompt-adherence and quality annotations remain unavailable.

## Active scope update — 2026-10-16

User steering reduced this work to an obvious engineering plumbing demonstration. The larger frozen discovery/held-out design above remains planned and unexecuted. See `../README.md` and the phase workflow for the active image budget; no research-quality effect claim is being pursued. Live ordinary-workflow results remain pending.
