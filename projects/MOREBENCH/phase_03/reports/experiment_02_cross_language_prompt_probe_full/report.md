# Experiment 02 Cross-Language Prompt Probe Full 30

Prompt-final residual probe on the translated `English / Spanish / Simplified Chinese` full-30 prompt set for `deontology` vs `virtue_ethics`.

## Run
- capture artifact: `capture_1_2c011b403d39`
- example count: `180`
- group count: `30`
- layers: `0, 4, 8, 16, 24, 32, 40, 44`

## Prompt Text Baseline
Raw prompt-text char-TF-IDF AUROC matrix:

| train -> test | `en` | `es` | `zh` |
| --- | ---: | ---: | ---: |
| `en` | `1.00` | `1.00` | `0.50` |
| `es` | `1.00` | `1.00` | `0.50` |
| `zh` | `0.42` | `0.43` | `1.00` |

- mean cross-language prompt-text AUROC: `0.6424`
- grouped bootstrap 95% CI: `[0.6140, 0.6628]`

Prompt-length-only baseline (`prompt_token_count`) cross-language AUROC matrix:

| train -> test | `en` | `es` | `zh` |
| --- | ---: | ---: | ---: |
| `en` | `0.55` | `0.54` | `0.59` |
| `es` | `0.55` | `0.54` | `0.59` |
| `zh` | `0.55` | `0.54` | `0.59` |

- mean cross-language prompt-length AUROC: `0.5583`

## Prompt-Final Probe
Mean cross-language prompt-final probe AUROC by layer:

| layer | mean cross-language AUROC |
| --- | ---: |
| `0` | `0.5000` |
| `4` | `0.7281` |
| `8` | `0.9019` |
| `16` | `0.9987` |
| `24` | `0.9235` |
| `32` | `1.0000` |
| `40` | `1.0000` |
| `44` | `0.9996` |

Best layer by cross-language delta over prompt text:
- best layer: `32`
- mean cross-language AUROC: `1.0000`
- grouped bootstrap 95% CI: `[1.0000, 1.0000]`
- delta vs prompt-text baseline: `0.3576`
- grouped bootstrap delta 95% CI: `[0.3372, 0.3860]`

Best-layer cross-language probe matrix:

| train -> test | `en` | `es` | `zh` |
| --- | ---: | ---: | ---: |
| `en` | `1.00` | `1.00` | `1.00` |
| `es` | `1.00` | `1.00` | `1.00` |
| `zh` | `1.00` | `1.00` | `1.00` |

Cross-script ordered pairs by layer:

| layer | `en->zh` | `zh->en` | `es->zh` | `zh->es` |
| --- | ---: | ---: | ---: | ---: |
| `0` | `0.50` | `0.50` | `0.50` | `0.50` |
| `4` | `0.86` | `0.70` | `0.76` | `0.64` |
| `8` | `0.98` | `0.81` | `0.93` | `0.88` |
| `16` | `1.00` | `1.00` | `1.00` | `1.00` |
| `24` | `0.95` | `0.94` | `0.82` | `0.94` |
| `32` | `1.00` | `1.00` | `1.00` | `1.00` |
| `40` | `1.00` | `1.00` | `1.00` | `1.00` |
| `44` | `1.00` | `1.00` | `1.00` | `1.00` |

Best-layer grouped bootstrap 95% CIs for cross-script ordered pairs:
- `en->zh`: `[1.0000, 1.0000]`
- `zh->en`: `[1.0000, 1.0000]`
- `es->zh`: `[1.0000, 1.0000]`
- `zh->es`: `[1.0000, 1.0000]`

## Red-Team Checks
- Spanish prompt rows with English theory-term residue: `0/60`
- Chinese prompt rows with English theory-term residue: `0/60`
- Chinese prompt rows with ASCII residue: `4/60`
- Top ASCII residues in Chinese prompts: `asap (2), back (2), call (2), dr (2), patel (2), urgent (2)`
- Random-label control at best layer:
  - mean cross-language AUROC under permutation: `0.4992`
  - 95th percentile: `0.6733`
  - max over `256` permutations: `0.8459`
  - share of permutations with mean cross-language AUROC `>= 0.60`: `0.1719`
  - share with `>= 0.80`: `0.0039`

## Interpretation
- The scale-up target is not whether any single pair is high, but whether the prompt-final probe still opens clear room over the prompt-text baseline once we use all `30` dilemmas.
- The most important structural check is the emergence curve. A signal that starts near chance at early layers and rises through the stack is much harder to explain as a surface-text shortcut than a flat ceiling from `L0`.
- The cross-script ordered pairs are the strongest subtest because they break the easy English-character path.
- The grouped bootstrap is the main guardrail here. We should trust the full run only if the best-layer delta stays comfortably above zero and the cross-script pairs remain high.

## Recommendation
- The full 30-dilemma prompt-side run supports a strong representational reopening on deontology vs virtue_ethics. The next step is a targeted follow-up control set, not another broad search.
