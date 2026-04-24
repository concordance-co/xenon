# Experiment 02 Cross-Language Prompt Probe

Prompt-final residual probe on the translated `English / Spanish / Simplified Chinese` pilot prompts for `deontology` vs `virtue_ethics`.

## Run
- capture artifact: `capture_1_7da39790c5d3`
- example count: `30`
- layers: `0, 4, 8, 16, 24, 32, 40, 44`

## Prompt Text Baseline
Raw prompt-text char-TF-IDF AUROC matrix:

| train -> test | `en` | `es` | `zh` |
| --- | ---: | ---: | ---: |
| `en` | `1.00` | `1.00` | `0.50` |
| `es` | `1.00` | `1.00` | `0.50` |
| `zh` | `0.28` | `0.36` | `1.00` |

- mean cross-language prompt-text AUROC: `0.6067`

Prompt-length-only baseline (`prompt_token_count`) cross-language AUROC matrix:

| train -> test | `en` | `es` | `zh` |
| --- | ---: | ---: | ---: |
| `en` | `0.64` | `0.64` | `0.60` |
| `es` | `0.64` | `0.64` | `0.60` |
| `zh` | `0.64` | `0.64` | `0.60` |

- mean cross-language prompt-length AUROC: `0.6267`

## Prompt-Final Probe
Mean cross-language prompt-final probe AUROC by layer:

| layer | mean cross-language AUROC |
| --- | ---: |
| `0` | `0.5000` |
| `4` | `0.7133` |
| `8` | `0.8133` |
| `16` | `0.9800` |
| `24` | `0.9800` |
| `32` | `1.0000` |
| `40` | `1.0000` |
| `44` | `1.0000` |

Best layer by cross-language delta over prompt text:
- best layer: `32`
- mean cross-language AUROC: `1.0000`
- delta vs prompt-text baseline: `0.3933`

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
| `4` | `0.80` | `0.72` | `0.72` | `0.72` |
| `8` | `1.00` | `0.64` | `0.92` | `0.84` |
| `16` | `1.00` | `0.88` | `1.00` | `1.00` |
| `24` | `1.00` | `1.00` | `0.88` | `1.00` |
| `32` | `1.00` | `1.00` | `1.00` | `1.00` |
| `40` | `1.00` | `1.00` | `1.00` | `1.00` |
| `44` | `1.00` | `1.00` | `1.00` | `1.00` |

## Red-Team Checks
- Chinese prompt rows with ASCII residue: `6/10`
- Chinese prompt rows with English theory-term residue: `0/10`
- Top ASCII residues in Chinese prompts: `priya (16), marcus (6), accommodation (2), amina (2), aurora (2), elena (2), hana (2), kim (2), lee (2), luis (2)`
- Random-label control at L32:
  - mean cross-language AUROC under permutation: `0.5019`
  - 95th percentile: `0.8733`
  - max over `256` permutations: `0.9933`
  - share of permutations with mean cross-language AUROC `>= 0.60`: `0.3398`
  - share with `>= 0.80`: `0.0859`

## Interpretation
- Prompt-side is the right substrate only if the prompt-final probe transfers across languages better than the prompt-text baseline.
- The critical comparison here is cross-language mean AUROC, not within-language diagonals.
- This pilot is tiny (`5` dilemmas per language pair), so exact cells are noisy; what matters is whether any layer clearly opens room over the prompt-text baseline.
- The key structural check passes: the cross-script pairs themselves rise through the stack and are already high by `16`, then saturate by `32`.
- The random-label control is mixed: its mean stays near chance, but the tail is wide enough that this `5`-dilemma pilot is not by itself sufficient to rule out small-N overfitting.
- The Chinese prompt audit is cleaner than the response-side case: some ASCII proper names remain, but English theory-term residue in the Chinese prompts is `0`, and the cross-script prompt-text baseline is still low.

## Recommendation
- Prompt-side looks promising enough to scale, but this pilot is not claim-ready; keep the interpretation frozen and verify on the 30-dilemma run.
