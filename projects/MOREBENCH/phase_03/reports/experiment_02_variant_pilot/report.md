# Experiment 02 Variant Pilot

Small generation pilot testing whether the new theory prompt augmentation materially lowers the response-side lexical ceiling.

## Run
- workflow run id: `wr_4cecba84bc24_001461b7`
- generation artifact: `generation_run_1_1fe7cbe7d1ea`
- transform artifact: `transform_f0f237de45e8_d319917b`
- runtime app id: `ap-xSwXWcBx70c1rsOZEbXaCO`

## Pilot Shape
- `5` theory groups:
  - `theory_group_005`
  - `theory_group_009`
  - `theory_group_013`
  - `theory_group_015`
  - `theory_group_022`
- `6` description-variant banks
- `3` name-only banks
- `5` theory primes + `1` generic control
- `270` prompts total
- all `270/270` generations completed with `stop`

## Question
Does the augmentation create enough response-side lexical diversity that held-out-bank or cross-format char-TF-IDF is no longer near ceiling?

## Main Result
No. The response-side lexical ceiling remains very high.

Per-theory response-side char-TF-IDF AUROC:

| Theory | Description Holdout Mean | Description -> Name | Name -> Description |
| --- | ---: | ---: | ---: |
| `contractarianism` | `0.9733` | `0.8711` | `0.7989` |
| `contractualism` | `0.9667` | `1.0000` | `0.8467` |
| `deontology` | `1.0000` | `1.0000` | `0.9767` |
| `utilitarian` | `0.9133` | `1.0000` | `0.8556` |
| `virtue_ethics` | `1.0000` | `1.0000` | `0.9822` |

## Interpretation
- The augmentation did not buy meaningful room on response text.
- Within-description held-out-bank transfer is still near ceiling for every theory.
- Cross-format transfer helped a little only for `contractarianism`, but even there the text baseline remains high.
- `deontology`, `virtue_ethics`, and most `theory -> name_only` comparisons are still effectively ceilinged.

## What This Means
- The new augmentation asset is useful as a methodological artifact, but it did **not** solve the lexical-confound problem on the actual generated responses in this pilot.
- The strongest possible optimistic read is that `name_only -> description` for `contractarianism` (`0.7989`) is lower than the rest, but it is still too high to count as clean room for a meaningful probe-over-text result.
- As a result, the current theory/generic response-side line is still not run-ready for a serious probe campaign.

## Recommendation
- Do not scale this exact response-side theory-vs-generic setup yet.
- If we stay on this track, the next sensible move is to switch the target:
  - theory-vs-theory rather than theory-vs-generic
  - or prompt-side / pre-generation state rather than response-side text
- Otherwise, we should treat this pilot as a useful negative and move to a different question family.
