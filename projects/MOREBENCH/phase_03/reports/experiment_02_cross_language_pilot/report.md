# Experiment 02 Cross-Language Pilot

Small fully translated diagonal pilot testing whether `English / Spanish / Simplified Chinese` output breaks the response-side lexical ceiling for `deontology` vs `virtue_ethics`.

## Run
- workflow run id: `wr_74c38b01483d_7e3195fe`
- generation artifact: `generation_run_1_6e2b8f5a2902`
- transform artifact: `transform_ca6b7a420e0a_c939a86f`
- runtime app id: `ap-mmO4OzSUvrr0k4sTT05AdK`

## Pilot Shape
- `5` theory groups:
  - `theory_group_005`
  - `theory_group_009`
  - `theory_group_013`
  - `theory_group_015`
  - `theory_group_022`
- `2` theories:
  - `deontology`
  - `virtue_ethics`
- `3` fully translated language conditions:
  - `English in / English out`
  - `Spanish in / Spanish out`
  - `Simplified Chinese in / Simplified Chinese out`
- `30` prompts total
- all `30/30` generations completed with `stop`

## Main Result
The cross-language move helped, but not enough to cleanly reopen response-side probing.

Raw cross-language char-TF-IDF AUROC matrix:

| train -> test | `en` | `es` | `zh` |
| --- | ---: | ---: | ---: |
| `en` | `1.00` | `1.00` | `0.88` |
| `es` | `1.00` | `1.00` | `0.56` |
| `zh` | `1.00` | `0.80` | `1.00` |

- mean cross-language AUROC: `0.8733`

Markdown-stripped cross-language AUROC matrix:

| train -> test | `en` | `es` | `zh` |
| --- | ---: | ---: | ---: |
| `en` | `1.00` | `1.00` | `0.68` |
| `es` | `1.00` | `1.00` | `0.80` |
| `zh` | `1.00` | `0.80` | `1.00` |

- mean cross-language markdown-stripped AUROC: `0.8800`

Non-English ASCII-token ablation matrix:

| train -> test | `en` | `es` | `zh` |
| --- | ---: | ---: | ---: |
| `en` | `1.00` | `1.00` | `0.56` |
| `es` | `1.00` | `1.00` | `0.00` |
| `zh` | `0.32` | `0.40` | `1.00` |

- mean cross-language AUROC after ablating ASCII tokens from non-English outputs: `0.5467`

## Interpretation
- This is the first response-side intervention that clearly reduced the lexical ceiling relative to the English-only pilots.
- The strongest drop was `Spanish -> Chinese`, which fell to `0.56` on raw text.
- But the overall cross-language ceiling stayed high: mean cross-language AUROC remained `0.8733`, well above the intended `<= 0.70` scale-up gate.
- The remaining signal is not just Markdown/header leakage: stripping Markdown still leaves high transfer in most cells.
- The asymmetry is important: `Chinese -> English` stayed at `1.00` on raw text, but after ablating ASCII tokens from non-English outputs it collapsed to `0.32`; `English -> Chinese` likewise fell from `0.88` to `0.56`.
- That pattern strongly supports the code-switching explanation: the surviving `en <-> zh` ceiling was largely carried by English theory terms embedded inside the Chinese outputs, not by genuine cross-script char-ngram transfer.

## Language / Fidelity Notes
- Script purity was strong for English and Spanish (`1.0` mean each).
- Chinese stayed mostly Chinese but not perfectly pure (`0.7753` mean Han-script share).
- Several Chinese rows still contained English philosophical terms like `Phronesis`, `Temperance`, `Virtue`, or `Categorical Imperative`, especially on `virtue_ethics` rows. That is a real residual leakage path.
- Manual spot-check of the response tails suggests theory fidelity mostly held: deontology rows still argued in principle / standing terms, and virtue rows still argued in practical-wisdom / balance terms, but the Chinese virtue outputs were the most likely to code-switch into English philosophical vocabulary.
- Prompt audit was clean on the specific theory terms: non-English prompts contained `0` rows with English philosophical anchor terms from the audit set.

## What This Means
- Full translation made an impact. This was not another pure `1.0` ceiling result.
- The English-token ablation clarifies the story: language variation *does* break the baseline when code-switching is controlled, but the current response-side outputs still reintroduce English lexical anchors.
- That is a real methodological win for the translation strategy, but still not a clean response-side win.
- The honest read is:
  - `not zero`: yes, the text ceiling moved
  - `methodological validation of cross-language variation`: yes, partial
  - `clean room for a probe`: not yet
  - `ready to scale response-side activation capture`: no

## Recommendation
- Do **not** scale response-side cross-language capture yet.
- If we stay on this line, the cleaner next move is prompt-side / pre-generation state on the same translated prompts.
- If we revisit response-side later, we should first tighten the non-English outputs further, especially Chinese code-switching on theory-specific terms.
