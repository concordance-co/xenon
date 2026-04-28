# Phase 03 Pre-Scale Robustness Checks

## Checks Run

- Batch-A-only PCA, dropping separately captured contractarianism conditions.
- Moral-active projection using the earlier diagnostic baseline and a clean neutrals-only baseline.
- Full-response lexical control using leave-one-dilemma-out TF-IDF ridge residualization.
- Full per-singleton principle-label correlations against L32 PC2.
- Logit-lens feasibility check in the local workspace.

## Main Conclusions

1. The contractarianism separate capture is not driving the PCA basis. Dropping contractarianism yields component cosines of 0.91-0.97 for PC1-PC5 against the all18 first16 basis.
2. The procedural-vs-decisive axis survives moral-active projection. With clean neutrals-only projection, the signal is weaker but still present across PC1/PC3: PC1 r = -0.219, PC3 r = 0.255.
3. Full-response pooling is lexically confounded. Raw full-response correlations are strong, but after TF-IDF residualization they collapse.
4. PC2 is specifically most associated with virtue_character among the old principle labels. The other principle labels are weak: rights 0.051, fairness 0.100, honesty 0.079, responsibility 0.120, loyalty 0.181, virtue 0.327.
5. First16 survives only as a weak top-3-subspace / early-state claim, not as a clean fixed-PC claim. Full future-response text residualization kills it; prefix-only text residualization leaves PC2 at about r = 0.25 in the clean-projected basis.
6. Behavioral substrate validation passes strongly: condition profiles over the model-judged labels are distinguishable well above a within-dilemma permutation null (p = 0.001).
7. Logit lens is not currently easy locally: no `/models` mount and no `transformers` package. It should be folded into a Modal/model-load job if needed.

## Lexical Control Result

Full-response L32 raw correlations looked strong:

- PC1 vs procedural_minus_decisive: r = 0.512
- PC2 vs principle_minus_outcome: r = 0.473
- PC2 vs virtue_character: r = 0.507

But after controlling for response text with leave-one-dilemma-out TF-IDF ridge:

- PC1 vs procedural_minus_decisive, both text-residualized: r = -0.001
- PC2 vs principle_minus_outcome, both text-residualized: r = -0.133
- PC2 vs virtue_character, both text-residualized: r = 0.147

Text predicts full-response PC scores well: PC1 R2 = 0.633, PC2 R2 = 0.643. Text also predicts the labels well, especially procedural_minus_decisive R2 = 0.509 and virtue_character R2 = 0.528.

Interpretation: full-response pooling is useful as a text/response-realized diagnostic, but it should not be primary for a non-lexical activation-state claim.

## First16 Lexical Control Result

The exact clean-projected first16 check used the L32 first16 all18 basis after projecting out the positive-theory-prime centroid vs neutrals-only centroid.

With prefix-only TF-IDF (`first16_words`), the procedural_decisive signal is not erased everywhere:

- PC1 raw r = -0.253, both prefix-text-residualized r = 0.073
- PC2 raw r = 0.196, both prefix-text-residualized r = 0.248
- PC3 raw r = 0.211, both prefix-text-residualized r = -0.005

With full-response TF-IDF, the signal collapses:

- PC1 both full-text-residualized r = -0.020
- PC2 both full-text-residualized r = 0.001
- PC3 both full-text-residualized r = -0.014

Interpretation: if the control is allowed to use future response text, the result is text-mediated. If the control is restricted to lexical content already present in the first 16 words, a weak procedural_decisive signal remains in the top-3 subspace. This supports a cautious early-state claim, not a strong fixed-PC non-lexical claim.

## Behavioral Substrate Validation

The 18 conditions do produce distinguishable behavior on the model-judged label space:

- global condition-separation stat: 1970.346
- within-dilemma permutation null p95: 283.640
- permutation p-value: 0.001

The most condition-sensitive labels are virtue_character (eta^2 = 0.566), loyalty_trust (0.236), fairness_justice (0.211), tradeoff_acknowledged (0.207), risk_mitigation (0.183), priority_resolution / decisive_resolution (0.166), and procedural_decisive_axis (0.164).

This means the substrate is behaviorally live, but the largest behavioral separation is virtue-prompt style/content, not procedural_decisive.

## Split Stability

On 100 random 15-dilemma / 15-dilemma splits of the clean-projected L32 first16 basis, individual PC indices rotate substantially. However, the top-3 subspace consistently contains a procedural_decisive correlate:

- train max |r| over PC1-PC3 median: 0.271
- test max |r| over PC1-PC3 median: 0.252
- test max |r| p05-p95: 0.184-0.335

Interpretation: pre-registration should not depend on "PC1 specifically." It should test whether at least one of PC1-PC3 carries the procedural_decisive signal after controls.

## Recommended Schema Going Forward

Primary behavioral axis:

- procedural_decisive_axis = procedural_risk_management - decisive_resolution
- procedural_risk_management = legality_compliance + procedural_escalation + risk_mitigation + conditional_recommendation + moral_uncertainty

Primary activation slice:

- generated L32 first16 mean-pool for the cautious early-state claim
- generated L24/L32/L40 first16 for layer-band stability
- full-response mean-pool only as supplementary text-realized behavior geometry, not primary

Candidate second axis:

- virtue_character singleton as an exploratory correlate of PC2
- do not call this principle-vs-outcome
- do not use principle_integrity, care_particularity, or dominant_mode

## Go / No-Go

Conditional go for the 240-dilemma unified capture, but pre-register a weaker and cleaner target:

- Primary success criterion should be top-3-subspace based, not PC1-specific.
- Require procedural_decisive_axis to appear in at least one of PC1-PC3 on L32 first16 after clean moral-active projection and prefix-only lexical control.
- Use full-response only as a supplementary text-realized behavioral geometry diagnostic.
- Treat virtue_character / PC2 as exploratory.
- Do not claim moral theory orientation, principle-vs-outcome, or non-lexical full-response state.
