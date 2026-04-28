# Within-Dilemma PCA Discovery

PCA is run on residual activations after centering each dilemma across the full condition manifold.
This removes the large scenario-content axis and asks which prompt-induced response-state modes remain.

## Inputs

- base capture: `capture_1_1d7271d73617`
- contractarian capture: `capture_1_c24f680774a7`
- base generation rows: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/all_theories_brief_recommendation_report/report_6aa730c32d87_8c1df9a2/results/generate_natural_responses_results.json`
- contractarian generation rows: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/all_theories_brief_recommendation_report/morebench_theory_persona_vectors_phase03_brief_recommendation_smoke_anti_contractarian_contractarian_contractarian/report_02aa68279c28_ff67f3ad/results/generate_natural_responses_results.json`
- conditions: `18`

## generated_sequence_residual L16 first_16

- rows: `540`
- dilemmas: `30`
- conditions: `18`
- features: `2048`

### Scree

| PC | variance ratio | cumulative | CV condition-loading corr |
|---:|---:|---:|---:|
| 1 | 0.073 | 0.073 | 0.791 |
| 2 | 0.068 | 0.142 | 0.890 |
| 3 | 0.049 | 0.190 | 0.953 |
| 4 | 0.038 | 0.228 | 0.808 |
| 5 | 0.031 | 0.260 | 0.675 |
| 6 | 0.025 | 0.284 | 0.576 |
| 7 | 0.023 | 0.307 | 0.484 |
| 8 | 0.021 | 0.328 | 0.767 |

### Condition Extremes

#### PC1 (0.073 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_contract_01` | 0.485 | `positive` | `contractualism` |
| positive | `P_util_01` | 0.478 | `positive` | `utilitarian` |
| positive | `P_util_02` | 0.243 | `positive_variant` | `utilitarian` |
| positive | `P_contractarian_01` | 0.224 | `positive` | `contractarianism` |
| positive | `N_neutral_01` | 0.210 | `neutral_negative` | `none` |
| negative | `P_virtue_02` | -0.636 | `positive_variant` | `virtue_ethics` |
| negative | `N_anti_contractarian_01` | -0.415 | `anti_theory_diagnostic` | `anti_contractarianism` |
| negative | `P_deont_02` | -0.391 | `positive_variant` | `deontology` |
| negative | `N_generic_moral_01` | -0.281 | `generic_moral_anchor` | `generic_moral` |
| negative | `N_anti_deont_01` | -0.123 | `anti_theory_diagnostic` | `anti_deontology` |

#### PC2 (0.068 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_contractarian_02` | 0.456 | `positive_variant` | `contractarianism` |
| positive | `P_contractarian_01` | 0.448 | `positive` | `contractarianism` |
| positive | `N_generic_moral_01` | 0.379 | `generic_moral_anchor` | `generic_moral` |
| positive | `P_virtue_01` | 0.325 | `positive` | `virtue_ethics` |
| positive | `P_contract_01` | 0.241 | `positive` | `contractualism` |
| negative | `N_anti_deont_01` | -0.486 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `P_util_02` | -0.448 | `positive_variant` | `utilitarian` |
| negative | `N_anti_contractarian_01` | -0.440 | `anti_theory_diagnostic` | `anti_contractarianism` |
| negative | `P_deont_02` | -0.368 | `positive_variant` | `deontology` |
| negative | `N_anti_util_01` | -0.197 | `anti_theory_diagnostic` | `anti_utilitarian` |

#### PC3 (0.049 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_virtue_02` | 0.954 | `positive_variant` | `virtue_ethics` |
| positive | `P_deont_02` | 0.609 | `positive_variant` | `deontology` |
| positive | `P_util_01` | 0.096 | `positive` | `utilitarian` |
| positive | `P_contract_02` | 0.074 | `positive_variant` | `contractualism` |
| positive | `P_util_02` | 0.073 | `positive_variant` | `utilitarian` |
| negative | `N_anti_contractarian_01` | -0.455 | `anti_theory_diagnostic` | `anti_contractarianism` |
| negative | `N_neutral_02` | -0.230 | `neutral_negative_length_matched` | `none` |
| negative | `N_anti_virtue_01` | -0.209 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| negative | `N_anti_contract_01` | -0.180 | `anti_theory_diagnostic` | `anti_contractualism` |
| negative | `P_contractarian_01` | -0.171 | `positive` | `contractarianism` |

#### PC4 (0.038 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_deont_01` | 0.551 | `positive` | `deontology` |
| positive | `P_deont_02` | 0.245 | `positive_variant` | `deontology` |
| positive | `P_virtue_01` | 0.222 | `positive` | `virtue_ethics` |
| positive | `N_anti_util_01` | 0.216 | `anti_theory_diagnostic` | `anti_utilitarian` |
| positive | `P_contract_02` | 0.182 | `positive_variant` | `contractualism` |
| negative | `N_anti_deont_01` | -0.382 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `N_anti_virtue_01` | -0.335 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| negative | `P_contractarian_02` | -0.322 | `positive_variant` | `contractarianism` |
| negative | `P_contractarian_01` | -0.137 | `positive` | `contractarianism` |
| negative | `N_neutral_01` | -0.125 | `neutral_negative` | `none` |

#### PC5 (0.031 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_generic_moral_01` | 0.339 | `generic_moral_anchor` | `generic_moral` |
| positive | `P_virtue_01` | 0.199 | `positive` | `virtue_ethics` |
| positive | `N_neutral_02` | 0.156 | `neutral_negative_length_matched` | `none` |
| positive | `N_neutral_01` | 0.119 | `neutral_negative` | `none` |
| positive | `P_contractarian_01` | 0.116 | `positive` | `contractarianism` |
| negative | `N_anti_contractarian_01` | -0.279 | `anti_theory_diagnostic` | `anti_contractarianism` |
| negative | `N_anti_virtue_01` | -0.260 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| negative | `N_anti_deont_01` | -0.217 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `P_contract_01` | -0.168 | `positive` | `contractualism` |
| negative | `P_virtue_02` | -0.119 | `positive_variant` | `virtue_ethics` |

## prompt_end_residual L16 full

- rows: `540`
- dilemmas: `30`
- conditions: `18`
- features: `2048`

### Scree

| PC | variance ratio | cumulative | CV condition-loading corr |
|---:|---:|---:|---:|
| 1 | 0.286 | 0.286 | 1.000 |
| 2 | 0.136 | 0.422 | 0.999 |
| 3 | 0.110 | 0.532 | 1.000 |
| 4 | 0.058 | 0.589 | 0.999 |
| 5 | 0.035 | 0.624 | 0.998 |
| 6 | 0.033 | 0.658 | 0.999 |
| 7 | 0.030 | 0.688 | 0.994 |
| 8 | 0.027 | 0.715 | 0.999 |

### Condition Extremes

#### PC1 (0.286 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_anti_contractarian_01` | 0.671 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `N_anti_deont_01` | 0.656 | `anti_theory_diagnostic` | `anti_deontology` |
| positive | `N_anti_util_01` | 0.653 | `anti_theory_diagnostic` | `anti_utilitarian` |
| positive | `N_anti_virtue_01` | 0.583 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| positive | `N_anti_contract_01` | 0.458 | `anti_theory_diagnostic` | `anti_contractualism` |
| negative | `P_contract_01` | -0.888 | `positive` | `contractualism` |
| negative | `P_virtue_01` | -0.839 | `positive` | `virtue_ethics` |
| negative | `N_generic_moral_01` | -0.797 | `generic_moral_anchor` | `generic_moral` |
| negative | `P_util_01` | -0.689 | `positive` | `utilitarian` |
| negative | `P_contractarian_01` | -0.583 | `positive` | `contractarianism` |

#### PC2 (0.136 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_neutral_02` | 0.696 | `neutral_negative_length_matched` | `none` |
| positive | `N_neutral_01` | 0.694 | `neutral_negative` | `none` |
| positive | `N_generic_moral_01` | 0.530 | `generic_moral_anchor` | `generic_moral` |
| positive | `N_anti_virtue_01` | 0.232 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| positive | `P_contract_01` | 0.231 | `positive` | `contractualism` |
| negative | `P_contractarian_02` | -0.536 | `positive_variant` | `contractarianism` |
| negative | `P_virtue_02` | -0.481 | `positive_variant` | `virtue_ethics` |
| negative | `P_util_02` | -0.469 | `positive_variant` | `utilitarian` |
| negative | `P_contract_02` | -0.422 | `positive_variant` | `contractualism` |
| negative | `P_deont_02` | -0.397 | `positive_variant` | `deontology` |

#### PC3 (0.110 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_neutral_01` | 0.641 | `neutral_negative` | `none` |
| positive | `N_neutral_02` | 0.598 | `neutral_negative_length_matched` | `none` |
| positive | `P_deont_02` | 0.348 | `positive_variant` | `deontology` |
| positive | `P_virtue_02` | 0.245 | `positive_variant` | `virtue_ethics` |
| positive | `P_contractarian_02` | 0.233 | `positive_variant` | `contractarianism` |
| negative | `N_anti_contractarian_01` | -0.521 | `anti_theory_diagnostic` | `anti_contractarianism` |
| negative | `N_anti_util_01` | -0.432 | `anti_theory_diagnostic` | `anti_utilitarian` |
| negative | `P_contract_01` | -0.401 | `positive` | `contractualism` |
| negative | `N_anti_deont_01` | -0.303 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `N_anti_virtue_01` | -0.293 | `anti_theory_diagnostic` | `anti_virtue_ethics` |

#### PC4 (0.058 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_contract_01` | 0.357 | `positive` | `contractualism` |
| positive | `N_anti_contractarian_01` | 0.320 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `P_contractarian_02` | 0.231 | `positive_variant` | `contractarianism` |
| positive | `P_util_01` | 0.214 | `positive` | `utilitarian` |
| positive | `P_util_02` | 0.199 | `positive_variant` | `utilitarian` |
| negative | `P_deont_01` | -0.458 | `positive` | `deontology` |
| negative | `P_virtue_02` | -0.317 | `positive_variant` | `virtue_ethics` |
| negative | `N_anti_virtue_01` | -0.299 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| negative | `N_anti_deont_01` | -0.276 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `P_virtue_01` | -0.253 | `positive` | `virtue_ethics` |

#### PC5 (0.035 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_util_01` | 0.399 | `positive` | `utilitarian` |
| positive | `P_deont_01` | 0.210 | `positive` | `deontology` |
| positive | `P_contractarian_01` | 0.146 | `positive` | `contractarianism` |
| positive | `N_anti_util_01` | 0.127 | `anti_theory_diagnostic` | `anti_utilitarian` |
| positive | `N_neutral_02` | 0.122 | `neutral_negative_length_matched` | `none` |
| negative | `P_contract_01` | -0.474 | `positive` | `contractualism` |
| negative | `P_deont_02` | -0.242 | `positive_variant` | `deontology` |
| negative | `P_virtue_02` | -0.205 | `positive_variant` | `virtue_ethics` |
| negative | `N_anti_contract_01` | -0.121 | `anti_theory_diagnostic` | `anti_contractualism` |
| negative | `N_neutral_01` | -0.076 | `neutral_negative` | `none` |

## generated_sequence_residual L32 first_16

- rows: `540`
- dilemmas: `30`
- conditions: `18`
- features: `2048`

### Scree

| PC | variance ratio | cumulative | CV condition-loading corr |
|---:|---:|---:|---:|
| 1 | 0.095 | 0.095 | 0.884 |
| 2 | 0.069 | 0.164 | 0.950 |
| 3 | 0.052 | 0.216 | 0.855 |
| 4 | 0.037 | 0.253 | 0.959 |
| 5 | 0.033 | 0.285 | 0.816 |
| 6 | 0.028 | 0.313 | 0.682 |
| 7 | 0.027 | 0.340 | 0.732 |
| 8 | 0.021 | 0.361 | 0.759 |

### Condition Extremes

#### PC1 (0.095 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_virtue_02` | 1.367 | `positive_variant` | `virtue_ethics` |
| positive | `N_anti_contractarian_01` | 1.261 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `P_deont_02` | 0.947 | `positive_variant` | `deontology` |
| positive | `N_anti_deont_01` | 0.720 | `anti_theory_diagnostic` | `anti_deontology` |
| positive | `N_anti_util_01` | 0.689 | `anti_theory_diagnostic` | `anti_utilitarian` |
| negative | `P_contractarian_01` | -1.105 | `positive` | `contractarianism` |
| negative | `P_contract_01` | -1.040 | `positive` | `contractualism` |
| negative | `N_neutral_01` | -0.835 | `neutral_negative` | `none` |
| negative | `P_util_01` | -0.738 | `positive` | `utilitarian` |
| negative | `P_contractarian_02` | -0.666 | `positive_variant` | `contractarianism` |

#### PC2 (0.069 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_virtue_02` | 1.445 | `positive_variant` | `virtue_ethics` |
| positive | `P_contractarian_02` | 0.692 | `positive_variant` | `contractarianism` |
| positive | `N_generic_moral_01` | 0.682 | `generic_moral_anchor` | `generic_moral` |
| positive | `P_virtue_01` | 0.584 | `positive` | `virtue_ethics` |
| positive | `P_deont_01` | 0.558 | `positive` | `deontology` |
| negative | `N_anti_deont_01` | -1.193 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `P_util_02` | -1.121 | `positive_variant` | `utilitarian` |
| negative | `N_anti_virtue_01` | -0.701 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| negative | `P_util_01` | -0.656 | `positive` | `utilitarian` |
| negative | `N_anti_contract_01` | -0.413 | `anti_theory_diagnostic` | `anti_contractualism` |

#### PC3 (0.052 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_anti_contractarian_01` | 0.859 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `N_generic_moral_01` | 0.640 | `generic_moral_anchor` | `generic_moral` |
| positive | `P_contractarian_02` | 0.526 | `positive_variant` | `contractarianism` |
| positive | `N_anti_virtue_01` | 0.510 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| positive | `N_anti_deont_01` | 0.448 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `P_virtue_02` | -1.620 | `positive_variant` | `virtue_ethics` |
| negative | `P_deont_02` | -1.100 | `positive_variant` | `deontology` |
| negative | `P_deont_01` | -0.427 | `positive` | `deontology` |
| negative | `P_contract_02` | -0.399 | `positive_variant` | `contractualism` |
| negative | `P_util_02` | -0.284 | `positive_variant` | `utilitarian` |

#### PC4 (0.037 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_deont_01` | 0.964 | `positive` | `deontology` |
| positive | `P_virtue_01` | 0.506 | `positive` | `virtue_ethics` |
| positive | `N_anti_util_01` | 0.320 | `anti_theory_diagnostic` | `anti_utilitarian` |
| positive | `P_contract_02` | 0.311 | `positive_variant` | `contractualism` |
| positive | `N_neutral_02` | 0.273 | `neutral_negative_length_matched` | `none` |
| negative | `N_anti_deont_01` | -0.825 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `P_virtue_02` | -0.739 | `positive_variant` | `virtue_ethics` |
| negative | `P_contractarian_02` | -0.566 | `positive_variant` | `contractarianism` |
| negative | `N_anti_virtue_01` | -0.404 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| negative | `P_util_02` | -0.117 | `positive_variant` | `utilitarian` |

#### PC5 (0.033 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_anti_virtue_01` | 0.594 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| positive | `N_anti_deont_01` | 0.587 | `anti_theory_diagnostic` | `anti_deontology` |
| positive | `N_anti_contractarian_01` | 0.503 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `P_contract_01` | 0.460 | `positive` | `contractualism` |
| positive | `N_anti_util_01` | 0.300 | `anti_theory_diagnostic` | `anti_utilitarian` |
| negative | `N_generic_moral_01` | -0.678 | `generic_moral_anchor` | `generic_moral` |
| negative | `P_deont_02` | -0.438 | `positive_variant` | `deontology` |
| negative | `P_virtue_01` | -0.375 | `positive` | `virtue_ethics` |
| negative | `N_neutral_02` | -0.364 | `neutral_negative_length_matched` | `none` |
| negative | `N_neutral_01` | -0.303 | `neutral_negative` | `none` |

## prompt_end_residual L32 full

- rows: `540`
- dilemmas: `30`
- conditions: `18`
- features: `2048`

### Scree

| PC | variance ratio | cumulative | CV condition-loading corr |
|---:|---:|---:|---:|
| 1 | 0.292 | 0.292 | 1.000 |
| 2 | 0.181 | 0.473 | 1.000 |
| 3 | 0.098 | 0.571 | 1.000 |
| 4 | 0.054 | 0.625 | 0.997 |
| 5 | 0.042 | 0.667 | 0.999 |
| 6 | 0.029 | 0.696 | 0.995 |
| 7 | 0.026 | 0.722 | 0.993 |
| 8 | 0.023 | 0.745 | 0.997 |

### Condition Extremes

#### PC1 (0.292 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_anti_deont_01` | 2.063 | `anti_theory_diagnostic` | `anti_deontology` |
| positive | `N_anti_virtue_01` | 1.874 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| positive | `N_anti_contractarian_01` | 1.808 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `N_anti_contract_01` | 1.593 | `anti_theory_diagnostic` | `anti_contractualism` |
| positive | `N_anti_util_01` | 1.509 | `anti_theory_diagnostic` | `anti_utilitarian` |
| negative | `P_contract_01` | -1.920 | `positive` | `contractualism` |
| negative | `P_virtue_01` | -1.682 | `positive` | `virtue_ethics` |
| negative | `N_generic_moral_01` | -1.301 | `generic_moral_anchor` | `generic_moral` |
| negative | `P_util_01` | -1.278 | `positive` | `utilitarian` |
| negative | `P_deont_01` | -0.897 | `positive` | `deontology` |

#### PC2 (0.181 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_contractarian_02` | 1.068 | `positive_variant` | `contractarianism` |
| positive | `P_deont_02` | 1.003 | `positive_variant` | `deontology` |
| positive | `P_contract_02` | 0.939 | `positive_variant` | `contractualism` |
| positive | `P_virtue_02` | 0.904 | `positive_variant` | `virtue_ethics` |
| positive | `P_util_02` | 0.642 | `positive_variant` | `utilitarian` |
| negative | `N_neutral_01` | -2.524 | `neutral_negative` | `none` |
| negative | `N_neutral_02` | -2.403 | `neutral_negative_length_matched` | `none` |
| negative | `N_generic_moral_01` | -0.499 | `generic_moral_anchor` | `generic_moral` |
| negative | `N_anti_util_01` | -0.203 | `anti_theory_diagnostic` | `anti_utilitarian` |
| negative | `N_anti_virtue_01` | -0.100 | `anti_theory_diagnostic` | `anti_virtue_ethics` |

#### PC3 (0.098 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_deont_02` | 1.065 | `positive_variant` | `deontology` |
| positive | `N_neutral_01` | 0.849 | `neutral_negative` | `none` |
| positive | `P_virtue_02` | 0.841 | `positive_variant` | `virtue_ethics` |
| positive | `N_neutral_02` | 0.837 | `neutral_negative_length_matched` | `none` |
| positive | `P_contractarian_02` | 0.829 | `positive_variant` | `contractarianism` |
| negative | `P_contract_01` | -1.162 | `positive` | `contractualism` |
| negative | `P_util_01` | -0.732 | `positive` | `utilitarian` |
| negative | `N_generic_moral_01` | -0.729 | `generic_moral_anchor` | `generic_moral` |
| negative | `N_anti_virtue_01` | -0.725 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| negative | `N_anti_deont_01` | -0.689 | `anti_theory_diagnostic` | `anti_deontology` |

#### PC4 (0.054 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_generic_moral_01` | 1.012 | `generic_moral_anchor` | `generic_moral` |
| positive | `P_deont_02` | 0.804 | `positive_variant` | `deontology` |
| positive | `P_virtue_02` | 0.637 | `positive_variant` | `virtue_ethics` |
| positive | `P_deont_01` | 0.538 | `positive` | `deontology` |
| positive | `P_virtue_01` | 0.405 | `positive` | `virtue_ethics` |
| negative | `P_util_01` | -0.805 | `positive` | `utilitarian` |
| negative | `P_util_02` | -0.656 | `positive_variant` | `utilitarian` |
| negative | `P_contractarian_01` | -0.626 | `positive` | `contractarianism` |
| negative | `N_anti_contractarian_01` | -0.601 | `anti_theory_diagnostic` | `anti_contractarianism` |
| negative | `P_contractarian_02` | -0.529 | `positive_variant` | `contractarianism` |

#### PC5 (0.042 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_anti_deont_01` | 1.033 | `anti_theory_diagnostic` | `anti_deontology` |
| positive | `N_anti_virtue_01` | 0.728 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| positive | `P_util_01` | 0.412 | `positive` | `utilitarian` |
| positive | `P_util_02` | 0.338 | `positive_variant` | `utilitarian` |
| positive | `P_deont_01` | 0.112 | `positive` | `deontology` |
| negative | `N_anti_contractarian_01` | -0.910 | `anti_theory_diagnostic` | `anti_contractarianism` |
| negative | `N_anti_util_01` | -0.740 | `anti_theory_diagnostic` | `anti_utilitarian` |
| negative | `N_anti_contract_01` | -0.569 | `anti_theory_diagnostic` | `anti_contractualism` |
| negative | `N_generic_moral_01` | -0.464 | `generic_moral_anchor` | `generic_moral` |
| negative | `P_contract_02` | -0.160 | `positive_variant` | `contractualism` |
