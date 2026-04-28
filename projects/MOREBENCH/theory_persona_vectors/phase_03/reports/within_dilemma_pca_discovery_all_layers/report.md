# Within-Dilemma PCA Discovery

PCA is run on residual activations after centering each dilemma across the full condition manifold.
This removes the large scenario-content axis and asks which prompt-induced response-state modes remain.

## Inputs

- base capture: `capture_1_1d7271d73617`
- contractarian capture: `capture_1_c24f680774a7`
- base generation rows: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/all_theories_brief_recommendation_report/report_6aa730c32d87_8c1df9a2/results/generate_natural_responses_results.json`
- contractarian generation rows: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/all_theories_brief_recommendation_report/morebench_theory_persona_vectors_phase03_brief_recommendation_smoke_anti_contractarian_contractarian_contractarian/report_02aa68279c28_ff67f3ad/results/generate_natural_responses_results.json`
- conditions: `18`

## generated_sequence_residual L0 first_16

- rows: `540`
- dilemmas: `30`
- conditions: `18`
- features: `2048`

### Scree

| PC | variance ratio | cumulative | CV condition-loading corr |
|---:|---:|---:|---:|
| 1 | 0.052 | 0.052 | 0.963 |
| 2 | 0.037 | 0.089 | 0.877 |
| 3 | 0.029 | 0.118 | 0.604 |
| 4 | 0.024 | 0.141 | 0.345 |
| 5 | 0.021 | 0.162 | 0.716 |
| 6 | 0.019 | 0.181 | 0.431 |
| 7 | 0.018 | 0.199 | 0.215 |
| 8 | 0.017 | 0.216 | 0.350 |

### Condition Extremes

#### PC1 (0.052 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_anti_virtue_01` | 0.047 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| positive | `N_anti_contractarian_01` | 0.035 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `N_anti_deont_01` | 0.033 | `anti_theory_diagnostic` | `anti_deontology` |
| positive | `P_virtue_02` | 0.032 | `positive_variant` | `virtue_ethics` |
| positive | `P_deont_02` | 0.024 | `positive_variant` | `deontology` |
| negative | `P_contractarian_01` | -0.020 | `positive` | `contractarianism` |
| negative | `P_contract_01` | -0.020 | `positive` | `contractualism` |
| negative | `P_virtue_01` | -0.018 | `positive` | `virtue_ethics` |
| negative | `P_util_01` | -0.018 | `positive` | `utilitarian` |
| negative | `N_neutral_01` | -0.017 | `neutral_negative` | `none` |

#### PC2 (0.037 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_virtue_02` | 0.037 | `positive_variant` | `virtue_ethics` |
| positive | `P_deont_02` | 0.015 | `positive_variant` | `deontology` |
| positive | `P_contractarian_02` | 0.008 | `positive_variant` | `contractarianism` |
| positive | `P_contract_02` | 0.007 | `positive_variant` | `contractualism` |
| positive | `P_util_01` | 0.005 | `positive` | `utilitarian` |
| negative | `N_anti_contractarian_01` | -0.035 | `anti_theory_diagnostic` | `anti_contractarianism` |
| negative | `N_anti_virtue_01` | -0.012 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| negative | `N_anti_contract_01` | -0.011 | `anti_theory_diagnostic` | `anti_contractualism` |
| negative | `N_neutral_02` | -0.009 | `neutral_negative_length_matched` | `none` |
| negative | `N_anti_deont_01` | -0.008 | `anti_theory_diagnostic` | `anti_deontology` |

#### PC3 (0.029 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_virtue_02` | 0.015 | `positive_variant` | `virtue_ethics` |
| positive | `P_deont_02` | 0.014 | `positive_variant` | `deontology` |
| positive | `P_contractarian_02` | 0.008 | `positive_variant` | `contractarianism` |
| positive | `N_anti_contractarian_01` | 0.006 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `P_deont_01` | 0.005 | `positive` | `deontology` |
| negative | `N_anti_deont_01` | -0.023 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `N_anti_virtue_01` | -0.022 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| negative | `N_anti_contract_01` | -0.008 | `anti_theory_diagnostic` | `anti_contractualism` |
| negative | `P_util_02` | -0.006 | `positive_variant` | `utilitarian` |
| negative | `P_util_01` | -0.006 | `positive` | `utilitarian` |

#### PC4 (0.024 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_anti_virtue_01` | 0.016 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| positive | `P_contractarian_02` | 0.012 | `positive_variant` | `contractarianism` |
| positive | `N_anti_deont_01` | 0.008 | `anti_theory_diagnostic` | `anti_deontology` |
| positive | `N_neutral_01` | 0.008 | `neutral_negative` | `none` |
| positive | `P_contractarian_01` | 0.008 | `positive` | `contractarianism` |
| negative | `P_deont_02` | -0.028 | `positive_variant` | `deontology` |
| negative | `P_virtue_02` | -0.024 | `positive_variant` | `virtue_ethics` |
| negative | `P_deont_01` | -0.008 | `positive` | `deontology` |
| negative | `N_anti_contractarian_01` | -0.005 | `anti_theory_diagnostic` | `anti_contractarianism` |
| negative | `P_contract_02` | -0.003 | `positive_variant` | `contractualism` |

#### PC5 (0.021 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_deont_02` | 0.012 | `positive_variant` | `deontology` |
| positive | `P_util_02` | 0.008 | `positive_variant` | `utilitarian` |
| positive | `P_util_01` | 0.008 | `positive` | `utilitarian` |
| positive | `N_anti_contract_01` | 0.006 | `anti_theory_diagnostic` | `anti_contractualism` |
| positive | `N_anti_util_01` | 0.004 | `anti_theory_diagnostic` | `anti_utilitarian` |
| negative | `P_contractarian_01` | -0.009 | `positive` | `contractarianism` |
| negative | `P_deont_01` | -0.009 | `positive` | `deontology` |
| negative | `P_virtue_01` | -0.006 | `positive` | `virtue_ethics` |
| negative | `P_contractarian_02` | -0.005 | `positive_variant` | `contractarianism` |
| negative | `N_generic_moral_01` | -0.004 | `generic_moral_anchor` | `generic_moral` |

## prompt_end_residual L0 full

- rows: `540`
- dilemmas: `30`
- conditions: `18`
- features: `2048`

### Scree

| PC | variance ratio | cumulative | CV condition-loading corr |
|---:|---:|---:|---:|
| 1 | 0.000 | 0.000 | nan |
| 2 | 0.000 | 0.000 | nan |
| 3 | 0.000 | 0.000 | nan |
| 4 | 0.000 | 0.000 | nan |
| 5 | 0.000 | 0.000 | nan |
| 6 | 0.000 | 0.000 | nan |
| 7 | 0.000 | 0.000 | nan |
| 8 | 0.000 | 0.000 | nan |

### Condition Extremes

#### PC1 (0.000 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_anti_contractarian_01` | 0.000 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `N_anti_contract_01` | 0.000 | `anti_theory_diagnostic` | `anti_contractualism` |
| positive | `N_anti_virtue_01` | 0.000 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| positive | `N_anti_util_01` | 0.000 | `anti_theory_diagnostic` | `anti_utilitarian` |
| positive | `N_anti_deont_01` | 0.000 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `N_neutral_01` | 0.000 | `neutral_negative` | `none` |
| negative | `N_neutral_02` | 0.000 | `neutral_negative_length_matched` | `none` |
| negative | `N_generic_moral_01` | 0.000 | `generic_moral_anchor` | `generic_moral` |
| negative | `P_deont_01` | 0.000 | `positive` | `deontology` |
| negative | `P_deont_02` | 0.000 | `positive_variant` | `deontology` |

#### PC2 (0.000 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_anti_contractarian_01` | 0.000 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `N_anti_contract_01` | 0.000 | `anti_theory_diagnostic` | `anti_contractualism` |
| positive | `N_anti_virtue_01` | 0.000 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| positive | `N_anti_util_01` | 0.000 | `anti_theory_diagnostic` | `anti_utilitarian` |
| positive | `N_anti_deont_01` | 0.000 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `N_neutral_01` | 0.000 | `neutral_negative` | `none` |
| negative | `N_neutral_02` | 0.000 | `neutral_negative_length_matched` | `none` |
| negative | `N_generic_moral_01` | 0.000 | `generic_moral_anchor` | `generic_moral` |
| negative | `P_deont_01` | 0.000 | `positive` | `deontology` |
| negative | `P_deont_02` | 0.000 | `positive_variant` | `deontology` |

#### PC3 (0.000 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_anti_contractarian_01` | 0.000 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `N_anti_contract_01` | 0.000 | `anti_theory_diagnostic` | `anti_contractualism` |
| positive | `N_anti_virtue_01` | 0.000 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| positive | `N_anti_util_01` | 0.000 | `anti_theory_diagnostic` | `anti_utilitarian` |
| positive | `N_anti_deont_01` | 0.000 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `N_neutral_01` | 0.000 | `neutral_negative` | `none` |
| negative | `N_neutral_02` | 0.000 | `neutral_negative_length_matched` | `none` |
| negative | `N_generic_moral_01` | 0.000 | `generic_moral_anchor` | `generic_moral` |
| negative | `P_deont_01` | 0.000 | `positive` | `deontology` |
| negative | `P_deont_02` | 0.000 | `positive_variant` | `deontology` |

#### PC4 (0.000 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_anti_contractarian_01` | 0.000 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `N_anti_contract_01` | 0.000 | `anti_theory_diagnostic` | `anti_contractualism` |
| positive | `N_anti_virtue_01` | 0.000 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| positive | `N_anti_util_01` | 0.000 | `anti_theory_diagnostic` | `anti_utilitarian` |
| positive | `N_anti_deont_01` | 0.000 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `N_neutral_01` | 0.000 | `neutral_negative` | `none` |
| negative | `N_neutral_02` | 0.000 | `neutral_negative_length_matched` | `none` |
| negative | `N_generic_moral_01` | 0.000 | `generic_moral_anchor` | `generic_moral` |
| negative | `P_deont_01` | 0.000 | `positive` | `deontology` |
| negative | `P_deont_02` | 0.000 | `positive_variant` | `deontology` |

#### PC5 (0.000 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_anti_contractarian_01` | 0.000 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `N_anti_contract_01` | 0.000 | `anti_theory_diagnostic` | `anti_contractualism` |
| positive | `N_anti_virtue_01` | 0.000 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| positive | `N_anti_util_01` | 0.000 | `anti_theory_diagnostic` | `anti_utilitarian` |
| positive | `N_anti_deont_01` | 0.000 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `N_neutral_01` | 0.000 | `neutral_negative` | `none` |
| negative | `N_neutral_02` | 0.000 | `neutral_negative_length_matched` | `none` |
| negative | `N_generic_moral_01` | 0.000 | `generic_moral_anchor` | `generic_moral` |
| negative | `P_deont_01` | 0.000 | `positive` | `deontology` |
| negative | `P_deont_02` | 0.000 | `positive_variant` | `deontology` |

## generated_sequence_residual L4 first_16

- rows: `540`
- dilemmas: `30`
- conditions: `18`
- features: `2048`

### Scree

| PC | variance ratio | cumulative | CV condition-loading corr |
|---:|---:|---:|---:|
| 1 | 0.065 | 0.065 | 0.807 |
| 2 | 0.053 | 0.118 | 0.954 |
| 3 | 0.042 | 0.160 | 0.937 |
| 4 | 0.035 | 0.195 | 0.839 |
| 5 | 0.028 | 0.223 | 0.535 |
| 6 | 0.026 | 0.249 | 0.718 |
| 7 | 0.024 | 0.273 | 0.687 |
| 8 | 0.021 | 0.294 | 0.658 |

### Condition Extremes

#### PC1 (0.065 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_contract_01` | 0.130 | `positive` | `contractualism` |
| positive | `P_util_01` | 0.129 | `positive` | `utilitarian` |
| positive | `P_contractarian_01` | 0.095 | `positive` | `contractarianism` |
| positive | `P_virtue_01` | 0.094 | `positive` | `virtue_ethics` |
| positive | `N_neutral_01` | 0.074 | `neutral_negative` | `none` |
| negative | `N_anti_contractarian_01` | -0.243 | `anti_theory_diagnostic` | `anti_contractarianism` |
| negative | `N_anti_deont_01` | -0.120 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `P_deont_02` | -0.113 | `positive_variant` | `deontology` |
| negative | `N_anti_virtue_01` | -0.091 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| negative | `P_virtue_02` | -0.091 | `positive_variant` | `virtue_ethics` |

#### PC2 (0.053 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_contractarian_01` | 0.092 | `positive` | `contractarianism` |
| positive | `N_neutral_02` | 0.088 | `neutral_negative_length_matched` | `none` |
| positive | `N_neutral_01` | 0.074 | `neutral_negative` | `none` |
| positive | `P_virtue_01` | 0.073 | `positive` | `virtue_ethics` |
| positive | `P_contract_01` | 0.064 | `positive` | `contractualism` |
| negative | `P_virtue_02` | -0.274 | `positive_variant` | `virtue_ethics` |
| negative | `P_deont_02` | -0.259 | `positive_variant` | `deontology` |
| negative | `P_util_02` | -0.048 | `positive_variant` | `utilitarian` |
| negative | `N_anti_deont_01` | -0.043 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `P_util_01` | -0.023 | `positive` | `utilitarian` |

#### PC3 (0.042 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_anti_virtue_01` | 0.155 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| positive | `N_anti_deont_01` | 0.154 | `anti_theory_diagnostic` | `anti_deontology` |
| positive | `N_anti_contractarian_01` | 0.127 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `N_anti_contract_01` | 0.083 | `anti_theory_diagnostic` | `anti_contractualism` |
| positive | `P_util_02` | 0.081 | `positive_variant` | `utilitarian` |
| negative | `P_virtue_02` | -0.250 | `positive_variant` | `virtue_ethics` |
| negative | `P_contractarian_02` | -0.105 | `positive_variant` | `contractarianism` |
| negative | `P_deont_02` | -0.098 | `positive_variant` | `deontology` |
| negative | `N_generic_moral_01` | -0.063 | `generic_moral_anchor` | `generic_moral` |
| negative | `P_contractarian_01` | -0.059 | `positive` | `contractarianism` |

#### PC4 (0.035 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_deont_02` | 0.117 | `positive_variant` | `deontology` |
| positive | `P_deont_01` | 0.117 | `positive` | `deontology` |
| positive | `P_virtue_01` | 0.059 | `positive` | `virtue_ethics` |
| positive | `N_anti_contractarian_01` | 0.051 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `P_contract_02` | 0.047 | `positive_variant` | `contractualism` |
| negative | `N_anti_virtue_01` | -0.180 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| negative | `N_anti_deont_01` | -0.134 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `P_contractarian_02` | -0.058 | `positive_variant` | `contractarianism` |
| negative | `N_neutral_01` | -0.034 | `neutral_negative` | `none` |
| negative | `P_contractarian_01` | -0.023 | `positive` | `contractarianism` |

#### PC5 (0.028 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_anti_contractarian_01` | 0.045 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `P_contract_01` | 0.042 | `positive` | `contractualism` |
| positive | `P_util_02` | 0.025 | `positive_variant` | `utilitarian` |
| positive | `P_virtue_02` | 0.023 | `positive_variant` | `virtue_ethics` |
| positive | `P_util_01` | 0.022 | `positive` | `utilitarian` |
| negative | `N_generic_moral_01` | -0.095 | `generic_moral_anchor` | `generic_moral` |
| negative | `P_virtue_01` | -0.042 | `positive` | `virtue_ethics` |
| negative | `P_contractarian_01` | -0.026 | `positive` | `contractarianism` |
| negative | `N_neutral_01` | -0.022 | `neutral_negative` | `none` |
| negative | `N_neutral_02` | -0.019 | `neutral_negative_length_matched` | `none` |

## prompt_end_residual L4 full

- rows: `540`
- dilemmas: `30`
- conditions: `18`
- features: `2048`

### Scree

| PC | variance ratio | cumulative | CV condition-loading corr |
|---:|---:|---:|---:|
| 1 | 0.323 | 0.323 | 1.000 |
| 2 | 0.209 | 0.532 | 1.000 |
| 3 | 0.130 | 0.663 | 0.999 |
| 4 | 0.078 | 0.740 | 0.999 |
| 5 | 0.039 | 0.780 | 0.998 |
| 6 | 0.034 | 0.813 | 0.997 |
| 7 | 0.023 | 0.836 | 0.954 |
| 8 | 0.020 | 0.856 | 0.995 |

### Condition Extremes

#### PC1 (0.323 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_anti_virtue_01` | 0.104 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| positive | `N_anti_contractarian_01` | 0.099 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `N_anti_deont_01` | 0.098 | `anti_theory_diagnostic` | `anti_deontology` |
| positive | `N_anti_util_01` | 0.095 | `anti_theory_diagnostic` | `anti_utilitarian` |
| positive | `N_neutral_01` | 0.095 | `neutral_negative` | `none` |
| negative | `N_generic_moral_01` | -0.144 | `generic_moral_anchor` | `generic_moral` |
| negative | `P_contractarian_01` | -0.143 | `positive` | `contractarianism` |
| negative | `P_contract_01` | -0.141 | `positive` | `contractualism` |
| negative | `P_virtue_01` | -0.140 | `positive` | `virtue_ethics` |
| negative | `P_util_01` | -0.139 | `positive` | `utilitarian` |

#### PC2 (0.209 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_deont_02` | 0.074 | `positive_variant` | `deontology` |
| positive | `N_anti_contractarian_01` | 0.055 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `P_contract_02` | 0.052 | `positive_variant` | `contractualism` |
| positive | `P_util_02` | 0.050 | `positive_variant` | `utilitarian` |
| positive | `P_contractarian_02` | 0.049 | `positive_variant` | `contractarianism` |
| negative | `N_neutral_01` | -0.281 | `neutral_negative` | `none` |
| negative | `N_neutral_02` | -0.115 | `neutral_negative_length_matched` | `none` |
| negative | `N_generic_moral_01` | -0.043 | `generic_moral_anchor` | `generic_moral` |
| negative | `P_contract_01` | -0.014 | `positive` | `contractualism` |
| negative | `P_util_01` | -0.013 | `positive` | `utilitarian` |

#### PC3 (0.130 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_neutral_02` | 0.145 | `neutral_negative_length_matched` | `none` |
| positive | `P_deont_02` | 0.077 | `positive_variant` | `deontology` |
| positive | `P_contract_02` | 0.069 | `positive_variant` | `contractualism` |
| positive | `P_util_02` | 0.052 | `positive_variant` | `utilitarian` |
| positive | `P_virtue_02` | 0.051 | `positive_variant` | `virtue_ethics` |
| negative | `N_anti_virtue_01` | -0.090 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| negative | `N_anti_util_01` | -0.083 | `anti_theory_diagnostic` | `anti_utilitarian` |
| negative | `N_generic_moral_01` | -0.078 | `generic_moral_anchor` | `generic_moral` |
| negative | `N_anti_contract_01` | -0.068 | `anti_theory_diagnostic` | `anti_contractualism` |
| negative | `N_anti_contractarian_01` | -0.047 | `anti_theory_diagnostic` | `anti_contractarianism` |

#### PC4 (0.078 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_neutral_01` | 0.080 | `neutral_negative` | `none` |
| positive | `P_contractarian_02` | 0.065 | `positive_variant` | `contractarianism` |
| positive | `P_util_02` | 0.044 | `positive_variant` | `utilitarian` |
| positive | `P_virtue_02` | 0.042 | `positive_variant` | `virtue_ethics` |
| positive | `P_contract_02` | 0.039 | `positive_variant` | `contractualism` |
| negative | `N_neutral_02` | -0.132 | `neutral_negative_length_matched` | `none` |
| negative | `N_anti_virtue_01` | -0.046 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| negative | `N_anti_deont_01` | -0.046 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `N_generic_moral_01` | -0.040 | `generic_moral_anchor` | `generic_moral` |
| negative | `N_anti_util_01` | -0.027 | `anti_theory_diagnostic` | `anti_utilitarian` |

#### PC5 (0.039 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_virtue_01` | 0.065 | `positive` | `virtue_ethics` |
| positive | `P_deont_01` | 0.059 | `positive` | `deontology` |
| positive | `N_anti_deont_01` | 0.057 | `anti_theory_diagnostic` | `anti_deontology` |
| positive | `P_deont_02` | 0.030 | `positive_variant` | `deontology` |
| positive | `N_neutral_01` | 0.028 | `neutral_negative` | `none` |
| negative | `N_neutral_02` | -0.046 | `neutral_negative_length_matched` | `none` |
| negative | `P_contractarian_01` | -0.040 | `positive` | `contractarianism` |
| negative | `P_contractarian_02` | -0.038 | `positive_variant` | `contractarianism` |
| negative | `P_contract_01` | -0.026 | `positive` | `contractualism` |
| negative | `N_anti_contractarian_01` | -0.026 | `anti_theory_diagnostic` | `anti_contractarianism` |

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

## generated_sequence_residual L24 first_16

- rows: `540`
- dilemmas: `30`
- conditions: `18`
- features: `2048`

### Scree

| PC | variance ratio | cumulative | CV condition-loading corr |
|---:|---:|---:|---:|
| 1 | 0.083 | 0.083 | 0.836 |
| 2 | 0.071 | 0.154 | 0.931 |
| 3 | 0.049 | 0.203 | 0.913 |
| 4 | 0.039 | 0.242 | 0.770 |
| 5 | 0.033 | 0.275 | 0.886 |
| 6 | 0.026 | 0.301 | 0.767 |
| 7 | 0.024 | 0.325 | 0.579 |
| 8 | 0.021 | 0.346 | 0.934 |

### Condition Extremes

#### PC1 (0.083 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_deont_02` | 0.780 | `positive_variant` | `deontology` |
| positive | `N_anti_contractarian_01` | 0.773 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `P_virtue_02` | 0.731 | `positive_variant` | `virtue_ethics` |
| positive | `N_anti_deont_01` | 0.485 | `anti_theory_diagnostic` | `anti_deontology` |
| positive | `N_anti_util_01` | 0.325 | `anti_theory_diagnostic` | `anti_utilitarian` |
| negative | `P_contract_01` | -0.735 | `positive` | `contractualism` |
| negative | `P_contractarian_01` | -0.700 | `positive` | `contractarianism` |
| negative | `N_neutral_01` | -0.501 | `neutral_negative` | `none` |
| negative | `P_contractarian_02` | -0.416 | `positive_variant` | `contractarianism` |
| negative | `P_util_01` | -0.411 | `positive` | `utilitarian` |

#### PC2 (0.071 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_generic_moral_01` | 0.668 | `generic_moral_anchor` | `generic_moral` |
| positive | `P_contractarian_02` | 0.575 | `positive_variant` | `contractarianism` |
| positive | `P_virtue_02` | 0.416 | `positive_variant` | `virtue_ethics` |
| positive | `P_contractarian_01` | 0.315 | `positive` | `contractarianism` |
| positive | `P_virtue_01` | 0.195 | `positive` | `virtue_ethics` |
| negative | `P_util_02` | -0.680 | `positive_variant` | `utilitarian` |
| negative | `P_util_01` | -0.488 | `positive` | `utilitarian` |
| negative | `N_anti_deont_01` | -0.398 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `P_deont_02` | -0.208 | `positive_variant` | `deontology` |
| negative | `N_anti_contract_01` | -0.201 | `anti_theory_diagnostic` | `anti_contractualism` |

#### PC3 (0.049 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_virtue_02` | 1.299 | `positive_variant` | `virtue_ethics` |
| positive | `P_deont_02` | 0.760 | `positive_variant` | `deontology` |
| positive | `P_contract_02` | 0.137 | `positive_variant` | `contractualism` |
| positive | `P_deont_01` | 0.085 | `positive` | `deontology` |
| positive | `P_util_01` | 0.076 | `positive` | `utilitarian` |
| negative | `N_anti_contractarian_01` | -0.590 | `anti_theory_diagnostic` | `anti_contractarianism` |
| negative | `N_neutral_02` | -0.314 | `neutral_negative_length_matched` | `none` |
| negative | `N_anti_contract_01` | -0.273 | `anti_theory_diagnostic` | `anti_contractualism` |
| negative | `N_anti_deont_01` | -0.264 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `N_anti_virtue_01` | -0.249 | `anti_theory_diagnostic` | `anti_virtue_ethics` |

#### PC4 (0.039 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_deont_01` | 0.648 | `positive` | `deontology` |
| positive | `P_virtue_01` | 0.323 | `positive` | `virtue_ethics` |
| positive | `P_contract_02` | 0.220 | `positive_variant` | `contractualism` |
| positive | `N_anti_util_01` | 0.218 | `anti_theory_diagnostic` | `anti_utilitarian` |
| positive | `P_contract_01` | 0.127 | `positive` | `contractualism` |
| negative | `N_anti_deont_01` | -0.454 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `P_contractarian_02` | -0.411 | `positive_variant` | `contractarianism` |
| negative | `N_anti_virtue_01` | -0.340 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| negative | `N_generic_moral_01` | -0.118 | `generic_moral_anchor` | `generic_moral` |
| negative | `P_contractarian_01` | -0.097 | `positive` | `contractarianism` |

#### PC5 (0.033 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_generic_moral_01` | 0.469 | `generic_moral_anchor` | `generic_moral` |
| positive | `P_virtue_01` | 0.335 | `positive` | `virtue_ethics` |
| positive | `N_neutral_02` | 0.230 | `neutral_negative_length_matched` | `none` |
| positive | `P_deont_02` | 0.193 | `positive_variant` | `deontology` |
| positive | `N_neutral_01` | 0.166 | `neutral_negative` | `none` |
| negative | `N_anti_virtue_01` | -0.415 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| negative | `N_anti_contractarian_01` | -0.398 | `anti_theory_diagnostic` | `anti_contractarianism` |
| negative | `N_anti_deont_01` | -0.391 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `P_contract_01` | -0.246 | `positive` | `contractualism` |
| negative | `P_virtue_02` | -0.151 | `positive_variant` | `virtue_ethics` |

## prompt_end_residual L24 full

- rows: `540`
- dilemmas: `30`
- conditions: `18`
- features: `2048`

### Scree

| PC | variance ratio | cumulative | CV condition-loading corr |
|---:|---:|---:|---:|
| 1 | 0.350 | 0.350 | 0.999 |
| 2 | 0.154 | 0.505 | 0.999 |
| 3 | 0.096 | 0.601 | 0.999 |
| 4 | 0.051 | 0.652 | 0.998 |
| 5 | 0.040 | 0.692 | 0.999 |
| 6 | 0.028 | 0.720 | 0.995 |
| 7 | 0.024 | 0.744 | 0.997 |
| 8 | 0.019 | 0.763 | 0.994 |

### Condition Extremes

#### PC1 (0.350 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_anti_contractarian_01` | 1.392 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `N_anti_deont_01` | 1.211 | `anti_theory_diagnostic` | `anti_deontology` |
| positive | `N_anti_util_01` | 1.125 | `anti_theory_diagnostic` | `anti_utilitarian` |
| positive | `N_anti_virtue_01` | 1.029 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| positive | `N_anti_contract_01` | 1.010 | `anti_theory_diagnostic` | `anti_contractualism` |
| negative | `P_contract_01` | -1.745 | `positive` | `contractualism` |
| negative | `P_virtue_01` | -1.482 | `positive` | `virtue_ethics` |
| negative | `N_generic_moral_01` | -1.424 | `generic_moral_anchor` | `generic_moral` |
| negative | `P_util_01` | -1.147 | `positive` | `utilitarian` |
| negative | `P_contractarian_01` | -0.821 | `positive` | `contractarianism` |

#### PC2 (0.154 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_contractarian_02` | 0.595 | `positive_variant` | `contractarianism` |
| positive | `P_contract_02` | 0.478 | `positive_variant` | `contractualism` |
| positive | `P_virtue_02` | 0.474 | `positive_variant` | `virtue_ethics` |
| positive | `N_anti_contractarian_01` | 0.465 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `P_contract_01` | 0.438 | `positive` | `contractualism` |
| negative | `N_neutral_01` | -1.761 | `neutral_negative` | `none` |
| negative | `N_neutral_02` | -1.541 | `neutral_negative_length_matched` | `none` |
| negative | `N_generic_moral_01` | -0.458 | `generic_moral_anchor` | `generic_moral` |
| negative | `N_anti_virtue_01` | -0.243 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| negative | `N_anti_util_01` | -0.077 | `anti_theory_diagnostic` | `anti_utilitarian` |

#### PC3 (0.096 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_anti_contractarian_01` | 0.711 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `P_contract_01` | 0.685 | `positive` | `contractualism` |
| positive | `N_anti_util_01` | 0.562 | `anti_theory_diagnostic` | `anti_utilitarian` |
| positive | `N_anti_contract_01` | 0.484 | `anti_theory_diagnostic` | `anti_contractualism` |
| positive | `N_anti_virtue_01` | 0.438 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| negative | `P_deont_02` | -0.889 | `positive_variant` | `deontology` |
| negative | `P_virtue_02` | -0.673 | `positive_variant` | `virtue_ethics` |
| negative | `P_contractarian_02` | -0.621 | `positive_variant` | `contractarianism` |
| negative | `P_contract_02` | -0.549 | `positive_variant` | `contractualism` |
| negative | `P_util_02` | -0.526 | `positive_variant` | `utilitarian` |

#### PC4 (0.051 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_deont_01` | 0.806 | `positive` | `deontology` |
| positive | `N_anti_deont_01` | 0.474 | `anti_theory_diagnostic` | `anti_deontology` |
| positive | `N_anti_virtue_01` | 0.470 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| positive | `N_generic_moral_01` | 0.340 | `generic_moral_anchor` | `generic_moral` |
| positive | `P_virtue_01` | 0.332 | `positive` | `virtue_ethics` |
| negative | `P_contract_01` | -0.601 | `positive` | `contractualism` |
| negative | `N_anti_contractarian_01` | -0.385 | `anti_theory_diagnostic` | `anti_contractarianism` |
| negative | `P_contract_02` | -0.359 | `positive_variant` | `contractualism` |
| negative | `P_util_02` | -0.310 | `positive_variant` | `utilitarian` |
| negative | `P_contractarian_02` | -0.305 | `positive_variant` | `contractarianism` |

#### PC5 (0.040 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_deont_02` | 0.552 | `positive_variant` | `deontology` |
| positive | `N_generic_moral_01` | 0.467 | `generic_moral_anchor` | `generic_moral` |
| positive | `P_contract_01` | 0.430 | `positive` | `contractualism` |
| positive | `N_anti_contract_01` | 0.281 | `anti_theory_diagnostic` | `anti_contractualism` |
| positive | `N_neutral_02` | 0.194 | `neutral_negative_length_matched` | `none` |
| negative | `P_util_01` | -0.664 | `positive` | `utilitarian` |
| negative | `P_contractarian_01` | -0.432 | `positive` | `contractarianism` |
| negative | `P_util_02` | -0.343 | `positive_variant` | `utilitarian` |
| negative | `N_neutral_01` | -0.340 | `neutral_negative` | `none` |
| negative | `P_deont_01` | -0.313 | `positive` | `deontology` |

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

## generated_sequence_residual L40 first_16

- rows: `540`
- dilemmas: `30`
- conditions: `18`
- features: `2048`

### Scree

| PC | variance ratio | cumulative | CV condition-loading corr |
|---:|---:|---:|---:|
| 1 | 0.097 | 0.097 | 0.949 |
| 2 | 0.070 | 0.167 | 0.956 |
| 3 | 0.051 | 0.218 | 0.980 |
| 4 | 0.039 | 0.257 | 0.911 |
| 5 | 0.033 | 0.290 | 0.866 |
| 6 | 0.026 | 0.316 | 0.563 |
| 7 | 0.026 | 0.342 | 0.649 |
| 8 | 0.021 | 0.363 | 0.953 |

### Condition Extremes

#### PC1 (0.097 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_contract_01` | 2.192 | `positive` | `contractualism` |
| positive | `P_contractarian_01` | 2.105 | `positive` | `contractarianism` |
| positive | `N_neutral_01` | 1.632 | `neutral_negative` | `none` |
| positive | `P_contractarian_02` | 1.558 | `positive_variant` | `contractarianism` |
| positive | `N_neutral_02` | 1.420 | `neutral_negative_length_matched` | `none` |
| negative | `P_virtue_02` | -2.786 | `positive_variant` | `virtue_ethics` |
| negative | `N_anti_contractarian_01` | -2.501 | `anti_theory_diagnostic` | `anti_contractarianism` |
| negative | `P_deont_02` | -2.126 | `positive_variant` | `deontology` |
| negative | `N_anti_util_01` | -1.447 | `anti_theory_diagnostic` | `anti_utilitarian` |
| negative | `N_anti_deont_01` | -1.395 | `anti_theory_diagnostic` | `anti_deontology` |

#### PC2 (0.070 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_virtue_02` | 3.244 | `positive_variant` | `virtue_ethics` |
| positive | `N_generic_moral_01` | 1.278 | `generic_moral_anchor` | `generic_moral` |
| positive | `P_contractarian_02` | 1.277 | `positive_variant` | `contractarianism` |
| positive | `P_deont_01` | 1.246 | `positive` | `deontology` |
| positive | `P_virtue_01` | 1.026 | `positive` | `virtue_ethics` |
| negative | `N_anti_deont_01` | -2.385 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `P_util_02` | -2.148 | `positive_variant` | `utilitarian` |
| negative | `N_anti_virtue_01` | -1.466 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| negative | `P_util_01` | -1.369 | `positive` | `utilitarian` |
| negative | `N_anti_contract_01` | -0.966 | `anti_theory_diagnostic` | `anti_contractualism` |

#### PC3 (0.051 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_anti_contractarian_01` | 1.685 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `N_anti_deont_01` | 1.382 | `anti_theory_diagnostic` | `anti_deontology` |
| positive | `N_generic_moral_01` | 1.251 | `generic_moral_anchor` | `generic_moral` |
| positive | `P_contractarian_02` | 1.241 | `positive_variant` | `contractarianism` |
| positive | `N_anti_virtue_01` | 1.081 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| negative | `P_virtue_02` | -2.260 | `positive_variant` | `virtue_ethics` |
| negative | `P_deont_02` | -2.079 | `positive_variant` | `deontology` |
| negative | `P_deont_01` | -1.143 | `positive` | `deontology` |
| negative | `P_contract_02` | -1.111 | `positive_variant` | `contractualism` |
| negative | `P_virtue_01` | -0.743 | `positive` | `virtue_ethics` |

#### PC4 (0.039 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_deont_01` | 1.750 | `positive` | `deontology` |
| positive | `P_virtue_01` | 1.101 | `positive` | `virtue_ethics` |
| positive | `N_neutral_02` | 0.886 | `neutral_negative_length_matched` | `none` |
| positive | `N_generic_moral_01` | 0.803 | `generic_moral_anchor` | `generic_moral` |
| positive | `P_contract_02` | 0.540 | `positive_variant` | `contractualism` |
| negative | `P_virtue_02` | -2.113 | `positive_variant` | `virtue_ethics` |
| negative | `N_anti_deont_01` | -1.751 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `N_anti_virtue_01` | -1.086 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| negative | `P_deont_02` | -0.494 | `positive_variant` | `deontology` |
| negative | `P_contractarian_02` | -0.420 | `positive_variant` | `contractarianism` |

#### PC5 (0.033 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_anti_contractarian_01` | 1.006 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `N_anti_virtue_01` | 0.994 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| positive | `N_anti_deont_01` | 0.829 | `anti_theory_diagnostic` | `anti_deontology` |
| positive | `P_contract_01` | 0.758 | `positive` | `contractualism` |
| positive | `N_anti_util_01` | 0.751 | `anti_theory_diagnostic` | `anti_utilitarian` |
| negative | `P_deont_02` | -1.377 | `positive_variant` | `deontology` |
| negative | `N_generic_moral_01` | -1.241 | `generic_moral_anchor` | `generic_moral` |
| negative | `N_neutral_02` | -0.554 | `neutral_negative_length_matched` | `none` |
| negative | `P_contractarian_02` | -0.518 | `positive_variant` | `contractarianism` |
| negative | `N_neutral_01` | -0.463 | `neutral_negative` | `none` |

## prompt_end_residual L40 full

- rows: `540`
- dilemmas: `30`
- conditions: `18`
- features: `2048`

### Scree

| PC | variance ratio | cumulative | CV condition-loading corr |
|---:|---:|---:|---:|
| 1 | 0.305 | 0.305 | 0.999 |
| 2 | 0.186 | 0.492 | 0.998 |
| 3 | 0.091 | 0.583 | 0.998 |
| 4 | 0.038 | 0.621 | 0.997 |
| 5 | 0.035 | 0.656 | 0.988 |
| 6 | 0.025 | 0.680 | 0.996 |
| 7 | 0.024 | 0.704 | 0.995 |
| 8 | 0.021 | 0.725 | 0.990 |

### Condition Extremes

#### PC1 (0.305 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_anti_deont_01` | 3.357 | `anti_theory_diagnostic` | `anti_deontology` |
| positive | `N_anti_contractarian_01` | 3.345 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `N_anti_virtue_01` | 3.220 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| positive | `N_anti_contract_01` | 2.698 | `anti_theory_diagnostic` | `anti_contractualism` |
| positive | `N_anti_util_01` | 2.637 | `anti_theory_diagnostic` | `anti_utilitarian` |
| negative | `P_contract_01` | -3.854 | `positive` | `contractualism` |
| negative | `P_virtue_01` | -3.237 | `positive` | `virtue_ethics` |
| negative | `P_util_01` | -2.563 | `positive` | `utilitarian` |
| negative | `N_generic_moral_01` | -2.379 | `generic_moral_anchor` | `generic_moral` |
| negative | `P_deont_01` | -1.499 | `positive` | `deontology` |

#### PC2 (0.186 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_deont_02` | 2.022 | `positive_variant` | `deontology` |
| positive | `P_contractarian_02` | 1.639 | `positive_variant` | `contractarianism` |
| positive | `P_contract_02` | 1.575 | `positive_variant` | `contractualism` |
| positive | `P_virtue_02` | 1.483 | `positive_variant` | `virtue_ethics` |
| positive | `P_util_02` | 1.088 | `positive_variant` | `utilitarian` |
| negative | `N_neutral_01` | -4.600 | `neutral_negative` | `none` |
| negative | `N_neutral_02` | -4.395 | `neutral_negative_length_matched` | `none` |
| negative | `N_anti_virtue_01` | -0.521 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| negative | `N_generic_moral_01` | -0.478 | `generic_moral_anchor` | `generic_moral` |
| negative | `N_anti_util_01` | -0.347 | `anti_theory_diagnostic` | `anti_utilitarian` |

#### PC3 (0.091 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_contract_01` | 2.084 | `positive` | `contractualism` |
| positive | `N_anti_deont_01` | 1.499 | `anti_theory_diagnostic` | `anti_deontology` |
| positive | `N_anti_virtue_01` | 1.305 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| positive | `P_util_01` | 1.183 | `positive` | `utilitarian` |
| positive | `N_anti_util_01` | 1.140 | `anti_theory_diagnostic` | `anti_utilitarian` |
| negative | `P_deont_02` | -1.793 | `positive_variant` | `deontology` |
| negative | `P_contract_02` | -1.463 | `positive_variant` | `contractualism` |
| negative | `P_contractarian_02` | -1.439 | `positive_variant` | `contractarianism` |
| negative | `P_virtue_02` | -1.256 | `positive_variant` | `virtue_ethics` |
| negative | `N_neutral_02` | -1.227 | `neutral_negative_length_matched` | `none` |

#### PC4 (0.038 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `P_util_01` | 1.254 | `positive` | `utilitarian` |
| positive | `P_contractarian_01` | 0.962 | `positive` | `contractarianism` |
| positive | `P_util_02` | 0.941 | `positive_variant` | `utilitarian` |
| positive | `P_contractarian_02` | 0.811 | `positive_variant` | `contractarianism` |
| positive | `N_anti_contractarian_01` | 0.595 | `anti_theory_diagnostic` | `anti_contractarianism` |
| negative | `P_deont_02` | -1.550 | `positive_variant` | `deontology` |
| negative | `N_generic_moral_01` | -1.333 | `generic_moral_anchor` | `generic_moral` |
| negative | `P_virtue_02` | -1.084 | `positive_variant` | `virtue_ethics` |
| negative | `N_anti_virtue_01` | -0.400 | `anti_theory_diagnostic` | `anti_virtue_ethics` |
| negative | `N_neutral_01` | -0.380 | `neutral_negative` | `none` |

#### PC5 (0.035 variance)

| side | condition | score | role | theory |
|---|---|---:|---|---|
| positive | `N_anti_contractarian_01` | 1.199 | `anti_theory_diagnostic` | `anti_contractarianism` |
| positive | `P_contract_01` | 1.194 | `positive` | `contractualism` |
| positive | `N_neutral_01` | 0.597 | `neutral_negative` | `none` |
| positive | `P_contractarian_02` | 0.511 | `positive_variant` | `contractarianism` |
| positive | `P_deont_02` | 0.487 | `positive_variant` | `deontology` |
| negative | `P_deont_01` | -1.581 | `positive` | `deontology` |
| negative | `N_anti_deont_01` | -0.998 | `anti_theory_diagnostic` | `anti_deontology` |
| negative | `P_virtue_01` | -0.826 | `positive` | `virtue_ethics` |
| negative | `P_virtue_02` | -0.701 | `positive_variant` | `virtue_ethics` |
| negative | `N_neutral_02` | -0.405 | `neutral_negative_length_matched` | `none` |
