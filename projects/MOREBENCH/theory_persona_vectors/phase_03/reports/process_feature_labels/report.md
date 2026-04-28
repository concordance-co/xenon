# Process Feature Labels

- scored rows: 540 / 540
- scores JSONL: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/process_feature_labels/process_feature_scores.jsonl`
- scorer type for this run: `keyword_baseline`

## Feature Distributions

| feature | mean | score 0 | score 1 | score 2 |
|---|---:|---:|---:|---:|
| stakeholder_identification | 1.981 | 0 | 10 | 530 |
| consequence_forecasting | 0.287 | 391 | 143 | 6 |
| tradeoff_acknowledged | 1.259 | 66 | 268 | 206 |
| priority_resolution | 0.406 | 332 | 197 | 11 |
| moral_uncertainty | 0.185 | 449 | 82 | 9 |
| risk_mitigation | 0.278 | 414 | 102 | 24 |
| conditional_recommendation | 0.463 | 317 | 196 | 27 |
| procedural_escalation | 0.402 | 363 | 137 | 40 |

## Condition Mean Table

| condition | n | stakeholder_identification | consequence_forecasting | tradeoff_acknowledged | priority_resolution | moral_uncertainty | risk_mitigation | conditional_recommendation | procedural_escalation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N_anti_contract_01 | 30 | 1.933 | 0.167 | 1.133 | 0.433 | 0.133 | 0.167 | 0.300 | 0.267 |
| N_anti_contractarian_01 | 30 | 1.967 | 0.167 | 1.167 | 0.833 | 0.100 | 0.333 | 0.500 | 0.433 |
| N_anti_deont_01 | 30 | 1.967 | 0.333 | 0.967 | 0.467 | 0.133 | 0.167 | 0.333 | 0.233 |
| N_anti_util_01 | 30 | 1.967 | 0.433 | 0.800 | 0.567 | 0.167 | 0.167 | 0.400 | 0.300 |
| N_anti_virtue_01 | 30 | 1.967 | 0.200 | 0.867 | 0.367 | 0.067 | 0.133 | 0.433 | 0.467 |
| N_generic_moral_01 | 30 | 2.000 | 0.167 | 1.533 | 0.300 | 0.267 | 0.400 | 0.633 | 0.467 |
| N_neutral_01 | 30 | 1.967 | 0.233 | 1.333 | 0.233 | 0.133 | 0.267 | 0.467 | 0.467 |
| N_neutral_02 | 30 | 2.000 | 0.300 | 1.267 | 0.267 | 0.100 | 0.300 | 0.433 | 0.533 |
| P_contract_01 | 30 | 1.967 | 0.200 | 1.567 | 0.300 | 0.067 | 0.367 | 0.200 | 0.467 |
| P_contract_02 | 30 | 2.000 | 0.467 | 1.033 | 0.367 | 0.200 | 0.267 | 0.833 | 0.333 |
| P_contractarian_01 | 30 | 2.000 | 0.233 | 1.567 | 0.300 | 0.067 | 0.300 | 0.333 | 0.433 |
| P_contractarian_02 | 30 | 2.000 | 0.200 | 1.600 | 0.100 | 0.133 | 0.200 | 0.567 | 0.467 |
| P_deont_01 | 30 | 2.000 | 0.267 | 1.300 | 0.467 | 0.300 | 0.533 | 0.600 | 0.700 |
| P_deont_02 | 30 | 2.000 | 0.533 | 0.600 | 0.500 | 0.267 | 0.267 | 0.667 | 0.267 |
| P_util_01 | 30 | 2.000 | 0.400 | 1.700 | 0.333 | 0.200 | 0.433 | 0.267 | 0.433 |
| P_util_02 | 30 | 2.000 | 0.233 | 1.933 | 0.767 | 0.267 | 0.267 | 0.533 | 0.367 |
| P_virtue_01 | 30 | 1.967 | 0.133 | 1.400 | 0.300 | 0.433 | 0.333 | 0.600 | 0.433 |
| P_virtue_02 | 30 | 1.967 | 0.500 | 0.900 | 0.400 | 0.300 | 0.100 | 0.233 | 0.167 |
