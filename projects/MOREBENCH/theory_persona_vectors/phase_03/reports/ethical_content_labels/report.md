# Ethical Content Labels

- scores JSONL: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/ethical_content_labels/scores.jsonl`
- loaded rows: `540`
- scored rows in JSONL: `540`
- scorer counts: `{"keyword_baseline": 540}`

## Per-Dimension Distributions

| dimension | score 0 | score 1 | score 2 |
|---|---:|---:|---:|
| harm_welfare | 141 | 160 | 239 |
| rights_autonomy | 344 | 152 | 44 |
| fairness_justice | 381 | 116 | 43 |
| honesty_truthfulness | 381 | 115 | 44 |
| responsibility_accountability | 234 | 195 | 111 |
| loyalty_trust | 340 | 146 | 54 |
| legality_compliance | 428 | 79 | 33 |
| public_interest_social_impact | 289 | 197 | 54 |
| virtue_character | 393 | 118 | 29 |
| care_compassion | 347 | 134 | 59 |

## Condition Mean Scores

| condition | n | harm_welfare | rights_autonomy | fairness_justice | honesty_truthfulness | responsibility_accountability | loyalty_trust | legality_compliance | public_interest_social_impact | virtue_character | care_compassion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N_anti_contract_01 | 30 | 0.967 | 0.233 | 0.200 | 0.167 | 0.400 | 0.200 | 0.267 | 0.600 | 0.133 | 0.400 |
| N_anti_contractarian_01 | 30 | 1.333 | 0.333 | 0.233 | 0.333 | 0.767 | 0.300 | 0.300 | 0.600 | 0.133 | 0.267 |
| N_anti_deont_01 | 30 | 1.167 | 0.167 | 0.000 | 0.200 | 0.233 | 0.133 | 0.167 | 0.500 | 0.033 | 0.167 |
| N_anti_util_01 | 30 | 1.167 | 0.467 | 0.300 | 0.233 | 0.700 | 0.333 | 0.167 | 0.467 | 0.200 | 0.467 |
| N_anti_virtue_01 | 30 | 1.033 | 0.200 | 0.100 | 0.267 | 0.733 | 0.133 | 0.333 | 0.367 | 0.200 | 0.333 |
| N_generic_moral_01 | 30 | 1.233 | 0.633 | 0.300 | 0.433 | 1.100 | 0.433 | 0.267 | 0.700 | 0.267 | 0.733 |
| N_neutral_01 | 30 | 0.900 | 0.200 | 0.167 | 0.367 | 0.533 | 0.167 | 0.233 | 0.533 | 0.100 | 0.467 |
| N_neutral_02 | 30 | 0.933 | 0.233 | 0.167 | 0.267 | 0.567 | 0.267 | 0.300 | 0.500 | 0.100 | 0.600 |
| P_contract_01 | 30 | 1.067 | 0.600 | 0.433 | 0.467 | 1.133 | 0.467 | 0.300 | 0.700 | 0.233 | 0.633 |
| P_contract_02 | 30 | 1.133 | 0.700 | 1.433 | 0.333 | 0.633 | 0.433 | 0.333 | 0.367 | 0.167 | 0.400 |
| P_contractarian_01 | 30 | 1.133 | 0.333 | 0.533 | 0.467 | 0.767 | 0.800 | 0.400 | 0.600 | 0.133 | 0.333 |
| P_contractarian_02 | 30 | 0.867 | 0.400 | 0.333 | 0.467 | 0.667 | 1.100 | 0.300 | 0.900 | 0.300 | 0.467 |
| P_deont_01 | 30 | 1.433 | 1.233 | 0.433 | 0.500 | 1.767 | 0.733 | 0.267 | 0.667 | 0.267 | 0.667 |
| P_deont_02 | 30 | 1.233 | 0.467 | 0.133 | 0.333 | 1.067 | 1.300 | 0.267 | 0.400 | 0.500 | 0.367 |
| P_util_01 | 30 | 1.767 | 0.267 | 0.367 | 0.300 | 0.567 | 0.300 | 0.267 | 0.833 | 0.167 | 0.433 |
| P_util_02 | 30 | 1.833 | 0.267 | 0.167 | 0.267 | 0.400 | 0.067 | 0.167 | 0.733 | 0.100 | 0.233 |
| P_virtue_01 | 30 | 1.100 | 0.567 | 1.000 | 0.733 | 0.767 | 0.433 | 0.100 | 0.433 | 1.667 | 0.767 |
| P_virtue_02 | 30 | 0.967 | 0.700 | 0.433 | 0.633 | 1.100 | 0.867 | 0.400 | 0.267 | 1.167 | 0.667 |
