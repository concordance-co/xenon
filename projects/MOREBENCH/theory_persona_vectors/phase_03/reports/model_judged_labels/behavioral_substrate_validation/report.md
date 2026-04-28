# Behavioral Substrate Validation

- rows: `540`
- dilemmas: `30`
- conditions: `18`
- global condition-separation stat: `1970.346`
- within-dilemma permutation null p95: `283.640`
- permutation p-value: `0.001`

## Most Condition-Sensitive Labels

| feature | eta^2 by condition |
|---|---:|
| `virtue_character` | 0.566 |
| `virtue_character_singleton` | 0.566 |
| `loyalty_trust` | 0.236 |
| `fairness_justice` | 0.211 |
| `tradeoff_acknowledged` | 0.207 |
| `risk_mitigation` | 0.183 |
| `priority_resolution` | 0.166 |
| `decisive_resolution` | 0.166 |
| `procedural_decisive_axis` | 0.164 |
| `responsibility_accountability` | 0.159 |
| `consequence_forecasting` | 0.140 |
| `procedural_risk_management` | 0.140 |
| `stakeholder_identification` | 0.118 |
| `rights_autonomy` | 0.113 |
| `conditional_recommendation` | 0.105 |
| `harm_welfare` | 0.082 |
| `care_compassion` | 0.073 |
| `public_interest_social_impact` | 0.065 |
| `honesty_truthfulness` | 0.059 |
| `procedural_escalation` | 0.050 |

## Condition Means: Primary Axes

| condition | n | procedural_risk | decisive | procedural_decisive | virtue |
|---|---:|---:|---:|---:|---:|
| `N_anti_contract_01` | 30 | 0.520 | 1.567 | -1.047 | 0.133 |
| `N_anti_contractarian_01` | 30 | 0.420 | 1.867 | -1.447 | 0.233 |
| `N_anti_deont_01` | 30 | 0.460 | 1.467 | -1.007 | 0.000 |
| `N_anti_util_01` | 30 | 0.300 | 1.733 | -1.433 | 0.233 |
| `N_anti_virtue_01` | 30 | 0.547 | 1.400 | -0.853 | 0.067 |
| `N_generic_moral_01` | 30 | 0.927 | 1.433 | -0.507 | 0.300 |
| `N_neutral_01` | 30 | 0.727 | 1.233 | -0.507 | 0.100 |
| `N_neutral_02` | 30 | 0.773 | 1.433 | -0.660 | 0.100 |
| `P_contract_01` | 30 | 0.660 | 1.633 | -0.973 | 0.267 |
| `P_contract_02` | 30 | 0.467 | 1.733 | -1.267 | 0.100 |
| `P_contractarian_01` | 30 | 0.780 | 1.400 | -0.620 | 0.167 |
| `P_contractarian_02` | 30 | 0.887 | 1.333 | -0.447 | 0.233 |
| `P_deont_01` | 30 | 0.720 | 1.900 | -1.180 | 0.433 |
| `P_deont_02` | 30 | 0.460 | 1.867 | -1.407 | 0.833 |
| `P_util_01` | 30 | 0.547 | 1.733 | -1.187 | 0.200 |
| `P_util_02` | 30 | 0.360 | 1.867 | -1.507 | 0.067 |
| `P_virtue_01` | 30 | 0.707 | 1.567 | -0.860 | 1.833 |
| `P_virtue_02` | 30 | 0.287 | 1.567 | -1.280 | 1.833 |

## Largest Pairwise Behavioral Distances

| condition A | condition B | euclidean z | ridge mahalanobis | proc-decisive delta | virtue delta |
|---|---|---:|---:|---:|---:|
| `N_anti_deont_01` | `P_virtue_01` | 4.683 | 10.106 | -0.147 | -1.833 |
| `P_contractarian_02` | `P_virtue_02` | 4.558 | 8.885 | 0.833 | -1.600 |
| `P_util_02` | `P_virtue_01` | 4.473 | 8.158 | -0.647 | -1.767 |
| `P_util_02` | `P_virtue_02` | 4.426 | 8.989 | -0.227 | -1.767 |
| `N_generic_moral_01` | `P_virtue_02` | 4.371 | 7.165 | 0.773 | -1.533 |
| `P_contractarian_01` | `P_virtue_02` | 4.336 | 8.462 | 0.660 | -1.667 |
| `N_neutral_01` | `P_virtue_02` | 4.237 | 6.745 | 0.773 | -1.733 |
| `N_neutral_02` | `P_virtue_02` | 4.199 | 7.050 | 0.620 | -1.733 |
| `N_anti_deont_01` | `P_virtue_02` | 4.178 | 8.038 | 0.273 | -1.833 |
| `P_contract_02` | `P_virtue_02` | 4.148 | 9.820 | 0.013 | -1.733 |
| `N_anti_virtue_01` | `P_virtue_01` | 4.142 | 8.573 | 0.007 | -1.767 |
| `P_util_01` | `P_virtue_02` | 4.128 | 7.981 | 0.093 | -1.633 |
| `N_anti_util_01` | `P_virtue_01` | 3.954 | 7.260 | -0.573 | -1.600 |
| `P_contract_02` | `P_virtue_01` | 3.942 | 8.106 | -0.407 | -1.733 |
| `P_deont_01` | `P_virtue_02` | 3.932 | 7.079 | 0.100 | -1.400 |
| `N_anti_contractarian_01` | `P_virtue_01` | 3.930 | 7.158 | -0.587 | -1.600 |
| `P_contract_01` | `P_virtue_02` | 3.914 | 7.076 | 0.307 | -1.567 |
| `N_anti_virtue_01` | `P_virtue_02` | 3.906 | 7.165 | 0.427 | -1.767 |
| `N_neutral_01` | `P_virtue_01` | 3.891 | 7.118 | 0.353 | -1.733 |
| `N_anti_contract_01` | `P_virtue_01` | 3.850 | 6.948 | -0.187 | -1.700 |
| `P_contractarian_02` | `P_virtue_01` | 3.792 | 8.157 | 0.413 | -1.600 |
| `N_anti_contractarian_01` | `P_virtue_02` | 3.784 | 6.832 | -0.167 | -1.600 |
| `N_anti_contract_01` | `P_virtue_02` | 3.769 | 6.198 | 0.233 | -1.700 |
| `P_util_01` | `P_virtue_01` | 3.713 | 6.345 | -0.327 | -1.633 |
| `N_neutral_02` | `P_virtue_01` | 3.710 | 6.739 | 0.200 | -1.733 |
