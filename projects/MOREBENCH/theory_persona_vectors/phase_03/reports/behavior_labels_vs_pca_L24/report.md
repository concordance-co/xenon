# Behavioral Labels vs PCA

- ethical labels: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/ethical_content_labels/scores.jsonl`
- process labels: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/process_feature_labels/process_feature_scores.jsonl`
- layer: `L24 generated first16`
- joined rows: `540`
- dilemmas: `30`
- conditions: `18`

## Feature Distributions

| feature | mean | std | counts | majority rate | drop? |
|---|---:|---:|---|---:|---|
| `harm_welfare` | 1.181 | 0.819 | `{"0": 141, "1": 160, "2": 239}` | 0.443 | `False` |
| `rights_autonomy` | 0.444 | 0.640 | `{"0": 344, "1": 152, "2": 44}` | 0.637 | `False` |
| `fairness_justice` | 0.374 | 0.627 | `{"0": 381, "1": 116, "2": 43}` | 0.706 | `False` |
| `honesty_truthfulness` | 0.376 | 0.631 | `{"0": 381, "1": 115, "2": 44}` | 0.706 | `False` |
| `responsibility_accountability` | 0.772 | 0.766 | `{"0": 234, "1": 195, "2": 111}` | 0.433 | `False` |
| `loyalty_trust` | 0.470 | 0.670 | `{"0": 340, "1": 146, "2": 54}` | 0.630 | `False` |
| `legality_compliance` | 0.269 | 0.564 | `{"0": 428, "1": 79, "2": 33}` | 0.793 | `False` |
| `public_interest_social_impact` | 0.565 | 0.668 | `{"0": 289, "1": 197, "2": 54}` | 0.535 | `False` |
| `virtue_character` | 0.326 | 0.572 | `{"0": 393, "1": 118, "2": 29}` | 0.728 | `False` |
| `care_compassion` | 0.467 | 0.684 | `{"0": 347, "1": 134, "2": 59}` | 0.643 | `False` |
| `stakeholder_identification` | 1.981 | 0.135 | `{"1": 10, "2": 530}` | 0.981 | `True` |
| `consequence_forecasting` | 0.287 | 0.476 | `{"0": 391, "1": 143, "2": 6}` | 0.724 | `False` |
| `tradeoff_acknowledged` | 1.259 | 0.661 | `{"0": 66, "1": 268, "2": 206}` | 0.496 | `False` |
| `priority_resolution` | 0.406 | 0.531 | `{"0": 332, "1": 197, "2": 11}` | 0.615 | `False` |
| `moral_uncertainty` | 0.185 | 0.429 | `{"0": 449, "1": 82, "2": 9}` | 0.831 | `False` |
| `risk_mitigation` | 0.278 | 0.538 | `{"0": 414, "1": 102, "2": 24}` | 0.767 | `False` |
| `conditional_recommendation` | 0.463 | 0.590 | `{"0": 317, "1": 196, "2": 27}` | 0.587 | `False` |
| `procedural_escalation` | 0.402 | 0.623 | `{"0": 363, "1": 137, "2": 40}` | 0.672 | `False` |

## LOO Dilemma R2 Predicting Activation PCs

| PC | content | process | content+process | theory one-hot | condition one-hot |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.023 | 0.078 | 0.068 | 0.197 | 0.279 |
| 2 | 0.049 | -0.019 | 0.022 | 0.155 | 0.151 |
| 3 | 0.073 | -0.006 | 0.061 | 0.242 | 0.383 |
| 4 | 0.045 | -0.007 | 0.045 | 0.122 | 0.148 |
| 5 | -0.002 | -0.011 | -0.011 | 0.145 | 0.168 |

## Strongest Feature Correlations By PC

### PC1

| feature | corr |
|---|---:|
| `priority_resolution` | 0.220 |
| `tradeoff_acknowledged` | -0.208 |
| `harm_welfare` | 0.157 |
| `consequence_forecasting` | 0.131 |
| `procedural_escalation` | -0.128 |
| `moral_uncertainty` | 0.100 |
| `loyalty_trust` | 0.067 |
| `fairness_justice` | -0.059 |
| `conditional_recommendation` | 0.058 |
| `virtue_character` | 0.054 |

### PC2

| feature | corr |
|---|---:|
| `responsibility_accountability` | 0.165 |
| `loyalty_trust` | 0.146 |
| `rights_autonomy` | 0.139 |
| `virtue_character` | 0.122 |
| `harm_welfare` | -0.114 |
| `fairness_justice` | 0.092 |
| `care_compassion` | 0.080 |
| `procedural_escalation` | 0.073 |
| `conditional_recommendation` | 0.064 |
| `priority_resolution` | -0.059 |

### PC3

| feature | corr |
|---|---:|
| `virtue_character` | 0.254 |
| `loyalty_trust` | 0.214 |
| `responsibility_accountability` | 0.142 |
| `rights_autonomy` | 0.117 |
| `honesty_truthfulness` | 0.112 |
| `tradeoff_acknowledged` | -0.110 |
| `moral_uncertainty` | 0.093 |
| `procedural_escalation` | -0.086 |
| `consequence_forecasting` | 0.083 |
| `fairness_justice` | 0.082 |

### PC4

| feature | corr |
|---|---:|
| `virtue_character` | 0.145 |
| `fairness_justice` | 0.144 |
| `rights_autonomy` | 0.143 |
| `priority_resolution` | 0.111 |
| `risk_mitigation` | 0.102 |
| `responsibility_accountability` | 0.097 |
| `harm_welfare` | 0.070 |
| `procedural_escalation` | 0.067 |
| `tradeoff_acknowledged` | -0.062 |
| `loyalty_trust` | 0.060 |

### PC5

| feature | corr |
|---|---:|
| `stakeholder_identification` | 0.113 |
| `conditional_recommendation` | 0.097 |
| `virtue_character` | 0.093 |
| `tradeoff_acknowledged` | 0.089 |
| `priority_resolution` | -0.081 |
| `moral_uncertainty` | 0.076 |
| `harm_welfare` | 0.076 |
| `fairness_justice` | 0.052 |
| `loyalty_trust` | 0.048 |
| `consequence_forecasting` | 0.046 |
