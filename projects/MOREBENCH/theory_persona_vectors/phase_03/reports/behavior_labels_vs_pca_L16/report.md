# Behavioral Labels vs PCA

- ethical labels: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/ethical_content_labels/scores.jsonl`
- process labels: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/process_feature_labels/process_feature_scores.jsonl`
- layer: `L16 generated first16`
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
| 1 | 0.016 | 0.019 | 0.021 | 0.127 | 0.190 |
| 2 | 0.072 | 0.020 | 0.065 | 0.218 | 0.242 |
| 3 | 0.053 | 0.001 | 0.049 | 0.215 | 0.377 |
| 4 | 0.076 | 0.010 | 0.089 | 0.218 | 0.232 |
| 5 | -0.013 | -0.007 | -0.016 | 0.100 | 0.116 |

## Strongest Feature Correlations By PC

### PC1

| feature | corr |
|---|---:|
| `loyalty_trust` | -0.141 |
| `tradeoff_acknowledged` | 0.136 |
| `priority_resolution` | -0.117 |
| `moral_uncertainty` | -0.112 |
| `virtue_character` | -0.110 |
| `consequence_forecasting` | -0.110 |
| `conditional_recommendation` | -0.088 |
| `rights_autonomy` | -0.088 |
| `responsibility_accountability` | -0.083 |
| `public_interest_social_impact` | 0.063 |

### PC2

| feature | corr |
|---|---:|
| `harm_welfare` | -0.192 |
| `priority_resolution` | -0.179 |
| `responsibility_accountability` | 0.169 |
| `tradeoff_acknowledged` | 0.133 |
| `fairness_justice` | 0.132 |
| `rights_autonomy` | 0.128 |
| `procedural_escalation` | 0.122 |
| `virtue_character` | 0.120 |
| `loyalty_trust` | 0.094 |
| `care_compassion` | 0.088 |

### PC3

| feature | corr |
|---|---:|
| `virtue_character` | 0.218 |
| `loyalty_trust` | 0.202 |
| `responsibility_accountability` | 0.122 |
| `honesty_truthfulness` | 0.121 |
| `moral_uncertainty` | 0.109 |
| `procedural_escalation` | -0.104 |
| `tradeoff_acknowledged` | -0.099 |
| `rights_autonomy` | 0.094 |
| `priority_resolution` | -0.076 |
| `consequence_forecasting` | 0.074 |

### PC4

| feature | corr |
|---|---:|
| `virtue_character` | 0.176 |
| `rights_autonomy` | 0.161 |
| `fairness_justice` | 0.155 |
| `priority_resolution` | 0.152 |
| `responsibility_accountability` | 0.111 |
| `loyalty_trust` | 0.102 |
| `harm_welfare` | 0.096 |
| `tradeoff_acknowledged` | -0.093 |
| `consequence_forecasting` | 0.087 |
| `risk_mitigation` | 0.084 |

### PC5

| feature | corr |
|---|---:|
| `priority_resolution` | -0.101 |
| `stakeholder_identification` | 0.092 |
| `conditional_recommendation` | 0.080 |
| `tradeoff_acknowledged` | 0.079 |
| `moral_uncertainty` | 0.074 |
| `harm_welfare` | 0.067 |
| `virtue_character` | 0.057 |
| `legality_compliance` | -0.045 |
| `consequence_forecasting` | 0.041 |
| `rights_autonomy` | -0.035 |
