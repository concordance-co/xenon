# Behavioral Labels vs PCA

- ethical labels: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/ethical_content_labels/scores.jsonl`
- process labels: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/process_feature_labels/process_feature_scores.jsonl`
- layer: `L40 generated first16`
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
| 1 | 0.032 | 0.122 | 0.113 | 0.292 | 0.396 |
| 2 | 0.225 | -0.015 | 0.192 | 0.403 | 0.434 |
| 3 | 0.090 | -0.013 | 0.079 | 0.383 | 0.412 |
| 4 | 0.025 | -0.003 | 0.020 | 0.151 | 0.346 |
| 5 | -0.006 | 0.006 | -0.004 | 0.164 | 0.205 |

## Strongest Feature Correlations By PC

### PC1

| feature | corr |
|---|---:|
| `tradeoff_acknowledged` | 0.265 |
| `priority_resolution` | -0.252 |
| `harm_welfare` | -0.149 |
| `procedural_escalation` | 0.144 |
| `consequence_forecasting` | -0.136 |
| `moral_uncertainty` | -0.093 |
| `virtue_character` | -0.089 |
| `loyalty_trust` | -0.076 |
| `public_interest_social_impact` | 0.069 |
| `honesty_truthfulness` | 0.048 |

### PC2

| feature | corr |
|---|---:|
| `virtue_character` | 0.304 |
| `loyalty_trust` | 0.295 |
| `responsibility_accountability` | 0.278 |
| `rights_autonomy` | 0.257 |
| `fairness_justice` | 0.191 |
| `harm_welfare` | -0.121 |
| `care_compassion` | 0.117 |
| `honesty_truthfulness` | 0.112 |
| `moral_uncertainty` | 0.090 |
| `priority_resolution` | -0.084 |

### PC3

| feature | corr |
|---|---:|
| `virtue_character` | -0.261 |
| `loyalty_trust` | -0.191 |
| `fairness_justice` | -0.149 |
| `tradeoff_acknowledged` | 0.124 |
| `responsibility_accountability` | -0.119 |
| `rights_autonomy` | -0.105 |
| `procedural_escalation` | 0.064 |
| `honesty_truthfulness` | -0.063 |
| `moral_uncertainty` | -0.058 |
| `consequence_forecasting` | -0.050 |

### PC4

| feature | corr |
|---|---:|
| `risk_mitigation` | 0.141 |
| `fairness_justice` | 0.120 |
| `harm_welfare` | 0.115 |
| `procedural_escalation` | 0.112 |
| `tradeoff_acknowledged` | 0.101 |
| `care_compassion` | 0.096 |
| `honesty_truthfulness` | -0.081 |
| `conditional_recommendation` | 0.078 |
| `rights_autonomy` | 0.072 |
| `stakeholder_identification` | 0.068 |

### PC5

| feature | corr |
|---|---:|
| `moral_uncertainty` | -0.111 |
| `priority_resolution` | 0.107 |
| `loyalty_trust` | -0.107 |
| `conditional_recommendation` | -0.081 |
| `tradeoff_acknowledged` | -0.075 |
| `virtue_character` | -0.066 |
| `stakeholder_identification` | -0.062 |
| `honesty_truthfulness` | -0.044 |
| `procedural_escalation` | 0.042 |
| `responsibility_accountability` | -0.042 |
