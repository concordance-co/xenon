# Behavioral Labels vs PCA

- ethical labels: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/model_judged_labels/content_scores.jsonl`
- process labels: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/model_judged_labels/process_scores.jsonl`
- layer: `L40 generated first16`
- joined rows: `540`
- dilemmas: `30`
- conditions: `18`

## Feature Distributions

| feature | mean | std | counts | majority rate | drop? |
|---|---:|---:|---|---:|---|
| `harm_welfare` | 1.596 | 0.632 | `{"0": 43, "1": 132, "2": 365}` | 0.676 | `False` |
| `rights_autonomy` | 0.657 | 0.809 | `{"0": 301, "1": 123, "2": 116}` | 0.557 | `False` |
| `fairness_justice` | 0.665 | 0.886 | `{"0": 333, "1": 55, "2": 152}` | 0.617 | `False` |
| `honesty_truthfulness` | 0.631 | 0.862 | `{"0": 337, "1": 65, "2": 138}` | 0.624 | `False` |
| `responsibility_accountability` | 1.206 | 0.839 | `{"0": 146, "1": 137, "2": 257}` | 0.476 | `False` |
| `loyalty_trust` | 0.652 | 0.842 | `{"0": 318, "1": 92, "2": 130}` | 0.589 | `False` |
| `legality_compliance` | 0.459 | 0.745 | `{"0": 375, "1": 82, "2": 83}` | 0.694 | `False` |
| `public_interest_social_impact` | 0.728 | 0.766 | `{"0": 252, "1": 183, "2": 105}` | 0.467 | `False` |
| `virtue_character` | 0.396 | 0.716 | `{"0": 400, "1": 66, "2": 74}` | 0.741 | `False` |
| `care_compassion` | 0.522 | 0.711 | `{"0": 327, "1": 144, "2": 69}` | 0.606 | `False` |
| `stakeholder_identification` | 1.806 | 0.432 | `{"0": 8, "1": 89, "2": 443}` | 0.820 | `False` |
| `consequence_forecasting` | 1.181 | 0.534 | `{"0": 37, "1": 368, "2": 135}` | 0.681 | `False` |
| `tradeoff_acknowledged` | 1.857 | 0.360 | `{"0": 2, "1": 73, "2": 465}` | 0.861 | `False` |
| `priority_resolution` | 1.596 | 0.491 | `{"1": 218, "2": 322}` | 0.596 | `False` |
| `moral_uncertainty` | 0.222 | 0.493 | `{"0": 439, "1": 82, "2": 19}` | 0.813 | `False` |
| `risk_mitigation` | 1.217 | 0.869 | `{"0": 158, "1": 107, "2": 275}` | 0.509 | `False` |
| `conditional_recommendation` | 0.504 | 0.783 | `{"0": 366, "1": 76, "2": 98}` | 0.678 | `False` |
| `procedural_escalation` | 0.528 | 0.806 | `{"0": 363, "1": 69, "2": 108}` | 0.672 | `False` |

## LOO Dilemma R2 Predicting Activation PCs

| PC | content | process | content+process | theory one-hot | condition one-hot |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.057 | 0.147 | 0.127 | 0.292 | 0.396 |
| 2 | 0.226 | 0.024 | 0.251 | 0.403 | 0.434 |
| 3 | 0.095 | 0.061 | 0.130 | 0.383 | 0.412 |
| 4 | 0.014 | 0.067 | 0.054 | 0.151 | 0.346 |
| 5 | 0.007 | 0.025 | 0.018 | 0.164 | 0.205 |

## Strongest Feature Correlations By PC

### PC1

| feature | corr |
|---|---:|
| `risk_mitigation` | 0.325 |
| `priority_resolution` | -0.307 |
| `procedural_escalation` | 0.225 |
| `conditional_recommendation` | 0.201 |
| `stakeholder_identification` | 0.159 |
| `virtue_character` | -0.155 |
| `tradeoff_acknowledged` | 0.128 |
| `honesty_truthfulness` | 0.117 |
| `fairness_justice` | 0.113 |
| `consequence_forecasting` | 0.084 |

### PC2

| feature | corr |
|---|---:|
| `virtue_character` | 0.385 |
| `loyalty_trust` | 0.243 |
| `responsibility_accountability` | 0.209 |
| `risk_mitigation` | 0.179 |
| `harm_welfare` | -0.161 |
| `rights_autonomy` | 0.153 |
| `fairness_justice` | 0.139 |
| `care_compassion` | 0.139 |
| `priority_resolution` | -0.133 |
| `moral_uncertainty` | 0.129 |

### PC3

| feature | corr |
|---|---:|
| `virtue_character` | -0.307 |
| `priority_resolution` | -0.258 |
| `conditional_recommendation` | 0.248 |
| `moral_uncertainty` | 0.205 |
| `harm_welfare` | 0.130 |
| `fairness_justice` | -0.120 |
| `risk_mitigation` | 0.114 |
| `procedural_escalation` | 0.111 |
| `stakeholder_identification` | 0.088 |
| `rights_autonomy` | -0.079 |

### PC4

| feature | corr |
|---|---:|
| `tradeoff_acknowledged` | 0.212 |
| `stakeholder_identification` | 0.200 |
| `priority_resolution` | 0.132 |
| `responsibility_accountability` | 0.108 |
| `procedural_escalation` | 0.101 |
| `consequence_forecasting` | 0.101 |
| `rights_autonomy` | 0.091 |
| `conditional_recommendation` | 0.089 |
| `fairness_justice` | 0.084 |
| `honesty_truthfulness` | -0.073 |

### PC5

| feature | corr |
|---|---:|
| `moral_uncertainty` | -0.183 |
| `conditional_recommendation` | -0.153 |
| `tradeoff_acknowledged` | -0.116 |
| `risk_mitigation` | -0.114 |
| `responsibility_accountability` | -0.087 |
| `virtue_character` | -0.086 |
| `honesty_truthfulness` | -0.086 |
| `loyalty_trust` | -0.079 |
| `consequence_forecasting` | -0.060 |
| `priority_resolution` | 0.054 |
