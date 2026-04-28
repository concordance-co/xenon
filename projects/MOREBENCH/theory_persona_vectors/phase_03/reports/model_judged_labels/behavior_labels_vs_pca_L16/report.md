# Behavioral Labels vs PCA

- ethical labels: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/model_judged_labels/content_scores.jsonl`
- process labels: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/model_judged_labels/process_scores.jsonl`
- layer: `L16 generated first16`
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
| 1 | 0.024 | 0.020 | 0.020 | 0.127 | 0.190 |
| 2 | 0.071 | 0.114 | 0.153 | 0.218 | 0.242 |
| 3 | 0.076 | 0.014 | 0.078 | 0.215 | 0.377 |
| 4 | 0.048 | 0.053 | 0.081 | 0.218 | 0.232 |
| 5 | -0.001 | 0.063 | 0.065 | 0.100 | 0.116 |

## Strongest Feature Correlations By PC

### PC1

| feature | corr |
|---|---:|
| `virtue_character` | -0.173 |
| `moral_uncertainty` | -0.172 |
| `loyalty_trust` | -0.117 |
| `consequence_forecasting` | 0.112 |
| `fairness_justice` | 0.077 |
| `risk_mitigation` | 0.068 |
| `procedural_escalation` | 0.068 |
| `stakeholder_identification` | 0.056 |
| `public_interest_social_impact` | 0.055 |
| `tradeoff_acknowledged` | 0.043 |

### PC2

| feature | corr |
|---|---:|
| `risk_mitigation` | 0.320 |
| `priority_resolution` | -0.307 |
| `conditional_recommendation` | 0.225 |
| `procedural_escalation` | 0.177 |
| `responsibility_accountability` | 0.165 |
| `stakeholder_identification` | 0.154 |
| `care_compassion` | 0.138 |
| `fairness_justice` | 0.132 |
| `rights_autonomy` | 0.130 |
| `virtue_character` | 0.122 |

### PC3

| feature | corr |
|---|---:|
| `virtue_character` | 0.302 |
| `stakeholder_identification` | -0.148 |
| `conditional_recommendation` | -0.135 |
| `procedural_escalation` | -0.124 |
| `tradeoff_acknowledged` | -0.113 |
| `harm_welfare` | -0.102 |
| `honesty_truthfulness` | 0.082 |
| `legality_compliance` | -0.076 |
| `priority_resolution` | 0.072 |
| `care_compassion` | 0.067 |

### PC4

| feature | corr |
|---|---:|
| `priority_resolution` | 0.283 |
| `virtue_character` | 0.200 |
| `conditional_recommendation` | -0.167 |
| `fairness_justice` | 0.102 |
| `honesty_truthfulness` | -0.097 |
| `moral_uncertainty` | -0.097 |
| `risk_mitigation` | -0.089 |
| `rights_autonomy` | 0.088 |
| `responsibility_accountability` | 0.082 |
| `loyalty_trust` | 0.070 |

### PC5

| feature | corr |
|---|---:|
| `moral_uncertainty` | 0.258 |
| `conditional_recommendation` | 0.212 |
| `tradeoff_acknowledged` | 0.171 |
| `stakeholder_identification` | 0.087 |
| `risk_mitigation` | 0.084 |
| `priority_resolution` | -0.067 |
| `responsibility_accountability` | 0.063 |
| `harm_welfare` | 0.058 |
| `care_compassion` | 0.056 |
| `consequence_forecasting` | 0.052 |
