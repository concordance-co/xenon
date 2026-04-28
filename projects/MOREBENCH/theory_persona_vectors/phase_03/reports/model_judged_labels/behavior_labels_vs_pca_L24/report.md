# Behavioral Labels vs PCA

- ethical labels: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/model_judged_labels/content_scores.jsonl`
- process labels: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/model_judged_labels/process_scores.jsonl`
- layer: `L24 generated first16`
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
| 1 | 0.034 | 0.109 | 0.084 | 0.197 | 0.279 |
| 2 | 0.043 | 0.056 | 0.093 | 0.155 | 0.151 |
| 3 | 0.102 | 0.020 | 0.110 | 0.242 | 0.383 |
| 4 | 0.030 | 0.038 | 0.050 | 0.122 | 0.148 |
| 5 | 0.008 | 0.050 | 0.056 | 0.145 | 0.168 |

## Strongest Feature Correlations By PC

### PC1

| feature | corr |
|---|---:|
| `risk_mitigation` | -0.277 |
| `priority_resolution` | 0.270 |
| `procedural_escalation` | -0.193 |
| `conditional_recommendation` | -0.153 |
| `stakeholder_identification` | -0.141 |
| `fairness_justice` | -0.119 |
| `virtue_character` | 0.109 |
| `tradeoff_acknowledged` | -0.098 |
| `honesty_truthfulness` | -0.093 |
| `responsibility_accountability` | -0.079 |

### PC2

| feature | corr |
|---|---:|
| `priority_resolution` | -0.224 |
| `risk_mitigation` | 0.205 |
| `conditional_recommendation` | 0.203 |
| `moral_uncertainty` | 0.198 |
| `loyalty_trust` | 0.165 |
| `virtue_character` | 0.152 |
| `responsibility_accountability` | 0.126 |
| `procedural_escalation` | 0.106 |
| `stakeholder_identification` | 0.098 |
| `care_compassion` | 0.098 |

### PC3

| feature | corr |
|---|---:|
| `virtue_character` | 0.336 |
| `stakeholder_identification` | -0.149 |
| `conditional_recommendation` | -0.146 |
| `tradeoff_acknowledged` | -0.120 |
| `harm_welfare` | -0.119 |
| `procedural_escalation` | -0.111 |
| `care_compassion` | 0.085 |
| `legality_compliance` | -0.079 |
| `moral_uncertainty` | -0.076 |
| `loyalty_trust` | 0.070 |

### PC4

| feature | corr |
|---|---:|
| `priority_resolution` | 0.233 |
| `virtue_character` | 0.150 |
| `conditional_recommendation` | -0.140 |
| `moral_uncertainty` | -0.114 |
| `fairness_justice` | 0.098 |
| `honesty_truthfulness` | -0.097 |
| `rights_autonomy` | 0.093 |
| `responsibility_accountability` | 0.070 |
| `consequence_forecasting` | 0.064 |
| `risk_mitigation` | -0.061 |

### PC5

| feature | corr |
|---|---:|
| `moral_uncertainty` | 0.239 |
| `conditional_recommendation` | 0.202 |
| `tradeoff_acknowledged` | 0.174 |
| `risk_mitigation` | 0.108 |
| `virtue_character` | 0.093 |
| `responsibility_accountability` | 0.086 |
| `stakeholder_identification` | 0.085 |
| `care_compassion` | 0.081 |
| `harm_welfare` | 0.058 |
| `loyalty_trust` | 0.053 |
