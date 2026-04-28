# Behavioral Labels vs PCA

- ethical labels: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/model_judged_labels/content_scores.jsonl`
- process labels: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/model_judged_labels/process_scores.jsonl`
- layer: `L32 generated first16`
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
| 1 | 0.048 | 0.123 | 0.103 | 0.259 | 0.348 |
| 2 | 0.193 | 0.029 | 0.222 | 0.346 | 0.366 |
| 3 | 0.113 | 0.048 | 0.134 | 0.362 | 0.440 |
| 4 | 0.010 | 0.059 | 0.053 | 0.136 | 0.271 |
| 5 | 0.010 | 0.052 | 0.055 | 0.194 | 0.223 |

## Strongest Feature Correlations By PC

### PC1

| feature | corr |
|---|---:|
| `risk_mitigation` | -0.300 |
| `priority_resolution` | 0.275 |
| `procedural_escalation` | -0.201 |
| `conditional_recommendation` | -0.172 |
| `stakeholder_identification` | -0.151 |
| `virtue_character` | 0.149 |
| `tradeoff_acknowledged` | -0.118 |
| `fairness_justice` | -0.113 |
| `honesty_truthfulness` | -0.106 |
| `consequence_forecasting` | -0.087 |

### PC2

| feature | corr |
|---|---:|
| `virtue_character` | 0.344 |
| `loyalty_trust` | 0.230 |
| `risk_mitigation` | 0.193 |
| `responsibility_accountability` | 0.193 |
| `priority_resolution` | -0.160 |
| `harm_welfare` | -0.158 |
| `rights_autonomy` | 0.149 |
| `care_compassion` | 0.148 |
| `moral_uncertainty` | 0.135 |
| `conditional_recommendation` | 0.134 |

### PC3

| feature | corr |
|---|---:|
| `virtue_character` | -0.347 |
| `conditional_recommendation` | 0.236 |
| `priority_resolution` | -0.220 |
| `moral_uncertainty` | 0.150 |
| `procedural_escalation` | 0.143 |
| `stakeholder_identification` | 0.137 |
| `harm_welfare` | 0.129 |
| `risk_mitigation` | 0.112 |
| `tradeoff_acknowledged` | 0.092 |
| `fairness_justice` | -0.085 |

### PC4

| feature | corr |
|---|---:|
| `priority_resolution` | 0.209 |
| `stakeholder_identification` | 0.146 |
| `tradeoff_acknowledged` | 0.126 |
| `responsibility_accountability` | 0.111 |
| `rights_autonomy` | 0.105 |
| `honesty_truthfulness` | -0.099 |
| `consequence_forecasting` | 0.095 |
| `fairness_justice` | 0.093 |
| `moral_uncertainty` | -0.070 |
| `procedural_escalation` | 0.063 |

### PC5

| feature | corr |
|---|---:|
| `conditional_recommendation` | -0.222 |
| `moral_uncertainty` | -0.220 |
| `tradeoff_acknowledged` | -0.173 |
| `risk_mitigation` | -0.106 |
| `loyalty_trust` | -0.096 |
| `responsibility_accountability` | -0.087 |
| `stakeholder_identification` | -0.087 |
| `virtue_character` | -0.078 |
| `care_compassion` | -0.063 |
| `honesty_truthfulness` | -0.061 |
