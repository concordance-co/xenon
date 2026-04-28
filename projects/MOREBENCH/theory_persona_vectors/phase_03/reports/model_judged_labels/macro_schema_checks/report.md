# Macro Schema Checks

- n: `540`
- activation locus: `L32 generated first16`

## Internal Consistency

| composite | features | alpha | mean inter-item r |
|---|---|---:|---:|
| `outcome_original` | `harm_welfare, public_interest_social_impact, consequence_forecasting` | 0.364 | 0.161 |
| `outcome_content_only` | `harm_welfare, public_interest_social_impact` | 0.298 | 0.178 |
| `principle_integrity` | `rights_autonomy, fairness_justice, honesty_truthfulness, responsibility_accountability, loyalty_trust, virtue_character` | 0.373 | 0.091 |
| `procedural_risk_management` | `legality_compliance, procedural_escalation, risk_mitigation, conditional_recommendation, moral_uncertainty` | 0.697 | 0.319 |
| `care_original` | `care_compassion, rights_autonomy, stakeholder_identification` | 0.087 | 0.047 |
| `care_repaired` | `care_compassion, stakeholder_identification` | 0.110 | 0.065 |
| `tradeoff_mapping` | `tradeoff_acknowledged, stakeholder_identification, consequence_forecasting` | 0.385 | 0.205 |

## Length Residualization: Key Correlations

| PC | composite | raw r | PC length-resid r | both length-resid r | comp-length r | pc-length r |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `outcome_content_only` | -0.048 | -0.034 | -0.034 | 0.122 | -0.120 |
| 1 | `principle_integrity` | -0.072 | -0.030 | -0.032 | 0.354 | -0.120 |
| 1 | `procedural_risk_management` | -0.234 | -0.184 | -0.203 | 0.421 | -0.120 |
| 1 | `decisive_resolution` | 0.275 | 0.287 | 0.288 | 0.084 | -0.120 |
| 1 | `principle_minus_outcome` | -0.005 | 0.008 | 0.008 | 0.107 | -0.120 |
| 1 | `procedural_minus_decisive` | -0.298 | -0.276 | -0.282 | 0.202 | -0.120 |
| 2 | `outcome_content_only` | -0.123 | -0.176 | -0.177 | 0.122 | 0.342 |
| 2 | `principle_integrity` | 0.381 | 0.277 | 0.296 | 0.354 | 0.342 |
| 2 | `procedural_risk_management` | 0.157 | 0.014 | 0.015 | 0.421 | 0.342 |
| 2 | `decisive_resolution` | -0.160 | -0.200 | -0.201 | 0.084 | 0.342 |
| 2 | `principle_minus_outcome` | 0.303 | 0.283 | 0.285 | 0.107 | 0.342 |
| 2 | `procedural_minus_decisive` | 0.186 | 0.124 | 0.127 | 0.202 | 0.342 |
| 3 | `outcome_content_only` | 0.089 | 0.098 | 0.099 | 0.122 | -0.072 |
| 3 | `principle_integrity` | -0.188 | -0.163 | -0.174 | 0.354 | -0.072 |
| 3 | `procedural_risk_management` | 0.207 | 0.238 | 0.263 | 0.421 | -0.072 |
| 3 | `decisive_resolution` | -0.220 | -0.215 | -0.215 | 0.084 | -0.072 |
| 3 | `principle_minus_outcome` | -0.170 | -0.163 | -0.164 | 0.107 | -0.072 |
| 3 | `procedural_minus_decisive` | 0.251 | 0.266 | 0.272 | 0.202 | -0.072 |

## Composite Histograms / Shape

| composite | min | p25 | median | p75 | max | mean | std |
|---|---:|---:|---:|---:|---:|---:|---:|
| `outcome_content_only` | 0.000 | 1.000 | 1.000 | 1.500 | 2.000 | 1.162 | 0.538 |
| `principle_integrity` | 0.000 | 0.333 | 0.667 | 1.000 | 1.833 | 0.701 | 0.407 |
| `procedural_risk_management` | 0.000 | 0.200 | 0.500 | 0.800 | 2.000 | 0.586 | 0.505 |
| `care_repaired` | 0.000 | 1.000 | 1.000 | 1.500 | 2.000 | 1.164 | 0.428 |
| `decisive_resolution` | 1.000 | 1.000 | 2.000 | 2.000 | 2.000 | 1.596 | 0.491 |
| `principle_minus_outcome` | -2.000 | -1.000 | -0.500 | 0.000 | 1.500 | -0.461 | 0.732 |
| `procedural_minus_decisive` | -2.000 | -1.800 | -1.200 | -0.200 | 1.000 | -1.010 | 0.847 |
