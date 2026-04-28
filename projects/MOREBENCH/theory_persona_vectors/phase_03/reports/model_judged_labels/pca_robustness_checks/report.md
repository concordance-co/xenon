# PCA Robustness Checks

- layer: `L32`
- components: `8`

## all18_first16

- rows: `540`
- dilemmas: `30`
- conditions: `18`
- EVR top 5: `['0.095', '0.069', '0.052', '0.037', '0.033']`

| PC | EVR | proc-minus-decisive r | principle-minus-outcome r | procedural r | decisive r |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.095 | -0.282 | 0.008 | -0.203 | 0.288 |
| 2 | 0.069 | 0.127 | 0.285 | 0.015 | -0.201 |
| 3 | 0.052 | 0.272 | -0.164 | 0.263 | -0.215 |
| 4 | 0.037 | -0.161 | 0.018 | -0.081 | 0.198 |
| 5 | 0.033 | -0.081 | -0.006 | -0.066 | 0.076 |

### Condition Extremes

- PC1 negative: P_contractarian_01 (-1.105), P_contract_01 (-1.040), N_neutral_01 (-0.835), P_util_01 (-0.738), P_contractarian_02 (-0.666)
- PC1 positive: P_virtue_02 (1.367), N_anti_contractarian_01 (1.261), P_deont_02 (0.947), N_anti_deont_01 (0.720), N_anti_util_01 (0.689)
- PC2 negative: N_anti_deont_01 (-1.193), P_util_02 (-1.121), N_anti_virtue_01 (-0.701), P_util_01 (-0.656), N_anti_contract_01 (-0.413)
- PC2 positive: P_virtue_02 (1.445), P_contractarian_02 (0.692), N_generic_moral_01 (0.682), P_virtue_01 (0.584), P_deont_01 (0.558)
- PC3 negative: P_virtue_02 (-1.620), P_deont_02 (-1.100), P_deont_01 (-0.427), P_contract_02 (-0.399), P_util_02 (-0.284)
- PC3 positive: N_anti_contractarian_01 (0.859), N_generic_moral_01 (0.640), P_contractarian_02 (0.526), N_anti_virtue_01 (0.510), N_anti_deont_01 (0.448)

## batchA_15_first16_no_contractarian

- rows: `450`
- dilemmas: `30`
- conditions: `15`
- EVR top 5: `['0.093', '0.071', '0.052', '0.039', '0.035']`

| PC | EVR | proc-minus-decisive r | principle-minus-outcome r | procedural r | decisive r |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.093 | -0.220 | 0.158 | -0.204 | 0.180 |
| 2 | 0.071 | -0.186 | -0.315 | -0.081 | 0.241 |
| 3 | 0.052 | -0.245 | 0.159 | -0.251 | 0.179 |
| 4 | 0.039 | -0.234 | 0.045 | -0.140 | 0.266 |
| 5 | 0.035 | -0.088 | -0.009 | -0.078 | 0.077 |

### Condition Extremes

- PC1 negative: P_contract_01 (-0.976), P_util_01 (-0.908), N_neutral_01 (-0.884), N_neutral_02 (-0.742), P_util_02 (-0.364)
- PC1 positive: P_virtue_02 (2.041), P_deont_02 (1.089), P_deont_01 (0.640), N_anti_util_01 (0.574), N_anti_deont_01 (0.159)
- PC2 negative: P_virtue_02 (-0.914), P_virtue_01 (-0.818), N_generic_moral_01 (-0.739), P_contract_01 (-0.510), P_deont_01 (-0.498)
- PC2 positive: N_anti_deont_01 (1.452), P_util_02 (1.081), N_anti_virtue_01 (0.629), N_anti_contract_01 (0.362), P_util_01 (0.332)
- PC3 negative: N_generic_moral_01 (-0.801), N_neutral_02 (-0.535), N_anti_virtue_01 (-0.516), N_anti_deont_01 (-0.431), N_anti_contract_01 (-0.394)
- PC3 positive: P_virtue_02 (1.331), P_deont_02 (0.840), P_contract_02 (0.281), P_util_02 (0.261), P_util_01 (0.209)

## all18_first16_project_out_moral_active

- rows: `540`
- dilemmas: `30`
- conditions: `18`
- EVR top 5: `['0.096', '0.066', '0.040', '0.038', '0.030']`

| PC | EVR | proc-minus-decisive r | principle-minus-outcome r | procedural r | decisive r |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.096 | 0.284 | -0.073 | 0.226 | -0.270 |
| 2 | 0.066 | -0.285 | -0.151 | -0.180 | 0.315 |
| 3 | 0.040 | -0.026 | -0.137 | 0.048 | 0.089 |
| 4 | 0.038 | -0.187 | 0.129 | -0.143 | 0.184 |
| 5 | 0.030 | -0.032 | -0.259 | -0.007 | 0.049 |

### Condition Extremes

- PC1 negative: P_virtue_02 (-1.826), P_deont_02 (-1.112), N_anti_contractarian_01 (-0.974), P_deont_01 (-0.660), N_anti_util_01 (-0.621)
- PC1 positive: P_contractarian_01 (0.986), P_contract_01 (0.977), N_neutral_01 (0.885), P_util_01 (0.795), N_neutral_02 (0.715)
- PC2 negative: N_generic_moral_01 (-0.888), P_contractarian_02 (-0.821), P_contractarian_01 (-0.511), P_virtue_01 (-0.310), N_neutral_01 (-0.267)
- PC2 positive: P_util_02 (1.159), N_anti_deont_01 (0.661), P_deont_02 (0.649), P_util_01 (0.621), P_contract_02 (0.170)
- PC3 negative: P_virtue_02 (-1.476), N_anti_deont_01 (-0.541), N_anti_virtue_01 (-0.374), P_deont_02 (-0.272), N_anti_util_01 (-0.099)
- PC3 positive: P_deont_01 (0.532), N_generic_moral_01 (0.429), N_neutral_02 (0.414), P_virtue_01 (0.322), P_contractarian_01 (0.270)

## all18_first16_project_out_positive_vs_neutral_only

- rows: `540`
- dilemmas: `30`
- conditions: `18`
- EVR top 5: `['0.088', '0.070', '0.042', '0.037', '0.032']`

| PC | EVR | proc-minus-decisive r | principle-minus-outcome r | procedural r | decisive r |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.088 | -0.219 | -0.099 | -0.118 | 0.261 |
| 2 | 0.070 | 0.147 | 0.223 | 0.056 | -0.197 |
| 3 | 0.042 | 0.255 | -0.163 | 0.212 | -0.234 |
| 4 | 0.037 | 0.102 | 0.038 | 0.034 | -0.141 |
| 5 | 0.032 | -0.102 | 0.048 | -0.090 | 0.088 |

### Condition Extremes

- PC1 negative: P_contractarian_01 (-1.059), P_contract_01 (-1.024), P_contractarian_02 (-0.754), P_util_01 (-0.581), P_virtue_01 (-0.528)
- PC1 positive: N_anti_contractarian_01 (1.445), N_anti_deont_01 (1.017), N_anti_util_01 (0.595), P_deont_02 (0.501), N_anti_contract_01 (0.390)
- PC2 negative: P_util_02 (-1.183), N_anti_deont_01 (-0.940), P_util_01 (-0.792), N_anti_virtue_01 (-0.522), N_anti_contract_01 (-0.253)
- PC2 positive: P_virtue_02 (1.065), N_generic_moral_01 (0.859), P_contractarian_02 (0.618), P_deont_01 (0.480), P_virtue_01 (0.462)
- PC3 negative: P_virtue_02 (-1.027), P_deont_02 (-1.012), P_deont_01 (-0.780), P_virtue_01 (-0.599), N_anti_util_01 (-0.272)
- PC3 positive: N_anti_deont_01 (0.899), P_contractarian_02 (0.829), N_anti_virtue_01 (0.686), N_anti_contractarian_01 (0.655), P_contractarian_01 (0.458)

## all18_full_response

- rows: `540`
- dilemmas: `30`
- conditions: `18`
- EVR top 5: `['0.128', '0.097', '0.046', '0.035', '0.031']`

| PC | EVR | proc-minus-decisive r | principle-minus-outcome r | procedural r | decisive r |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.128 | 0.510 | -0.115 | 0.431 | -0.461 |
| 2 | 0.097 | 0.121 | 0.474 | 0.003 | -0.203 |
| 3 | 0.046 | 0.127 | 0.067 | 0.186 | -0.042 |
| 4 | 0.035 | -0.160 | 0.001 | -0.052 | 0.223 |
| 5 | 0.031 | -0.029 | -0.078 | 0.014 | 0.062 |

### Condition Extremes

- PC1 negative: P_virtue_02 (-1.620), P_deont_02 (-1.347), N_anti_util_01 (-0.717), N_anti_contractarian_01 (-0.607), P_util_02 (-0.605)
- PC1 positive: N_neutral_02 (1.116), N_neutral_01 (1.114), P_contractarian_02 (0.899), N_generic_moral_01 (0.755), P_contractarian_01 (0.697)
- PC2 negative: P_util_02 (-1.422), N_anti_deont_01 (-1.379), N_anti_contractarian_01 (-0.773), P_util_01 (-0.689), N_anti_virtue_01 (-0.615)
- PC2 positive: P_virtue_02 (1.499), P_virtue_01 (0.991), P_deont_02 (0.713), P_deont_01 (0.611), P_contractarian_02 (0.423)
- PC3 negative: P_contractarian_01 (-0.890), P_contractarian_02 (-0.782), P_util_02 (-0.717), P_contract_02 (-0.617), P_util_01 (-0.598)
- PC3 positive: N_anti_virtue_01 (0.626), P_deont_02 (0.556), N_neutral_01 (0.488), N_anti_contract_01 (0.439), N_anti_deont_01 (0.438)
