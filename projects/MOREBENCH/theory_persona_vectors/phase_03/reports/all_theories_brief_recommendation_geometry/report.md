# Prompt-End vs Generated Geometry

- capture artifact: `capture_1_1d7271d73617`
- generation rows: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/all_theories_brief_recommendation_report/report_6aa730c32d87_8c1df9a2/results/generate_natural_responses_results.json`

## Cross-Locus Cosines

| layer | theory | construction | cos(prompt_end, generated) | prompt gap | generated gap |
|---:|---|---|---:|---:|---:|
| 0 | contract | contract_alt_deont | nan | nan | 0.425 |
| 0 | contract | contract_alt_util | nan | nan | 0.453 |
| 0 | contract | contract_alt_virtue | nan | nan | 0.387 |
| 0 | contract | contract_anti | nan | nan | 0.236 |
| 0 | contract | contract_generic_moral | nan | nan | 0.167 |
| 0 | contract | contract_neutral_length | nan | nan | 0.258 |
| 0 | contract | contract_neutral_short | nan | nan | 0.295 |
| 0 | contract | contract_positive_variant | nan | nan | 0.389 |
| 0 | deont | deont_alt_contract | nan | nan | 0.425 |
| 0 | deont | deont_alt_util | nan | nan | 0.480 |
| 0 | deont | deont_alt_virtue | nan | nan | 0.494 |
| 0 | deont | deont_anti | nan | nan | 0.464 |
| 0 | deont | deont_generic_moral | nan | nan | 0.413 |
| 0 | deont | deont_neutral_length | nan | nan | 0.449 |
| 0 | deont | deont_neutral_short | nan | nan | 0.444 |
| 0 | deont | deont_positive_variant | nan | nan | 0.531 |
| 0 | util | util_alt_contract | nan | nan | 0.453 |
| 0 | util | util_alt_deont | nan | nan | 0.480 |
| 0 | util | util_alt_virtue | nan | nan | 0.509 |
| 0 | util | util_anti | nan | nan | 0.344 |
| 0 | util | util_generic_moral | nan | nan | 0.371 |
| 0 | util | util_neutral_length | nan | nan | 0.373 |
| 0 | util | util_neutral_short | nan | nan | 0.340 |
| 0 | util | util_positive_variant | nan | nan | 0.444 |
| 0 | virtue | virtue_alt_contract | nan | nan | 0.387 |
| 0 | virtue | virtue_alt_deont | nan | nan | 0.494 |
| 0 | virtue | virtue_alt_util | nan | nan | 0.509 |
| 0 | virtue | virtue_anti | nan | nan | 0.477 |
| 0 | virtue | virtue_generic_moral | nan | nan | 0.348 |
| 0 | virtue | virtue_neutral_length | nan | nan | 0.452 |
| 0 | virtue | virtue_neutral_short | nan | nan | 0.475 |
| 0 | virtue | virtue_positive_variant | nan | nan | 0.497 |
| 4 | contract | contract_alt_deont | 0.044 | 0.083 | 0.429 |
| 4 | contract | contract_alt_util | 0.055 | 0.171 | 0.487 |
| 4 | contract | contract_alt_virtue | 0.120 | 0.131 | 0.410 |
| 4 | contract | contract_anti | 0.041 | 0.067 | 0.377 |
| 4 | contract | contract_generic_moral | 0.071 | 0.142 | 0.300 |
| 4 | contract | contract_neutral_length | 0.113 | 0.045 | 0.424 |
| 4 | contract | contract_neutral_short | 0.136 | 0.046 | 0.428 |
| 4 | contract | contract_positive_variant | 0.055 | 0.065 | 0.465 |
| 4 | deont | deont_alt_contract | 0.044 | 0.083 | 0.429 |
| 4 | deont | deont_alt_util | 0.080 | 0.168 | 0.479 |
| 4 | deont | deont_alt_virtue | 0.064 | 0.154 | 0.461 |
| 4 | deont | deont_anti | 0.027 | 0.056 | 0.509 |
| 4 | deont | deont_generic_moral | 0.074 | 0.105 | 0.475 |
| 4 | deont | deont_neutral_length | 0.076 | 0.043 | 0.507 |
| 4 | deont | deont_neutral_short | 0.135 | 0.048 | 0.515 |
| 4 | deont | deont_positive_variant | 0.067 | 0.066 | 0.503 |
| 4 | util | util_alt_contract | 0.055 | 0.171 | 0.487 |
| 4 | util | util_alt_deont | 0.080 | 0.168 | 0.479 |
| 4 | util | util_alt_virtue | 0.123 | 0.097 | 0.515 |
| 4 | util | util_anti | 0.053 | 0.034 | 0.452 |
| 4 | util | util_generic_moral | 0.084 | 0.106 | 0.370 |
| 4 | util | util_neutral_length | 0.104 | 0.057 | 0.431 |
| 4 | util | util_neutral_short | 0.130 | 0.046 | 0.391 |
| 4 | util | util_positive_variant | 0.051 | 0.058 | 0.424 |
| 4 | virtue | virtue_alt_contract | 0.120 | 0.131 | 0.410 |
| 4 | virtue | virtue_alt_deont | 0.064 | 0.154 | 0.461 |
| 4 | virtue | virtue_alt_util | 0.123 | 0.097 | 0.515 |
| 4 | virtue | virtue_anti | 0.083 | 0.037 | 0.469 |
| 4 | virtue | virtue_generic_moral | 0.106 | 0.078 | 0.426 |
| 4 | virtue | virtue_neutral_length | 0.114 | 0.053 | 0.448 |
| 4 | virtue | virtue_neutral_short | 0.161 | 0.051 | 0.504 |
| 4 | virtue | virtue_positive_variant | 0.025 | 0.049 | 0.446 |
| 16 | contract | contract_alt_deont | 0.144 | 0.136 | 0.520 |
| 16 | contract | contract_alt_util | 0.132 | 0.203 | 0.489 |
| 16 | contract | contract_alt_virtue | 0.155 | 0.223 | 0.521 |
| 16 | contract | contract_anti | 0.145 | 0.155 | 0.491 |
| 16 | contract | contract_generic_moral | 0.103 | 0.203 | 0.484 |
| 16 | contract | contract_neutral_length | 0.149 | 0.133 | 0.513 |
| 16 | contract | contract_neutral_short | 0.187 | 0.150 | 0.502 |
| 16 | contract | contract_positive_variant | 0.153 | 0.166 | 0.507 |
| 16 | deont | deont_alt_contract | 0.144 | 0.136 | 0.520 |
| 16 | deont | deont_alt_util | 0.079 | 0.206 | 0.523 |
| 16 | deont | deont_alt_virtue | 0.067 | 0.229 | 0.491 |
| 16 | deont | deont_anti | 0.110 | 0.135 | 0.487 |
| 16 | deont | deont_generic_moral | 0.087 | 0.176 | 0.501 |
| 16 | deont | deont_neutral_length | 0.090 | 0.138 | 0.451 |
| 16 | deont | deont_neutral_short | 0.122 | 0.185 | 0.500 |
| 16 | deont | deont_positive_variant | 0.033 | 0.191 | 0.453 |
| 16 | util | util_alt_contract | 0.132 | 0.203 | 0.489 |
| 16 | util | util_alt_deont | 0.079 | 0.206 | 0.523 |
| 16 | util | util_alt_virtue | 0.133 | 0.211 | 0.497 |
| 16 | util | util_anti | 0.093 | 0.125 | 0.489 |
| 16 | util | util_generic_moral | 0.109 | 0.186 | 0.513 |
| 16 | util | util_neutral_length | 0.101 | 0.136 | 0.490 |
| 16 | util | util_neutral_short | 0.140 | 0.161 | 0.493 |
| 16 | util | util_positive_variant | 0.027 | 0.177 | 0.432 |
| 16 | virtue | virtue_alt_contract | 0.155 | 0.223 | 0.521 |
| 16 | virtue | virtue_alt_deont | 0.067 | 0.229 | 0.491 |
| 16 | virtue | virtue_alt_util | 0.133 | 0.211 | 0.497 |
| 16 | virtue | virtue_anti | 0.100 | 0.154 | 0.520 |
| 16 | virtue | virtue_generic_moral | 0.083 | 0.204 | 0.498 |
| 16 | virtue | virtue_neutral_length | 0.117 | 0.131 | 0.543 |
| 16 | virtue | virtue_neutral_short | 0.138 | 0.160 | 0.568 |
| 16 | virtue | virtue_positive_variant | -0.021 | 0.141 | 0.442 |
| 24 | contract | contract_alt_deont | 0.111 | 0.119 | 0.520 |
| 24 | contract | contract_alt_util | 0.086 | 0.191 | 0.518 |
| 24 | contract | contract_alt_virtue | 0.077 | 0.230 | 0.521 |
| 24 | contract | contract_anti | 0.077 | 0.101 | 0.518 |
| 24 | contract | contract_generic_moral | 0.046 | 0.183 | 0.471 |
| 24 | contract | contract_neutral_length | 0.075 | 0.100 | 0.475 |
| 24 | contract | contract_neutral_short | 0.113 | 0.105 | 0.538 |
| 24 | contract | contract_positive_variant | 0.046 | 0.127 | 0.422 |
| 24 | deont | deont_alt_contract | 0.111 | 0.119 | 0.520 |
| 24 | deont | deont_alt_util | 0.083 | 0.208 | 0.511 |
| 24 | deont | deont_alt_virtue | 0.133 | 0.239 | 0.482 |
| 24 | deont | deont_anti | 0.084 | 0.116 | 0.527 |
| 24 | deont | deont_generic_moral | 0.106 | 0.197 | 0.533 |
| 24 | deont | deont_neutral_length | 0.104 | 0.132 | 0.487 |
| 24 | deont | deont_neutral_short | 0.145 | 0.137 | 0.538 |
| 24 | deont | deont_positive_variant | 0.056 | 0.153 | 0.487 |
| 24 | util | util_alt_contract | 0.086 | 0.191 | 0.518 |
| 24 | util | util_alt_deont | 0.083 | 0.208 | 0.511 |
| 24 | util | util_alt_virtue | 0.123 | 0.270 | 0.494 |
| 24 | util | util_anti | 0.056 | 0.119 | 0.416 |
| 24 | util | util_generic_moral | 0.057 | 0.191 | 0.501 |
| 24 | util | util_neutral_length | 0.066 | 0.137 | 0.556 |
| 24 | util | util_neutral_short | 0.088 | 0.147 | 0.468 |
| 24 | util | util_positive_variant | 0.025 | 0.174 | 0.443 |
| 24 | virtue | virtue_alt_contract | 0.077 | 0.230 | 0.521 |
| 24 | virtue | virtue_alt_deont | 0.133 | 0.239 | 0.482 |
| 24 | virtue | virtue_alt_util | 0.123 | 0.270 | 0.494 |
| 24 | virtue | virtue_anti | 0.070 | 0.115 | 0.558 |
| 24 | virtue | virtue_generic_moral | 0.066 | 0.248 | 0.529 |
| 24 | virtue | virtue_neutral_length | 0.103 | 0.150 | 0.491 |
| 24 | virtue | virtue_neutral_short | 0.153 | 0.142 | 0.519 |
| 24 | virtue | virtue_positive_variant | -0.004 | 0.134 | 0.455 |
| 32 | contract | contract_alt_deont | 0.161 | 0.092 | 0.398 |
| 32 | contract | contract_alt_util | 0.159 | 0.125 | 0.446 |
| 32 | contract | contract_alt_virtue | 0.173 | 0.157 | 0.423 |
| 32 | contract | contract_anti | 0.147 | 0.063 | 0.390 |
| 32 | contract | contract_generic_moral | 0.105 | 0.114 | 0.373 |
| 32 | contract | contract_neutral_length | 0.113 | 0.067 | 0.387 |
| 32 | contract | contract_neutral_short | 0.132 | 0.069 | 0.408 |
| 32 | contract | contract_positive_variant | 0.130 | 0.080 | 0.419 |
| 32 | deont | deont_alt_contract | 0.161 | 0.092 | 0.398 |
| 32 | deont | deont_alt_util | 0.221 | 0.109 | 0.368 |
| 32 | deont | deont_alt_virtue | 0.149 | 0.131 | 0.433 |
| 32 | deont | deont_anti | 0.202 | 0.065 | 0.347 |
| 32 | deont | deont_generic_moral | 0.118 | 0.117 | 0.407 |
| 32 | deont | deont_neutral_length | 0.134 | 0.068 | 0.370 |
| 32 | deont | deont_neutral_short | 0.148 | 0.078 | 0.360 |
| 32 | deont | deont_positive_variant | 0.125 | 0.121 | 0.365 |
| 32 | util | util_alt_contract | 0.159 | 0.125 | 0.446 |
| 32 | util | util_alt_deont | 0.221 | 0.109 | 0.368 |
| 32 | util | util_alt_virtue | 0.258 | 0.163 | 0.366 |
| 32 | util | util_anti | 0.204 | 0.083 | 0.382 |
| 32 | util | util_generic_moral | 0.169 | 0.112 | 0.476 |
| 32 | util | util_neutral_length | 0.122 | 0.087 | 0.466 |
| 32 | util | util_neutral_short | 0.144 | 0.084 | 0.438 |
| 32 | util | util_positive_variant | 0.066 | 0.133 | 0.433 |
| 32 | virtue | virtue_alt_contract | 0.173 | 0.157 | 0.423 |
| 32 | virtue | virtue_alt_deont | 0.149 | 0.131 | 0.433 |
| 32 | virtue | virtue_alt_util | 0.258 | 0.163 | 0.366 |
| 32 | virtue | virtue_anti | 0.218 | 0.058 | 0.369 |
| 32 | virtue | virtue_generic_moral | 0.141 | 0.159 | 0.413 |
| 32 | virtue | virtue_neutral_length | 0.195 | 0.086 | 0.368 |
| 32 | virtue | virtue_neutral_short | 0.199 | 0.086 | 0.445 |
| 32 | virtue | virtue_positive_variant | 0.062 | 0.087 | 0.432 |
| 40 | contract | contract_alt_deont | 0.137 | 0.153 | 0.454 |
| 40 | contract | contract_alt_util | 0.161 | 0.238 | 0.518 |
| 40 | contract | contract_alt_virtue | 0.172 | 0.240 | 0.457 |
| 40 | contract | contract_anti | 0.163 | 0.078 | 0.479 |
| 40 | contract | contract_generic_moral | 0.124 | 0.192 | 0.474 |
| 40 | contract | contract_neutral_length | 0.113 | 0.094 | 0.462 |
| 40 | contract | contract_neutral_short | 0.088 | 0.101 | 0.471 |
| 40 | contract | contract_positive_variant | 0.091 | 0.118 | 0.513 |
| 40 | deont | deont_alt_contract | 0.137 | 0.153 | 0.454 |
| 40 | deont | deont_alt_util | 0.196 | 0.227 | 0.476 |
| 40 | deont | deont_alt_virtue | 0.104 | 0.225 | 0.474 |
| 40 | deont | deont_anti | 0.122 | 0.116 | 0.452 |
| 40 | deont | deont_generic_moral | 0.136 | 0.203 | 0.492 |
| 40 | deont | deont_neutral_length | 0.050 | 0.123 | 0.405 |
| 40 | deont | deont_neutral_short | 0.035 | 0.139 | 0.418 |
| 40 | deont | deont_positive_variant | 0.126 | 0.175 | 0.466 |
| 40 | util | util_alt_contract | 0.161 | 0.238 | 0.518 |
| 40 | util | util_alt_deont | 0.196 | 0.227 | 0.476 |
| 40 | util | util_alt_virtue | 0.234 | 0.248 | 0.420 |
| 40 | util | util_anti | 0.220 | 0.154 | 0.507 |
| 40 | util | util_generic_moral | 0.184 | 0.220 | 0.509 |
| 40 | util | util_neutral_length | 0.097 | 0.134 | 0.480 |
| 40 | util | util_neutral_short | 0.073 | 0.116 | 0.487 |
| 40 | util | util_positive_variant | 0.097 | 0.180 | 0.488 |
| 40 | virtue | virtue_alt_contract | 0.172 | 0.240 | 0.457 |
| 40 | virtue | virtue_alt_deont | 0.104 | 0.225 | 0.474 |
| 40 | virtue | virtue_alt_util | 0.234 | 0.248 | 0.420 |
| 40 | virtue | virtue_anti | 0.161 | 0.114 | 0.463 |
| 40 | virtue | virtue_generic_moral | 0.165 | 0.192 | 0.428 |
| 40 | virtue | virtue_neutral_length | 0.122 | 0.127 | 0.389 |
| 40 | virtue | virtue_neutral_short | 0.099 | 0.114 | 0.421 |
| 40 | virtue | virtue_positive_variant | 0.123 | 0.147 | 0.413 |

## Generic-Moral Alignment

| site | layer | theory | cos(theory-neutral, generic-neutral) |
|---|---:|---|---:|
| prompt_end_residual | 0 | deont | nan |
| prompt_end_residual | 0 | util | nan |
| prompt_end_residual | 0 | virtue | nan |
| prompt_end_residual | 0 | contract | nan |
| prompt_end_residual | 4 | deont | 0.886 |
| prompt_end_residual | 4 | util | 0.931 |
| prompt_end_residual | 4 | virtue | 0.880 |
| prompt_end_residual | 4 | contract | 0.909 |
| prompt_end_residual | 16 | deont | 0.728 |
| prompt_end_residual | 16 | util | 0.783 |
| prompt_end_residual | 16 | virtue | 0.815 |
| prompt_end_residual | 16 | contract | 0.785 |
| prompt_end_residual | 24 | deont | 0.763 |
| prompt_end_residual | 24 | util | 0.796 |
| prompt_end_residual | 24 | virtue | 0.849 |
| prompt_end_residual | 24 | contract | 0.818 |
| prompt_end_residual | 32 | deont | 0.802 |
| prompt_end_residual | 32 | util | 0.774 |
| prompt_end_residual | 32 | virtue | 0.851 |
| prompt_end_residual | 32 | contract | 0.812 |
| prompt_end_residual | 40 | deont | 0.828 |
| prompt_end_residual | 40 | util | 0.836 |
| prompt_end_residual | 40 | virtue | 0.865 |
| prompt_end_residual | 40 | contract | 0.865 |
| generated_sequence_residual | 0 | deont | 0.550 |
| generated_sequence_residual | 0 | util | 0.441 |
| generated_sequence_residual | 0 | virtue | 0.670 |
| generated_sequence_residual | 0 | contract | 0.683 |
| generated_sequence_residual | 4 | deont | 0.626 |
| generated_sequence_residual | 4 | util | 0.584 |
| generated_sequence_residual | 4 | virtue | 0.673 |
| generated_sequence_residual | 4 | contract | 0.715 |
| generated_sequence_residual | 16 | deont | 0.536 |
| generated_sequence_residual | 16 | util | 0.440 |
| generated_sequence_residual | 16 | virtue | 0.620 |
| generated_sequence_residual | 16 | contract | 0.604 |
| generated_sequence_residual | 24 | deont | 0.549 |
| generated_sequence_residual | 24 | util | 0.445 |
| generated_sequence_residual | 24 | virtue | 0.620 |
| generated_sequence_residual | 24 | contract | 0.605 |
| generated_sequence_residual | 32 | deont | 0.537 |
| generated_sequence_residual | 32 | util | 0.436 |
| generated_sequence_residual | 32 | virtue | 0.589 |
| generated_sequence_residual | 32 | contract | 0.618 |
| generated_sequence_residual | 40 | deont | 0.508 |
| generated_sequence_residual | 40 | util | 0.441 |
| generated_sequence_residual | 40 | virtue | 0.548 |
| generated_sequence_residual | 40 | contract | 0.586 |

## Cross-Theory Cosines: Neutral-Short Construction

### prompt_end_residual L0

| | contract | deont | util | virtue |
|---|---:|---:|---:|---:|
| contract | nan | nan | nan | nan |
| deont | nan | nan | nan | nan |
| util | nan | nan | nan | nan |
| virtue | nan | nan | nan | nan |

### prompt_end_residual L4

| | contract | deont | util | virtue |
|---|---:|---:|---:|---:|
| contract | 1.000 | 0.922 | 0.953 | 0.932 |
| deont | 0.922 | 1.000 | 0.935 | 0.953 |
| util | 0.953 | 0.935 | 1.000 | 0.914 |
| virtue | 0.932 | 0.953 | 0.914 | 1.000 |

### prompt_end_residual L16

| | contract | deont | util | virtue |
|---|---:|---:|---:|---:|
| contract | 1.000 | 0.702 | 0.832 | 0.820 |
| deont | 0.702 | 1.000 | 0.790 | 0.841 |
| util | 0.832 | 0.790 | 1.000 | 0.838 |
| virtue | 0.820 | 0.841 | 0.838 | 1.000 |

### prompt_end_residual L24

| | contract | deont | util | virtue |
|---|---:|---:|---:|---:|
| contract | 1.000 | 0.742 | 0.859 | 0.859 |
| deont | 0.742 | 1.000 | 0.813 | 0.845 |
| util | 0.859 | 0.813 | 1.000 | 0.868 |
| virtue | 0.859 | 0.845 | 0.868 | 1.000 |

### prompt_end_residual L32

| | contract | deont | util | virtue |
|---|---:|---:|---:|---:|
| contract | 1.000 | 0.773 | 0.869 | 0.869 |
| deont | 0.773 | 1.000 | 0.820 | 0.865 |
| util | 0.869 | 0.820 | 1.000 | 0.864 |
| virtue | 0.869 | 0.865 | 0.864 | 1.000 |

### prompt_end_residual L40

| | contract | deont | util | virtue |
|---|---:|---:|---:|---:|
| contract | 1.000 | 0.784 | 0.892 | 0.887 |
| deont | 0.784 | 1.000 | 0.854 | 0.879 |
| util | 0.892 | 0.854 | 1.000 | 0.896 |
| virtue | 0.887 | 0.879 | 0.896 | 1.000 |

### generated_sequence_residual L0

| | contract | deont | util | virtue |
|---|---:|---:|---:|---:|
| contract | 1.000 | 0.584 | 0.388 | 0.570 |
| deont | 0.584 | 1.000 | 0.336 | 0.504 |
| util | 0.388 | 0.336 | 1.000 | 0.300 |
| virtue | 0.570 | 0.504 | 0.300 | 1.000 |

### generated_sequence_residual L4

| | contract | deont | util | virtue |
|---|---:|---:|---:|---:|
| contract | 1.000 | 0.735 | 0.558 | 0.703 |
| deont | 0.735 | 1.000 | 0.428 | 0.684 |
| util | 0.558 | 0.428 | 1.000 | 0.396 |
| virtue | 0.703 | 0.684 | 0.396 | 1.000 |

### generated_sequence_residual L16

| | contract | deont | util | virtue |
|---|---:|---:|---:|---:|
| contract | 1.000 | 0.668 | 0.612 | 0.664 |
| deont | 0.668 | 1.000 | 0.447 | 0.710 |
| util | 0.612 | 0.447 | 1.000 | 0.400 |
| virtue | 0.664 | 0.710 | 0.400 | 1.000 |

### generated_sequence_residual L24

| | contract | deont | util | virtue |
|---|---:|---:|---:|---:|
| contract | 1.000 | 0.692 | 0.666 | 0.678 |
| deont | 0.692 | 1.000 | 0.520 | 0.736 |
| util | 0.666 | 0.520 | 1.000 | 0.417 |
| virtue | 0.678 | 0.736 | 0.417 | 1.000 |

### generated_sequence_residual L32

| | contract | deont | util | virtue |
|---|---:|---:|---:|---:|
| contract | 1.000 | 0.670 | 0.640 | 0.642 |
| deont | 0.670 | 1.000 | 0.430 | 0.717 |
| util | 0.640 | 0.430 | 1.000 | 0.342 |
| virtue | 0.642 | 0.717 | 0.342 | 1.000 |

### generated_sequence_residual L40

| | contract | deont | util | virtue |
|---|---:|---:|---:|---:|
| contract | 1.000 | 0.633 | 0.616 | 0.602 |
| deont | 0.633 | 1.000 | 0.401 | 0.679 |
| util | 0.616 | 0.401 | 1.000 | 0.305 |
| virtue | 0.602 | 0.679 | 0.305 | 1.000 |
