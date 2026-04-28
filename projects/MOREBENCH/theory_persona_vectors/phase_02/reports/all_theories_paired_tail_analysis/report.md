# All-Theories Paired Tail Analysis

- capture artifact: `capture_1_c2684db0530c`
- generation rows: `projects/MOREBENCH/theory_persona_vectors/phase_02/reports/all_theories_pole_pilot_report/report_0a6a1bb7ed4a_f2f091e0/results/generate_terse_responses_results.json`
- primary site/layer: `generated_sequence_residual` / L32

## Response Token Distribution

- n=420, mean=10.436, median=9.000, share_lt_10=0.552, share_lt_20=0.912

| condition | n | mean | median | min | max | share_ge_20 |
|---|---:|---:|---:|---:|---:|---:|
| N_anti_contract_01 | 30 | 8.033 | 7.500 | 4 | 20 | 0.033 |
| N_anti_deont_01 | 30 | 7.067 | 7.000 | 4 | 16 | 0.000 |
| N_anti_util_01 | 30 | 7.200 | 7.000 | 4 | 13 | 0.000 |
| N_anti_virtue_01 | 30 | 8.867 | 8.000 | 5 | 24 | 0.033 |
| N_neutral_01 | 30 | 10.333 | 9.500 | 5 | 24 | 0.067 |
| N_neutral_02 | 30 | 11.733 | 10.000 | 5 | 37 | 0.133 |
| P_contract_01 | 30 | 13.367 | 12.000 | 5 | 33 | 0.167 |
| P_contract_02 | 30 | 13.100 | 11.000 | 5 | 31 | 0.233 |
| P_deont_01 | 30 | 12.900 | 10.000 | 5 | 37 | 0.167 |
| P_deont_02 | 30 | 11.167 | 10.000 | 5 | 25 | 0.033 |
| P_util_01 | 30 | 10.067 | 8.000 | 5 | 21 | 0.067 |
| P_util_02 | 30 | 10.500 | 10.500 | 5 | 20 | 0.033 |
| P_virtue_01 | 30 | 14.433 | 14.500 | 5 | 27 | 0.267 |
| P_virtue_02 | 30 | 7.333 | 6.500 | 4 | 16 | 0.000 |

## Paired Smoke: `none` min_tokens=0

| theory | construction | n | real_median | null_p95 | gap | pos_tok | neg_tok |
|---|---|---:|---:|---:|---:|---:|---:|
| deontology | deont_neutral_short | 30 | 0.875 | 0.532 | 0.343 | 12.900 | 10.333 |
| deontology | deont_neutral_length_matched | 30 | 0.825 | 0.375 | 0.449 | 12.900 | 11.733 |
| deontology | deont_anti | 30 | 0.784 | 0.342 | 0.442 | 12.900 | 7.067 |
| deontology | deont_alt_util | 30 | 0.691 | 0.277 | 0.414 | 12.900 | 10.067 |
| deontology | deont_alt_virtue | 30 | 0.613 | 0.272 | 0.341 | 12.900 | 14.433 |
| deontology | deont_alt_contract | 30 | 0.566 | 0.273 | 0.293 | 12.900 | 13.367 |
| utilitarian | util_neutral_short | 30 | 0.811 | 0.459 | 0.352 | 10.067 | 10.333 |
| utilitarian | util_neutral_length_matched | 30 | 0.770 | 0.345 | 0.424 | 10.067 | 11.733 |
| utilitarian | util_anti | 30 | 0.671 | 0.312 | 0.359 | 10.067 | 7.200 |
| utilitarian | util_alt_deont | 30 | 0.691 | 0.277 | 0.414 | 10.067 | 12.900 |
| utilitarian | util_alt_virtue | 30 | 0.720 | 0.293 | 0.427 | 10.067 | 14.433 |
| utilitarian | util_alt_contract | 30 | 0.608 | 0.274 | 0.334 | 10.067 | 13.367 |
| virtue_ethics | virtue_neutral_short | 30 | 0.874 | 0.549 | 0.326 | 14.433 | 10.333 |
| virtue_ethics | virtue_neutral_length_matched | 30 | 0.847 | 0.424 | 0.423 | 14.433 | 11.733 |
| virtue_ethics | virtue_anti | 30 | 0.789 | 0.370 | 0.419 | 14.433 | 8.867 |
| virtue_ethics | virtue_alt_deont | 30 | 0.613 | 0.272 | 0.341 | 14.433 | 12.900 |
| virtue_ethics | virtue_alt_util | 30 | 0.720 | 0.293 | 0.427 | 14.433 | 10.067 |
| virtue_ethics | virtue_alt_contract | 30 | 0.582 | 0.235 | 0.348 | 14.433 | 13.367 |
| contractualism | contract_neutral_short | 30 | 0.859 | 0.471 | 0.388 | 13.367 | 10.333 |
| contractualism | contract_neutral_length_matched | 30 | 0.834 | 0.402 | 0.432 | 13.367 | 11.733 |
| contractualism | contract_anti | 30 | 0.769 | 0.300 | 0.469 | 13.367 | 8.033 |
| contractualism | contract_alt_deont | 30 | 0.566 | 0.273 | 0.293 | 13.367 | 12.900 |
| contractualism | contract_alt_util | 30 | 0.608 | 0.274 | 0.334 | 13.367 | 10.067 |
| contractualism | contract_alt_virtue | 30 | 0.582 | 0.235 | 0.348 | 13.367 | 14.433 |

Cross-theory cosines:

### neutral_short

| | contract | deont | util | virtue |
|---|---:|---:|---:|---:|
| contract | 1.000 | 0.858 | 0.837 | 0.895 |
| deont | 0.858 | 1.000 | 0.754 | 0.868 |
| util | 0.837 | 0.754 | 1.000 | 0.711 |
| virtue | 0.895 | 0.868 | 0.711 | 1.000 |

### neutral_length_matched

| | contract | deont | util | virtue |
|---|---:|---:|---:|---:|
| contract | 1.000 | 0.829 | 0.798 | 0.870 |
| deont | 0.829 | 1.000 | 0.709 | 0.838 |
| util | 0.798 | 0.709 | 1.000 | 0.639 |
| virtue | 0.870 | 0.838 | 0.639 | 1.000 |

### anti

| | contract | deont | util | virtue |
|---|---:|---:|---:|---:|
| contract | 1.000 | 0.646 | 0.635 | 0.751 |
| deont | 0.646 | 1.000 | 0.345 | 0.726 |
| util | 0.635 | 0.345 | 1.000 | 0.441 |
| virtue | 0.751 | 0.726 | 0.441 | 1.000 |

## Paired Smoke: `pos` min_tokens=10

| theory | construction | n | real_median | null_p95 | gap | pos_tok | neg_tok |
|---|---|---:|---:|---:|---:|---:|---:|
| deontology | deont_neutral_short | 17 | 0.830 | 0.527 | 0.303 | 17.882 | 12.471 |
| deontology | deont_neutral_length_matched | 17 | 0.756 | 0.400 | 0.356 | 17.882 | 13.235 |
| deontology | deont_anti | 17 | 0.771 | 0.433 | 0.339 | 17.882 | 7.647 |
| deontology | deont_alt_util | 17 | 0.611 | 0.357 | 0.254 | 17.882 | 12.000 |
| deontology | deont_alt_virtue | 17 | 0.365 | 0.167 | 0.198 | 17.882 | 15.471 |
| deontology | deont_alt_contract | 17 | 0.468 | 0.237 | 0.231 | 17.882 | 14.118 |
| utilitarian | util_neutral_short | 13 | 0.750 | 0.456 | 0.294 | 14.538 | 13.462 |
| utilitarian | util_neutral_length_matched | 13 | 0.613 | 0.353 | 0.260 | 14.538 | 13.385 |
| utilitarian | util_anti | 13 | 0.610 | 0.310 | 0.300 | 14.538 | 7.769 |
| utilitarian | util_alt_deont | 13 | 0.395 | 0.270 | 0.126 | 14.538 | 16.615 |
| utilitarian | util_alt_virtue | 13 | 0.435 | 0.244 | 0.191 | 14.538 | 17.000 |
| utilitarian | util_alt_contract | 13 | 0.348 | 0.176 | 0.171 | 14.538 | 15.462 |
| virtue_ethics | virtue_neutral_short | 20 | 0.859 | 0.532 | 0.327 | 18.350 | 11.450 |
| virtue_ethics | virtue_neutral_length_matched | 20 | 0.813 | 0.405 | 0.409 | 18.350 | 13.350 |
| virtue_ethics | virtue_anti | 20 | 0.746 | 0.417 | 0.329 | 18.350 | 9.750 |
| virtue_ethics | virtue_alt_deont | 20 | 0.570 | 0.328 | 0.242 | 18.350 | 15.400 |
| virtue_ethics | virtue_alt_util | 20 | 0.711 | 0.344 | 0.368 | 18.350 | 11.550 |
| virtue_ethics | virtue_alt_contract | 20 | 0.531 | 0.227 | 0.304 | 18.350 | 16.050 |
| contractualism | contract_neutral_short | 21 | 0.810 | 0.442 | 0.368 | 16.238 | 11.381 |
| contractualism | contract_neutral_length_matched | 21 | 0.765 | 0.406 | 0.359 | 16.238 | 13.381 |
| contractualism | contract_anti | 21 | 0.749 | 0.353 | 0.396 | 16.238 | 8.524 |
| contractualism | contract_alt_deont | 21 | 0.466 | 0.292 | 0.174 | 16.238 | 14.143 |
| contractualism | contract_alt_util | 21 | 0.594 | 0.347 | 0.247 | 16.238 | 11.095 |
| contractualism | contract_alt_virtue | 21 | 0.410 | 0.157 | 0.253 | 16.238 | 15.810 |

Cross-theory cosines:

### neutral_short

| | contract | deont | util | virtue |
|---|---:|---:|---:|---:|
| contract | 1.000 | 0.858 | 0.862 | 0.906 |
| deont | 0.858 | 1.000 | 0.783 | 0.885 |
| util | 0.862 | 0.783 | 1.000 | 0.797 |
| virtue | 0.906 | 0.885 | 0.797 | 1.000 |

### neutral_length_matched

| | contract | deont | util | virtue |
|---|---:|---:|---:|---:|
| contract | 1.000 | 0.811 | 0.839 | 0.874 |
| deont | 0.811 | 1.000 | 0.705 | 0.859 |
| util | 0.839 | 0.705 | 1.000 | 0.751 |
| virtue | 0.874 | 0.859 | 0.751 | 1.000 |

### anti

| | contract | deont | util | virtue |
|---|---:|---:|---:|---:|
| contract | 1.000 | 0.738 | 0.751 | 0.807 |
| deont | 0.738 | 1.000 | 0.584 | 0.784 |
| util | 0.751 | 0.584 | 1.000 | 0.654 |
| virtue | 0.807 | 0.784 | 0.654 | 1.000 |

## Paired Smoke: `pos` min_tokens=20

| theory | construction | n | real_median | null_p95 | gap | pos_tok | neg_tok |
|---|---|---:|---:|---:|---:|---:|---:|
| deontology | deont_neutral_short | 5 | 0.713 | 0.662 | 0.051 | 27.600 | 12.600 |
| deontology | deont_neutral_length_matched | 5 | 0.510 | 0.467 | 0.043 | 27.600 | 11.000 |
| deontology | deont_anti | 5 | 0.590 | 0.582 | 0.008 | 27.600 | 7.800 |
| deontology | deont_alt_util | 5 | 0.388 | 0.379 | 0.009 | 27.600 | 14.200 |
| deontology | deont_alt_virtue | 5 | 0.055 | 0.180 | -0.124 | 27.600 | 20.200 |
| deontology | deont_alt_contract | 5 | 0.392 | 0.403 | -0.010 | 27.600 | 13.400 |
| utilitarian | util_neutral_short | 2 | NA | NA | NA | NA | NA |
| utilitarian | util_neutral_length_matched | 2 | NA | NA | NA | NA | NA |
| utilitarian | util_anti | 2 | NA | NA | NA | NA | NA |
| utilitarian | util_alt_deont | 2 | NA | NA | NA | NA | NA |
| utilitarian | util_alt_virtue | 2 | NA | NA | NA | NA | NA |
| utilitarian | util_alt_contract | 2 | NA | NA | NA | NA | NA |
| virtue_ethics | virtue_neutral_short | 8 | 0.795 | 0.600 | 0.195 | 23.250 | 11.125 |
| virtue_ethics | virtue_neutral_length_matched | 8 | 0.678 | 0.512 | 0.166 | 23.250 | 17.000 |
| virtue_ethics | virtue_anti | 8 | 0.718 | 0.494 | 0.225 | 23.250 | 7.125 |
| virtue_ethics | virtue_alt_deont | 8 | 0.370 | 0.351 | 0.019 | 23.250 | 18.500 |
| virtue_ethics | virtue_alt_util | 8 | 0.464 | 0.304 | 0.160 | 23.250 | 12.375 |
| virtue_ethics | virtue_alt_contract | 8 | 0.373 | 0.288 | 0.085 | 23.250 | 18.250 |
| contractualism | contract_neutral_short | 5 | 0.508 | 0.487 | 0.021 | 25.000 | 12.400 |
| contractualism | contract_neutral_length_matched | 5 | 0.247 | 0.323 | -0.076 | 25.000 | 20.800 |
| contractualism | contract_anti | 5 | 0.440 | 0.444 | -0.004 | 25.000 | 10.000 |
| contractualism | contract_alt_deont | 5 | 0.372 | 0.352 | 0.020 | 25.000 | 13.400 |
| contractualism | contract_alt_util | 5 | 0.160 | 0.150 | 0.010 | 25.000 | 15.400 |
| contractualism | contract_alt_virtue | 5 | 0.208 | 0.175 | 0.034 | 25.000 | 22.600 |

Cross-theory cosines:

### neutral_short

| | contract | deont | virtue |
|---|---:|---:|---:|
| contract | 1.000 | 0.721 | 0.817 |
| deont | 0.721 | 1.000 | 0.862 |
| virtue | 0.817 | 0.862 | 1.000 |

### neutral_length_matched

| | contract | deont | virtue |
|---|---:|---:|---:|
| contract | 1.000 | 0.512 | 0.694 |
| deont | 0.512 | 1.000 | 0.741 |
| virtue | 0.694 | 0.741 | 1.000 |

### anti

| | contract | deont | virtue |
|---|---:|---:|---:|
| contract | 1.000 | 0.559 | 0.741 |
| deont | 0.559 | 1.000 | 0.728 |
| virtue | 0.741 | 0.728 | 1.000 |

## Paired Smoke: `both` min_tokens=10

| theory | construction | n | real_median | null_p95 | gap | pos_tok | neg_tok |
|---|---|---:|---:|---:|---:|---:|---:|
| deontology | deont_neutral_short | 12 | 0.748 | 0.435 | 0.314 | 16.250 | 14.333 |
| deontology | deont_neutral_length_matched | 15 | 0.704 | 0.427 | 0.277 | 18.533 | 13.800 |
| deontology | deont_anti | 4 | 0.350 | 0.350 | 0.000 | 21.500 | 12.500 |
| deontology | deont_alt_util | 11 | 0.430 | 0.322 | 0.108 | 18.182 | 14.455 |
| deontology | deont_alt_virtue | 14 | 0.335 | 0.197 | 0.138 | 19.143 | 17.286 |
| deontology | deont_alt_contract | 16 | 0.459 | 0.197 | 0.262 | 16.688 | 14.500 |
| utilitarian | util_neutral_short | 10 | 0.658 | 0.466 | 0.192 | 14.300 | 14.900 |
| utilitarian | util_neutral_length_matched | 11 | 0.526 | 0.319 | 0.207 | 14.455 | 14.182 |
| utilitarian | util_anti | 3 | NA | NA | NA | NA | NA |
| utilitarian | util_alt_deont | 11 | 0.430 | 0.322 | 0.108 | 14.455 | 18.182 |
| utilitarian | util_alt_virtue | 11 | 0.430 | 0.271 | 0.159 | 15.364 | 18.727 |
| utilitarian | util_alt_contract | 12 | 0.308 | 0.167 | 0.140 | 14.417 | 16.000 |
| virtue_ethics | virtue_neutral_short | 13 | 0.800 | 0.510 | 0.289 | 17.538 | 13.462 |
| virtue_ethics | virtue_neutral_length_matched | 17 | 0.773 | 0.422 | 0.351 | 18.294 | 14.294 |
| virtue_ethics | virtue_anti | 9 | 0.289 | 0.241 | 0.048 | 15.667 | 12.889 |
| virtue_ethics | virtue_alt_deont | 14 | 0.335 | 0.197 | 0.138 | 17.286 | 19.143 |
| virtue_ethics | virtue_alt_util | 11 | 0.430 | 0.271 | 0.159 | 18.727 | 15.364 |
| virtue_ethics | virtue_alt_contract | 17 | 0.392 | 0.158 | 0.233 | 17.765 | 17.412 |
| contractualism | contract_neutral_short | 13 | 0.763 | 0.509 | 0.254 | 14.538 | 14.000 |
| contractualism | contract_neutral_length_matched | 16 | 0.659 | 0.330 | 0.330 | 15.500 | 15.188 |
| contractualism | contract_anti | 6 | 0.503 | 0.496 | 0.006 | 17.500 | 12.500 |
| contractualism | contract_alt_deont | 16 | 0.459 | 0.197 | 0.262 | 14.500 | 16.688 |
| contractualism | contract_alt_util | 12 | 0.308 | 0.167 | 0.140 | 16.000 | 14.417 |
| contractualism | contract_alt_virtue | 17 | 0.392 | 0.158 | 0.233 | 17.412 | 17.765 |

Cross-theory cosines:

### neutral_short

| | contract | deont | util | virtue |
|---|---:|---:|---:|---:|
| contract | 1.000 | 0.836 | 0.863 | 0.855 |
| deont | 0.836 | 1.000 | 0.759 | 0.870 |
| util | 0.863 | 0.759 | 1.000 | 0.748 |
| virtue | 0.855 | 0.870 | 0.748 | 1.000 |

### neutral_length_matched

| | contract | deont | util | virtue |
|---|---:|---:|---:|---:|
| contract | 1.000 | 0.737 | 0.809 | 0.807 |
| deont | 0.737 | 1.000 | 0.645 | 0.848 |
| util | 0.809 | 0.645 | 1.000 | 0.685 |
| virtue | 0.807 | 0.848 | 0.685 | 1.000 |

### anti

| | contract | deont | virtue |
|---|---:|---:|---:|
| contract | 1.000 | 0.285 | 0.618 |
| deont | 0.285 | 1.000 | 0.418 |
| virtue | 0.618 | 0.418 | 1.000 |

## Paired Smoke: `both` min_tokens=20

| theory | construction | n | real_median | null_p95 | gap | pos_tok | neg_tok |
|---|---|---:|---:|---:|---:|---:|---:|
| deontology | deont_neutral_short | 1 | NA | NA | NA | NA | NA |
| deontology | deont_neutral_length_matched | 0 | NA | NA | NA | NA | NA |
| deontology | deont_anti | 0 | NA | NA | NA | NA | NA |
| deontology | deont_alt_util | 1 | NA | NA | NA | NA | NA |
| deontology | deont_alt_virtue | 3 | NA | NA | NA | NA | NA |
| deontology | deont_alt_contract | 1 | NA | NA | NA | NA | NA |
| utilitarian | util_neutral_short | 1 | NA | NA | NA | NA | NA |
| utilitarian | util_neutral_length_matched | 1 | NA | NA | NA | NA | NA |
| utilitarian | util_anti | 0 | NA | NA | NA | NA | NA |
| utilitarian | util_alt_deont | 1 | NA | NA | NA | NA | NA |
| utilitarian | util_alt_virtue | 1 | NA | NA | NA | NA | NA |
| utilitarian | util_alt_contract | 2 | NA | NA | NA | NA | NA |
| virtue_ethics | virtue_neutral_short | 1 | NA | NA | NA | NA | NA |
| virtue_ethics | virtue_neutral_length_matched | 3 | NA | NA | NA | NA | NA |
| virtue_ethics | virtue_anti | 0 | NA | NA | NA | NA | NA |
| virtue_ethics | virtue_alt_deont | 3 | NA | NA | NA | NA | NA |
| virtue_ethics | virtue_alt_util | 1 | NA | NA | NA | NA | NA |
| virtue_ethics | virtue_alt_contract | 4 | 0.162 | 0.162 | 0.000 | 23.500 | 26.000 |
| contractualism | contract_neutral_short | 1 | NA | NA | NA | NA | NA |
| contractualism | contract_neutral_length_matched | 3 | NA | NA | NA | NA | NA |
| contractualism | contract_anti | 1 | NA | NA | NA | NA | NA |
| contractualism | contract_alt_deont | 1 | NA | NA | NA | NA | NA |
| contractualism | contract_alt_util | 2 | NA | NA | NA | NA | NA |
| contractualism | contract_alt_virtue | 4 | 0.162 | 0.162 | 0.000 | 26.000 | 23.500 |

Cross-theory cosines:

