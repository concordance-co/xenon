# All-Theories Persona-Vector Pole Pilot Report (Phase 02)

- generation artifact: `generation_run_1_15bc125de56b`
- capture artifact: `capture_1_c2684db0530c`
- captured layers: `[0, 4, 16, 24, 32, 40]`
- primary layer: `32` / primary site: `generated_sequence_residual`
- theories: `['deontology', 'utilitarian', 'virtue_ethics', 'contractualism']`

## Headline smoke @ L32 generated_sequence_residual

Pass criterion (soft): split_half − null_p95 ≥ 0.2; marginal in [0.1, 0.2); fail < 0.1.

| theory | construction | split_half | null_p95 | gap | verdict |
|---|---|---:|---:|---:|---|
| deontology | deont_neutral_short | 0.687 | 0.626 | 0.061 | fail |
| deontology | deont_neutral_length_matched | 0.654 | 0.538 | 0.116 | marginal |
| deontology | deont_anti | 0.685 | 0.563 | 0.123 | marginal |
| deontology | deont_alt_util | 0.304 | 0.407 | -0.103 | fail |
| deontology | deont_alt_virtue | 0.302 | 0.382 | -0.081 | fail |
| deontology | deont_alt_contract | 0.271 | 0.346 | -0.075 | fail |
| utilitarian | util_neutral_short | 0.559 | 0.517 | 0.042 | fail |
| utilitarian | util_neutral_length_matched | 0.589 | 0.463 | 0.126 | marginal |
| utilitarian | util_anti | 0.444 | 0.465 | -0.021 | fail |
| utilitarian | util_alt_deont | 0.387 | 0.336 | 0.051 | fail |
| utilitarian | util_alt_virtue | 0.588 | 0.506 | 0.083 | fail |
| utilitarian | util_alt_contract | 0.258 | 0.400 | -0.141 | fail |
| virtue_ethics | virtue_neutral_short | 0.720 | 0.681 | 0.039 | fail |
| virtue_ethics | virtue_neutral_length_matched | 0.700 | 0.637 | 0.064 | fail |
| virtue_ethics | virtue_anti | 0.655 | 0.560 | 0.095 | fail |
| virtue_ethics | virtue_alt_deont | 0.172 | 0.442 | -0.270 | fail |
| virtue_ethics | virtue_alt_util | 0.325 | 0.503 | -0.178 | fail |
| virtue_ethics | virtue_alt_contract | 0.111 | 0.365 | -0.254 | fail |
| contractualism | contract_neutral_short | 0.733 | 0.599 | 0.134 | marginal |
| contractualism | contract_neutral_length_matched | 0.675 | 0.566 | 0.108 | marginal |
| contractualism | contract_anti | 0.645 | 0.482 | 0.163 | marginal |
| contractualism | contract_alt_deont | 0.251 | 0.322 | -0.071 | fail |
| contractualism | contract_alt_util | 0.133 | 0.410 | -0.277 | fail |
| contractualism | contract_alt_virtue | 0.273 | 0.413 | -0.140 | fail |

## Cross-theory direction cosines @ L32 generated_sequence_residual

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

## Generation sanity

- rows: `420`
- nonempty: `420` (1.000)
- finish reasons: `{'stop': 420}`

Response length by condition:

| condition | n | mean_chars | median_chars | min | max |
|---|---:|---:|---:|---:|---:|
| N_anti_contract_01 | 30 | 36.167 | 32.500 | 9 | 115 |
| N_anti_deont_01 | 30 | 30.267 | 27.000 | 11 | 76 |
| N_anti_util_01 | 30 | 30.367 | 29.500 | 9 | 69 |
| N_anti_virtue_01 | 30 | 40.333 | 37.000 | 14 | 124 |
| N_neutral_01 | 30 | 48.300 | 42.000 | 14 | 142 |
| N_neutral_02 | 30 | 56.633 | 47.000 | 15 | 202 |
| P_contract_01 | 30 | 67.200 | 56.500 | 15 | 184 |
| P_contract_02 | 30 | 65.400 | 49.000 | 15 | 190 |
| P_deont_01 | 30 | 64.400 | 46.000 | 14 | 188 |
| P_deont_02 | 30 | 53.733 | 40.000 | 15 | 124 |
| P_util_01 | 30 | 47.833 | 37.000 | 14 | 124 |
| P_util_02 | 30 | 48.733 | 48.500 | 16 | 110 |
| P_virtue_01 | 30 | 74.233 | 75.500 | 15 | 162 |
| P_virtue_02 | 30 | 32.233 | 28.500 | 9 | 83 |

## Behavioral divergence per theory (diagnostic)

### deontology

| pair | n | diverged_share | avg_jaccard |
|---|---:|---:|---:|
| deont_p1_vs_p2 | 30 | 0.667 | 0.460 |
| deont_vs_N_anti | 30 | 0.667 | 0.439 |
| deont_vs_N_neutral_length_matched | 30 | 0.800 | 0.403 |
| deont_vs_N_neutral_short | 30 | 0.733 | 0.451 |
| deont_vs_alt_contract | 30 | 0.600 | 0.527 |
| deont_vs_alt_util | 30 | 0.600 | 0.538 |
| deont_vs_alt_virtue | 30 | 0.633 | 0.497 |

### utilitarian

| pair | n | diverged_share | avg_jaccard |
|---|---:|---:|---:|
| util_p1_vs_p2 | 30 | 0.400 | 0.628 |
| util_vs_N_anti | 30 | 0.700 | 0.442 |
| util_vs_N_neutral_length_matched | 30 | 0.700 | 0.488 |
| util_vs_N_neutral_short | 30 | 0.633 | 0.516 |
| util_vs_alt_contract | 30 | 0.567 | 0.560 |
| util_vs_alt_deont | 30 | 0.600 | 0.538 |
| util_vs_alt_virtue | 30 | 0.733 | 0.412 |

### virtue_ethics

| pair | n | diverged_share | avg_jaccard |
|---|---:|---:|---:|
| virtue_p1_vs_p2 | 30 | 0.733 | 0.438 |
| virtue_vs_N_anti | 30 | 0.733 | 0.430 |
| virtue_vs_N_neutral_length_matched | 30 | 0.867 | 0.344 |
| virtue_vs_N_neutral_short | 30 | 0.833 | 0.368 |
| virtue_vs_alt_contract | 30 | 0.600 | 0.546 |
| virtue_vs_alt_deont | 30 | 0.633 | 0.497 |
| virtue_vs_alt_util | 30 | 0.733 | 0.412 |

### contractualism

| pair | n | diverged_share | avg_jaccard |
|---|---:|---:|---:|
| contract_p1_vs_p2 | 30 | 0.633 | 0.515 |
| contract_vs_N_anti | 30 | 0.833 | 0.355 |
| contract_vs_N_neutral_length_matched | 30 | 0.800 | 0.404 |
| contract_vs_N_neutral_short | 30 | 0.767 | 0.418 |
| contract_vs_alt_deont | 30 | 0.600 | 0.527 |
| contract_vs_alt_util | 30 | 0.567 | 0.560 |
| contract_vs_alt_virtue | 30 | 0.600 | 0.546 |

## Per-theory directions: deontology

### Site `generated_sequence_residual`

#### Layer 0

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| deont_alt_contract | -0.557 | 0.252 | 0.400 | -0.809 |
| deont_alt_util | -0.361 | 0.249 | 0.301 | -0.610 |
| deont_alt_virtue | -0.431 | 0.290 | 0.518 | -0.722 |
| deont_anti | -0.073 | 0.310 | 0.462 | -0.383 |
| deont_anti_p_variant | -0.198 | 0.338 | 0.445 | -0.536 |
| deont_neutral_length_matched | 0.413 | 0.440 | 0.669 | -0.027 |
| deont_neutral_length_matched_p_variant | 0.379 | 0.419 | 0.624 | -0.040 |
| deont_neutral_short | 0.429 | 0.409 | 0.687 | 0.020 |
| deont_neutral_short_p_variant | 0.375 | 0.410 | 0.649 | -0.034 |

Pole-construction cosine matrix:

| | deont_alt_contract | deont_alt_util | deont_alt_virtue | deont_anti | deont_neutral_length_matched | deont_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| deont_alt_contract | 1.000 | 0.349 | 0.572 | 0.067 | 0.120 | 0.095 |
| deont_alt_util | 0.349 | 1.000 | 0.224 | 0.494 | 0.559 | 0.569 |
| deont_alt_virtue | 0.572 | 0.224 | 1.000 | 0.032 | 0.136 | 0.106 |
| deont_anti | 0.067 | 0.494 | 0.032 | 1.000 | 0.410 | 0.444 |
| deont_neutral_length_matched | 0.120 | 0.559 | 0.136 | 0.410 | 1.000 | 0.960 |
| deont_neutral_short | 0.095 | 0.569 | 0.106 | 0.444 | 0.960 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.906`
- `neutral_length_matched`: `0.898`
- `anti`: `0.809`

#### Layer 4

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| deont_alt_contract | -0.274 | 0.289 | 0.383 | -0.562 |
| deont_alt_util | 0.044 | 0.370 | 0.534 | -0.327 |
| deont_alt_virtue | -0.075 | 0.394 | 0.575 | -0.470 |
| deont_anti | 0.372 | 0.400 | 0.602 | -0.029 |
| deont_anti_p_variant | 0.283 | 0.456 | 0.494 | -0.173 |
| deont_neutral_length_matched | 0.639 | 0.567 | 0.659 | 0.073 |
| deont_neutral_length_matched_p_variant | 0.627 | 0.545 | 0.614 | 0.083 |
| deont_neutral_short | 0.728 | 0.572 | 0.619 | 0.155 |
| deont_neutral_short_p_variant | 0.700 | 0.587 | 0.639 | 0.112 |

Pole-construction cosine matrix:

| | deont_alt_contract | deont_alt_util | deont_alt_virtue | deont_anti | deont_neutral_length_matched | deont_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| deont_alt_contract | 1.000 | 0.356 | 0.531 | -0.063 | 0.018 | -0.024 |
| deont_alt_util | 0.356 | 1.000 | 0.059 | 0.584 | 0.568 | 0.595 |
| deont_alt_virtue | 0.531 | 0.059 | 1.000 | -0.133 | 0.033 | -0.041 |
| deont_anti | -0.063 | 0.584 | -0.133 | 1.000 | 0.527 | 0.590 |
| deont_neutral_length_matched | 0.018 | 0.568 | 0.033 | 0.527 | 1.000 | 0.963 |
| deont_neutral_short | -0.024 | 0.595 | -0.041 | 0.590 | 0.963 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.919`
- `neutral_length_matched`: `0.906`
- `anti`: `0.833`

#### Layer 16

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| deont_alt_contract | -0.476 | 0.316 | 0.471 | -0.792 |
| deont_alt_util | -0.293 | 0.363 | 0.404 | -0.656 |
| deont_alt_virtue | -0.375 | 0.532 | 0.554 | -0.907 |
| deont_anti | 0.097 | 0.358 | 0.481 | -0.261 |
| deont_anti_p_variant | 0.145 | 0.442 | 0.472 | -0.297 |
| deont_neutral_length_matched | 0.273 | 0.517 | 0.591 | -0.245 |
| deont_neutral_length_matched_p_variant | 0.423 | 0.538 | 0.638 | -0.116 |
| deont_neutral_short | 0.351 | 0.552 | 0.617 | -0.201 |
| deont_neutral_short_p_variant | 0.477 | 0.515 | 0.648 | -0.038 |

Pole-construction cosine matrix:

| | deont_alt_contract | deont_alt_util | deont_alt_virtue | deont_anti | deont_neutral_length_matched | deont_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| deont_alt_contract | 1.000 | 0.395 | 0.487 | -0.037 | -0.005 | -0.067 |
| deont_alt_util | 0.395 | 1.000 | -0.050 | 0.514 | 0.377 | 0.390 |
| deont_alt_virtue | 0.487 | -0.050 | 1.000 | -0.212 | 0.000 | -0.087 |
| deont_anti | -0.037 | 0.514 | -0.212 | 1.000 | 0.404 | 0.459 |
| deont_neutral_length_matched | -0.005 | 0.377 | 0.000 | 0.404 | 1.000 | 0.954 |
| deont_neutral_short | -0.067 | 0.390 | -0.087 | 0.459 | 0.954 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.891`
- `neutral_length_matched`: `0.876`
- `anti`: `0.823`

#### Layer 24

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| deont_alt_contract | -0.322 | 0.293 | 0.381 | -0.615 |
| deont_alt_util | -0.101 | 0.314 | 0.461 | -0.415 |
| deont_alt_virtue | -0.139 | 0.377 | 0.524 | -0.517 |
| deont_anti | 0.281 | 0.509 | 0.614 | -0.229 |
| deont_anti_p_variant | 0.349 | 0.435 | 0.645 | -0.086 |
| deont_neutral_length_matched | 0.454 | 0.495 | 0.685 | -0.041 |
| deont_neutral_length_matched_p_variant | 0.607 | 0.514 | 0.733 | 0.093 |
| deont_neutral_short | 0.526 | 0.565 | 0.689 | -0.039 |
| deont_neutral_short_p_variant | 0.596 | 0.602 | 0.733 | -0.006 |

Pole-construction cosine matrix:

| | deont_alt_contract | deont_alt_util | deont_alt_virtue | deont_anti | deont_neutral_length_matched | deont_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| deont_alt_contract | 1.000 | 0.318 | 0.540 | -0.112 | -0.012 | -0.067 |
| deont_alt_util | 0.318 | 1.000 | -0.087 | 0.538 | 0.429 | 0.461 |
| deont_alt_virtue | 0.540 | -0.087 | 1.000 | -0.280 | -0.020 | -0.120 |
| deont_anti | -0.112 | 0.538 | -0.280 | 1.000 | 0.389 | 0.452 |
| deont_neutral_length_matched | -0.012 | 0.429 | -0.020 | 0.389 | 1.000 | 0.946 |
| deont_neutral_short | -0.067 | 0.461 | -0.120 | 0.452 | 0.946 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.903`
- `neutral_length_matched`: `0.882`
- `anti`: `0.845`

#### Layer 32

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| deont_alt_contract | 0.271 | 0.346 | 0.583 | -0.075 |
| deont_alt_util | 0.304 | 0.407 | 0.593 | -0.103 |
| deont_alt_virtue | 0.302 | 0.382 | 0.518 | -0.081 |
| deont_anti | 0.685 | 0.563 | 0.615 | 0.123 |
| deont_anti_p_variant | 0.729 | 0.519 | 0.617 | 0.209 |
| deont_neutral_length_matched | 0.654 | 0.538 | 0.620 | 0.116 |
| deont_neutral_length_matched_p_variant | 0.680 | 0.541 | 0.552 | 0.139 |
| deont_neutral_short | 0.687 | 0.626 | 0.663 | 0.061 |
| deont_neutral_short_p_variant | 0.709 | 0.591 | 0.657 | 0.119 |

Pole-construction cosine matrix:

| | deont_alt_contract | deont_alt_util | deont_alt_virtue | deont_anti | deont_neutral_length_matched | deont_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| deont_alt_contract | 1.000 | 0.561 | 0.576 | 0.202 | 0.234 | 0.184 |
| deont_alt_util | 0.561 | 1.000 | 0.141 | 0.642 | 0.550 | 0.560 |
| deont_alt_virtue | 0.576 | 0.141 | 1.000 | -0.080 | 0.076 | -0.003 |
| deont_anti | 0.202 | 0.642 | -0.080 | 1.000 | 0.517 | 0.552 |
| deont_neutral_length_matched | 0.234 | 0.550 | 0.076 | 0.517 | 1.000 | 0.945 |
| deont_neutral_short | 0.184 | 0.560 | -0.003 | 0.552 | 0.945 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.873`
- `neutral_length_matched`: `0.847`
- `anti`: `0.870`

#### Layer 40

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| deont_alt_contract | -0.079 | 0.282 | 0.345 | -0.361 |
| deont_alt_util | 0.194 | 0.370 | 0.445 | -0.176 |
| deont_alt_virtue | -0.007 | 0.398 | 0.569 | -0.405 |
| deont_anti | 0.532 | 0.419 | 0.584 | 0.113 |
| deont_anti_p_variant | 0.606 | 0.438 | 0.541 | 0.168 |
| deont_neutral_length_matched | 0.491 | 0.510 | 0.609 | -0.019 |
| deont_neutral_length_matched_p_variant | 0.588 | 0.488 | 0.522 | 0.099 |
| deont_neutral_short | 0.522 | 0.499 | 0.568 | 0.023 |
| deont_neutral_short_p_variant | 0.612 | 0.500 | 0.565 | 0.113 |

Pole-construction cosine matrix:

| | deont_alt_contract | deont_alt_util | deont_alt_virtue | deont_anti | deont_neutral_length_matched | deont_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| deont_alt_contract | 1.000 | 0.549 | 0.550 | 0.263 | 0.314 | 0.261 |
| deont_alt_util | 0.549 | 1.000 | 0.092 | 0.679 | 0.583 | 0.587 |
| deont_alt_virtue | 0.550 | 0.092 | 1.000 | -0.098 | 0.085 | 0.013 |
| deont_anti | 0.263 | 0.679 | -0.098 | 1.000 | 0.556 | 0.602 |
| deont_neutral_length_matched | 0.314 | 0.583 | 0.085 | 0.556 | 1.000 | 0.958 |
| deont_neutral_short | 0.261 | 0.587 | 0.013 | 0.602 | 0.958 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.849`
- `neutral_length_matched`: `0.830`
- `anti`: `0.854`

### Site `prompt_end_residual`

#### Layer 0

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| deont_alt_contract | nan | nan | nan | nan |
| deont_alt_util | nan | nan | nan | nan |
| deont_alt_virtue | nan | nan | nan | nan |
| deont_anti | nan | nan | nan | nan |
| deont_anti_p_variant | nan | nan | nan | nan |
| deont_neutral_length_matched | nan | nan | nan | nan |
| deont_neutral_length_matched_p_variant | nan | nan | nan | nan |
| deont_neutral_short | nan | nan | nan | nan |
| deont_neutral_short_p_variant | nan | nan | nan | nan |

Pole-construction cosine matrix:

| | deont_alt_contract | deont_alt_util | deont_alt_virtue | deont_anti | deont_neutral_length_matched | deont_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| deont_alt_contract | nan | nan | nan | nan | nan | nan |
| deont_alt_util | nan | nan | nan | nan | nan | nan |
| deont_alt_virtue | nan | nan | nan | nan | nan | nan |
| deont_anti | nan | nan | nan | nan | nan | nan |
| deont_neutral_length_matched | nan | nan | nan | nan | nan | nan |
| deont_neutral_short | nan | nan | nan | nan | nan | nan |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `nan`
- `neutral_length_matched`: `nan`
- `anti`: `nan`

#### Layer 4

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| deont_alt_contract | 0.724 | 0.493 | 0.558 | 0.231 |
| deont_alt_util | 0.658 | 0.440 | 0.553 | 0.219 |
| deont_alt_virtue | 0.705 | 0.478 | 0.536 | 0.226 |
| deont_anti | 0.853 | 0.641 | 0.731 | 0.212 |
| deont_anti_p_variant | 0.917 | 0.744 | 0.839 | 0.172 |
| deont_neutral_length_matched | 0.902 | 0.700 | 0.775 | 0.201 |
| deont_neutral_length_matched_p_variant | 0.923 | 0.723 | 0.843 | 0.201 |
| deont_neutral_short | 0.942 | 0.820 | 0.869 | 0.122 |
| deont_neutral_short_p_variant | 0.961 | 0.864 | 0.909 | 0.097 |

Pole-construction cosine matrix:

| | deont_alt_contract | deont_alt_util | deont_alt_virtue | deont_anti | deont_neutral_length_matched | deont_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| deont_alt_contract | 1.000 | 0.653 | 0.254 | -0.066 | 0.090 | 0.227 |
| deont_alt_util | 0.653 | 1.000 | -0.008 | 0.062 | 0.134 | 0.367 |
| deont_alt_virtue | 0.254 | -0.008 | 1.000 | -0.162 | -0.002 | -0.211 |
| deont_anti | -0.066 | 0.062 | -0.162 | 1.000 | 0.658 | 0.757 |
| deont_neutral_length_matched | 0.090 | 0.134 | -0.002 | 0.658 | 1.000 | 0.613 |
| deont_neutral_short | 0.227 | 0.367 | -0.211 | 0.757 | 0.613 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.954`
- `neutral_length_matched`: `0.879`
- `anti`: `0.882`

#### Layer 16

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| deont_alt_contract | 0.663 | 0.618 | 0.712 | 0.044 |
| deont_alt_util | 0.638 | 0.580 | 0.702 | 0.058 |
| deont_alt_virtue | 0.504 | 0.493 | 0.649 | 0.012 |
| deont_anti | 0.707 | 0.647 | 0.752 | 0.060 |
| deont_anti_p_variant | 0.764 | 0.679 | 0.782 | 0.086 |
| deont_neutral_length_matched | 0.803 | 0.742 | 0.827 | 0.061 |
| deont_neutral_length_matched_p_variant | 0.806 | 0.732 | 0.824 | 0.074 |
| deont_neutral_short | 0.779 | 0.712 | 0.811 | 0.067 |
| deont_neutral_short_p_variant | 0.808 | 0.729 | 0.822 | 0.079 |

Pole-construction cosine matrix:

| | deont_alt_contract | deont_alt_util | deont_alt_virtue | deont_anti | deont_neutral_length_matched | deont_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| deont_alt_contract | 1.000 | 0.510 | 0.518 | 0.186 | 0.156 | 0.153 |
| deont_alt_util | 0.510 | 1.000 | 0.313 | 0.194 | 0.251 | 0.200 |
| deont_alt_virtue | 0.518 | 0.313 | 1.000 | 0.089 | 0.076 | 0.092 |
| deont_anti | 0.186 | 0.194 | 0.089 | 1.000 | 0.399 | 0.468 |
| deont_neutral_length_matched | 0.156 | 0.251 | 0.076 | 0.399 | 1.000 | 0.782 |
| deont_neutral_short | 0.153 | 0.200 | 0.092 | 0.468 | 0.782 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.803`
- `neutral_length_matched`: `0.815`
- `anti`: `0.726`

#### Layer 24

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| deont_alt_contract | 0.855 | 0.672 | 0.755 | 0.183 |
| deont_alt_util | 0.831 | 0.650 | 0.740 | 0.181 |
| deont_alt_virtue | 0.780 | 0.586 | 0.659 | 0.194 |
| deont_anti | 0.871 | 0.706 | 0.794 | 0.165 |
| deont_anti_p_variant | 0.910 | 0.767 | 0.834 | 0.142 |
| deont_neutral_length_matched | 0.931 | 0.794 | 0.871 | 0.137 |
| deont_neutral_length_matched_p_variant | 0.947 | 0.831 | 0.892 | 0.116 |
| deont_neutral_short | 0.922 | 0.785 | 0.849 | 0.137 |
| deont_neutral_short_p_variant | 0.944 | 0.835 | 0.882 | 0.109 |

Pole-construction cosine matrix:

| | deont_alt_contract | deont_alt_util | deont_alt_virtue | deont_anti | deont_neutral_length_matched | deont_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| deont_alt_contract | 1.000 | 0.506 | 0.444 | 0.141 | 0.207 | 0.200 |
| deont_alt_util | 0.506 | 1.000 | 0.312 | 0.168 | 0.363 | 0.300 |
| deont_alt_virtue | 0.444 | 0.312 | 1.000 | 0.047 | 0.260 | 0.178 |
| deont_anti | 0.141 | 0.168 | 0.047 | 1.000 | 0.339 | 0.424 |
| deont_neutral_length_matched | 0.207 | 0.363 | 0.260 | 0.339 | 1.000 | 0.849 |
| deont_neutral_short | 0.200 | 0.300 | 0.178 | 0.424 | 0.849 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.839`
- `neutral_length_matched`: `0.848`
- `anti`: `0.706`

#### Layer 32

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| deont_alt_contract | 0.890 | 0.750 | 0.855 | 0.140 |
| deont_alt_util | 0.895 | 0.791 | 0.866 | 0.104 |
| deont_alt_virtue | 0.795 | 0.660 | 0.778 | 0.135 |
| deont_anti | 0.951 | 0.888 | 0.938 | 0.063 |
| deont_anti_p_variant | 0.947 | 0.884 | 0.936 | 0.063 |
| deont_neutral_length_matched | 0.942 | 0.872 | 0.933 | 0.070 |
| deont_neutral_length_matched_p_variant | 0.946 | 0.888 | 0.940 | 0.058 |
| deont_neutral_short | 0.940 | 0.849 | 0.920 | 0.090 |
| deont_neutral_short_p_variant | 0.944 | 0.872 | 0.932 | 0.072 |

Pole-construction cosine matrix:

| | deont_alt_contract | deont_alt_util | deont_alt_virtue | deont_anti | deont_neutral_length_matched | deont_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| deont_alt_contract | 1.000 | 0.602 | 0.490 | 0.167 | 0.207 | 0.129 |
| deont_alt_util | 0.602 | 1.000 | 0.467 | 0.261 | 0.365 | 0.282 |
| deont_alt_virtue | 0.490 | 0.467 | 1.000 | 0.148 | 0.307 | 0.196 |
| deont_anti | 0.167 | 0.261 | 0.148 | 1.000 | 0.508 | 0.563 |
| deont_neutral_length_matched | 0.207 | 0.365 | 0.307 | 0.508 | 1.000 | 0.792 |
| deont_neutral_short | 0.129 | 0.282 | 0.196 | 0.563 | 0.792 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.862`
- `neutral_length_matched`: `0.879`
- `anti`: `0.874`

#### Layer 40

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| deont_alt_contract | 0.907 | 0.732 | 0.873 | 0.175 |
| deont_alt_util | 0.897 | 0.766 | 0.867 | 0.130 |
| deont_alt_virtue | 0.832 | 0.626 | 0.771 | 0.207 |
| deont_anti | 0.963 | 0.885 | 0.933 | 0.078 |
| deont_anti_p_variant | 0.962 | 0.861 | 0.926 | 0.101 |
| deont_neutral_length_matched | 0.971 | 0.906 | 0.956 | 0.065 |
| deont_neutral_length_matched_p_variant | 0.980 | 0.912 | 0.962 | 0.068 |
| deont_neutral_short | 0.978 | 0.915 | 0.956 | 0.064 |
| deont_neutral_short_p_variant | 0.984 | 0.925 | 0.964 | 0.058 |

Pole-construction cosine matrix:

| | deont_alt_contract | deont_alt_util | deont_alt_virtue | deont_anti | deont_neutral_length_matched | deont_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| deont_alt_contract | 1.000 | 0.591 | 0.564 | 0.293 | 0.144 | 0.087 |
| deont_alt_util | 0.591 | 1.000 | 0.354 | 0.286 | 0.346 | 0.334 |
| deont_alt_virtue | 0.564 | 0.354 | 1.000 | 0.259 | 0.187 | 0.101 |
| deont_anti | 0.293 | 0.286 | 0.259 | 1.000 | 0.495 | 0.461 |
| deont_neutral_length_matched | 0.144 | 0.346 | 0.187 | 0.495 | 1.000 | 0.813 |
| deont_neutral_short | 0.087 | 0.334 | 0.101 | 0.461 | 0.813 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.909`
- `neutral_length_matched`: `0.889`
- `anti`: `0.810`

## Per-theory directions: utilitarian

### Site `generated_sequence_residual`

#### Layer 0

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| util_alt_contract | -0.371 | 0.307 | 0.418 | -0.678 |
| util_alt_deont | -0.339 | 0.284 | 0.378 | -0.623 |
| util_alt_virtue | -0.241 | 0.307 | 0.433 | -0.548 |
| util_anti | -0.293 | 0.296 | 0.377 | -0.589 |
| util_anti_p_variant | -0.012 | 0.397 | 0.489 | -0.409 |
| util_neutral_length_matched | 0.126 | 0.490 | 0.556 | -0.364 |
| util_neutral_length_matched_p_variant | -0.373 | 0.354 | 0.436 | -0.727 |
| util_neutral_short | 0.145 | 0.439 | 0.640 | -0.294 |
| util_neutral_short_p_variant | -0.302 | 0.392 | 0.448 | -0.694 |

Pole-construction cosine matrix:

| | util_alt_contract | util_alt_deont | util_alt_virtue | util_anti | util_neutral_length_matched | util_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| util_alt_contract | 1.000 | 0.713 | 0.755 | 0.144 | -0.115 | -0.166 |
| util_alt_deont | 0.713 | 1.000 | 0.660 | 0.250 | -0.067 | -0.102 |
| util_alt_virtue | 0.755 | 0.660 | 1.000 | 0.233 | -0.034 | -0.087 |
| util_anti | 0.144 | 0.250 | 0.233 | 1.000 | 0.187 | 0.204 |
| util_neutral_length_matched | -0.115 | -0.067 | -0.034 | 0.187 | 1.000 | 0.942 |
| util_neutral_short | -0.166 | -0.102 | -0.087 | 0.204 | 0.942 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.691`
- `neutral_length_matched`: `0.670`
- `anti`: `0.657`

#### Layer 4

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| util_alt_contract | -0.035 | 0.355 | 0.534 | -0.391 |
| util_alt_deont | -0.033 | 0.331 | 0.425 | -0.364 |
| util_alt_virtue | 0.286 | 0.434 | 0.542 | -0.147 |
| util_anti | 0.239 | 0.458 | 0.593 | -0.220 |
| util_anti_p_variant | 0.321 | 0.519 | 0.667 | -0.198 |
| util_neutral_length_matched | 0.496 | 0.552 | 0.623 | -0.056 |
| util_neutral_length_matched_p_variant | 0.016 | 0.389 | 0.460 | -0.373 |
| util_neutral_short | 0.639 | 0.552 | 0.676 | 0.087 |
| util_neutral_short_p_variant | 0.182 | 0.422 | 0.535 | -0.240 |

Pole-construction cosine matrix:

| | util_alt_contract | util_alt_deont | util_alt_virtue | util_anti | util_neutral_length_matched | util_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| util_alt_contract | 1.000 | 0.729 | 0.788 | -0.162 | -0.245 | -0.350 |
| util_alt_deont | 0.729 | 1.000 | 0.730 | 0.064 | -0.114 | -0.191 |
| util_alt_virtue | 0.788 | 0.730 | 1.000 | -0.042 | -0.084 | -0.203 |
| util_anti | -0.162 | 0.064 | -0.042 | 1.000 | 0.209 | 0.271 |
| util_neutral_length_matched | -0.245 | -0.114 | -0.084 | 0.209 | 1.000 | 0.943 |
| util_neutral_short | -0.350 | -0.191 | -0.203 | 0.271 | 0.943 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.793`
- `neutral_length_matched`: `0.757`
- `anti`: `0.737`

#### Layer 16

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| util_alt_contract | -0.388 | 0.408 | 0.485 | -0.796 |
| util_alt_deont | -0.338 | 0.343 | 0.447 | -0.681 |
| util_alt_virtue | -0.029 | 0.563 | 0.653 | -0.592 |
| util_anti | -0.084 | 0.339 | 0.472 | -0.423 |
| util_anti_p_variant | 0.179 | 0.458 | 0.533 | -0.279 |
| util_neutral_length_matched | 0.192 | 0.506 | 0.605 | -0.314 |
| util_neutral_length_matched_p_variant | 0.058 | 0.420 | 0.576 | -0.363 |
| util_neutral_short | 0.279 | 0.523 | 0.565 | -0.244 |
| util_neutral_short_p_variant | 0.100 | 0.417 | 0.578 | -0.317 |

Pole-construction cosine matrix:

| | util_alt_contract | util_alt_deont | util_alt_virtue | util_anti | util_neutral_length_matched | util_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| util_alt_contract | 1.000 | 0.653 | 0.762 | -0.174 | -0.011 | -0.106 |
| util_alt_deont | 0.653 | 1.000 | 0.731 | 0.099 | 0.189 | 0.138 |
| util_alt_virtue | 0.762 | 0.731 | 1.000 | -0.058 | 0.152 | 0.051 |
| util_anti | -0.174 | 0.099 | -0.058 | 1.000 | 0.241 | 0.295 |
| util_neutral_length_matched | -0.011 | 0.189 | 0.152 | 0.241 | 1.000 | 0.946 |
| util_neutral_short | -0.106 | 0.138 | 0.051 | 0.295 | 0.946 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.757`
- `neutral_length_matched`: `0.725`
- `anti`: `0.706`

#### Layer 24

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| util_alt_contract | -0.121 | 0.461 | 0.506 | -0.582 |
| util_alt_deont | -0.183 | 0.332 | 0.516 | -0.514 |
| util_alt_virtue | 0.173 | 0.441 | 0.524 | -0.268 |
| util_anti | 0.062 | 0.429 | 0.482 | -0.367 |
| util_anti_p_variant | 0.303 | 0.454 | 0.543 | -0.151 |
| util_neutral_length_matched | 0.317 | 0.469 | 0.649 | -0.152 |
| util_neutral_length_matched_p_variant | 0.207 | 0.423 | 0.486 | -0.216 |
| util_neutral_short | 0.393 | 0.547 | 0.695 | -0.154 |
| util_neutral_short_p_variant | 0.163 | 0.449 | 0.556 | -0.286 |

Pole-construction cosine matrix:

| | util_alt_contract | util_alt_deont | util_alt_virtue | util_anti | util_neutral_length_matched | util_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| util_alt_contract | 1.000 | 0.661 | 0.791 | -0.256 | -0.041 | -0.160 |
| util_alt_deont | 0.661 | 1.000 | 0.732 | 0.034 | 0.132 | 0.045 |
| util_alt_virtue | 0.791 | 0.732 | 1.000 | -0.146 | 0.110 | -0.029 |
| util_anti | -0.256 | 0.034 | -0.146 | 1.000 | 0.186 | 0.229 |
| util_neutral_length_matched | -0.041 | 0.132 | 0.110 | 0.186 | 1.000 | 0.930 |
| util_neutral_short | -0.160 | 0.045 | -0.029 | 0.229 | 0.930 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.788`
- `neutral_length_matched`: `0.755`
- `anti`: `0.728`

#### Layer 32

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| util_alt_contract | 0.258 | 0.400 | 0.600 | -0.141 |
| util_alt_deont | 0.387 | 0.336 | 0.390 | 0.051 |
| util_alt_virtue | 0.588 | 0.506 | 0.598 | 0.083 |
| util_anti | 0.444 | 0.465 | 0.540 | -0.021 |
| util_anti_p_variant | 0.612 | 0.435 | 0.567 | 0.176 |
| util_neutral_length_matched | 0.589 | 0.463 | 0.572 | 0.126 |
| util_neutral_length_matched_p_variant | 0.573 | 0.401 | 0.505 | 0.172 |
| util_neutral_short | 0.559 | 0.517 | 0.630 | 0.042 |
| util_neutral_short_p_variant | 0.598 | 0.469 | 0.498 | 0.129 |

Pole-construction cosine matrix:

| | util_alt_contract | util_alt_deont | util_alt_virtue | util_anti | util_neutral_length_matched | util_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| util_alt_contract | 1.000 | 0.616 | 0.775 | -0.066 | 0.039 | -0.075 |
| util_alt_deont | 0.616 | 1.000 | 0.716 | 0.250 | 0.200 | 0.121 |
| util_alt_virtue | 0.775 | 0.716 | 1.000 | -0.011 | 0.142 | 0.017 |
| util_anti | -0.066 | 0.250 | -0.011 | 1.000 | 0.321 | 0.361 |
| util_neutral_length_matched | 0.039 | 0.200 | 0.142 | 0.321 | 1.000 | 0.920 |
| util_neutral_short | -0.075 | 0.121 | 0.017 | 0.361 | 0.920 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.758`
- `neutral_length_matched`: `0.728`
- `anti`: `0.765`

#### Layer 40

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| util_alt_contract | 0.050 | 0.367 | 0.415 | -0.317 |
| util_alt_deont | 0.184 | 0.372 | 0.418 | -0.188 |
| util_alt_virtue | 0.335 | 0.455 | 0.526 | -0.120 |
| util_anti | 0.192 | 0.423 | 0.502 | -0.231 |
| util_anti_p_variant | 0.349 | 0.395 | 0.515 | -0.045 |
| util_neutral_length_matched | 0.421 | 0.464 | 0.564 | -0.043 |
| util_neutral_length_matched_p_variant | 0.302 | 0.416 | 0.518 | -0.114 |
| util_neutral_short | 0.465 | 0.472 | 0.543 | -0.007 |
| util_neutral_short_p_variant | 0.304 | 0.414 | 0.541 | -0.109 |

Pole-construction cosine matrix:

| | util_alt_contract | util_alt_deont | util_alt_virtue | util_anti | util_neutral_length_matched | util_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| util_alt_contract | 1.000 | 0.611 | 0.770 | -0.096 | 0.072 | -0.026 |
| util_alt_deont | 0.611 | 1.000 | 0.719 | 0.100 | 0.128 | 0.077 |
| util_alt_virtue | 0.770 | 0.719 | 1.000 | -0.117 | 0.119 | 0.020 |
| util_anti | -0.096 | 0.100 | -0.117 | 1.000 | 0.339 | 0.413 |
| util_neutral_length_matched | 0.072 | 0.128 | 0.119 | 0.339 | 1.000 | 0.935 |
| util_neutral_short | -0.026 | 0.077 | 0.020 | 0.413 | 0.935 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.747`
- `neutral_length_matched`: `0.724`
- `anti`: `0.726`

### Site `prompt_end_residual`

#### Layer 0

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| util_alt_contract | nan | nan | nan | nan |
| util_alt_deont | nan | nan | nan | nan |
| util_alt_virtue | nan | nan | nan | nan |
| util_anti | nan | nan | nan | nan |
| util_anti_p_variant | nan | nan | nan | nan |
| util_neutral_length_matched | nan | nan | nan | nan |
| util_neutral_length_matched_p_variant | nan | nan | nan | nan |
| util_neutral_short | nan | nan | nan | nan |
| util_neutral_short_p_variant | nan | nan | nan | nan |

Pole-construction cosine matrix:

| | util_alt_contract | util_alt_deont | util_alt_virtue | util_anti | util_neutral_length_matched | util_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| util_alt_contract | nan | nan | nan | nan | nan | nan |
| util_alt_deont | nan | nan | nan | nan | nan | nan |
| util_alt_virtue | nan | nan | nan | nan | nan | nan |
| util_anti | nan | nan | nan | nan | nan | nan |
| util_neutral_length_matched | nan | nan | nan | nan | nan | nan |
| util_neutral_short | nan | nan | nan | nan | nan | nan |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `nan`
- `neutral_length_matched`: `nan`
- `anti`: `nan`

#### Layer 4

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| util_alt_contract | 0.548 | 0.443 | 0.533 | 0.105 |
| util_alt_deont | 0.661 | 0.440 | 0.612 | 0.221 |
| util_alt_virtue | 0.830 | 0.617 | 0.707 | 0.213 |
| util_anti | 0.871 | 0.671 | 0.752 | 0.200 |
| util_anti_p_variant | 0.916 | 0.766 | 0.831 | 0.150 |
| util_neutral_length_matched | 0.907 | 0.703 | 0.814 | 0.203 |
| util_neutral_length_matched_p_variant | 0.925 | 0.762 | 0.845 | 0.163 |
| util_neutral_short | 0.932 | 0.798 | 0.849 | 0.134 |
| util_neutral_short_p_variant | 0.953 | 0.856 | 0.899 | 0.097 |

Pole-construction cosine matrix:

| | util_alt_contract | util_alt_deont | util_alt_virtue | util_anti | util_neutral_length_matched | util_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| util_alt_contract | 1.000 | 0.296 | 0.441 | -0.029 | 0.105 | -0.023 |
| util_alt_deont | 0.296 | 1.000 | 0.686 | 0.094 | 0.346 | -0.018 |
| util_alt_virtue | 0.441 | 0.686 | 1.000 | -0.093 | 0.236 | -0.175 |
| util_anti | -0.029 | 0.094 | -0.093 | 1.000 | 0.643 | 0.809 |
| util_neutral_length_matched | 0.105 | 0.346 | 0.236 | 0.643 | 1.000 | 0.568 |
| util_neutral_short | -0.023 | -0.018 | -0.175 | 0.809 | 0.568 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.956`
- `neutral_length_matched`: `0.898`
- `anti`: `0.890`

#### Layer 16

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| util_alt_contract | 0.678 | 0.625 | 0.693 | 0.053 |
| util_alt_deont | 0.678 | 0.605 | 0.712 | 0.072 |
| util_alt_virtue | 0.704 | 0.623 | 0.737 | 0.081 |
| util_anti | 0.766 | 0.678 | 0.777 | 0.088 |
| util_anti_p_variant | 0.787 | 0.696 | 0.797 | 0.091 |
| util_neutral_length_matched | 0.831 | 0.767 | 0.841 | 0.064 |
| util_neutral_length_matched_p_variant | 0.848 | 0.792 | 0.855 | 0.055 |
| util_neutral_short | 0.828 | 0.743 | 0.834 | 0.084 |
| util_neutral_short_p_variant | 0.855 | 0.780 | 0.857 | 0.075 |

Pole-construction cosine matrix:

| | util_alt_contract | util_alt_deont | util_alt_virtue | util_anti | util_neutral_length_matched | util_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| util_alt_contract | 1.000 | 0.502 | 0.624 | 0.248 | 0.233 | 0.287 |
| util_alt_deont | 0.502 | 1.000 | 0.714 | 0.327 | 0.411 | 0.480 |
| util_alt_virtue | 0.624 | 0.714 | 1.000 | 0.244 | 0.292 | 0.362 |
| util_anti | 0.248 | 0.327 | 0.244 | 1.000 | 0.439 | 0.537 |
| util_neutral_length_matched | 0.233 | 0.411 | 0.292 | 0.439 | 1.000 | 0.814 |
| util_neutral_short | 0.287 | 0.480 | 0.362 | 0.537 | 0.814 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.886`
- `neutral_length_matched`: `0.883`
- `anti`: `0.817`

#### Layer 24

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| util_alt_contract | 0.836 | 0.671 | 0.769 | 0.164 |
| util_alt_deont | 0.818 | 0.673 | 0.738 | 0.145 |
| util_alt_virtue | 0.849 | 0.671 | 0.759 | 0.178 |
| util_anti | 0.881 | 0.715 | 0.816 | 0.166 |
| util_anti_p_variant | 0.897 | 0.765 | 0.844 | 0.132 |
| util_neutral_length_matched | 0.921 | 0.780 | 0.866 | 0.141 |
| util_neutral_length_matched_p_variant | 0.945 | 0.846 | 0.905 | 0.099 |
| util_neutral_short | 0.918 | 0.792 | 0.857 | 0.126 |
| util_neutral_short_p_variant | 0.945 | 0.860 | 0.903 | 0.085 |

Pole-construction cosine matrix:

| | util_alt_contract | util_alt_deont | util_alt_virtue | util_anti | util_neutral_length_matched | util_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| util_alt_contract | 1.000 | 0.476 | 0.549 | 0.184 | 0.137 | 0.202 |
| util_alt_deont | 0.476 | 1.000 | 0.678 | 0.354 | 0.234 | 0.322 |
| util_alt_virtue | 0.549 | 0.678 | 1.000 | 0.263 | 0.278 | 0.283 |
| util_anti | 0.184 | 0.354 | 0.263 | 1.000 | 0.392 | 0.476 |
| util_neutral_length_matched | 0.137 | 0.234 | 0.278 | 0.392 | 1.000 | 0.842 |
| util_neutral_short | 0.202 | 0.322 | 0.283 | 0.476 | 0.842 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.875`
- `neutral_length_matched`: `0.871`
- `anti`: `0.732`

#### Layer 32

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| util_alt_contract | 0.838 | 0.720 | 0.840 | 0.118 |
| util_alt_deont | 0.835 | 0.753 | 0.867 | 0.082 |
| util_alt_virtue | 0.833 | 0.736 | 0.832 | 0.097 |
| util_anti | 0.921 | 0.853 | 0.919 | 0.068 |
| util_anti_p_variant | 0.917 | 0.869 | 0.915 | 0.048 |
| util_neutral_length_matched | 0.932 | 0.873 | 0.930 | 0.059 |
| util_neutral_length_matched_p_variant | 0.939 | 0.888 | 0.944 | 0.051 |
| util_neutral_short | 0.938 | 0.866 | 0.930 | 0.072 |
| util_neutral_short_p_variant | 0.946 | 0.890 | 0.940 | 0.056 |

Pole-construction cosine matrix:

| | util_alt_contract | util_alt_deont | util_alt_virtue | util_anti | util_neutral_length_matched | util_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| util_alt_contract | 1.000 | 0.498 | 0.528 | 0.186 | 0.101 | 0.129 |
| util_alt_deont | 0.498 | 1.000 | 0.688 | 0.423 | 0.231 | 0.344 |
| util_alt_virtue | 0.528 | 0.688 | 1.000 | 0.338 | 0.276 | 0.288 |
| util_anti | 0.186 | 0.423 | 0.338 | 1.000 | 0.561 | 0.616 |
| util_neutral_length_matched | 0.101 | 0.231 | 0.276 | 0.561 | 1.000 | 0.785 |
| util_neutral_short | 0.129 | 0.344 | 0.288 | 0.616 | 0.785 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.903`
- `neutral_length_matched`: `0.905`
- `anti`: `0.868`

#### Layer 40

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| util_alt_contract | 0.852 | 0.712 | 0.853 | 0.139 |
| util_alt_deont | 0.880 | 0.746 | 0.843 | 0.133 |
| util_alt_virtue | 0.886 | 0.764 | 0.845 | 0.122 |
| util_anti | 0.953 | 0.851 | 0.927 | 0.103 |
| util_anti_p_variant | 0.939 | 0.828 | 0.918 | 0.111 |
| util_neutral_length_matched | 0.960 | 0.894 | 0.948 | 0.066 |
| util_neutral_length_matched_p_variant | 0.950 | 0.891 | 0.945 | 0.059 |
| util_neutral_short | 0.971 | 0.903 | 0.947 | 0.068 |
| util_neutral_short_p_variant | 0.965 | 0.899 | 0.936 | 0.066 |

Pole-construction cosine matrix:

| | util_alt_contract | util_alt_deont | util_alt_virtue | util_anti | util_neutral_length_matched | util_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| util_alt_contract | 1.000 | 0.446 | 0.613 | 0.062 | 0.006 | -0.067 |
| util_alt_deont | 0.446 | 1.000 | 0.720 | 0.231 | 0.171 | 0.140 |
| util_alt_virtue | 0.613 | 0.720 | 1.000 | 0.169 | 0.174 | 0.087 |
| util_anti | 0.062 | 0.231 | 0.169 | 1.000 | 0.677 | 0.743 |
| util_neutral_length_matched | 0.006 | 0.171 | 0.174 | 0.677 | 1.000 | 0.793 |
| util_neutral_short | -0.067 | 0.140 | 0.087 | 0.743 | 0.793 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.926`
- `neutral_length_matched`: `0.915`
- `anti`: `0.875`

## Per-theory directions: virtue_ethics

### Site `generated_sequence_residual`

#### Layer 0

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| virtue_alt_contract | -0.465 | 0.295 | 0.355 | -0.760 |
| virtue_alt_deont | -0.357 | 0.305 | 0.421 | -0.662 |
| virtue_alt_util | -0.209 | 0.365 | 0.449 | -0.573 |
| virtue_anti | -0.022 | 0.304 | 0.579 | -0.327 |
| virtue_anti_p_variant | -0.328 | 0.277 | 0.431 | -0.605 |
| virtue_neutral_length_matched | 0.463 | 0.430 | 0.669 | 0.032 |
| virtue_neutral_length_matched_p_variant | 0.296 | 0.454 | 0.617 | -0.158 |
| virtue_neutral_short | 0.454 | 0.421 | 0.699 | 0.032 |
| virtue_neutral_short_p_variant | 0.291 | 0.414 | 0.618 | -0.123 |

Pole-construction cosine matrix:

| | virtue_alt_contract | virtue_alt_deont | virtue_alt_util | virtue_anti | virtue_neutral_length_matched | virtue_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| virtue_alt_contract | 1.000 | 0.611 | 0.535 | 0.358 | 0.233 | 0.231 |
| virtue_alt_deont | 0.611 | 1.000 | 0.585 | 0.504 | 0.316 | 0.326 |
| virtue_alt_util | 0.535 | 0.585 | 1.000 | 0.629 | 0.606 | 0.622 |
| virtue_anti | 0.358 | 0.504 | 0.629 | 1.000 | 0.741 | 0.763 |
| virtue_neutral_length_matched | 0.233 | 0.316 | 0.606 | 0.741 | 1.000 | 0.964 |
| virtue_neutral_short | 0.231 | 0.326 | 0.622 | 0.763 | 0.964 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.768`
- `neutral_length_matched`: `0.754`
- `anti`: `0.367`

#### Layer 4

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| virtue_alt_contract | -0.173 | 0.328 | 0.585 | -0.500 |
| virtue_alt_deont | -0.113 | 0.422 | 0.483 | -0.535 |
| virtue_alt_util | 0.312 | 0.526 | 0.637 | -0.214 |
| virtue_anti | 0.450 | 0.578 | 0.665 | -0.127 |
| virtue_anti_p_variant | 0.078 | 0.397 | 0.491 | -0.318 |
| virtue_neutral_length_matched | 0.735 | 0.536 | 0.649 | 0.199 |
| virtue_neutral_length_matched_p_variant | 0.598 | 0.538 | 0.594 | 0.060 |
| virtue_neutral_short | 0.784 | 0.596 | 0.668 | 0.188 |
| virtue_neutral_short_p_variant | 0.662 | 0.529 | 0.599 | 0.133 |

Pole-construction cosine matrix:

| | virtue_alt_contract | virtue_alt_deont | virtue_alt_util | virtue_anti | virtue_neutral_length_matched | virtue_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| virtue_alt_contract | 1.000 | 0.619 | 0.627 | 0.306 | 0.224 | 0.243 |
| virtue_alt_deont | 0.619 | 1.000 | 0.639 | 0.535 | 0.361 | 0.396 |
| virtue_alt_util | 0.627 | 0.639 | 1.000 | 0.702 | 0.637 | 0.676 |
| virtue_anti | 0.306 | 0.535 | 0.702 | 1.000 | 0.794 | 0.830 |
| virtue_neutral_length_matched | 0.224 | 0.361 | 0.637 | 0.794 | 1.000 | 0.970 |
| virtue_neutral_short | 0.243 | 0.396 | 0.676 | 0.830 | 0.970 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.736`
- `neutral_length_matched`: `0.697`
- `anti`: `0.292`

#### Layer 16

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| virtue_alt_contract | -0.290 | 0.426 | 0.583 | -0.715 |
| virtue_alt_deont | -0.197 | 0.473 | 0.591 | -0.671 |
| virtue_alt_util | 0.194 | 0.501 | 0.597 | -0.307 |
| virtue_anti | 0.248 | 0.480 | 0.548 | -0.232 |
| virtue_anti_p_variant | -0.188 | 0.363 | 0.528 | -0.551 |
| virtue_neutral_length_matched | 0.493 | 0.560 | 0.636 | -0.067 |
| virtue_neutral_length_matched_p_variant | 0.452 | 0.567 | 0.641 | -0.115 |
| virtue_neutral_short | 0.557 | 0.572 | 0.635 | -0.014 |
| virtue_neutral_short_p_variant | 0.450 | 0.546 | 0.695 | -0.096 |

Pole-construction cosine matrix:

| | virtue_alt_contract | virtue_alt_deont | virtue_alt_util | virtue_anti | virtue_neutral_length_matched | virtue_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| virtue_alt_contract | 1.000 | 0.613 | 0.705 | 0.436 | 0.289 | 0.295 |
| virtue_alt_deont | 0.613 | 1.000 | 0.718 | 0.651 | 0.478 | 0.513 |
| virtue_alt_util | 0.705 | 0.718 | 1.000 | 0.716 | 0.574 | 0.600 |
| virtue_anti | 0.436 | 0.651 | 0.716 | 1.000 | 0.749 | 0.780 |
| virtue_neutral_length_matched | 0.289 | 0.478 | 0.574 | 0.749 | 1.000 | 0.967 |
| virtue_neutral_short | 0.295 | 0.513 | 0.600 | 0.780 | 0.967 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.634`
- `neutral_length_matched`: `0.585`
- `anti`: `0.124`

#### Layer 24

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| virtue_alt_contract | -0.209 | 0.448 | 0.511 | -0.657 |
| virtue_alt_deont | -0.141 | 0.449 | 0.609 | -0.590 |
| virtue_alt_util | 0.270 | 0.536 | 0.605 | -0.266 |
| virtue_anti | 0.325 | 0.444 | 0.608 | -0.119 |
| virtue_anti_p_variant | -0.099 | 0.380 | 0.465 | -0.479 |
| virtue_neutral_length_matched | 0.576 | 0.590 | 0.733 | -0.015 |
| virtue_neutral_length_matched_p_variant | 0.479 | 0.520 | 0.692 | -0.040 |
| virtue_neutral_short | 0.633 | 0.574 | 0.752 | 0.059 |
| virtue_neutral_short_p_variant | 0.480 | 0.577 | 0.728 | -0.097 |

Pole-construction cosine matrix:

| | virtue_alt_contract | virtue_alt_deont | virtue_alt_util | virtue_anti | virtue_neutral_length_matched | virtue_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| virtue_alt_contract | 1.000 | 0.601 | 0.679 | 0.445 | 0.300 | 0.320 |
| virtue_alt_deont | 0.601 | 1.000 | 0.742 | 0.675 | 0.501 | 0.536 |
| virtue_alt_util | 0.679 | 0.742 | 1.000 | 0.748 | 0.620 | 0.655 |
| virtue_anti | 0.445 | 0.675 | 0.748 | 1.000 | 0.751 | 0.775 |
| virtue_neutral_length_matched | 0.300 | 0.501 | 0.620 | 0.751 | 1.000 | 0.963 |
| virtue_neutral_short | 0.320 | 0.536 | 0.655 | 0.775 | 0.963 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.622`
- `neutral_length_matched`: `0.559`
- `anti`: `0.075`

#### Layer 32

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| virtue_alt_contract | 0.111 | 0.365 | 0.522 | -0.254 |
| virtue_alt_deont | 0.172 | 0.442 | 0.624 | -0.270 |
| virtue_alt_util | 0.325 | 0.503 | 0.640 | -0.178 |
| virtue_anti | 0.655 | 0.560 | 0.644 | 0.095 |
| virtue_anti_p_variant | 0.418 | 0.393 | 0.599 | 0.025 |
| virtue_neutral_length_matched | 0.700 | 0.637 | 0.662 | 0.064 |
| virtue_neutral_length_matched_p_variant | 0.646 | 0.549 | 0.596 | 0.097 |
| virtue_neutral_short | 0.720 | 0.681 | 0.752 | 0.039 |
| virtue_neutral_short_p_variant | 0.633 | 0.562 | 0.684 | 0.071 |

Pole-construction cosine matrix:

| | virtue_alt_contract | virtue_alt_deont | virtue_alt_util | virtue_anti | virtue_neutral_length_matched | virtue_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| virtue_alt_contract | 1.000 | 0.498 | 0.710 | 0.529 | 0.417 | 0.419 |
| virtue_alt_deont | 0.498 | 1.000 | 0.591 | 0.579 | 0.481 | 0.499 |
| virtue_alt_util | 0.710 | 0.591 | 1.000 | 0.744 | 0.670 | 0.691 |
| virtue_anti | 0.529 | 0.579 | 0.744 | 1.000 | 0.768 | 0.783 |
| virtue_neutral_length_matched | 0.417 | 0.481 | 0.670 | 0.768 | 1.000 | 0.960 |
| virtue_neutral_short | 0.419 | 0.499 | 0.691 | 0.783 | 0.960 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.613`
- `neutral_length_matched`: `0.555`
- `anti`: `0.368`

#### Layer 40

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| virtue_alt_contract | -0.113 | 0.393 | 0.501 | -0.506 |
| virtue_alt_deont | -0.058 | 0.511 | 0.660 | -0.568 |
| virtue_alt_util | 0.355 | 0.534 | 0.705 | -0.178 |
| virtue_anti | 0.445 | 0.560 | 0.718 | -0.115 |
| virtue_anti_p_variant | 0.204 | 0.407 | 0.513 | -0.204 |
| virtue_neutral_length_matched | 0.567 | 0.569 | 0.761 | -0.002 |
| virtue_neutral_length_matched_p_variant | 0.575 | 0.474 | 0.563 | 0.101 |
| virtue_neutral_short | 0.598 | 0.593 | 0.726 | 0.005 |
| virtue_neutral_short_p_variant | 0.578 | 0.496 | 0.620 | 0.082 |

Pole-construction cosine matrix:

| | virtue_alt_contract | virtue_alt_deont | virtue_alt_util | virtue_anti | virtue_neutral_length_matched | virtue_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| virtue_alt_contract | 1.000 | 0.523 | 0.726 | 0.599 | 0.487 | 0.483 |
| virtue_alt_deont | 0.523 | 1.000 | 0.626 | 0.584 | 0.468 | 0.492 |
| virtue_alt_util | 0.726 | 0.626 | 1.000 | 0.786 | 0.692 | 0.708 |
| virtue_anti | 0.599 | 0.584 | 0.786 | 1.000 | 0.799 | 0.822 |
| virtue_neutral_length_matched | 0.487 | 0.468 | 0.692 | 0.799 | 1.000 | 0.969 |
| virtue_neutral_short | 0.483 | 0.492 | 0.708 | 0.822 | 0.969 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.535`
- `neutral_length_matched`: `0.488`
- `anti`: `0.254`

### Site `prompt_end_residual`

#### Layer 0

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| virtue_alt_contract | nan | nan | nan | nan |
| virtue_alt_deont | nan | nan | nan | nan |
| virtue_alt_util | nan | nan | nan | nan |
| virtue_anti | nan | nan | nan | nan |
| virtue_anti_p_variant | nan | nan | nan | nan |
| virtue_neutral_length_matched | nan | nan | nan | nan |
| virtue_neutral_length_matched_p_variant | nan | nan | nan | nan |
| virtue_neutral_short | nan | nan | nan | nan |
| virtue_neutral_short_p_variant | nan | nan | nan | nan |

Pole-construction cosine matrix:

| | virtue_alt_contract | virtue_alt_deont | virtue_alt_util | virtue_anti | virtue_neutral_length_matched | virtue_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| virtue_alt_contract | nan | nan | nan | nan | nan | nan |
| virtue_alt_deont | nan | nan | nan | nan | nan | nan |
| virtue_alt_util | nan | nan | nan | nan | nan | nan |
| virtue_anti | nan | nan | nan | nan | nan | nan |
| virtue_neutral_length_matched | nan | nan | nan | nan | nan | nan |
| virtue_neutral_short | nan | nan | nan | nan | nan | nan |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `nan`
- `neutral_length_matched`: `nan`
- `anti`: `nan`

#### Layer 4

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| virtue_alt_contract | 0.784 | 0.602 | 0.660 | 0.183 |
| virtue_alt_deont | 0.713 | 0.471 | 0.615 | 0.241 |
| virtue_alt_util | 0.818 | 0.599 | 0.687 | 0.220 |
| virtue_anti | 0.934 | 0.788 | 0.841 | 0.147 |
| virtue_anti_p_variant | 0.908 | 0.739 | 0.805 | 0.169 |
| virtue_neutral_length_matched | 0.925 | 0.741 | 0.840 | 0.185 |
| virtue_neutral_length_matched_p_variant | 0.909 | 0.713 | 0.821 | 0.197 |
| virtue_neutral_short | 0.958 | 0.853 | 0.896 | 0.105 |
| virtue_neutral_short_p_variant | 0.947 | 0.830 | 0.876 | 0.117 |

Pole-construction cosine matrix:

| | virtue_alt_contract | virtue_alt_deont | virtue_alt_util | virtue_anti | virtue_neutral_length_matched | virtue_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| virtue_alt_contract | 1.000 | 0.585 | 0.803 | 0.498 | 0.343 | 0.507 |
| virtue_alt_deont | 0.585 | 1.000 | 0.733 | 0.563 | 0.472 | 0.515 |
| virtue_alt_util | 0.803 | 0.733 | 1.000 | 0.580 | 0.426 | 0.595 |
| virtue_anti | 0.498 | 0.563 | 0.580 | 1.000 | 0.748 | 0.906 |
| virtue_neutral_length_matched | 0.343 | 0.472 | 0.426 | 0.748 | 1.000 | 0.717 |
| virtue_neutral_short | 0.507 | 0.515 | 0.595 | 0.906 | 0.717 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.916`
- `neutral_length_matched`: `0.826`
- `anti`: `0.857`

#### Layer 16

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| virtue_alt_contract | 0.614 | 0.601 | 0.667 | 0.013 |
| virtue_alt_deont | 0.518 | 0.543 | 0.616 | -0.026 |
| virtue_alt_util | 0.688 | 0.621 | 0.711 | 0.067 |
| virtue_anti | 0.779 | 0.695 | 0.794 | 0.084 |
| virtue_anti_p_variant | 0.727 | 0.660 | 0.758 | 0.067 |
| virtue_neutral_length_matched | 0.840 | 0.771 | 0.848 | 0.069 |
| virtue_neutral_length_matched_p_variant | 0.819 | 0.746 | 0.831 | 0.073 |
| virtue_neutral_short | 0.824 | 0.734 | 0.834 | 0.090 |
| virtue_neutral_short_p_variant | 0.790 | 0.720 | 0.809 | 0.070 |

Pole-construction cosine matrix:

| | virtue_alt_contract | virtue_alt_deont | virtue_alt_util | virtue_anti | virtue_neutral_length_matched | virtue_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| virtue_alt_contract | 1.000 | 0.301 | 0.499 | 0.261 | 0.245 | 0.237 |
| virtue_alt_deont | 0.301 | 1.000 | 0.441 | 0.391 | 0.419 | 0.436 |
| virtue_alt_util | 0.499 | 0.441 | 1.000 | 0.307 | 0.381 | 0.339 |
| virtue_anti | 0.261 | 0.391 | 0.307 | 1.000 | 0.533 | 0.652 |
| virtue_neutral_length_matched | 0.245 | 0.419 | 0.381 | 0.533 | 1.000 | 0.820 |
| virtue_neutral_short | 0.237 | 0.436 | 0.339 | 0.652 | 0.820 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.822`
- `neutral_length_matched`: `0.845`
- `anti`: `0.748`

#### Layer 24

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| virtue_alt_contract | 0.845 | 0.664 | 0.749 | 0.180 |
| virtue_alt_deont | 0.773 | 0.598 | 0.662 | 0.175 |
| virtue_alt_util | 0.856 | 0.681 | 0.764 | 0.174 |
| virtue_anti | 0.920 | 0.774 | 0.844 | 0.146 |
| virtue_anti_p_variant | 0.922 | 0.776 | 0.847 | 0.146 |
| virtue_neutral_length_matched | 0.929 | 0.787 | 0.871 | 0.141 |
| virtue_neutral_length_matched_p_variant | 0.951 | 0.850 | 0.904 | 0.101 |
| virtue_neutral_short | 0.930 | 0.798 | 0.864 | 0.132 |
| virtue_neutral_short_p_variant | 0.946 | 0.846 | 0.892 | 0.101 |

Pole-construction cosine matrix:

| | virtue_alt_contract | virtue_alt_deont | virtue_alt_util | virtue_anti | virtue_neutral_length_matched | virtue_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| virtue_alt_contract | 1.000 | 0.385 | 0.535 | 0.235 | 0.186 | 0.247 |
| virtue_alt_deont | 0.385 | 1.000 | 0.487 | 0.348 | 0.236 | 0.332 |
| virtue_alt_util | 0.535 | 0.487 | 1.000 | 0.285 | 0.376 | 0.377 |
| virtue_anti | 0.235 | 0.348 | 0.285 | 1.000 | 0.443 | 0.528 |
| virtue_neutral_length_matched | 0.186 | 0.236 | 0.376 | 0.443 | 1.000 | 0.852 |
| virtue_neutral_short | 0.247 | 0.332 | 0.377 | 0.528 | 0.852 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.846`
- `neutral_length_matched`: `0.862`
- `anti`: `0.758`

#### Layer 32

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| virtue_alt_contract | 0.884 | 0.722 | 0.825 | 0.162 |
| virtue_alt_deont | 0.789 | 0.669 | 0.771 | 0.119 |
| virtue_alt_util | 0.891 | 0.764 | 0.845 | 0.127 |
| virtue_anti | 0.942 | 0.878 | 0.933 | 0.064 |
| virtue_anti_p_variant | 0.943 | 0.885 | 0.937 | 0.057 |
| virtue_neutral_length_matched | 0.944 | 0.868 | 0.925 | 0.076 |
| virtue_neutral_length_matched_p_variant | 0.955 | 0.892 | 0.943 | 0.064 |
| virtue_neutral_short | 0.947 | 0.856 | 0.919 | 0.091 |
| virtue_neutral_short_p_variant | 0.956 | 0.878 | 0.937 | 0.077 |

Pole-construction cosine matrix:

| | virtue_alt_contract | virtue_alt_deont | virtue_alt_util | virtue_anti | virtue_neutral_length_matched | virtue_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| virtue_alt_contract | 1.000 | 0.348 | 0.541 | 0.163 | 0.115 | 0.134 |
| virtue_alt_deont | 0.348 | 1.000 | 0.320 | 0.264 | 0.148 | 0.283 |
| virtue_alt_util | 0.541 | 0.320 | 1.000 | 0.253 | 0.295 | 0.290 |
| virtue_anti | 0.163 | 0.264 | 0.253 | 1.000 | 0.558 | 0.644 |
| virtue_neutral_length_matched | 0.115 | 0.148 | 0.295 | 0.558 | 1.000 | 0.786 |
| virtue_neutral_short | 0.134 | 0.283 | 0.290 | 0.644 | 0.786 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.860`
- `neutral_length_matched`: `0.870`
- `anti`: `0.857`

#### Layer 40

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| virtue_alt_contract | 0.871 | 0.678 | 0.833 | 0.193 |
| virtue_alt_deont | 0.839 | 0.641 | 0.797 | 0.198 |
| virtue_alt_util | 0.899 | 0.757 | 0.875 | 0.143 |
| virtue_anti | 0.965 | 0.877 | 0.929 | 0.088 |
| virtue_anti_p_variant | 0.969 | 0.888 | 0.937 | 0.081 |
| virtue_neutral_length_matched | 0.971 | 0.901 | 0.956 | 0.071 |
| virtue_neutral_length_matched_p_variant | 0.979 | 0.914 | 0.962 | 0.065 |
| virtue_neutral_short | 0.980 | 0.917 | 0.958 | 0.063 |
| virtue_neutral_short_p_variant | 0.984 | 0.927 | 0.964 | 0.058 |

Pole-construction cosine matrix:

| | virtue_alt_contract | virtue_alt_deont | virtue_alt_util | virtue_anti | virtue_neutral_length_matched | virtue_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| virtue_alt_contract | 1.000 | 0.219 | 0.541 | 0.138 | 0.088 | 0.088 |
| virtue_alt_deont | 0.219 | 1.000 | 0.394 | 0.238 | 0.196 | 0.246 |
| virtue_alt_util | 0.541 | 0.394 | 1.000 | 0.308 | 0.351 | 0.383 |
| virtue_anti | 0.138 | 0.238 | 0.308 | 1.000 | 0.697 | 0.785 |
| virtue_neutral_length_matched | 0.088 | 0.196 | 0.351 | 0.697 | 1.000 | 0.820 |
| virtue_neutral_short | 0.088 | 0.246 | 0.383 | 0.785 | 0.820 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.916`
- `neutral_length_matched`: `0.901`
- `anti`: `0.852`

## Per-theory directions: contractualism

### Site `generated_sequence_residual`

#### Layer 0

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| contract_alt_deont | -0.505 | 0.258 | 0.383 | -0.763 |
| contract_alt_util | -0.366 | 0.310 | 0.384 | -0.676 |
| contract_alt_virtue | -0.473 | 0.247 | 0.510 | -0.720 |
| contract_anti | 0.143 | 0.317 | 0.597 | -0.173 |
| contract_anti_p_variant | -0.132 | 0.261 | 0.532 | -0.392 |
| contract_neutral_length_matched | 0.449 | 0.462 | 0.622 | -0.013 |
| contract_neutral_length_matched_p_variant | 0.057 | 0.484 | 0.526 | -0.428 |
| contract_neutral_short | 0.440 | 0.390 | 0.650 | 0.050 |
| contract_neutral_short_p_variant | 0.032 | 0.423 | 0.502 | -0.390 |

Pole-construction cosine matrix:

| | contract_alt_deont | contract_alt_util | contract_alt_virtue | contract_anti | contract_neutral_length_matched | contract_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| contract_alt_deont | 1.000 | 0.408 | 0.300 | 0.339 | 0.259 | 0.268 |
| contract_alt_util | 0.408 | 1.000 | 0.150 | 0.549 | 0.596 | 0.615 |
| contract_alt_virtue | 0.300 | 0.150 | 1.000 | 0.212 | 0.154 | 0.139 |
| contract_anti | 0.339 | 0.549 | 0.212 | 1.000 | 0.746 | 0.749 |
| contract_neutral_length_matched | 0.259 | 0.596 | 0.154 | 0.746 | 1.000 | 0.963 |
| contract_neutral_short | 0.268 | 0.615 | 0.139 | 0.749 | 0.963 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.894`
- `neutral_length_matched`: `0.882`
- `anti`: `0.768`

#### Layer 4

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| contract_alt_deont | -0.340 | 0.298 | 0.351 | -0.638 |
| contract_alt_util | 0.026 | 0.430 | 0.666 | -0.404 |
| contract_alt_virtue | -0.138 | 0.315 | 0.398 | -0.453 |
| contract_anti | 0.540 | 0.498 | 0.545 | 0.043 |
| contract_anti_p_variant | 0.220 | 0.385 | 0.456 | -0.165 |
| contract_neutral_length_matched | 0.725 | 0.555 | 0.633 | 0.170 |
| contract_neutral_length_matched_p_variant | 0.386 | 0.538 | 0.737 | -0.153 |
| contract_neutral_short | 0.755 | 0.589 | 0.641 | 0.167 |
| contract_neutral_short_p_variant | 0.519 | 0.567 | 0.750 | -0.049 |

Pole-construction cosine matrix:

| | contract_alt_deont | contract_alt_util | contract_alt_virtue | contract_anti | contract_neutral_length_matched | contract_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| contract_alt_deont | 1.000 | 0.381 | 0.337 | 0.387 | 0.315 | 0.330 |
| contract_alt_util | 0.381 | 1.000 | -0.014 | 0.660 | 0.647 | 0.690 |
| contract_alt_virtue | 0.337 | -0.014 | 1.000 | 0.131 | 0.131 | 0.082 |
| contract_anti | 0.387 | 0.660 | 0.131 | 1.000 | 0.783 | 0.804 |
| contract_neutral_length_matched | 0.315 | 0.647 | 0.131 | 0.783 | 1.000 | 0.968 |
| contract_neutral_short | 0.330 | 0.690 | 0.082 | 0.804 | 0.968 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.927`
- `neutral_length_matched`: `0.908`
- `anti`: `0.828`

#### Layer 16

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| contract_alt_deont | -0.361 | 0.291 | 0.402 | -0.651 |
| contract_alt_util | -0.178 | 0.368 | 0.508 | -0.546 |
| contract_alt_virtue | -0.294 | 0.441 | 0.527 | -0.735 |
| contract_anti | 0.237 | 0.449 | 0.613 | -0.213 |
| contract_anti_p_variant | 0.006 | 0.396 | 0.421 | -0.390 |
| contract_neutral_length_matched | 0.441 | 0.577 | 0.633 | -0.136 |
| contract_neutral_length_matched_p_variant | 0.173 | 0.494 | 0.542 | -0.322 |
| contract_neutral_short | 0.499 | 0.572 | 0.676 | -0.073 |
| contract_neutral_short_p_variant | 0.242 | 0.496 | 0.580 | -0.254 |

Pole-construction cosine matrix:

| | contract_alt_deont | contract_alt_util | contract_alt_virtue | contract_anti | contract_neutral_length_matched | contract_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| contract_alt_deont | 1.000 | 0.438 | 0.392 | 0.492 | 0.428 | 0.450 |
| contract_alt_util | 0.438 | 1.000 | -0.078 | 0.630 | 0.523 | 0.562 |
| contract_alt_virtue | 0.392 | -0.078 | 1.000 | 0.115 | 0.170 | 0.125 |
| contract_anti | 0.492 | 0.630 | 0.115 | 1.000 | 0.747 | 0.771 |
| contract_neutral_length_matched | 0.428 | 0.523 | 0.170 | 0.747 | 1.000 | 0.964 |
| contract_neutral_short | 0.450 | 0.562 | 0.125 | 0.771 | 0.964 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.861`
- `neutral_length_matched`: `0.828`
- `anti`: `0.761`

#### Layer 24

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| contract_alt_deont | -0.351 | 0.258 | 0.425 | -0.609 |
| contract_alt_util | -0.072 | 0.426 | 0.491 | -0.498 |
| contract_alt_virtue | -0.242 | 0.355 | 0.446 | -0.597 |
| contract_anti | 0.364 | 0.523 | 0.556 | -0.159 |
| contract_anti_p_variant | 0.101 | 0.340 | 0.461 | -0.239 |
| contract_neutral_length_matched | 0.515 | 0.552 | 0.689 | -0.037 |
| contract_neutral_length_matched_p_variant | 0.339 | 0.465 | 0.593 | -0.126 |
| contract_neutral_short | 0.563 | 0.562 | 0.697 | 0.001 |
| contract_neutral_short_p_variant | 0.329 | 0.515 | 0.595 | -0.186 |

Pole-construction cosine matrix:

| | contract_alt_deont | contract_alt_util | contract_alt_virtue | contract_anti | contract_neutral_length_matched | contract_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| contract_alt_deont | 1.000 | 0.501 | 0.349 | 0.519 | 0.439 | 0.449 |
| contract_alt_util | 0.501 | 1.000 | -0.088 | 0.692 | 0.575 | 0.619 |
| contract_alt_virtue | 0.349 | -0.088 | 1.000 | 0.057 | 0.140 | 0.073 |
| contract_anti | 0.519 | 0.692 | 0.057 | 1.000 | 0.750 | 0.773 |
| contract_neutral_length_matched | 0.439 | 0.575 | 0.140 | 0.750 | 1.000 | 0.958 |
| contract_neutral_short | 0.449 | 0.619 | 0.073 | 0.773 | 0.958 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.879`
- `neutral_length_matched`: `0.842`
- `anti`: `0.788`

#### Layer 32

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| contract_alt_deont | 0.251 | 0.322 | 0.431 | -0.071 |
| contract_alt_util | 0.133 | 0.410 | 0.493 | -0.277 |
| contract_alt_virtue | 0.273 | 0.413 | 0.452 | -0.140 |
| contract_anti | 0.645 | 0.482 | 0.603 | 0.163 |
| contract_anti_p_variant | 0.497 | 0.425 | 0.564 | 0.072 |
| contract_neutral_length_matched | 0.675 | 0.566 | 0.617 | 0.108 |
| contract_neutral_length_matched_p_variant | 0.532 | 0.495 | 0.617 | 0.037 |
| contract_neutral_short | 0.733 | 0.599 | 0.676 | 0.134 |
| contract_neutral_short_p_variant | 0.571 | 0.575 | 0.612 | -0.004 |

Pole-construction cosine matrix:

| | contract_alt_deont | contract_alt_util | contract_alt_virtue | contract_anti | contract_neutral_length_matched | contract_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| contract_alt_deont | 1.000 | 0.306 | 0.422 | 0.395 | 0.349 | 0.347 |
| contract_alt_util | 0.306 | 1.000 | -0.104 | 0.631 | 0.571 | 0.608 |
| contract_alt_virtue | 0.422 | -0.104 | 1.000 | 0.003 | 0.085 | 0.031 |
| contract_anti | 0.395 | 0.631 | 0.003 | 1.000 | 0.740 | 0.761 |
| contract_neutral_length_matched | 0.349 | 0.571 | 0.085 | 0.740 | 1.000 | 0.951 |
| contract_neutral_short | 0.347 | 0.608 | 0.031 | 0.761 | 0.951 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.829`
- `neutral_length_matched`: `0.789`
- `anti`: `0.782`

#### Layer 40

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| contract_alt_deont | -0.116 | 0.290 | 0.408 | -0.407 |
| contract_alt_util | 0.028 | 0.373 | 0.543 | -0.345 |
| contract_alt_virtue | -0.085 | 0.400 | 0.461 | -0.485 |
| contract_anti | 0.389 | 0.502 | 0.643 | -0.113 |
| contract_anti_p_variant | 0.314 | 0.434 | 0.561 | -0.120 |
| contract_neutral_length_matched | 0.459 | 0.475 | 0.649 | -0.016 |
| contract_neutral_length_matched_p_variant | 0.353 | 0.454 | 0.497 | -0.101 |
| contract_neutral_short | 0.504 | 0.507 | 0.582 | -0.004 |
| contract_neutral_short_p_variant | 0.389 | 0.425 | 0.548 | -0.035 |

Pole-construction cosine matrix:

| | contract_alt_deont | contract_alt_util | contract_alt_virtue | contract_anti | contract_neutral_length_matched | contract_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| contract_alt_deont | 1.000 | 0.326 | 0.424 | 0.288 | 0.266 | 0.282 |
| contract_alt_util | 0.326 | 1.000 | -0.120 | 0.630 | 0.558 | 0.591 |
| contract_alt_virtue | 0.424 | -0.120 | 1.000 | -0.104 | 0.016 | -0.021 |
| contract_anti | 0.288 | 0.630 | -0.104 | 1.000 | 0.760 | 0.791 |
| contract_neutral_length_matched | 0.266 | 0.558 | 0.016 | 0.760 | 1.000 | 0.959 |
| contract_neutral_short | 0.282 | 0.591 | -0.021 | 0.791 | 0.959 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.845`
- `neutral_length_matched`: `0.813`
- `anti`: `0.800`

### Site `prompt_end_residual`

#### Layer 0

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| contract_alt_deont | nan | nan | nan | nan |
| contract_alt_util | nan | nan | nan | nan |
| contract_alt_virtue | nan | nan | nan | nan |
| contract_anti | nan | nan | nan | nan |
| contract_anti_p_variant | nan | nan | nan | nan |
| contract_neutral_length_matched | nan | nan | nan | nan |
| contract_neutral_length_matched_p_variant | nan | nan | nan | nan |
| contract_neutral_short | nan | nan | nan | nan |
| contract_neutral_short_p_variant | nan | nan | nan | nan |

Pole-construction cosine matrix:

| | contract_alt_deont | contract_alt_util | contract_alt_virtue | contract_anti | contract_neutral_length_matched | contract_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| contract_alt_deont | nan | nan | nan | nan | nan | nan |
| contract_alt_util | nan | nan | nan | nan | nan | nan |
| contract_alt_virtue | nan | nan | nan | nan | nan | nan |
| contract_anti | nan | nan | nan | nan | nan | nan |
| contract_neutral_length_matched | nan | nan | nan | nan | nan | nan |
| contract_neutral_short | nan | nan | nan | nan | nan | nan |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `nan`
- `neutral_length_matched`: `nan`
- `anti`: `nan`

#### Layer 4

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| contract_alt_deont | 0.735 | 0.476 | 0.652 | 0.259 |
| contract_alt_util | 0.592 | 0.424 | 0.502 | 0.168 |
| contract_alt_virtue | 0.799 | 0.558 | 0.659 | 0.241 |
| contract_anti | 0.872 | 0.673 | 0.764 | 0.199 |
| contract_anti_p_variant | 0.902 | 0.729 | 0.813 | 0.173 |
| contract_neutral_length_matched | 0.918 | 0.718 | 0.832 | 0.200 |
| contract_neutral_length_matched_p_variant | 0.916 | 0.733 | 0.827 | 0.184 |
| contract_neutral_short | 0.942 | 0.807 | 0.867 | 0.135 |
| contract_neutral_short_p_variant | 0.954 | 0.848 | 0.901 | 0.107 |

Pole-construction cosine matrix:

| | contract_alt_deont | contract_alt_util | contract_alt_virtue | contract_anti | contract_neutral_length_matched | contract_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| contract_alt_deont | 1.000 | 0.529 | 0.636 | 0.290 | 0.428 | 0.170 |
| contract_alt_util | 0.529 | 1.000 | 0.182 | 0.355 | 0.303 | 0.338 |
| contract_alt_virtue | 0.636 | 0.182 | 1.000 | 0.076 | 0.255 | -0.109 |
| contract_anti | 0.290 | 0.355 | 0.076 | 1.000 | 0.702 | 0.828 |
| contract_neutral_length_matched | 0.428 | 0.303 | 0.255 | 0.702 | 1.000 | 0.617 |
| contract_neutral_short | 0.170 | 0.338 | -0.109 | 0.828 | 0.617 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.949`
- `neutral_length_matched`: `0.893`
- `anti`: `0.875`

#### Layer 16

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| contract_alt_deont | 0.667 | 0.573 | 0.725 | 0.094 |
| contract_alt_util | 0.647 | 0.562 | 0.696 | 0.085 |
| contract_alt_virtue | 0.600 | 0.511 | 0.702 | 0.089 |
| contract_anti | 0.733 | 0.638 | 0.772 | 0.094 |
| contract_anti_p_variant | 0.695 | 0.650 | 0.743 | 0.044 |
| contract_neutral_length_matched | 0.845 | 0.770 | 0.853 | 0.075 |
| contract_neutral_length_matched_p_variant | 0.833 | 0.779 | 0.848 | 0.054 |
| contract_neutral_short | 0.832 | 0.736 | 0.846 | 0.096 |
| contract_neutral_short_p_variant | 0.824 | 0.753 | 0.840 | 0.071 |

Pole-construction cosine matrix:

| | contract_alt_deont | contract_alt_util | contract_alt_virtue | contract_anti | contract_neutral_length_matched | contract_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| contract_alt_deont | 1.000 | 0.488 | 0.660 | 0.454 | 0.469 | 0.504 |
| contract_alt_util | 0.488 | 1.000 | 0.365 | 0.412 | 0.385 | 0.354 |
| contract_alt_virtue | 0.660 | 0.365 | 1.000 | 0.292 | 0.306 | 0.342 |
| contract_anti | 0.454 | 0.412 | 0.292 | 1.000 | 0.655 | 0.701 |
| contract_neutral_length_matched | 0.469 | 0.385 | 0.306 | 0.655 | 1.000 | 0.829 |
| contract_neutral_short | 0.504 | 0.354 | 0.342 | 0.701 | 0.829 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.781`
- `neutral_length_matched`: `0.798`
- `anti`: `0.579`

#### Layer 24

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| contract_alt_deont | 0.837 | 0.678 | 0.749 | 0.159 |
| contract_alt_util | 0.825 | 0.641 | 0.741 | 0.184 |
| contract_alt_virtue | 0.834 | 0.645 | 0.736 | 0.189 |
| contract_anti | 0.882 | 0.708 | 0.808 | 0.174 |
| contract_anti_p_variant | 0.881 | 0.685 | 0.801 | 0.195 |
| contract_neutral_length_matched | 0.936 | 0.813 | 0.886 | 0.124 |
| contract_neutral_length_matched_p_variant | 0.939 | 0.814 | 0.887 | 0.125 |
| contract_neutral_short | 0.932 | 0.805 | 0.873 | 0.126 |
| contract_neutral_short_p_variant | 0.937 | 0.817 | 0.876 | 0.120 |

Pole-construction cosine matrix:

| | contract_alt_deont | contract_alt_util | contract_alt_virtue | contract_anti | contract_neutral_length_matched | contract_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| contract_alt_deont | 1.000 | 0.518 | 0.656 | 0.442 | 0.376 | 0.411 |
| contract_alt_util | 0.518 | 1.000 | 0.412 | 0.366 | 0.438 | 0.396 |
| contract_alt_virtue | 0.656 | 0.412 | 1.000 | 0.291 | 0.381 | 0.339 |
| contract_anti | 0.442 | 0.366 | 0.291 | 1.000 | 0.581 | 0.628 |
| contract_neutral_length_matched | 0.376 | 0.438 | 0.381 | 0.581 | 1.000 | 0.866 |
| contract_neutral_short | 0.411 | 0.396 | 0.339 | 0.628 | 0.866 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.812`
- `neutral_length_matched`: `0.824`
- `anti`: `0.611`

#### Layer 32

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| contract_alt_deont | 0.852 | 0.741 | 0.851 | 0.111 |
| contract_alt_util | 0.872 | 0.740 | 0.828 | 0.131 |
| contract_alt_virtue | 0.856 | 0.705 | 0.820 | 0.152 |
| contract_anti | 0.940 | 0.862 | 0.926 | 0.078 |
| contract_anti_p_variant | 0.926 | 0.874 | 0.924 | 0.052 |
| contract_neutral_length_matched | 0.952 | 0.872 | 0.942 | 0.079 |
| contract_neutral_length_matched_p_variant | 0.940 | 0.891 | 0.941 | 0.049 |
| contract_neutral_short | 0.955 | 0.872 | 0.937 | 0.083 |
| contract_neutral_short_p_variant | 0.943 | 0.884 | 0.937 | 0.058 |

Pole-construction cosine matrix:

| | contract_alt_deont | contract_alt_util | contract_alt_virtue | contract_anti | contract_neutral_length_matched | contract_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| contract_alt_deont | 1.000 | 0.392 | 0.647 | 0.407 | 0.333 | 0.427 |
| contract_alt_util | 0.392 | 1.000 | 0.428 | 0.264 | 0.398 | 0.382 |
| contract_alt_virtue | 0.647 | 0.428 | 1.000 | 0.297 | 0.388 | 0.382 |
| contract_anti | 0.407 | 0.264 | 0.297 | 1.000 | 0.654 | 0.713 |
| contract_neutral_length_matched | 0.333 | 0.398 | 0.388 | 0.654 | 1.000 | 0.815 |
| contract_neutral_short | 0.427 | 0.382 | 0.382 | 0.713 | 0.815 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.826`
- `neutral_length_matched`: `0.834`
- `anti`: `0.782`

#### Layer 40

Direction stability:

| construction | split_half | null_p95 | null_max | gap |
|---|---:|---:|---:|---:|
| contract_alt_deont | 0.909 | 0.756 | 0.841 | 0.152 |
| contract_alt_util | 0.877 | 0.730 | 0.830 | 0.147 |
| contract_alt_virtue | 0.872 | 0.698 | 0.785 | 0.174 |
| contract_anti | 0.964 | 0.865 | 0.919 | 0.099 |
| contract_anti_p_variant | 0.936 | 0.809 | 0.916 | 0.127 |
| contract_neutral_length_matched | 0.975 | 0.914 | 0.958 | 0.061 |
| contract_neutral_length_matched_p_variant | 0.953 | 0.892 | 0.951 | 0.061 |
| contract_neutral_short | 0.983 | 0.925 | 0.958 | 0.058 |
| contract_neutral_short_p_variant | 0.966 | 0.897 | 0.946 | 0.068 |

Pole-construction cosine matrix:

| | contract_alt_deont | contract_alt_util | contract_alt_virtue | contract_anti | contract_neutral_length_matched | contract_neutral_short |
|---|---:|---:|---:|---:|---:|---:|
| contract_alt_deont | 1.000 | 0.458 | 0.682 | 0.369 | 0.349 | 0.360 |
| contract_alt_util | 0.458 | 1.000 | 0.333 | 0.338 | 0.432 | 0.457 |
| contract_alt_virtue | 0.682 | 0.333 | 1.000 | 0.267 | 0.327 | 0.289 |
| contract_anti | 0.369 | 0.338 | 0.267 | 1.000 | 0.751 | 0.770 |
| contract_neutral_length_matched | 0.349 | 0.432 | 0.327 | 0.751 | 1.000 | 0.835 |
| contract_neutral_short | 0.360 | 0.457 | 0.289 | 0.770 | 0.835 | 1.000 |

Cross-positive (P_01 vs P_02) cosines per anchor:
- `neutral_short`: `0.889`
- `neutral_length_matched`: `0.864`
- `anti`: `0.759`

