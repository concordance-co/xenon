# Deontology Persona-Vector Pole Pilot Report

- generation artifact: `generation_run_1_cdd0020853b6`
- capture artifact: `capture_1_88c42755786d`
- captured layers: `[0, 4, 16, 24, 32, 40]`
- primary layer: `32` / primary site: `prompt_end_residual`

## Generation sanity

- rows: `180`
- nonempty: `180` (1.000)
- finish reasons: `{'stop': 180}`

Response length by condition:

| condition | n | mean_chars | median_chars | min | max |
|---|---:|---:|---:|---:|---:|
| N_alt_util_01 | 30 | 47.133 | 37.000 | 14 | 106 |
| N_anti_01 | 30 | 31.767 | 29.000 | 11 | 82 |
| N_neutral_01 | 30 | 50.733 | 49.000 | 15 | 126 |
| N_neutral_02 | 30 | 58.100 | 47.500 | 15 | 193 |
| P_deont_01 | 30 | 67.267 | 47.500 | 14 | 246 |
| P_deont_02 | 30 | 51.800 | 39.000 | 14 | 200 |

## Behavioral divergence (diagnostic only)

- dilemmas evaluated: `30`
- all-converge dilemmas (pairwise jaccard ≥ 0.7): `1`
- all-diverge dilemmas (max pairwise jaccard < 0.4): `1`
- avg conditions per dilemma: `6.000`

Pair-level divergence share (jaccard < 0.6 over condition recommendations):

| pair | diverged_share | avg_jaccard |
|---|---:|---:|
| P_deont_vs_N_neutral_short | 0.833 | 0.369 |
| P_deont_vs_N_neutral_length_matched | 0.867 | 0.338 |
| P_deont_vs_N_anti | 0.700 | 0.370 |
| P_deont_vs_N_alt_util | 0.667 | 0.494 |
| P_deont_01_vs_P_deont_02 | 0.700 | 0.452 |

## Directions @ site `prompt_end_residual`

### Layer 0

Direction stability:

| construction | split_half_cos | null_p95 | null_max |
|---|---:|---:|---:|
| deont_anti | nan | nan | nan |
| deont_neutral_length_matched | nan | nan | nan |
| deont_neutral_length_matched_p_variant | nan | nan | nan |
| deont_neutral_short | nan | nan | nan |
| deont_neutral_short_p_variant | nan | nan | nan |
| deont_util | nan | nan | nan |

Pole-construction cosine matrix:

| | deont_anti | deont_neutral_length_matched | deont_neutral_short | deont_util |
|---|---:|---:|---:|---:|
| deont_anti | nan | nan | nan | nan |
| deont_neutral_length_matched | nan | nan | nan | nan |
| deont_neutral_short | nan | nan | nan | nan |
| deont_util | nan | nan | nan | nan |

Cross-positive (P_deont_01 vs P_deont_02) cosines:
- `neutral_short_p1_vs_p2`: `nan`
- `neutral_length_matched_p1_vs_p2`: `nan`

Cross-neutral cosine (deont_neutral_short vs deont_neutral_length_matched): `nan`

### Layer 4

Direction stability:

| construction | split_half_cos | null_p95 | null_max |
|---|---:|---:|---:|
| deont_anti | 0.861 | 0.653 | 0.738 |
| deont_neutral_length_matched | 0.901 | 0.710 | 0.779 |
| deont_neutral_length_matched_p_variant | 0.922 | 0.732 | 0.847 |
| deont_neutral_short | 0.943 | 0.822 | 0.868 |
| deont_neutral_short_p_variant | 0.962 | 0.863 | 0.909 |
| deont_util | 0.660 | 0.445 | 0.577 |

Pole-construction cosine matrix:

| | deont_anti | deont_neutral_length_matched | deont_neutral_short | deont_util |
|---|---:|---:|---:|---:|
| deont_anti | 1.000 | 0.658 | 0.757 | 0.084 |
| deont_neutral_length_matched | 0.658 | 1.000 | 0.604 | 0.141 |
| deont_neutral_short | 0.757 | 0.604 | 1.000 | 0.378 |
| deont_util | 0.084 | 0.141 | 0.378 | 1.000 |

Cross-positive (P_deont_01 vs P_deont_02) cosines:
- `neutral_short_p1_vs_p2`: `0.955`
- `neutral_length_matched_p1_vs_p2`: `0.882`

Cross-neutral cosine (deont_neutral_short vs deont_neutral_length_matched): `0.604`

### Layer 16

Direction stability:

| construction | split_half_cos | null_p95 | null_max |
|---|---:|---:|---:|
| deont_anti | 0.708 | 0.649 | 0.752 |
| deont_neutral_length_matched | 0.800 | 0.740 | 0.827 |
| deont_neutral_length_matched_p_variant | 0.802 | 0.727 | 0.822 |
| deont_neutral_short | 0.781 | 0.713 | 0.812 |
| deont_neutral_short_p_variant | 0.808 | 0.729 | 0.821 |
| deont_util | 0.638 | 0.574 | 0.702 |

Pole-construction cosine matrix:

| | deont_anti | deont_neutral_length_matched | deont_neutral_short | deont_util |
|---|---:|---:|---:|---:|
| deont_anti | 1.000 | 0.400 | 0.469 | 0.195 |
| deont_neutral_length_matched | 0.400 | 1.000 | 0.764 | 0.277 |
| deont_neutral_short | 0.469 | 0.764 | 1.000 | 0.199 |
| deont_util | 0.195 | 0.277 | 0.199 | 1.000 |

Cross-positive (P_deont_01 vs P_deont_02) cosines:
- `neutral_short_p1_vs_p2`: `0.803`
- `neutral_length_matched_p1_vs_p2`: `0.811`

Cross-neutral cosine (deont_neutral_short vs deont_neutral_length_matched): `0.764`

### Layer 24

Direction stability:

| construction | split_half_cos | null_p95 | null_max |
|---|---:|---:|---:|
| deont_anti | 0.870 | 0.704 | 0.793 |
| deont_neutral_length_matched | 0.929 | 0.783 | 0.867 |
| deont_neutral_length_matched_p_variant | 0.945 | 0.820 | 0.885 |
| deont_neutral_short | 0.924 | 0.781 | 0.849 |
| deont_neutral_short_p_variant | 0.946 | 0.830 | 0.880 |
| deont_util | 0.828 | 0.645 | 0.739 |

Pole-construction cosine matrix:

| | deont_anti | deont_neutral_length_matched | deont_neutral_short | deont_util |
|---|---:|---:|---:|---:|
| deont_anti | 1.000 | 0.350 | 0.424 | 0.173 |
| deont_neutral_length_matched | 0.350 | 1.000 | 0.833 | 0.387 |
| deont_neutral_short | 0.424 | 0.833 | 1.000 | 0.302 |
| deont_util | 0.173 | 0.387 | 0.302 | 1.000 |

Cross-positive (P_deont_01 vs P_deont_02) cosines:
- `neutral_short_p1_vs_p2`: `0.840`
- `neutral_length_matched_p1_vs_p2`: `0.837`

Cross-neutral cosine (deont_neutral_short vs deont_neutral_length_matched): `0.833`

### Layer 32

Direction stability:

| construction | split_half_cos | null_p95 | null_max |
|---|---:|---:|---:|
| deont_anti | 0.951 | 0.888 | 0.935 |
| deont_neutral_length_matched | 0.946 | 0.868 | 0.928 |
| deont_neutral_length_matched_p_variant | 0.948 | 0.880 | 0.936 |
| deont_neutral_short | 0.941 | 0.850 | 0.919 |
| deont_neutral_short_p_variant | 0.946 | 0.871 | 0.929 |
| deont_util | 0.890 | 0.786 | 0.857 |

Pole-construction cosine matrix:

| | deont_anti | deont_neutral_length_matched | deont_neutral_short | deont_util |
|---|---:|---:|---:|---:|
| deont_anti | 1.000 | 0.535 | 0.561 | 0.269 |
| deont_neutral_length_matched | 0.535 | 1.000 | 0.803 | 0.374 |
| deont_neutral_short | 0.561 | 0.803 | 1.000 | 0.291 |
| deont_util | 0.269 | 0.374 | 0.291 | 1.000 |

Cross-positive (P_deont_01 vs P_deont_02) cosines:
- `neutral_short_p1_vs_p2`: `0.863`
- `neutral_length_matched_p1_vs_p2`: `0.872`

Cross-neutral cosine (deont_neutral_short vs deont_neutral_length_matched): `0.803`

### Layer 40

Direction stability:

| construction | split_half_cos | null_p95 | null_max |
|---|---:|---:|---:|
| deont_anti | 0.960 | 0.881 | 0.931 |
| deont_neutral_length_matched | 0.967 | 0.897 | 0.948 |
| deont_neutral_length_matched_p_variant | 0.977 | 0.913 | 0.957 |
| deont_neutral_short | 0.977 | 0.915 | 0.956 |
| deont_neutral_short_p_variant | 0.983 | 0.928 | 0.965 |
| deont_util | 0.884 | 0.766 | 0.858 |

Pole-construction cosine matrix:

| | deont_anti | deont_neutral_length_matched | deont_neutral_short | deont_util |
|---|---:|---:|---:|---:|
| deont_anti | 1.000 | 0.546 | 0.464 | 0.288 |
| deont_neutral_length_matched | 0.546 | 1.000 | 0.787 | 0.331 |
| deont_neutral_short | 0.464 | 0.787 | 1.000 | 0.336 |
| deont_util | 0.288 | 0.331 | 0.336 | 1.000 |

Cross-positive (P_deont_01 vs P_deont_02) cosines:
- `neutral_short_p1_vs_p2`: `0.908`
- `neutral_length_matched_p1_vs_p2`: `0.871`

Cross-neutral cosine (deont_neutral_short vs deont_neutral_length_matched): `0.787`

## Directions @ site `generated_sequence_residual`

### Layer 0

Direction stability:

| construction | split_half_cos | null_p95 | null_max |
|---|---:|---:|---:|
| deont_anti | -0.017 | 0.299 | 0.464 |
| deont_neutral_length_matched | 0.548 | 0.479 | 0.753 |
| deont_neutral_length_matched_p_variant | 0.499 | 0.414 | 0.707 |
| deont_neutral_short | 0.491 | 0.456 | 0.705 |
| deont_neutral_short_p_variant | 0.431 | 0.430 | 0.662 |
| deont_util | -0.304 | 0.302 | 0.507 |

Pole-construction cosine matrix:

| | deont_anti | deont_neutral_length_matched | deont_neutral_short | deont_util |
|---|---:|---:|---:|---:|
| deont_anti | 1.000 | 0.321 | 0.329 | 0.536 |
| deont_neutral_length_matched | 0.321 | 1.000 | 0.976 | 0.414 |
| deont_neutral_short | 0.329 | 0.976 | 1.000 | 0.450 |
| deont_util | 0.536 | 0.414 | 0.450 | 1.000 |

Cross-positive (P_deont_01 vs P_deont_02) cosines:
- `neutral_short_p1_vs_p2`: `0.892`
- `neutral_length_matched_p1_vs_p2`: `0.905`

Cross-neutral cosine (deont_neutral_short vs deont_neutral_length_matched): `0.976`

### Layer 4

Direction stability:

| construction | split_half_cos | null_p95 | null_max |
|---|---:|---:|---:|
| deont_anti | 0.298 | 0.417 | 0.552 |
| deont_neutral_length_matched | 0.780 | 0.550 | 0.628 |
| deont_neutral_length_matched_p_variant | 0.720 | 0.557 | 0.620 |
| deont_neutral_short | 0.756 | 0.523 | 0.602 |
| deont_neutral_short_p_variant | 0.690 | 0.540 | 0.606 |
| deont_util | 0.175 | 0.361 | 0.524 |

Pole-construction cosine matrix:

| | deont_anti | deont_neutral_length_matched | deont_neutral_short | deont_util |
|---|---:|---:|---:|---:|
| deont_anti | 1.000 | 0.455 | 0.492 | 0.615 |
| deont_neutral_length_matched | 0.455 | 1.000 | 0.975 | 0.436 |
| deont_neutral_short | 0.492 | 0.975 | 1.000 | 0.494 |
| deont_util | 0.615 | 0.436 | 0.494 | 1.000 |

Cross-positive (P_deont_01 vs P_deont_02) cosines:
- `neutral_short_p1_vs_p2`: `0.904`
- `neutral_length_matched_p1_vs_p2`: `0.910`

Cross-neutral cosine (deont_neutral_short vs deont_neutral_length_matched): `0.975`

### Layer 16

Direction stability:

| construction | split_half_cos | null_p95 | null_max |
|---|---:|---:|---:|
| deont_anti | 0.221 | 0.364 | 0.529 |
| deont_neutral_length_matched | 0.474 | 0.608 | 0.677 |
| deont_neutral_length_matched_p_variant | 0.495 | 0.607 | 0.687 |
| deont_neutral_short | 0.465 | 0.554 | 0.622 |
| deont_neutral_short_p_variant | 0.510 | 0.584 | 0.684 |
| deont_util | -0.051 | 0.337 | 0.476 |

Pole-construction cosine matrix:

| | deont_anti | deont_neutral_length_matched | deont_neutral_short | deont_util |
|---|---:|---:|---:|---:|
| deont_anti | 1.000 | 0.354 | 0.389 | 0.560 |
| deont_neutral_length_matched | 0.354 | 1.000 | 0.961 | 0.290 |
| deont_neutral_short | 0.389 | 0.961 | 1.000 | 0.332 |
| deont_util | 0.560 | 0.290 | 0.332 | 1.000 |

Cross-positive (P_deont_01 vs P_deont_02) cosines:
- `neutral_short_p1_vs_p2`: `0.876`
- `neutral_length_matched_p1_vs_p2`: `0.883`

Cross-neutral cosine (deont_neutral_short vs deont_neutral_length_matched): `0.961`

### Layer 24

Direction stability:

| construction | split_half_cos | null_p95 | null_max |
|---|---:|---:|---:|
| deont_anti | 0.347 | 0.461 | 0.522 |
| deont_neutral_length_matched | 0.603 | 0.606 | 0.691 |
| deont_neutral_length_matched_p_variant | 0.671 | 0.557 | 0.736 |
| deont_neutral_short | 0.573 | 0.546 | 0.677 |
| deont_neutral_short_p_variant | 0.667 | 0.593 | 0.719 |
| deont_util | 0.017 | 0.344 | 0.488 |

Pole-construction cosine matrix:

| | deont_anti | deont_neutral_length_matched | deont_neutral_short | deont_util |
|---|---:|---:|---:|---:|
| deont_anti | 1.000 | 0.334 | 0.378 | 0.564 |
| deont_neutral_length_matched | 0.334 | 1.000 | 0.952 | 0.357 |
| deont_neutral_short | 0.378 | 0.952 | 1.000 | 0.417 |
| deont_util | 0.564 | 0.357 | 0.417 | 1.000 |

Cross-positive (P_deont_01 vs P_deont_02) cosines:
- `neutral_short_p1_vs_p2`: `0.886`
- `neutral_length_matched_p1_vs_p2`: `0.887`

Cross-neutral cosine (deont_neutral_short vs deont_neutral_length_matched): `0.952`

### Layer 32

Direction stability:

| construction | split_half_cos | null_p95 | null_max |
|---|---:|---:|---:|
| deont_anti | 0.663 | 0.524 | 0.580 |
| deont_neutral_length_matched | 0.707 | 0.583 | 0.651 |
| deont_neutral_length_matched_p_variant | 0.751 | 0.555 | 0.631 |
| deont_neutral_short | 0.695 | 0.579 | 0.614 |
| deont_neutral_short_p_variant | 0.739 | 0.558 | 0.618 |
| deont_util | 0.347 | 0.387 | 0.543 |

Pole-construction cosine matrix:

| | deont_anti | deont_neutral_length_matched | deont_neutral_short | deont_util |
|---|---:|---:|---:|---:|
| deont_anti | 1.000 | 0.488 | 0.514 | 0.655 |
| deont_neutral_length_matched | 0.488 | 1.000 | 0.947 | 0.515 |
| deont_neutral_short | 0.514 | 0.947 | 1.000 | 0.556 |
| deont_util | 0.655 | 0.515 | 0.556 | 1.000 |

Cross-positive (P_deont_01 vs P_deont_02) cosines:
- `neutral_short_p1_vs_p2`: `0.856`
- `neutral_length_matched_p1_vs_p2`: `0.854`

Cross-neutral cosine (deont_neutral_short vs deont_neutral_length_matched): `0.947`

### Layer 40

Direction stability:

| construction | split_half_cos | null_p95 | null_max |
|---|---:|---:|---:|
| deont_anti | 0.576 | 0.470 | 0.575 |
| deont_neutral_length_matched | 0.578 | 0.549 | 0.635 |
| deont_neutral_length_matched_p_variant | 0.627 | 0.513 | 0.621 |
| deont_neutral_short | 0.539 | 0.564 | 0.603 |
| deont_neutral_short_p_variant | 0.604 | 0.491 | 0.599 |
| deont_util | 0.238 | 0.404 | 0.452 |

Pole-construction cosine matrix:

| | deont_anti | deont_neutral_length_matched | deont_neutral_short | deont_util |
|---|---:|---:|---:|---:|
| deont_anti | 1.000 | 0.546 | 0.574 | 0.682 |
| deont_neutral_length_matched | 0.546 | 1.000 | 0.958 | 0.549 |
| deont_neutral_short | 0.574 | 0.958 | 1.000 | 0.567 |
| deont_util | 0.682 | 0.549 | 0.567 | 1.000 |

Cross-positive (P_deont_01 vs P_deont_02) cosines:
- `neutral_short_p1_vs_p2`: `0.837`
- `neutral_length_matched_p1_vs_p2`: `0.838`

Cross-neutral cosine (deont_neutral_short vs deont_neutral_length_matched): `0.958`

