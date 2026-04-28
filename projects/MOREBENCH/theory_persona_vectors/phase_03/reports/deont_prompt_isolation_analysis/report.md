# Deont Prompt Isolation Capture Analysis

- capture artifact: `capture_1_c479d296a725`
- generation rows: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/deont_prompt_isolation_report/report_4b6e5c6c9407_f2313986/results/generate_natural_responses_results.json`
- layers: `0, 4, 16, 24, 32, 40`

## Direct Readouts

| site | layer | task | pairs | pair acc | BA | AUROC | median margin |
|---|---:|---|---:|---:|---:|---:|---:|
| prompt_end | 0 | deont01_vs_generic | 0 | nan | nan | nan | nan |
| prompt_end | 0 | deont02_vs_generic | 0 | nan | nan | nan | nan |
| prompt_end | 0 | deont01_vs_neutral | 0 | nan | nan | nan | nan |
| prompt_end | 0 | deont02_vs_neutral | 0 | nan | nan | nan | nan |
| prompt_end | 0 | deont01_vs_deont02 | 0 | nan | nan | nan | nan |
| prompt_end | 0 | deont01_vs_anti | 0 | nan | nan | nan | nan |
| prompt_end | 0 | deont02_vs_anti | 0 | nan | nan | nan | nan |
| prompt_end | 4 | deont01_vs_generic | 30 | 1.000 | 0.500 | 1.000 | 0.203 |
| prompt_end | 4 | deont02_vs_generic | 30 | 1.000 | 0.500 | 1.000 | 0.164 |
| prompt_end | 4 | deont01_vs_neutral | 30 | 1.000 | 0.500 | 1.000 | 0.258 |
| prompt_end | 4 | deont02_vs_neutral | 30 | 1.000 | 0.500 | 1.000 | 0.236 |
| prompt_end | 4 | deont01_vs_deont02 | 30 | 1.000 | 0.500 | 1.000 | 0.109 |
| prompt_end | 4 | deont01_vs_anti | 30 | 1.000 | 0.500 | 0.999 | 0.102 |
| prompt_end | 4 | deont02_vs_anti | 30 | 1.000 | 0.500 | 1.000 | 0.177 |
| prompt_end | 16 | deont01_vs_generic | 30 | 1.000 | 0.500 | 1.000 | 0.986 |
| prompt_end | 16 | deont02_vs_generic | 30 | 1.000 | 0.767 | 1.000 | 0.828 |
| prompt_end | 16 | deont01_vs_neutral | 30 | 1.000 | 0.500 | 1.000 | 1.071 |
| prompt_end | 16 | deont02_vs_neutral | 30 | 1.000 | 0.500 | 1.000 | 0.932 |
| prompt_end | 16 | deont01_vs_deont02 | 30 | 1.000 | 0.500 | 1.000 | 0.769 |
| prompt_end | 16 | deont01_vs_anti | 30 | 1.000 | 0.500 | 1.000 | 0.973 |
| prompt_end | 16 | deont02_vs_anti | 30 | 1.000 | 0.500 | 1.000 | 1.093 |
| prompt_end | 24 | deont01_vs_generic | 30 | 1.000 | 0.500 | 1.000 | 2.534 |
| prompt_end | 24 | deont02_vs_generic | 30 | 1.000 | 0.500 | 1.000 | 2.067 |
| prompt_end | 24 | deont01_vs_neutral | 30 | 1.000 | 0.500 | 1.000 | 2.765 |
| prompt_end | 24 | deont02_vs_neutral | 30 | 1.000 | 0.500 | 1.000 | 2.292 |
| prompt_end | 24 | deont01_vs_deont02 | 30 | 1.000 | 0.500 | 1.000 | 1.791 |
| prompt_end | 24 | deont01_vs_anti | 30 | 1.000 | 0.500 | 1.000 | 2.119 |
| prompt_end | 24 | deont02_vs_anti | 30 | 1.000 | 0.500 | 1.000 | 2.398 |
| prompt_end | 32 | deont01_vs_generic | 30 | 1.000 | 1.000 | 1.000 | 4.554 |
| prompt_end | 32 | deont02_vs_generic | 30 | 1.000 | 0.500 | 1.000 | 4.202 |
| prompt_end | 32 | deont01_vs_neutral | 30 | 1.000 | 0.983 | 1.000 | 4.736 |
| prompt_end | 32 | deont02_vs_neutral | 30 | 1.000 | 0.500 | 1.000 | 4.482 |
| prompt_end | 32 | deont01_vs_deont02 | 30 | 1.000 | 0.500 | 1.000 | 3.108 |
| prompt_end | 32 | deont01_vs_anti | 30 | 1.000 | 0.500 | 1.000 | 4.208 |
| prompt_end | 32 | deont02_vs_anti | 30 | 1.000 | 0.500 | 1.000 | 4.588 |
| prompt_end | 40 | deont01_vs_generic | 30 | 1.000 | 0.500 | 1.000 | 7.592 |
| prompt_end | 40 | deont02_vs_generic | 30 | 1.000 | 0.500 | 1.000 | 6.908 |
| prompt_end | 40 | deont01_vs_neutral | 30 | 1.000 | 0.500 | 1.000 | 7.879 |
| prompt_end | 40 | deont02_vs_neutral | 30 | 1.000 | 0.500 | 1.000 | 7.178 |
| prompt_end | 40 | deont01_vs_deont02 | 30 | 1.000 | 0.983 | 1.000 | 5.184 |
| prompt_end | 40 | deont01_vs_anti | 30 | 1.000 | 0.983 | 1.000 | 6.497 |
| prompt_end | 40 | deont02_vs_anti | 30 | 1.000 | 1.000 | 1.000 | 6.803 |
| generated_full | 0 | deont01_vs_generic | 30 | 0.967 | 0.600 | 0.848 | 0.009 |
| generated_full | 0 | deont02_vs_generic | 30 | 0.900 | 0.683 | 0.859 | 0.016 |
| generated_full | 0 | deont01_vs_neutral | 30 | 0.800 | 0.583 | 0.773 | 0.008 |
| generated_full | 0 | deont02_vs_neutral | 30 | 0.867 | 0.683 | 0.779 | 0.013 |
| generated_full | 0 | deont01_vs_deont02 | 30 | 0.767 | 0.700 | 0.691 | 0.006 |
| generated_full | 0 | deont01_vs_anti | 30 | 0.967 | 0.550 | 0.861 | 0.010 |
| generated_full | 0 | deont02_vs_anti | 30 | 0.933 | 0.583 | 0.807 | 0.012 |
| generated_full | 4 | deont01_vs_generic | 30 | 1.000 | 0.500 | 0.858 | 0.076 |
| generated_full | 4 | deont02_vs_generic | 30 | 0.900 | 0.500 | 0.880 | 0.132 |
| generated_full | 4 | deont01_vs_neutral | 30 | 0.900 | 0.500 | 0.841 | 0.078 |
| generated_full | 4 | deont02_vs_neutral | 30 | 0.933 | 0.500 | 0.842 | 0.125 |
| generated_full | 4 | deont01_vs_deont02 | 30 | 0.833 | 0.550 | 0.734 | 0.063 |
| generated_full | 4 | deont01_vs_anti | 30 | 0.933 | 0.500 | 0.856 | 0.080 |
| generated_full | 4 | deont02_vs_anti | 30 | 0.867 | 0.500 | 0.819 | 0.095 |
| generated_full | 16 | deont01_vs_generic | 30 | 1.000 | 0.567 | 0.950 | 0.385 |
| generated_full | 16 | deont02_vs_generic | 30 | 1.000 | 0.883 | 0.958 | 0.590 |
| generated_full | 16 | deont01_vs_neutral | 30 | 0.967 | 0.583 | 0.924 | 0.372 |
| generated_full | 16 | deont02_vs_neutral | 30 | 0.967 | 0.883 | 0.947 | 0.567 |
| generated_full | 16 | deont01_vs_deont02 | 30 | 0.900 | 0.500 | 0.819 | 0.250 |
| generated_full | 16 | deont01_vs_anti | 30 | 1.000 | 0.500 | 0.967 | 0.340 |
| generated_full | 16 | deont02_vs_anti | 30 | 0.933 | 0.500 | 0.937 | 0.418 |
| generated_full | 24 | deont01_vs_generic | 30 | 1.000 | 0.500 | 0.951 | 0.525 |
| generated_full | 24 | deont02_vs_generic | 30 | 1.000 | 0.500 | 0.972 | 0.811 |
| generated_full | 24 | deont01_vs_neutral | 30 | 1.000 | 0.717 | 0.941 | 0.575 |
| generated_full | 24 | deont02_vs_neutral | 30 | 1.000 | 0.500 | 0.976 | 0.762 |
| generated_full | 24 | deont01_vs_deont02 | 30 | 1.000 | 0.500 | 0.814 | 0.393 |
| generated_full | 24 | deont01_vs_anti | 30 | 1.000 | 0.883 | 0.963 | 0.520 |
| generated_full | 24 | deont02_vs_anti | 30 | 0.933 | 0.500 | 0.947 | 0.576 |
| generated_full | 32 | deont01_vs_generic | 30 | 1.000 | 0.900 | 0.996 | 1.192 |
| generated_full | 32 | deont02_vs_generic | 30 | 1.000 | 0.950 | 0.993 | 1.488 |
| generated_full | 32 | deont01_vs_neutral | 30 | 1.000 | 0.917 | 0.987 | 1.137 |
| generated_full | 32 | deont02_vs_neutral | 30 | 1.000 | 0.883 | 0.990 | 1.492 |
| generated_full | 32 | deont01_vs_deont02 | 30 | 1.000 | 0.800 | 0.881 | 0.764 |
| generated_full | 32 | deont01_vs_anti | 30 | 1.000 | 0.500 | 0.992 | 1.320 |
| generated_full | 32 | deont02_vs_anti | 30 | 1.000 | 0.500 | 0.999 | 1.480 |
| generated_full | 40 | deont01_vs_generic | 30 | 1.000 | 0.500 | 1.000 | 2.592 |
| generated_full | 40 | deont02_vs_generic | 30 | 1.000 | 0.500 | 1.000 | 2.975 |
| generated_full | 40 | deont01_vs_neutral | 30 | 1.000 | 0.500 | 0.997 | 2.370 |
| generated_full | 40 | deont02_vs_neutral | 30 | 1.000 | 0.500 | 1.000 | 2.976 |
| generated_full | 40 | deont01_vs_deont02 | 30 | 1.000 | 0.500 | 0.921 | 1.514 |
| generated_full | 40 | deont01_vs_anti | 30 | 1.000 | 0.500 | 0.998 | 2.684 |
| generated_full | 40 | deont02_vs_anti | 30 | 1.000 | 0.500 | 0.999 | 3.024 |
| generated_first_16 | 0 | deont01_vs_generic | 30 | 0.733 | 0.583 | 0.699 | 0.006 |
| generated_first_16 | 0 | deont02_vs_generic | 30 | 0.767 | 0.617 | 0.724 | 0.021 |
| generated_first_16 | 0 | deont01_vs_neutral | 30 | 0.700 | 0.617 | 0.658 | 0.009 |
| generated_first_16 | 0 | deont02_vs_neutral | 30 | 0.867 | 0.683 | 0.789 | 0.026 |
| generated_first_16 | 0 | deont01_vs_deont02 | 30 | 0.633 | 0.583 | 0.650 | 0.010 |
| generated_first_16 | 0 | deont01_vs_anti | 30 | 0.667 | 0.567 | 0.609 | 0.004 |
| generated_first_16 | 0 | deont02_vs_anti | 30 | 0.733 | 0.600 | 0.662 | 0.012 |
| generated_first_16 | 4 | deont01_vs_generic | 30 | 0.867 | 0.500 | 0.786 | 0.085 |
| generated_first_16 | 4 | deont02_vs_generic | 30 | 0.867 | 0.600 | 0.790 | 0.119 |
| generated_first_16 | 4 | deont01_vs_neutral | 30 | 0.867 | 0.500 | 0.716 | 0.111 |
| generated_first_16 | 4 | deont02_vs_neutral | 30 | 0.933 | 0.733 | 0.801 | 0.200 |
| generated_first_16 | 4 | deont01_vs_deont02 | 30 | 0.867 | 0.500 | 0.642 | 0.059 |
| generated_first_16 | 4 | deont01_vs_anti | 30 | 0.767 | 0.500 | 0.660 | 0.046 |
| generated_first_16 | 4 | deont02_vs_anti | 30 | 0.733 | 0.600 | 0.642 | 0.054 |
| generated_first_16 | 16 | deont01_vs_generic | 30 | 1.000 | 0.733 | 0.864 | 0.456 |
| generated_first_16 | 16 | deont02_vs_generic | 30 | 0.967 | 0.617 | 0.891 | 0.626 |
| generated_first_16 | 16 | deont01_vs_neutral | 30 | 0.933 | 0.683 | 0.784 | 0.446 |
| generated_first_16 | 16 | deont02_vs_neutral | 30 | 0.967 | 0.567 | 0.880 | 0.772 |
| generated_first_16 | 16 | deont01_vs_deont02 | 30 | 0.933 | 0.500 | 0.683 | 0.156 |
| generated_first_16 | 16 | deont01_vs_anti | 30 | 0.967 | 0.500 | 0.867 | 0.354 |
| generated_first_16 | 16 | deont02_vs_anti | 30 | 0.933 | 0.750 | 0.877 | 0.381 |
| generated_first_16 | 24 | deont01_vs_generic | 30 | 0.967 | 0.500 | 0.826 | 0.561 |
| generated_first_16 | 24 | deont02_vs_generic | 30 | 0.967 | 0.500 | 0.867 | 0.895 |
| generated_first_16 | 24 | deont01_vs_neutral | 30 | 0.933 | 0.683 | 0.784 | 0.589 |
| generated_first_16 | 24 | deont02_vs_neutral | 30 | 0.933 | 0.733 | 0.858 | 0.857 |
| generated_first_16 | 24 | deont01_vs_deont02 | 30 | 0.933 | 0.633 | 0.692 | 0.254 |
| generated_first_16 | 24 | deont01_vs_anti | 30 | 0.933 | 0.500 | 0.818 | 0.364 |
| generated_first_16 | 24 | deont02_vs_anti | 30 | 0.933 | 0.500 | 0.890 | 0.564 |
| generated_first_16 | 32 | deont01_vs_generic | 30 | 1.000 | 0.850 | 0.924 | 1.417 |
| generated_first_16 | 32 | deont02_vs_generic | 30 | 1.000 | 0.600 | 0.966 | 1.550 |
| generated_first_16 | 32 | deont01_vs_neutral | 30 | 1.000 | 0.700 | 0.870 | 1.241 |
| generated_first_16 | 32 | deont02_vs_neutral | 30 | 1.000 | 0.567 | 0.939 | 1.418 |
| generated_first_16 | 32 | deont01_vs_deont02 | 30 | 1.000 | 0.500 | 0.808 | 0.710 |
| generated_first_16 | 32 | deont01_vs_anti | 30 | 1.000 | 0.500 | 0.984 | 1.455 |
| generated_first_16 | 32 | deont02_vs_anti | 30 | 1.000 | 0.767 | 0.994 | 1.597 |
| generated_first_16 | 40 | deont01_vs_generic | 30 | 1.000 | 0.500 | 0.953 | 3.040 |
| generated_first_16 | 40 | deont02_vs_generic | 30 | 1.000 | 0.500 | 0.976 | 3.335 |
| generated_first_16 | 40 | deont01_vs_neutral | 30 | 1.000 | 0.500 | 0.936 | 2.573 |
| generated_first_16 | 40 | deont02_vs_neutral | 30 | 1.000 | 0.500 | 0.942 | 3.277 |
| generated_first_16 | 40 | deont01_vs_deont02 | 30 | 1.000 | 0.717 | 0.814 | 1.311 |
| generated_first_16 | 40 | deont01_vs_anti | 30 | 1.000 | 0.500 | 0.992 | 2.890 |
| generated_first_16 | 40 | deont02_vs_anti | 30 | 1.000 | 0.500 | 0.994 | 3.249 |

## Transfer Readouts

| site | layer | train task | eval task | pairs | pair acc | BA | AUROC | median margin |
|---|---:|---|---|---:|---:|---:|---:|---:|
| prompt_end | 0 | deont01_vs_generic | deont02_vs_generic | 0 | nan | nan | nan | nan |
| prompt_end | 0 | deont01_vs_neutral | deont02_vs_neutral | 0 | nan | nan | nan | nan |
| prompt_end | 4 | deont01_vs_generic | deont02_vs_generic | 30 | 1.000 | 0.500 | 1.000 | 0.138 |
| prompt_end | 4 | deont01_vs_neutral | deont02_vs_neutral | 30 | 1.000 | 0.500 | 1.000 | 0.218 |
| prompt_end | 16 | deont01_vs_generic | deont02_vs_generic | 30 | 1.000 | 0.500 | 1.000 | 0.534 |
| prompt_end | 16 | deont01_vs_neutral | deont02_vs_neutral | 30 | 1.000 | 0.500 | 1.000 | 0.688 |
| prompt_end | 24 | deont01_vs_generic | deont02_vs_generic | 30 | 1.000 | 0.500 | 1.000 | 1.492 |
| prompt_end | 24 | deont01_vs_neutral | deont02_vs_neutral | 30 | 1.000 | 0.500 | 1.000 | 1.756 |
| prompt_end | 32 | deont01_vs_generic | deont02_vs_generic | 30 | 1.000 | 0.983 | 1.000 | 3.119 |
| prompt_end | 32 | deont01_vs_neutral | deont02_vs_neutral | 30 | 1.000 | 0.983 | 1.000 | 3.488 |
| prompt_end | 40 | deont01_vs_generic | deont02_vs_generic | 30 | 1.000 | 0.500 | 0.978 | 5.201 |
| prompt_end | 40 | deont01_vs_neutral | deont02_vs_neutral | 30 | 1.000 | 0.500 | 0.993 | 5.352 |
| generated_full | 0 | deont01_vs_generic | deont02_vs_generic | 30 | 0.867 | 0.600 | 0.853 | 0.011 |
| generated_full | 0 | deont01_vs_neutral | deont02_vs_neutral | 30 | 0.833 | 0.617 | 0.769 | 0.007 |
| generated_full | 4 | deont01_vs_generic | deont02_vs_generic | 30 | 0.900 | 0.500 | 0.823 | 0.091 |
| generated_full | 4 | deont01_vs_neutral | deont02_vs_neutral | 30 | 0.833 | 0.500 | 0.779 | 0.082 |
| generated_full | 16 | deont01_vs_generic | deont02_vs_generic | 30 | 0.933 | 0.567 | 0.941 | 0.467 |
| generated_full | 16 | deont01_vs_neutral | deont02_vs_neutral | 30 | 0.967 | 0.583 | 0.908 | 0.443 |
| generated_full | 24 | deont01_vs_generic | deont02_vs_generic | 30 | 1.000 | 0.500 | 0.950 | 0.635 |
| generated_full | 24 | deont01_vs_neutral | deont02_vs_neutral | 30 | 1.000 | 0.700 | 0.933 | 0.630 |
| generated_full | 32 | deont01_vs_generic | deont02_vs_generic | 30 | 1.000 | 0.867 | 0.992 | 1.231 |
| generated_full | 32 | deont01_vs_neutral | deont02_vs_neutral | 30 | 1.000 | 0.883 | 0.987 | 1.096 |
| generated_full | 40 | deont01_vs_generic | deont02_vs_generic | 30 | 1.000 | 0.500 | 1.000 | 2.634 |
| generated_full | 40 | deont01_vs_neutral | deont02_vs_neutral | 30 | 1.000 | 0.500 | 0.999 | 2.280 |
| generated_first_16 | 0 | deont01_vs_generic | deont02_vs_generic | 30 | 0.767 | 0.617 | 0.727 | 0.007 |
| generated_first_16 | 0 | deont01_vs_neutral | deont02_vs_neutral | 30 | 0.867 | 0.700 | 0.774 | 0.018 |
| generated_first_16 | 4 | deont01_vs_generic | deont02_vs_generic | 30 | 0.967 | 0.500 | 0.833 | 0.086 |
| generated_first_16 | 4 | deont01_vs_neutral | deont02_vs_neutral | 30 | 0.967 | 0.500 | 0.839 | 0.150 |
| generated_first_16 | 16 | deont01_vs_generic | deont02_vs_generic | 30 | 1.000 | 0.800 | 0.913 | 0.479 |
| generated_first_16 | 16 | deont01_vs_neutral | deont02_vs_neutral | 30 | 1.000 | 0.700 | 0.888 | 0.567 |
| generated_first_16 | 24 | deont01_vs_generic | deont02_vs_generic | 30 | 1.000 | 0.500 | 0.860 | 0.575 |
| generated_first_16 | 24 | deont01_vs_neutral | deont02_vs_neutral | 30 | 0.967 | 0.750 | 0.878 | 0.692 |
| generated_first_16 | 32 | deont01_vs_generic | deont02_vs_generic | 30 | 1.000 | 0.833 | 0.953 | 1.413 |
| generated_first_16 | 32 | deont01_vs_neutral | deont02_vs_neutral | 30 | 1.000 | 0.750 | 0.932 | 1.264 |
| generated_first_16 | 40 | deont01_vs_generic | deont02_vs_generic | 30 | 1.000 | 0.500 | 0.967 | 3.078 |
| generated_first_16 | 40 | deont01_vs_neutral | deont02_vs_neutral | 30 | 1.000 | 0.500 | 0.957 | 2.897 |

## Prompt-vs-Generated Direction Cosines

| layer | task | cos(prompt_end, generated_full) | cos(prompt_end, generated_first_16) | cos(generated_full, generated_first_16) |
|---:|---|---:|---:|---:|
| 0 | deont01_vs_generic | nan | nan | 0.584 |
| 0 | deont02_vs_generic | nan | nan | 0.662 |
| 0 | deont01_vs_neutral | nan | nan | 0.520 |
| 0 | deont02_vs_neutral | nan | nan | 0.661 |
| 0 | deont01_vs_deont02 | nan | nan | 0.621 |
| 0 | deont01_vs_anti | nan | nan | 0.541 |
| 0 | deont02_vs_anti | nan | nan | 0.601 |
| 4 | deont01_vs_generic | 0.011 | 0.040 | 0.661 |
| 4 | deont02_vs_generic | 0.040 | 0.011 | 0.712 |
| 4 | deont01_vs_neutral | 0.047 | 0.056 | 0.594 |
| 4 | deont02_vs_neutral | 0.075 | -0.006 | 0.708 |
| 4 | deont01_vs_deont02 | 0.017 | 0.012 | 0.667 |
| 4 | deont01_vs_anti | 0.061 | 0.052 | 0.637 |
| 4 | deont02_vs_anti | 0.093 | 0.047 | 0.651 |
| 16 | deont01_vs_generic | 0.076 | 0.064 | 0.771 |
| 16 | deont02_vs_generic | 0.082 | 0.067 | 0.794 |
| 16 | deont01_vs_neutral | 0.029 | 0.022 | 0.717 |
| 16 | deont02_vs_neutral | 0.033 | 0.024 | 0.767 |
| 16 | deont01_vs_deont02 | 0.051 | 0.084 | 0.695 |
| 16 | deont01_vs_anti | 0.097 | 0.082 | 0.675 |
| 16 | deont02_vs_anti | 0.075 | 0.089 | 0.717 |
| 24 | deont01_vs_generic | 0.036 | -0.002 | 0.781 |
| 24 | deont02_vs_generic | 0.031 | 0.004 | 0.801 |
| 24 | deont01_vs_neutral | 0.060 | 0.036 | 0.741 |
| 24 | deont02_vs_neutral | 0.053 | 0.030 | 0.790 |
| 24 | deont01_vs_deont02 | 0.071 | 0.089 | 0.748 |
| 24 | deont01_vs_anti | 0.069 | 0.000 | 0.690 |
| 24 | deont02_vs_anti | 0.050 | 0.007 | 0.756 |
| 32 | deont01_vs_generic | 0.246 | 0.213 | 0.855 |
| 32 | deont02_vs_generic | 0.255 | 0.211 | 0.856 |
| 32 | deont01_vs_neutral | 0.201 | 0.136 | 0.824 |
| 32 | deont02_vs_neutral | 0.201 | 0.132 | 0.840 |
| 32 | deont01_vs_deont02 | 0.209 | 0.239 | 0.829 |
| 32 | deont01_vs_anti | 0.292 | 0.266 | 0.860 |
| 32 | deont02_vs_anti | 0.233 | 0.223 | 0.878 |
| 40 | deont01_vs_generic | 0.269 | 0.216 | 0.868 |
| 40 | deont02_vs_generic | 0.286 | 0.242 | 0.863 |
| 40 | deont01_vs_neutral | 0.243 | 0.180 | 0.849 |
| 40 | deont02_vs_neutral | 0.241 | 0.191 | 0.850 |
| 40 | deont01_vs_deont02 | 0.263 | 0.232 | 0.824 |
| 40 | deont01_vs_anti | 0.281 | 0.257 | 0.871 |
| 40 | deont02_vs_anti | 0.242 | 0.250 | 0.893 |
