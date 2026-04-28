# Deontology Lexical Transfer

- capture artifact: `capture_1_1d7271d73617`
- generation rows: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/all_theories_brief_recommendation_report/report_6aa730c32d87_8c1df9a2/results/generate_natural_responses_results.json`
- site: `generated_sequence_residual`
- slice: `first_16`
- layers: `16, 32, 40`
- source lexical family: `\b(duty|duties|right|rights|promise|promises|obligation|obligations|constraint|constraints)\b`
- variant lexical family: `\b(commitment|commitments|boundary|boundaries|must not|mustn't|forbidden|off-limits|line(?:s)? not to cross)\b`

## Pair Availability

| eval pair | filter | available pairs |
|---|---|---:|
| deont_primary | all | 30 |
| deont_primary | source_lex_absent_first_n | 18 |
| deont_primary | source_lex_absent_full | 1 |
| deont_primary | source_lex_absent_first_n_and_variant_present | 5 |
| deont_variant | all | 30 |
| deont_variant | source_lex_absent_first_n | 29 |
| deont_variant | source_lex_absent_full | 18 |
| deont_variant | source_lex_absent_first_n_and_variant_present | 27 |
| generic_moral | all | 30 |
| generic_moral | source_lex_absent_first_n | 28 |
| generic_moral | source_lex_absent_full | 14 |
| generic_moral | source_lex_absent_first_n_and_variant_present | 5 |
| neutral_length | all | 30 |
| neutral_length | source_lex_absent_first_n | 30 |
| neutral_length | source_lex_absent_full | 22 |
| neutral_length | source_lex_absent_first_n_and_variant_present | 2 |
| anti_deont | all | 30 |
| anti_deont | source_lex_absent_first_n | 29 |
| anti_deont | source_lex_absent_full | 25 |
| anti_deont | source_lex_absent_first_n_and_variant_present | 1 |

## Transfer Results

| layer | train pair | train filter | eval pair | eval filter | eval pairs | LOO AUROC | mean margin | median margin |
|---:|---|---|---|---|---:|---:|---:|---:|
| 16 | deont_primary | all | deont_primary | all | 30 | 0.939 | 0.828 | 0.834 |
| 16 | deont_primary | all | deont_variant | all | 30 | 0.902 | 0.630 | 0.548 |
| 16 | deont_primary | all | deont_variant | source_lex_absent_first_n | 29 | 0.898 | 0.624 | 0.503 |
| 16 | deont_primary | all | deont_variant | source_lex_absent_full | 18 | 0.852 | 0.606 | 0.502 |
| 16 | deont_primary | all | deont_variant | source_lex_absent_first_n_and_variant_present | 27 | 0.925 | 0.644 | 0.503 |
| 16 | deont_primary | source_lex_absent_first_n | deont_variant | source_lex_absent_first_n | 29 | 0.854 | 0.569 | 0.501 |
| 16 | deont_primary | source_lex_absent_full | deont_variant | source_lex_absent_full | 17 | 0.651 | 0.206 | 0.142 |
| 16 | deont_primary | all | generic_moral | all | 30 | 0.680 | 0.229 | 0.192 |
| 16 | deont_primary | all | neutral_length | all | 30 | 0.570 | 0.091 | 0.078 |
| 16 | deont_primary | all | anti_deont | all | 30 | 0.686 | 0.216 | 0.233 |
| 16 | generic_moral | all | deont_variant | all | 30 | 0.759 | 0.390 | 0.351 |
| 16 | neutral_length | all | deont_variant | all | 30 | 0.640 | 0.137 | 0.075 |
| 32 | deont_primary | all | deont_primary | all | 30 | 0.970 | 2.082 | 2.068 |
| 32 | deont_primary | all | deont_variant | all | 30 | 0.926 | 1.540 | 1.434 |
| 32 | deont_primary | all | deont_variant | source_lex_absent_first_n | 29 | 0.922 | 1.506 | 1.428 |
| 32 | deont_primary | all | deont_variant | source_lex_absent_full | 18 | 0.895 | 1.514 | 1.434 |
| 32 | deont_primary | all | deont_variant | source_lex_absent_first_n_and_variant_present | 27 | 0.942 | 1.533 | 1.428 |
| 32 | deont_primary | source_lex_absent_first_n | deont_variant | source_lex_absent_first_n | 29 | 0.944 | 1.416 | 1.380 |
| 32 | deont_primary | source_lex_absent_full | deont_variant | source_lex_absent_full | 17 | 0.692 | 0.418 | 0.365 |
| 32 | deont_primary | all | generic_moral | all | 30 | 0.700 | 0.559 | 0.393 |
| 32 | deont_primary | all | neutral_length | all | 30 | 0.540 | 0.136 | 0.039 |
| 32 | deont_primary | all | anti_deont | all | 30 | 0.696 | 0.553 | 0.426 |
| 32 | generic_moral | all | deont_variant | all | 30 | 0.779 | 0.817 | 0.769 |
| 32 | neutral_length | all | deont_variant | all | 30 | 0.631 | 0.340 | 0.267 |
| 40 | deont_primary | all | deont_primary | all | 30 | 0.973 | 4.196 | 4.233 |
| 40 | deont_primary | all | deont_variant | all | 30 | 0.943 | 3.112 | 2.675 |
| 40 | deont_primary | all | deont_variant | source_lex_absent_first_n | 29 | 0.941 | 3.033 | 2.636 |
| 40 | deont_primary | all | deont_variant | source_lex_absent_full | 18 | 0.904 | 3.054 | 2.650 |
| 40 | deont_primary | all | deont_variant | source_lex_absent_first_n_and_variant_present | 27 | 0.966 | 3.081 | 2.636 |
| 40 | deont_primary | source_lex_absent_first_n | deont_variant | source_lex_absent_first_n | 29 | 0.943 | 2.889 | 2.727 |
| 40 | deont_primary | source_lex_absent_full | deont_variant | source_lex_absent_full | 17 | 0.744 | 1.055 | 0.852 |
| 40 | deont_primary | all | generic_moral | all | 30 | 0.722 | 1.041 | 0.760 |
| 40 | deont_primary | all | neutral_length | all | 30 | 0.528 | 0.220 | -0.021 |
| 40 | deont_primary | all | anti_deont | all | 30 | 0.644 | 0.676 | 0.608 |
| 40 | generic_moral | all | deont_variant | all | 30 | 0.799 | 1.338 | 1.071 |
| 40 | neutral_length | all | deont_variant | all | 30 | 0.519 | 0.073 | -0.092 |

## Text-Only Companion Check

- model: `char TF-IDF 3-5 + ridge classifier`
- max features: `4000`
- min df: `2`

| eval pair | filter | eval pairs | text BA | text AUROC |
|---|---|---:|---:|---:|
| deont_primary | all | 30 | 0.783 | 0.986 |
| deont_variant | all | 30 | 1.000 | 1.000 |
| deont_variant | source_lex_absent_first_n | 29 | 1.000 | 1.000 |
| deont_variant | source_lex_absent_full | 18 | 1.000 | 1.000 |
| deont_variant | source_lex_absent_first_n_and_variant_present | 27 | 1.000 | 1.000 |

## Direct `01 vs 02` Text Check

| task | filter | eval pairs | text BA | text AUROC |
|---|---|---:|---:|---:|
| primary_vs_variant | all | 30 | 0.967 | 0.998 |
| primary_vs_variant | source_lex_absent_first_n | 17 | 0.912 | 0.990 |
| primary_vs_variant | source_lex_absent_full | 0 | nan | nan |
| primary_vs_variant | source_lex_absent_first_n_and_variant_present | 5 | 0.700 | 0.880 |

## Transfer-Style Text Baselines

| train task | eval task | eval filter | eval pairs | text BA | text AUROC |
|---|---|---|---:|---:|---:|
| deont01_vs_generic | deont02_vs_generic | all | 30 | 0.733 | 0.854 |
| deont01_vs_generic | deont02_vs_generic | source_lex_absent_first_n | 27 | 0.722 | 0.842 |
| deont01_vs_generic | deont02_vs_generic | source_lex_absent_first_n_and_variant_present | 25 | 0.740 | 0.850 |
| deont01_vs_neutral | deont02_vs_neutral | all | 30 | 0.817 | 0.963 |
| deont01_vs_neutral | deont02_vs_neutral | source_lex_absent_first_n | 29 | 0.828 | 0.962 |
| deont01_vs_neutral | deont02_vs_neutral | source_lex_absent_first_n_and_variant_present | 27 | 0.833 | 0.966 |

## Readout Heuristic

- Good evidence for this strategy is: `P_deont_01 -> P_deont_02` stays clearly above chance, survives the source-lex suppression filters, and remains better aligned than generic or neutral control directions.
- Weak evidence is: transfer collapses to chance once source-family words are filtered, or generic-moral training transfers just as well as the deont-primary direction.
- Diagnostic caution: this is still a transfer-style confound audit on existing generations, not a claim that the direction is a clean latent deontology feature.
