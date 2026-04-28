# Natural-Prompt Lexical Smoke

- rows: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/all_theories_natural_prompt_behavior_smoke/report_6cd6f6de0d8f_03e7ab12/results/generate_natural_responses_results.json`
- split: leave-one-dilemma-out
- model: char TF-IDF 3-5 grams + balanced logistic regression
- caveat: only 8 dilemmas; this is a leak detector, not a final baseline

| task | n | classes | BA | AUROC | chance BA |
|---|---:|---|---:|---:|---:|
| primary_theory_4way_from_response_text | 32 | contractualism, deontology, utilitarian, virtue_ethics | 0.875 | nan | 0.250 |
| any_primary_theory_vs_neutral_short | 40 | neutral_short, theory_prime | 0.562 | 0.941 | 0.500 |
| any_primary_theory_vs_generic_moral | 40 | generic_moral, theory_prime | 0.500 | 0.965 | 0.500 |
| generic_moral_vs_neutral_short | 16 | N_generic_moral_01, N_neutral_01 | 0.688 | 0.984 | 0.500 |
| neutral_length_matched_vs_neutral_short | 16 | N_neutral_01, N_neutral_02 | 0.938 | 0.969 | 0.500 |
| deontology_primary_vs_neutral_short | 16 | neutral_short, positive | 0.875 | 1.000 | 0.500 |
| deontology_primary_vs_generic_moral | 16 | generic_moral, positive | 1.000 | 1.000 | 0.500 |
| deontology_primary_vs_anti | 16 | anti, positive | 0.812 | 1.000 | 0.500 |
| utilitarian_primary_vs_neutral_short | 16 | neutral_short, positive | 0.812 | 0.906 | 0.500 |
| utilitarian_primary_vs_generic_moral | 16 | generic_moral, positive | 0.812 | 0.984 | 0.500 |
| utilitarian_primary_vs_anti | 16 | anti, positive | 0.938 | 1.000 | 0.500 |
| virtue_ethics_primary_vs_neutral_short | 16 | neutral_short, positive | 0.875 | 0.953 | 0.500 |
| virtue_ethics_primary_vs_generic_moral | 16 | generic_moral, positive | 0.875 | 0.969 | 0.500 |
| virtue_ethics_primary_vs_anti | 16 | anti, positive | 0.688 | 0.984 | 0.500 |
| contractualism_primary_vs_neutral_short | 16 | neutral_short, positive | 0.875 | 1.000 | 0.500 |
| contractualism_primary_vs_generic_moral | 16 | generic_moral, positive | 0.875 | 0.953 | 0.500 |
| contractualism_primary_vs_anti | 16 | anti, positive | 0.562 | 0.922 | 0.500 |
