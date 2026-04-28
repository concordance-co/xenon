# Brief-Recommendation Lexical Smoke

- rows: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/all_theories_brief_recommendation_behavior_smoke/report_ab75bf6af63e_70a7e287/results/generate_natural_responses_results.json`
- split: leave-one-dilemma-out
- text model: char TF-IDF 3-5 grams + balanced logistic regression
- length baseline: token count + char count + balanced logistic regression
- caveat: only 8 dilemmas; leak detector, not final baseline

| task | n | classes | text BA | text AUROC | length BA | length AUROC | chance BA |
|---|---:|---|---:|---:|---:|---:|---:|
| primary_theory_4way_from_response_text | 32 | contractualism, deontology, utilitarian, virtue_ethics | 0.875 | nan | 0.188 | nan | 0.250 |
| any_primary_theory_vs_neutral_short | 40 | neutral_short, theory_prime | 0.500 | 0.895 | 0.766 | 0.812 | 0.500 |
| any_primary_theory_vs_generic_moral | 40 | generic_moral, theory_prime | 0.500 | 0.887 | 0.531 | 0.559 | 0.500 |
| generic_moral_vs_neutral_short | 16 | N_generic_moral_01, N_neutral_01 | 0.562 | 0.609 | 0.812 | 0.828 | 0.500 |
| neutral_length_matched_vs_neutral_short | 16 | N_neutral_01, N_neutral_02 | 0.438 | 0.391 | 0.688 | 0.703 | 0.500 |
| deontology_primary_vs_neutral_short | 16 | neutral_short, positive | 0.875 | 0.969 | 0.750 | 0.844 | 0.500 |
| deontology_primary_vs_generic_moral | 16 | generic_moral, positive | 0.875 | 0.938 | 0.375 | 0.328 | 0.500 |
| deontology_primary_vs_anti | 16 | anti, positive | 0.875 | 1.000 | 0.938 | 0.969 | 0.500 |
| utilitarian_primary_vs_neutral_short | 16 | neutral_short, positive | 0.812 | 0.984 | 0.750 | 0.797 | 0.500 |
| utilitarian_primary_vs_generic_moral | 16 | generic_moral, positive | 0.812 | 0.938 | 0.562 | 0.625 | 0.500 |
| utilitarian_primary_vs_anti | 16 | anti, positive | 0.938 | 0.984 | 0.750 | 0.828 | 0.500 |
| virtue_ethics_primary_vs_neutral_short | 16 | neutral_short, positive | 0.688 | 0.938 | 0.812 | 0.781 | 0.500 |
| virtue_ethics_primary_vs_generic_moral | 16 | generic_moral, positive | 0.938 | 0.984 | 0.500 | 0.594 | 0.500 |
| virtue_ethics_primary_vs_anti | 16 | anti, positive | 0.812 | 0.922 | 0.875 | 0.969 | 0.500 |
| contractualism_primary_vs_neutral_short | 16 | neutral_short, positive | 0.688 | 0.859 | 0.688 | 0.812 | 0.500 |
| contractualism_primary_vs_generic_moral | 16 | generic_moral, positive | 0.875 | 0.906 | 0.500 | 0.578 | 0.500 |
| contractualism_primary_vs_anti | 16 | anti, positive | 0.812 | 0.984 | 0.688 | 0.859 | 0.500 |
