# Pre-Theory-Token Window Check

- capture artifact: `capture_1_1d7271d73617`
- generation rows: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/all_theories_brief_recommendation_report/report_6aa730c32d87_8c1df9a2/results/generate_natural_responses_results.json`
- layer: L32
- filter: drop pairs where matched theory-canonical vocabulary appears in first 16 whitespace words of either response

| theory | all n | removed n | removed rate | all gap | kept gap | removed gap | cos(kept, all) | cos(removed, all) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deont | 30 | 12 | 0.400 | 0.419 | 0.335 | 0.297 | 0.863 | 0.903 |
| util | 30 | 3 | 0.100 | 0.363 | 0.343 | nan | 0.984 | 0.491 |
| virtue | 30 | 8 | 0.267 | 0.439 | 0.338 | 0.245 | 0.915 | 0.794 |
| contract | 30 | 1 | 0.033 | 0.423 | 0.430 | nan | 0.998 | 0.397 |
