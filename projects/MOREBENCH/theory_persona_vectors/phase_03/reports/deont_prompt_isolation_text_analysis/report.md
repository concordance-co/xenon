# Deont Prompt Isolation Text Analysis

- generation rows: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/deont_prompt_isolation_report/report_4b6e5c6c9407_f2313986/results/generate_natural_responses_results.json`
- text model: `char TF-IDF 3-5 + ridge classifier`

## Compliance

- total rows: `150`
- exact 3-line format: `150`
- banned-word leaks: `3`

## Recommendation Overlap

| pair | same recommendation count | total | rate |
|---|---:|---:|---:|
| deont01_vs_deont02 | 10 | 30 | 0.333 |
| deont01_vs_generic | 3 | 30 | 0.100 |
| deont02_vs_generic | 3 | 30 | 0.100 |

## Direct Text Readouts

| task | pairs | BA | AUROC |
|---|---:|---:|---:|
| deont01_vs_deont02 | 30 | 0.617 | 0.624 |
| deont01_vs_generic | 30 | 0.667 | 0.767 |
| deont02_vs_generic | 30 | 0.833 | 0.850 |
| deont01_vs_neutral | 30 | 0.717 | 0.767 |
| deont02_vs_neutral | 30 | 0.683 | 0.696 |

## Transfer-Style Text Readouts

| train task | eval task | pairs | BA | AUROC |
|---|---|---:|---:|---:|
| deont01_vs_generic | deont02_vs_generic | 30 | 0.650 | 0.784 |
| deont01_vs_neutral | deont02_vs_neutral | 30 | 0.700 | 0.682 |
