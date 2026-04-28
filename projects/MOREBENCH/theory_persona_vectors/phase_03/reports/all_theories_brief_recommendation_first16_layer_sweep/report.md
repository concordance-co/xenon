# Generated First-16 Layer Sweep

- capture artifact: `capture_1_1d7271d73617`
- generation rows: `projects/MOREBENCH/theory_persona_vectors/phase_03/reports/all_theories_brief_recommendation_report/report_6aa730c32d87_8c1df9a2/results/generate_natural_responses_results.json`
- slice: `first_16` generated tokens

| theory | layer | real median | null p95 | gap | cosine to L32 |
|---|---:|---:|---:|---:|---:|
| deont | 0 | 0.250 | 0.089 | 0.161 | 0.137 |
| util | 0 | -0.005 | 0.099 | -0.104 | 0.125 |
| generic | 0 | 0.169 | 0.113 | 0.055 | 0.163 |
| deont | 4 | 0.440 | 0.192 | 0.248 | 0.458 |
| util | 4 | 0.303 | 0.154 | 0.148 | 0.415 |
| generic | 4 | 0.340 | 0.188 | 0.152 | 0.491 |
| deont | 16 | 0.635 | 0.241 | 0.394 | 0.714 |
| util | 16 | 0.548 | 0.167 | 0.381 | 0.647 |
| generic | 16 | 0.508 | 0.229 | 0.279 | 0.624 |
| deont | 24 | 0.657 | 0.245 | 0.411 | 0.770 |
| util | 24 | 0.532 | 0.171 | 0.361 | 0.782 |
| generic | 24 | 0.504 | 0.287 | 0.218 | 0.781 |
| deont | 32 | 0.747 | 0.328 | 0.419 | 1.000 |
| util | 32 | 0.578 | 0.214 | 0.363 | 1.000 |
| generic | 32 | 0.593 | 0.247 | 0.346 | 1.000 |
| deont | 40 | 0.775 | 0.318 | 0.456 | 0.799 |
| util | 40 | 0.546 | 0.191 | 0.355 | 0.794 |
| generic | 40 | 0.573 | 0.256 | 0.318 | 0.791 |

## Suggested Write Layers

Use this as write-site evidence, not as a new readout claim. Earlier layers near the first sustained gap uptick are better first causal write candidates than the latest/highest readout layer.

- `deont`: `L16 first layer with gap>=0.30 and cos_to_L32>=0.35`
- `util`: `L16 first layer with gap>=0.30 and cos_to_L32>=0.35`
- `generic`: `L32 first layer with gap>=0.30 and cos_to_L32>=0.35`
