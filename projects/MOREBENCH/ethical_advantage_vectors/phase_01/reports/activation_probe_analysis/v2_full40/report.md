# Ethical Advantage Activation Probe Analysis

- capture: `capture_1_2461d8ccdc41`
- generation rows: `projects/MOREBENCH/ethical_advantage_vectors/phase_01/reports/v2_full_capture/report_12abda8dec51_9d7390f7/results/generate_v2_responses_results.json`
- strict behavior-gate dilemmas: `12`

## Prompt Pole Probe

Target: negative short-term self-advantage prompts vs ethical prompts, grouped by dilemma.

| slice | layer | n | positives | logistic AUROC | centroid AUROC | split gap | real median | null p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prompt_end | 16 | 400 | 240 | 1.000 | 1.000 | 0.137 | 0.994 | 0.857 |
| generated_first_16 | 16 | 400 | 240 | 0.998 | 0.996 | 0.569 | 0.911 | 0.342 |
| generated_full | 16 | 400 | 240 | 1.000 | 1.000 | 0.399 | 0.978 | 0.579 |
| prompt_end | 24 | 400 | 240 | 1.000 | 1.000 | 0.118 | 0.995 | 0.877 |
| generated_first_16 | 24 | 400 | 240 | 0.999 | 0.998 | 0.504 | 0.933 | 0.429 |
| generated_full | 24 | 400 | 240 | 1.000 | 1.000 | 0.310 | 0.981 | 0.671 |
| prompt_end | 32 | 400 | 240 | 1.000 | 1.000 | 0.177 | 0.993 | 0.815 |
| generated_first_16 | 32 | 400 | 240 | 1.000 | 1.000 | 0.506 | 0.951 | 0.445 |
| generated_full | 32 | 400 | 240 | 1.000 | 1.000 | 0.313 | 0.984 | 0.670 |
| prompt_end | 40 | 400 | 240 | 1.000 | 1.000 | 0.169 | 0.990 | 0.821 |
| generated_first_16 | 40 | 400 | 240 | 1.000 | 1.000 | 0.463 | 0.955 | 0.493 |
| generated_full | 40 | 400 | 240 | 1.000 | 1.000 | 0.279 | 0.985 | 0.707 |

## Observed Action Probe

Target: regex-labeled self-advantage action vs ethical action across all conditions; unknown labels excluded.

| slice | layer | n | positives | logistic AUROC | centroid AUROC |
|---|---:|---:|---:|---:|---:|
| prompt_end | 16 | 535 | 175 | 0.932 | 0.931 |
| generated_first_16 | 16 | 535 | 175 | 0.946 | 0.953 |
| generated_full | 16 | 535 | 175 | 0.980 | 0.972 |
| prompt_end | 24 | 535 | 175 | 0.944 | 0.939 |
| generated_first_16 | 24 | 535 | 175 | 0.967 | 0.971 |
| generated_full | 24 | 535 | 175 | 0.979 | 0.978 |
| prompt_end | 32 | 535 | 175 | 0.948 | 0.931 |
| generated_first_16 | 32 | 535 | 175 | 0.968 | 0.973 |
| generated_full | 32 | 535 | 175 | 0.974 | 0.978 |
| prompt_end | 40 | 535 | 175 | 0.951 | 0.930 |
| generated_first_16 | 40 | 535 | 175 | 0.963 | 0.968 |
| generated_full | 40 | 535 | 175 | 0.980 | 0.972 |

## Strict-Gate Prompt Pole Probe

Target: prompt pole restricted to dilemmas where behavior cleanly flipped in the smoke labeler.

| slice | layer | n | positives | logistic AUROC | centroid AUROC | split gap |
|---|---:|---:|---:|---:|---:|---:|
| prompt_end | 16 | 120 | 72 | 1.000 | 1.000 | 0.132 |
| generated_first_16 | 16 | 120 | 72 | 1.000 | 1.000 | 0.351 |
| generated_full | 16 | 120 | 72 | 1.000 | 1.000 | 0.284 |
| prompt_end | 24 | 120 | 72 | 1.000 | 1.000 | 0.123 |
| generated_first_16 | 24 | 120 | 72 | 1.000 | 1.000 | 0.377 |
| generated_full | 24 | 120 | 72 | 1.000 | 1.000 | 0.316 |
| prompt_end | 32 | 120 | 72 | 1.000 | 1.000 | 0.130 |
| generated_first_16 | 32 | 120 | 72 | 1.000 | 1.000 | 0.333 |
| generated_full | 32 | 120 | 72 | 1.000 | 1.000 | 0.273 |
| prompt_end | 40 | 120 | 72 | 1.000 | 1.000 | 0.222 |
| generated_first_16 | 40 | 120 | 72 | 1.000 | 1.000 | 0.438 |
| generated_full | 40 | 120 | 72 | 1.000 | 1.000 | 0.269 |
