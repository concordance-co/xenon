# Ethical Advantage Lexical-Confound Analysis

- generation rows: `projects/MOREBENCH/ethical_advantage_vectors/phase_01/reports/v2_full_capture/report_12abda8dec51_9d7390f7/results/generate_v2_responses_results.json`
- action rows: `projects/MOREBENCH/ethical_advantage_vectors/phase_01/reports/behavior_smoke_analysis/v2_full40/scored_rows.jsonl`
- capture: `capture_1_2461d8ccdc41`

## Text-Only Baselines

AUROC from TF-IDF text only, grouped by dilemma. For first-16 activation claims, the fair lexical comparator is `generated_prefix16_text`; for full-response claims it is `generated_text`; for prompt-end it is prompt/instruction text.

| target | text field | n | positives | AUROC |
|---|---|---:|---:|---:|
| prompt_pole | instruction_text | 400 | 240 | 1.000 |
| prompt_pole | prompt_text | 400 | 240 | 1.000 |
| prompt_pole | generated_prefix16_text | 400 | 240 | 0.934 |
| prompt_pole | generated_prefix32_text | 400 | 240 | 0.997 |
| prompt_pole | generated_text | 400 | 240 | 1.000 |
| observed_action | instruction_text | 535 | 175 | 0.934 |
| observed_action | prompt_text | 535 | 175 | 0.941 |
| observed_action | generated_prefix16_text | 535 | 175 | 0.857 |
| observed_action | generated_prefix32_text | 535 | 175 | 0.972 |
| observed_action | generated_text | 535 | 175 | 0.969 |
| observed_action_within_negative | instruction_text | 175 | 156 | 0.626 |
| observed_action_within_negative | prompt_text | 175 | 156 | 0.646 |
| observed_action_within_negative | generated_prefix16_text | 175 | 156 | 0.587 |
| observed_action_within_negative | generated_prefix32_text | 175 | 156 | 0.763 |
| observed_action_within_negative | generated_text | 175 | 156 | 0.778 |

## Condition-Only Baselines

| target | n | positives | AUROC |
|---|---:|---:|---:|
| prompt_pole | 400 | 240 | 1.000 |
| observed_action | 535 | 175 | 0.919 |
| observed_action_within_negative | 175 | 156 | 0.500 |

## Activation Beyond Text

Each row uses the matching text field for the activation slice. `combined delta` is AUROC(text+activation) - AUROC(text). `resid act abs` is activation-score AUROC after linearly removing the text score, sign-normalized away from 0.5.

| target | slice | layer | text field | n | text AUROC | act AUROC | text+act AUROC | combined delta | resid act abs |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| prompt_pole | prompt_end | 16 | prompt_text | 400 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| prompt_pole | prompt_end | 24 | prompt_text | 400 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| prompt_pole | prompt_end | 32 | prompt_text | 400 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| prompt_pole | prompt_end | 40 | prompt_text | 400 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| prompt_pole | generated_first_16 | 16 | generated_prefix16_text | 400 | 0.934 | 0.998 | 0.996 | 0.062 | 0.834 |
| prompt_pole | generated_first_16 | 24 | generated_prefix16_text | 400 | 0.934 | 0.999 | 0.997 | 0.063 | 0.863 |
| prompt_pole | generated_first_16 | 32 | generated_prefix16_text | 400 | 0.934 | 1.000 | 1.000 | 0.066 | 0.865 |
| prompt_pole | generated_first_16 | 40 | generated_prefix16_text | 400 | 0.934 | 1.000 | 1.000 | 0.066 | 0.864 |
| prompt_pole | generated_full | 16 | generated_text | 400 | 1.000 | 1.000 | 1.000 | 0.000 | 0.728 |
| prompt_pole | generated_full | 24 | generated_text | 400 | 1.000 | 1.000 | 1.000 | 0.000 | 0.736 |
| prompt_pole | generated_full | 32 | generated_text | 400 | 1.000 | 1.000 | 1.000 | 0.000 | 0.738 |
| prompt_pole | generated_full | 40 | generated_text | 400 | 1.000 | 1.000 | 1.000 | 0.000 | 0.741 |
| observed_action | prompt_end | 16 | prompt_text | 535 | 0.941 | 0.932 | 0.945 | 0.004 | 0.742 |
| observed_action | prompt_end | 24 | prompt_text | 535 | 0.941 | 0.944 | 0.949 | 0.008 | 0.743 |
| observed_action | prompt_end | 32 | prompt_text | 535 | 0.941 | 0.948 | 0.951 | 0.010 | 0.734 |
| observed_action | prompt_end | 40 | prompt_text | 535 | 0.941 | 0.951 | 0.954 | 0.013 | 0.747 |
| observed_action | generated_first_16 | 16 | generated_prefix16_text | 535 | 0.857 | 0.946 | 0.936 | 0.080 | 0.776 |
| observed_action | generated_first_16 | 24 | generated_prefix16_text | 535 | 0.857 | 0.967 | 0.958 | 0.101 | 0.861 |
| observed_action | generated_first_16 | 32 | generated_prefix16_text | 535 | 0.857 | 0.968 | 0.957 | 0.100 | 0.834 |
| observed_action | generated_first_16 | 40 | generated_prefix16_text | 535 | 0.857 | 0.963 | 0.954 | 0.097 | 0.812 |
| observed_action | generated_full | 16 | generated_text | 535 | 0.969 | 0.980 | 0.979 | 0.009 | 0.680 |
| observed_action | generated_full | 24 | generated_text | 535 | 0.969 | 0.979 | 0.979 | 0.010 | 0.695 |
| observed_action | generated_full | 32 | generated_text | 535 | 0.969 | 0.974 | 0.977 | 0.007 | 0.665 |
| observed_action | generated_full | 40 | generated_text | 535 | 0.969 | 0.980 | 0.980 | 0.011 | 0.677 |
| observed_action_within_negative | prompt_end | 16 | prompt_text | 175 | 0.646 | 0.460 | 0.460 | -0.185 | 0.643 |
| observed_action_within_negative | prompt_end | 24 | prompt_text | 175 | 0.646 | 0.540 | 0.553 | -0.092 | 0.612 |
| observed_action_within_negative | prompt_end | 32 | prompt_text | 175 | 0.646 | 0.476 | 0.484 | -0.161 | 0.661 |
| observed_action_within_negative | prompt_end | 40 | prompt_text | 175 | 0.646 | 0.576 | 0.575 | -0.070 | 0.665 |
| observed_action_within_negative | generated_first_16 | 16 | generated_prefix16_text | 175 | 0.587 | 0.610 | 0.600 | 0.012 | 0.708 |
| observed_action_within_negative | generated_first_16 | 24 | generated_prefix16_text | 175 | 0.587 | 0.706 | 0.706 | 0.118 | 0.790 |
| observed_action_within_negative | generated_first_16 | 32 | generated_prefix16_text | 175 | 0.587 | 0.661 | 0.698 | 0.111 | 0.813 |
| observed_action_within_negative | generated_first_16 | 40 | generated_prefix16_text | 175 | 0.587 | 0.689 | 0.696 | 0.108 | 0.758 |
| observed_action_within_negative | generated_full | 16 | generated_text | 175 | 0.778 | 0.753 | 0.764 | -0.015 | 0.744 |
| observed_action_within_negative | generated_full | 24 | generated_text | 175 | 0.778 | 0.803 | 0.809 | 0.031 | 0.784 |
| observed_action_within_negative | generated_full | 32 | generated_text | 175 | 0.778 | 0.735 | 0.791 | 0.013 | 0.778 |
| observed_action_within_negative | generated_full | 40 | generated_text | 175 | 0.778 | 0.782 | 0.784 | 0.006 | 0.813 |

## Readout

- If text-only AUROC is already near the activation AUROC, the probe is lexically vulnerable.
- If `combined delta` is near zero, activation adds little beyond text for that target/slice.
- Full-response generated activations are the highest-risk slice because the behavior labeler also reads the full response text.
