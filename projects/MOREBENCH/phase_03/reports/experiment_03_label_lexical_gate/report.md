# Experiment 03 Label Lexical Gate

Exploratory lexical preflight for benchmark-native MoReBench labels. Theory-augmentation labels are excluded.

Important: the dilemma-family and structural-axis labels in this report are heuristic prompt-text flags, not gold annotations. Their purpose is to map leakage and split viability before deciding whether to invest in real annotation.

## Triage Summary

- `augmentation_candidate`: `6`
- `borderline`: `7`
- `needs_split_review`: `1`
- `probe_candidate`: `10`
- `unsupported`: `8`

## Supported Labels

| label | type | triage | counts | primary BA | primary AUROC | CV BA | final/prompt BA |
|---|---:|---:|---:|---:|---:|---:|---:|
| heuristic_helpful_harm_dynamic | generation | probe_candidate | harm_only=151, both=247, helpful_only=59, neither=43 | 0.338 |  | 0.374 | 0.324 |
| heuristic_commitment_style | generation | probe_candidate | defer_or_refuse=129, qualified_recommendation=326, direct_recommendation=45 | 0.442 |  | 0.383 | 0.482 |
| strong_helpful | generation | probe_candidate | True=371, False=129 | 0.516 | 0.583 | 0.573 | 0.490 |
| heuristic_tradeoff_engagement | generation | probe_candidate | True=418, False=82 | 0.519 | 0.587 | 0.540 | 0.554 |
| helpful_harmless_off_diagonal | generation | probe_candidate | harmless_over_helpful=57, helpful_over_harmless=33 | 0.586 | 0.579 | 0.585 | 0.525 |
| heuristic_refuses_or_hedges | generation | probe_candidate | True=129, False=371 | 0.590 | 0.624 | 0.550 | 0.626 |
| heuristic_helpfulness_invoked | generation | probe_candidate | False=205, True=295 | 0.596 | 0.599 | 0.552 | 0.569 |
| heuristic_harm_avoidance_invoked | generation | probe_candidate | True=332, False=168 | 0.600 | 0.687 | 0.644 | 0.612 |
| authority_constraint | prompt | probe_candidate | True=126, False=374 | 0.622 | 0.699 | 0.715 |  |
| strong_harmless | generation | probe_candidate | True=395, False=105 | 0.638 | 0.735 | 0.592 | 0.588 |
| dilemma_type | prompt | augmentation_candidate | long_case=200, short_case=200, expert_case=100 | 0.385 |  | 0.973 |  |
| fairness_access_conflict | prompt | augmentation_candidate | False=418, True=82 | 0.548 | 0.694 | 0.665 |  |
| public_safety | prompt | augmentation_candidate | False=174, True=326 | 0.601 | 0.750 | 0.761 |  |
| privacy_monitoring_conflict | prompt | augmentation_candidate | False=325, True=175 | 0.650 | 0.784 | 0.769 |  |
| close_relationship_obligation | prompt | augmentation_candidate | True=190, False=310 | 0.665 | 0.782 | 0.842 |  |
| loyalty_relationship | prompt | augmentation_candidate | True=254, False=246 | 0.695 | 0.783 | 0.804 |  |
| autonomy_boundary | prompt | borderline | True=142, False=358 | 0.653 | 0.747 | 0.677 |  |
| heuristic_uncertainty_and_scope_calibration | generation | borderline | False=381, True=119 | 0.671 | 0.791 | 0.737 | 0.594 |
| rubric_tailored_actionability | generation | borderline | True=444, False=56 | 0.673 | 0.813 | 0.626 | 0.649 |
| public_harm_resource_tradeoff | prompt | borderline | False=298, True=202 | 0.683 | 0.775 | 0.766 |  |
| uncertainty_incomplete_info | prompt | borderline | True=174, False=326 | 0.688 | 0.778 | 0.784 |  |
| disclosure_transparency_conflict | prompt | borderline | False=373, True=127 | 0.710 | 0.811 | 0.689 |  |
| institutional_policy_constraint | prompt | borderline | True=231, False=269 | 0.714 | 0.844 | 0.775 |  |
| role_domain | prompt | needs_split_review | ai_advisor=293, ai_agent=207 |  |  | 0.980 |  |

## Unsupported / Low-Support Labels

- `source_family` (prompt): daily_dilemmas=200, expert_written_ethic_bowl=51, ai_risk_dilemmas=200, expert_written_ethic_unwrapped=30, expert_written_literature=12, expert_written_collab=7
- `context` (prompt): Right & Duty & Justice=44, Interpersonal relationship=66, Business & Workplace=59, Bioethics & Healthcare=48, Animal & Environment=37, Sports=15, Media & Journalism=13, Science & Techonology=90, Education=35, Entertainment=14, Organization=11, Art & Culture=6, Professional Ethics=28, Others=8, Transport=14, Religion=12
- `helpful_binary` (generation): True=479, False=21
- `harmless_binary` (generation): True=451, False=49
- `helpful_harmless_cell` (generation): h3_s3=338, h2_s2=28, h1_s1=6, h1_s0=10, h2_s3=54, h2_s1=20, h3_s2=26, h1_s2=2, h3_s1=4, h2_s0=6, h3_s0=3, h1_s3=3
- `rubric_recommendation_clarity` (generation): True=492, False=8
- `rubric_tradeoff_balance` (generation): True=499, False=1
- `rubric_harm_vector_addressed` (generation): True=492, False=8

## Notes

- `primary BA` is source-family holdout balanced accuracy for char-TFIDF on the relevant text surface: prompt text for prompt labels, response text for generation labels.
- `CV BA` is random stratified CV on the same surface and feature class.
- `final/prompt BA` is final-span char-TFIDF for generation labels, and dilemma-text char-TFIDF for prompt labels.
- Labels marked `probe_candidate` still need human review of the label definition before activation capture if they are heuristic or support annotations.
