# MoReBench Phase 01 Design

## Goal

Translate `MoReBench` from benchmark-native surfaces into a first-pass mechanistically useful label ontology.

## Deliverables

- `outputs/candidate_mechanistic_questions.json`
- `outputs/prompt_side_labels.json`
- `outputs/response_side_labels.json`
- `outputs/validation_signals.json`
- `outputs/nuisance_variables.json`
- `outputs/follow_on_data_plan.json`
- `outputs/recommended_first_experiments.json`
- `outputs/frozen_label_set_placeholder.json`
- `specs/labeling-functions.md`
- `reports/gap-list.md`
- `reports/phase_01_benchmark_to_latent_labels.md`

## Scope

- extract implicit mechanistic questions from the benchmark design itself
- preserve prompt-side vs response-side separation
- demote grader-designed rubric surfaces to validation when appropriate
- identify which labels are direct, derived, deferred, or nuisance-only
- keep helpfulness and harm avoidance separable unless data later justifies a joint relation label

## Explicit Non-Goals

- no fresh-generation labeling yet
- no probe or intervention methodology writeup
- no expansion into augmentation or intervention phases
