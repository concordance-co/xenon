# default

- template: `default`
- input_count: 6
- example_count: 30
- manifest: `assets/manifest.json`
- summary: `summary.json`

## Inputs

### generate_theory_primed_responses

- artifact_id: `generation_run_1_35985c0c3945`
- artifact_kind: `generation_run`
- provenance: run `wr_a3343c545c25_4f4ca71f` / step `generate_theory_primed_responses` / index `0`
- runtime: runner `modal` / app `ap-5q11wbBvfZWGU7nanLd1sD`
- results: `results/generate_theory_primed_responses_results.json`

### build_theory_persistence_capture_dataset

- artifact_id: `transform_1_2e9298c2`
- artifact_kind: `transform`
- provenance: run `wr_a3343c545c25_4f4ca71f` / step `build_theory_persistence_capture_dataset` / index `1`
- runtime: runner `modal` / app `ap-UHSKioguPiiSstHolBJZp7`
- results: `results/build_theory_persistence_capture_dataset_results.json`

### capture_generated_sequence_residual

- artifact_id: `capture_1_9e63d326abde`
- artifact_kind: `capture`
- provenance: run `wr_a3343c545c25_4f4ca71f` / step `capture_generated_sequence_residual` / index `2`
- runtime: runner `modal` / app `ap-PxaG0gXeZFyLD0CtGMxUFA`

### text_baseline_generation_prime_condition

- artifact_id: `text_baseline_1_ad6b1575`
- artifact_kind: `text_baseline`
- provenance: run `wr_a3343c545c25_4f4ca71f` / step `text_baseline_generation_prime_condition` / index `3`
- runtime: runner `modal` / app `ap-DgyhoBku3obaR772SjbE7M`
- results: `results/text_baseline_generation_prime_condition_results.json`
- table: `tables/text_baseline_generation_prime_condition.json`
- headline_metrics:
```json
{
  "best_cross_transfer_balanced_accuracy": null,
  "best_split_balanced_accuracy": {
    "cohort": null,
    "direction": null,
    "split_name": "split",
    "value": 0.6
  },
  "example_count": 23,
  "mode": "split_holdout",
  "model": "countvectorizer_logreg"
}
```
- figures:
  - `assets/text_baseline_generation_prime_condition/split_balanced_accuracy.png` (primary): Balanced Accuracy Split

### probe_generation_prime_condition_residual

- artifact_id: `probe_1_406968a4`
- artifact_kind: `probe`
- provenance: run `wr_a3343c545c25_4f4ca71f` / step `probe_generation_prime_condition_residual` / index `4`
- runtime: runner `modal` / app `ap-mvlzFHBj1uRiUcacPenarU`
- results: `results/probe_generation_prime_condition_residual_results.json`
- table: `tables/probe_generation_prime_condition_residual.json`
- headline_metrics:
```json
{
  "best_layer": 8,
  "best_metric": "balanced_accuracy",
  "best_value": 0.9,
  "example_count": 23,
  "group_count": 5,
  "split_mode": "fixed"
}
```
- figures:
  - `assets/probe_generation_prime_condition_residual/balanced_accuracy_by_layer.png` (primary): Balanced Accuracy by layer
  - `assets/probe_generation_prime_condition_residual/accuracy_by_layer.png`: Accuracy by layer
  - `assets/probe_generation_prime_condition_residual/probe_metrics_by_layer.png`: Probe metrics by layer

### summarize_capture_smoke

- artifact_id: `transform_1_5714e0c7`
- artifact_kind: `transform`
- provenance: run `wr_a3343c545c25_4f4ca71f` / step `summarize_capture_smoke` / index `5`
- runtime: runner `modal` / app `ap-9iF5a1lZrvuBQFpw8aZZBO`
- results: `results/summarize_capture_smoke_results.json`

## Summary

```json
{
  "example_count": 30,
  "figures": {
    "probe_generation_prime_condition_residual/accuracy_by_layer": {
      "caption": "Accuracy across captured layers for probe step probe_generation_prime_condition_residual.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "probe_generation_prime_condition_residual/accuracy_by_layer",
      "path": "assets/probe_generation_prime_condition_residual/accuracy_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "probe_generation_prime_condition_residual",
      "title": "Accuracy by layer"
    },
    "probe_generation_prime_condition_residual/balanced_accuracy_by_layer": {
      "caption": "Balanced Accuracy across captured layers for probe step probe_generation_prime_condition_residual.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "probe_generation_prime_condition_residual/balanced_accuracy_by_layer",
      "path": "assets/probe_generation_prime_condition_residual/balanced_accuracy_by_layer.png",
      "primary": true,
      "result_kind": "probe_result",
      "step_name": "probe_generation_prime_condition_residual",
      "title": "Balanced Accuracy by layer"
    },
    "probe_generation_prime_condition_residual/probe_metrics_by_layer": {
      "caption": "Available probe metrics across captured layers for step probe_generation_prime_condition_residual.",
      "chart_kind": "probe_metrics_by_layer",
      "figure_id": "probe_generation_prime_condition_residual/probe_metrics_by_layer",
      "path": "assets/probe_generation_prime_condition_residual/probe_metrics_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "probe_generation_prime_condition_residual",
      "title": "Probe metrics by layer"
    },
    "text_baseline_generation_prime_condition/split_balanced_accuracy": {
      "caption": "Balanced Accuracy for text split split in step text_baseline_generation_prime_condition.",
      "chart_kind": "text_split_metric",
      "figure_id": "text_baseline_generation_prime_condition/split_balanced_accuracy",
      "path": "assets/text_baseline_generation_prime_condition/split_balanced_accuracy.png",
      "primary": true,
      "result_kind": "text_baseline_result",
      "step_name": "text_baseline_generation_prime_condition",
      "title": "Balanced Accuracy Split"
    }
  },
  "input_count": 6,
  "step_summaries": {
    "probe_generation_prime_condition_residual": {
      "headline_metrics": {
        "best_layer": 8,
        "best_metric": "balanced_accuracy",
        "best_value": 0.9,
        "example_count": 23,
        "group_count": 5,
        "split_mode": "fixed"
      },
      "kind": "probe_result",
      "primary_figure_id": "probe_generation_prime_condition_residual/balanced_accuracy_by_layer",
      "step_name": "probe_generation_prime_condition_residual",
      "table_path": "tables/probe_generation_prime_condition_residual.json"
    },
    "text_baseline_generation_prime_condition": {
      "headline_metrics": {
        "best_cross_transfer_balanced_accuracy": null,
        "best_split_balanced_accuracy": {
          "cohort": null,
          "direction": null,
          "split_name": "split",
          "value": 0.6
        },
        "example_count": 23,
        "mode": "split_holdout",
        "model": "countvectorizer_logreg"
      },
      "kind": "text_baseline_result",
      "primary_figure_id": "text_baseline_generation_prime_condition/split_balanced_accuracy",
      "step_name": "text_baseline_generation_prime_condition",
      "table_path": "tables/text_baseline_generation_prime_condition.json"
    }
  },
  "tables": {
    "probe_generation_prime_condition_residual": {
      "columns": [
        "accuracy",
        "balanced_accuracy",
        "baseline_majority",
        "baseline_shuffled",
        "class_count",
        "example_count",
        "layer",
        "selectivity",
        "split_mode"
      ],
      "path": "tables/probe_generation_prime_condition_residual.json",
      "result_kind": "probe_result",
      "rows": 6,
      "step_name": "probe_generation_prime_condition_residual"
    },
    "text_baseline_generation_prime_condition": {
      "columns": [
        "C",
        "auroc",
        "balanced_accuracy",
        "cohort",
        "cross_transfer_auroc",
        "cross_transfer_balanced_accuracy",
        "direction",
        "model",
        "row_kind",
        "split_name",
        "transfer_delta_balanced_accuracy",
        "within_baseline_auroc",
        "within_baseline_balanced_accuracy"
      ],
      "path": "tables/text_baseline_generation_prime_condition.json",
      "result_kind": "text_baseline_result",
      "rows": 1,
      "step_name": "text_baseline_generation_prime_condition"
    }
  },
  "template": "default"
}
```
