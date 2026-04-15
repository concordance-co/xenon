# default

- template: `default`
- input_count: 1
- example_count: 864
- manifest: `assets/manifest.json`
- summary: `summary.json`

## Inputs

### cross_dimension_similarity

- artifact_id: `transfer_probe_1_12051749`
- artifact_kind: `transfer_probe`
- provenance: run `wr_8810ccd5558b_2de7fc4d` / step `cross_dimension_similarity` / index `2`
- runtime: runner `modal` / app `ap-hJQTT8R5FiNvPwGw84hxsa`
- results: `results/cross_dimension_similarity_results.json`
- table: `tables/cross_dimension_similarity.json`
- headline_metrics:
```json
{
  "best_cross_transfer_balanced_accuracy": {
    "cohort": "trading_activity",
    "direction": "trade_size_to_trading_activity",
    "layer": 44,
    "split_name": null,
    "value": 0.5896
  },
  "best_split_balanced_accuracy": null,
  "cohort_count": 2,
  "layer_count": 12,
  "max_direction_similarity": {
    "cohort": null,
    "direction": "trade_size_vs_trading_activity",
    "layer": 32,
    "split_name": null,
    "value": 0.0835
  },
  "mode": "cross_cohort_transfer",
  "regularization": [
    1.0
  ]
}
```
- figures:
  - `assets/cross_dimension_similarity/balanced_accuracy_cross_cohort.png` (primary): Balanced Accuracy cross cohort
  - `assets/cross_dimension_similarity/auroc_cross_cohort.png`: AUROC cross cohort
  - `assets/cross_dimension_similarity/transfer_delta_balanced_accuracy.png`: Transfer delta vs within baseline
  - `assets/cross_dimension_similarity/direction_similarity.png`: Direction similarity

## Summary

```json
{
  "example_count": 864,
  "figures": {
    "cross_dimension_similarity/auroc_cross_cohort": {
      "caption": "AUROC across layers for cross-cohort transfer in step cross_dimension_similarity.",
      "chart_kind": "transfer_cross_cohort_metric",
      "figure_id": "cross_dimension_similarity/auroc_cross_cohort",
      "path": "assets/cross_dimension_similarity/auroc_cross_cohort.png",
      "primary": false,
      "result_kind": "transfer_probe_result",
      "step_name": "cross_dimension_similarity",
      "title": "AUROC cross cohort"
    },
    "cross_dimension_similarity/balanced_accuracy_cross_cohort": {
      "caption": "Balanced Accuracy across layers for cross-cohort transfer in step cross_dimension_similarity.",
      "chart_kind": "transfer_cross_cohort_metric",
      "figure_id": "cross_dimension_similarity/balanced_accuracy_cross_cohort",
      "path": "assets/cross_dimension_similarity/balanced_accuracy_cross_cohort.png",
      "primary": true,
      "result_kind": "transfer_probe_result",
      "step_name": "cross_dimension_similarity",
      "title": "Balanced Accuracy cross cohort"
    },
    "cross_dimension_similarity/direction_similarity": {
      "caption": "Per-layer direction similarity for transfer comparisons in step cross_dimension_similarity.",
      "chart_kind": "transfer_direction_similarity",
      "figure_id": "cross_dimension_similarity/direction_similarity",
      "path": "assets/cross_dimension_similarity/direction_similarity.png",
      "primary": false,
      "result_kind": "transfer_probe_result",
      "step_name": "cross_dimension_similarity",
      "title": "Direction similarity"
    },
    "cross_dimension_similarity/transfer_delta_balanced_accuracy": {
      "caption": "Balanced-accuracy transfer delta against the test-cohort within baseline for step cross_dimension_similarity.",
      "chart_kind": "transfer_delta_by_layer",
      "figure_id": "cross_dimension_similarity/transfer_delta_balanced_accuracy",
      "path": "assets/cross_dimension_similarity/transfer_delta_balanced_accuracy.png",
      "primary": false,
      "result_kind": "transfer_probe_result",
      "step_name": "cross_dimension_similarity",
      "title": "Transfer delta vs within baseline"
    }
  },
  "input_count": 1,
  "step_summaries": {
    "cross_dimension_similarity": {
      "headline_metrics": {
        "best_cross_transfer_balanced_accuracy": {
          "cohort": "trading_activity",
          "direction": "trade_size_to_trading_activity",
          "layer": 44,
          "split_name": null,
          "value": 0.5896
        },
        "best_split_balanced_accuracy": null,
        "cohort_count": 2,
        "layer_count": 12,
        "max_direction_similarity": {
          "cohort": null,
          "direction": "trade_size_vs_trading_activity",
          "layer": 32,
          "split_name": null,
          "value": 0.0835
        },
        "mode": "cross_cohort_transfer",
        "regularization": [
          1.0
        ]
      },
      "kind": "transfer_probe_result",
      "primary_figure_id": "cross_dimension_similarity/balanced_accuracy_cross_cohort",
      "table_path": "tables/cross_dimension_similarity.json"
    }
  },
  "tables": {
    "cross_dimension_similarity": {
      "path": "tables/cross_dimension_similarity.json",
      "result_kind": "transfer_probe_result",
      "step_name": "cross_dimension_similarity"
    }
  },
  "template": "default"
}
```
