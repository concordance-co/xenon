# default

- template: `default`
- input_count: 17
- example_count: 500
- manifest: `assets/manifest.json`
- summary: `summary.json`

## Inputs

### capture_response_label_residuals

- artifact_id: `capture_1_f2a9e4531dec`
- artifact_kind: `capture`
- provenance: run `wr_5e81c613af54_54fc9204` / step `capture_response_label_residuals` / index `0`
- runtime: runner `modal` / app `ap-ornV5vmzD21qczGamOZ3yz`

### summarize_capture

- artifact_id: `transform_1_e40d6e62`
- artifact_kind: `transform`
- provenance: run `wr_5e81c613af54_54fc9204` / step `summarize_capture` / index `1`
- runtime: runner `modal` / app `ap-59dxIgoBt6bBaKp7Pdn2ZL`
- results: `results/summarize_capture_results.json`

### report_probe_helpful_harmless_off_diagonal_prompt_end

- artifact_id: `transform_469a52e972f0_ba224958`
- artifact_kind: `transform`
- provenance: run `wr_1978c6512339_7ce43404` / step `report_probe_helpful_harmless_off_diagonal_prompt_end` / index `2`
- runtime: runner `local`
- results: `results/report_probe_helpful_harmless_off_diagonal_prompt_end_results.json`
- table: `tables/report_probe_helpful_harmless_off_diagonal_prompt_end.json`
- headline_metrics:
```json
{
  "best_layer": 16,
  "best_metric": "source_family_holdout_balanced_accuracy",
  "best_value": 0.6408730158730158,
  "example_count": 90,
  "group_count": null,
  "split_mode": "source_family_holdout"
}
```
- figures:
  - `assets/report_probe_helpful_harmless_off_diagonal_prompt_end/balanced_accuracy_by_layer.png` (primary): Balanced Accuracy by layer
  - `assets/report_probe_helpful_harmless_off_diagonal_prompt_end/auroc_by_layer.png`: AUROC by layer
  - `assets/report_probe_helpful_harmless_off_diagonal_prompt_end/probe_metrics_by_layer.png`: Probe metrics by layer

### report_probe_helpful_harmless_off_diagonal_generated_first_third

- artifact_id: `transform_40dae0c3b937_80608250`
- artifact_kind: `transform`
- provenance: run `wr_1978c6512339_7ce43404` / step `report_probe_helpful_harmless_off_diagonal_generated_first_third` / index `3`
- runtime: runner `local`
- results: `results/report_probe_helpful_harmless_off_diagonal_generated_first_third_results.json`
- table: `tables/report_probe_helpful_harmless_off_diagonal_generated_first_third.json`
- headline_metrics:
```json
{
  "best_layer": 36,
  "best_metric": "source_family_holdout_balanced_accuracy",
  "best_value": 0.6175595238095237,
  "example_count": 90,
  "group_count": null,
  "split_mode": "source_family_holdout"
}
```
- figures:
  - `assets/report_probe_helpful_harmless_off_diagonal_generated_first_third/balanced_accuracy_by_layer.png` (primary): Balanced Accuracy by layer
  - `assets/report_probe_helpful_harmless_off_diagonal_generated_first_third/auroc_by_layer.png`: AUROC by layer
  - `assets/report_probe_helpful_harmless_off_diagonal_generated_first_third/probe_metrics_by_layer.png`: Probe metrics by layer

### report_probe_helpful_harmless_off_diagonal_generated_middle_third

- artifact_id: `transform_76e17851e357_81e9978a`
- artifact_kind: `transform`
- provenance: run `wr_1978c6512339_7ce43404` / step `report_probe_helpful_harmless_off_diagonal_generated_middle_third` / index `4`
- runtime: runner `local`
- results: `results/report_probe_helpful_harmless_off_diagonal_generated_middle_third_results.json`
- table: `tables/report_probe_helpful_harmless_off_diagonal_generated_middle_third.json`
- headline_metrics:
```json
{
  "best_layer": 44,
  "best_metric": "source_family_holdout_balanced_accuracy",
  "best_value": 0.621031746031746,
  "example_count": 90,
  "group_count": null,
  "split_mode": "source_family_holdout"
}
```
- figures:
  - `assets/report_probe_helpful_harmless_off_diagonal_generated_middle_third/balanced_accuracy_by_layer.png` (primary): Balanced Accuracy by layer
  - `assets/report_probe_helpful_harmless_off_diagonal_generated_middle_third/auroc_by_layer.png`: AUROC by layer
  - `assets/report_probe_helpful_harmless_off_diagonal_generated_middle_third/probe_metrics_by_layer.png`: Probe metrics by layer

### report_probe_helpful_harmless_off_diagonal_generated_last_third

- artifact_id: `transform_8c94fbead150_7d0e57cf`
- artifact_kind: `transform`
- provenance: run `wr_1978c6512339_7ce43404` / step `report_probe_helpful_harmless_off_diagonal_generated_last_third` / index `5`
- runtime: runner `local`
- results: `results/report_probe_helpful_harmless_off_diagonal_generated_last_third_results.json`
- table: `tables/report_probe_helpful_harmless_off_diagonal_generated_last_third.json`
- headline_metrics:
```json
{
  "best_layer": 44,
  "best_metric": "source_family_holdout_balanced_accuracy",
  "best_value": 0.6284722222222222,
  "example_count": 90,
  "group_count": null,
  "split_mode": "source_family_holdout"
}
```
- figures:
  - `assets/report_probe_helpful_harmless_off_diagonal_generated_last_third/balanced_accuracy_by_layer.png` (primary): Balanced Accuracy by layer
  - `assets/report_probe_helpful_harmless_off_diagonal_generated_last_third/auroc_by_layer.png`: AUROC by layer
  - `assets/report_probe_helpful_harmless_off_diagonal_generated_last_third/probe_metrics_by_layer.png`: Probe metrics by layer

### report_probe_helpful_harmless_off_diagonal_generated_total

- artifact_id: `transform_6f8f2e0fc725_231c66f4`
- artifact_kind: `transform`
- provenance: run `wr_1978c6512339_7ce43404` / step `report_probe_helpful_harmless_off_diagonal_generated_total` / index `6`
- runtime: runner `local`
- results: `results/report_probe_helpful_harmless_off_diagonal_generated_total_results.json`
- table: `tables/report_probe_helpful_harmless_off_diagonal_generated_total.json`
- headline_metrics:
```json
{
  "best_layer": 44,
  "best_metric": "source_family_holdout_balanced_accuracy",
  "best_value": 0.6453373015873016,
  "example_count": 90,
  "group_count": null,
  "split_mode": "source_family_holdout"
}
```
- figures:
  - `assets/report_probe_helpful_harmless_off_diagonal_generated_total/balanced_accuracy_by_layer.png` (primary): Balanced Accuracy by layer
  - `assets/report_probe_helpful_harmless_off_diagonal_generated_total/auroc_by_layer.png`: AUROC by layer
  - `assets/report_probe_helpful_harmless_off_diagonal_generated_total/probe_metrics_by_layer.png`: Probe metrics by layer

### report_probe_strong_helpful_prompt_end

- artifact_id: `transform_d67917336d3c_7f211e33`
- artifact_kind: `transform`
- provenance: run `wr_1978c6512339_7ce43404` / step `report_probe_strong_helpful_prompt_end` / index `7`
- runtime: runner `local`
- results: `results/report_probe_strong_helpful_prompt_end_results.json`
- table: `tables/report_probe_strong_helpful_prompt_end.json`
- headline_metrics:
```json
{
  "best_layer": 36,
  "best_metric": "source_family_holdout_balanced_accuracy",
  "best_value": 0.5922926906258534,
  "example_count": 500,
  "group_count": null,
  "split_mode": "source_family_holdout"
}
```
- figures:
  - `assets/report_probe_strong_helpful_prompt_end/balanced_accuracy_by_layer.png` (primary): Balanced Accuracy by layer
  - `assets/report_probe_strong_helpful_prompt_end/auroc_by_layer.png`: AUROC by layer
  - `assets/report_probe_strong_helpful_prompt_end/probe_metrics_by_layer.png`: Probe metrics by layer

### report_probe_strong_helpful_generated_first_third

- artifact_id: `transform_d22ee5879406_ff552c1c`
- artifact_kind: `transform`
- provenance: run `wr_1978c6512339_7ce43404` / step `report_probe_strong_helpful_generated_first_third` / index `8`
- runtime: runner `local`
- results: `results/report_probe_strong_helpful_generated_first_third_results.json`
- table: `tables/report_probe_strong_helpful_generated_first_third.json`
- headline_metrics:
```json
{
  "best_layer": 36,
  "best_metric": "source_family_holdout_balanced_accuracy",
  "best_value": 0.5592526406446845,
  "example_count": 500,
  "group_count": null,
  "split_mode": "source_family_holdout"
}
```
- figures:
  - `assets/report_probe_strong_helpful_generated_first_third/balanced_accuracy_by_layer.png` (primary): Balanced Accuracy by layer
  - `assets/report_probe_strong_helpful_generated_first_third/auroc_by_layer.png`: AUROC by layer
  - `assets/report_probe_strong_helpful_generated_first_third/probe_metrics_by_layer.png`: Probe metrics by layer

### report_probe_strong_helpful_generated_middle_third

- artifact_id: `transform_bb08318c8e19_ed32b76f`
- artifact_kind: `transform`
- provenance: run `wr_1978c6512339_7ce43404` / step `report_probe_strong_helpful_generated_middle_third` / index `9`
- runtime: runner `local`
- results: `results/report_probe_strong_helpful_generated_middle_third_results.json`
- table: `tables/report_probe_strong_helpful_generated_middle_third.json`
- headline_metrics:
```json
{
  "best_layer": 8,
  "best_metric": "source_family_holdout_balanced_accuracy",
  "best_value": 0.5741081161753798,
  "example_count": 500,
  "group_count": null,
  "split_mode": "source_family_holdout"
}
```
- figures:
  - `assets/report_probe_strong_helpful_generated_middle_third/balanced_accuracy_by_layer.png` (primary): Balanced Accuracy by layer
  - `assets/report_probe_strong_helpful_generated_middle_third/auroc_by_layer.png`: AUROC by layer
  - `assets/report_probe_strong_helpful_generated_middle_third/probe_metrics_by_layer.png`: Probe metrics by layer

### report_probe_strong_helpful_generated_last_third

- artifact_id: `transform_4713b9e99669_b482e85d`
- artifact_kind: `transform`
- provenance: run `wr_1978c6512339_7ce43404` / step `report_probe_strong_helpful_generated_last_third` / index `10`
- runtime: runner `local`
- results: `results/report_probe_strong_helpful_generated_last_third_results.json`
- table: `tables/report_probe_strong_helpful_generated_last_third.json`
- headline_metrics:
```json
{
  "best_layer": 8,
  "best_metric": "source_family_holdout_balanced_accuracy",
  "best_value": 0.5898776693344356,
  "example_count": 500,
  "group_count": null,
  "split_mode": "source_family_holdout"
}
```
- figures:
  - `assets/report_probe_strong_helpful_generated_last_third/balanced_accuracy_by_layer.png` (primary): Balanced Accuracy by layer
  - `assets/report_probe_strong_helpful_generated_last_third/auroc_by_layer.png`: AUROC by layer
  - `assets/report_probe_strong_helpful_generated_last_third/probe_metrics_by_layer.png`: Probe metrics by layer

### report_probe_strong_helpful_generated_total

- artifact_id: `transform_a0bec147cb22_3db5fbea`
- artifact_kind: `transform`
- provenance: run `wr_1978c6512339_7ce43404` / step `report_probe_strong_helpful_generated_total` / index `11`
- runtime: runner `local`
- results: `results/report_probe_strong_helpful_generated_total_results.json`
- table: `tables/report_probe_strong_helpful_generated_total.json`
- headline_metrics:
```json
{
  "best_layer": 16,
  "best_metric": "source_family_holdout_balanced_accuracy",
  "best_value": 0.6100958182916448,
  "example_count": 500,
  "group_count": null,
  "split_mode": "source_family_holdout"
}
```
- figures:
  - `assets/report_probe_strong_helpful_generated_total/balanced_accuracy_by_layer.png` (primary): Balanced Accuracy by layer
  - `assets/report_probe_strong_helpful_generated_total/auroc_by_layer.png`: AUROC by layer
  - `assets/report_probe_strong_helpful_generated_total/probe_metrics_by_layer.png`: Probe metrics by layer

### report_probe_strong_harmless_prompt_end

- artifact_id: `transform_5b52cb7ea253_8bd01cb3`
- artifact_kind: `transform`
- provenance: run `wr_1978c6512339_7ce43404` / step `report_probe_strong_harmless_prompt_end` / index `12`
- runtime: runner `local`
- results: `results/report_probe_strong_harmless_prompt_end_results.json`
- table: `tables/report_probe_strong_harmless_prompt_end.json`
- headline_metrics:
```json
{
  "best_layer": 28,
  "best_metric": "source_family_holdout_balanced_accuracy",
  "best_value": 0.5735146366440623,
  "example_count": 500,
  "group_count": null,
  "split_mode": "source_family_holdout"
}
```
- figures:
  - `assets/report_probe_strong_harmless_prompt_end/balanced_accuracy_by_layer.png` (primary): Balanced Accuracy by layer
  - `assets/report_probe_strong_harmless_prompt_end/auroc_by_layer.png`: AUROC by layer
  - `assets/report_probe_strong_harmless_prompt_end/probe_metrics_by_layer.png`: Probe metrics by layer

### report_probe_strong_harmless_generated_first_third

- artifact_id: `transform_da8a97388617_6eb8304a`
- artifact_kind: `transform`
- provenance: run `wr_1978c6512339_7ce43404` / step `report_probe_strong_harmless_generated_first_third` / index `13`
- runtime: runner `local`
- results: `results/report_probe_strong_harmless_generated_first_third_results.json`
- table: `tables/report_probe_strong_harmless_generated_first_third.json`
- headline_metrics:
```json
{
  "best_layer": 0,
  "best_metric": "source_family_holdout_balanced_accuracy",
  "best_value": 0.5506750126021369,
  "example_count": 500,
  "group_count": null,
  "split_mode": "source_family_holdout"
}
```
- figures:
  - `assets/report_probe_strong_harmless_generated_first_third/balanced_accuracy_by_layer.png` (primary): Balanced Accuracy by layer
  - `assets/report_probe_strong_harmless_generated_first_third/auroc_by_layer.png`: AUROC by layer
  - `assets/report_probe_strong_harmless_generated_first_third/probe_metrics_by_layer.png`: Probe metrics by layer

### report_probe_strong_harmless_generated_middle_third

- artifact_id: `transform_76ad273f18d3_6e0cdf9c`
- artifact_kind: `transform`
- provenance: run `wr_1978c6512339_7ce43404` / step `report_probe_strong_harmless_generated_middle_third` / index `14`
- runtime: runner `local`
- results: `results/report_probe_strong_harmless_generated_middle_third_results.json`
- table: `tables/report_probe_strong_harmless_generated_middle_third.json`
- headline_metrics:
```json
{
  "best_layer": 4,
  "best_metric": "source_family_holdout_balanced_accuracy",
  "best_value": 0.6090817732512264,
  "example_count": 500,
  "group_count": null,
  "split_mode": "source_family_holdout"
}
```
- figures:
  - `assets/report_probe_strong_harmless_generated_middle_third/balanced_accuracy_by_layer.png` (primary): Balanced Accuracy by layer
  - `assets/report_probe_strong_harmless_generated_middle_third/auroc_by_layer.png`: AUROC by layer
  - `assets/report_probe_strong_harmless_generated_middle_third/probe_metrics_by_layer.png`: Probe metrics by layer

### report_probe_strong_harmless_generated_last_third

- artifact_id: `transform_1a7d61893d27_09a52df7`
- artifact_kind: `transform`
- provenance: run `wr_1978c6512339_7ce43404` / step `report_probe_strong_harmless_generated_last_third` / index `15`
- runtime: runner `local`
- results: `results/report_probe_strong_harmless_generated_last_third_results.json`
- table: `tables/report_probe_strong_harmless_generated_last_third.json`
- headline_metrics:
```json
{
  "best_layer": 0,
  "best_metric": "source_family_holdout_balanced_accuracy",
  "best_value": 0.663826232994211,
  "example_count": 500,
  "group_count": null,
  "split_mode": "source_family_holdout"
}
```
- figures:
  - `assets/report_probe_strong_harmless_generated_last_third/balanced_accuracy_by_layer.png` (primary): Balanced Accuracy by layer
  - `assets/report_probe_strong_harmless_generated_last_third/auroc_by_layer.png`: AUROC by layer
  - `assets/report_probe_strong_harmless_generated_last_third/probe_metrics_by_layer.png`: Probe metrics by layer

### report_probe_strong_harmless_generated_total

- artifact_id: `transform_fc724fb65cdd_7b93fbcf`
- artifact_kind: `transform`
- provenance: run `wr_1978c6512339_7ce43404` / step `report_probe_strong_harmless_generated_total` / index `16`
- runtime: runner `local`
- results: `results/report_probe_strong_harmless_generated_total_results.json`
- table: `tables/report_probe_strong_harmless_generated_total.json`
- headline_metrics:
```json
{
  "best_layer": 44,
  "best_metric": "source_family_holdout_balanced_accuracy",
  "best_value": 0.6142569585650821,
  "example_count": 500,
  "group_count": null,
  "split_mode": "source_family_holdout"
}
```
- figures:
  - `assets/report_probe_strong_harmless_generated_total/balanced_accuracy_by_layer.png` (primary): Balanced Accuracy by layer
  - `assets/report_probe_strong_harmless_generated_total/auroc_by_layer.png`: AUROC by layer
  - `assets/report_probe_strong_harmless_generated_total/probe_metrics_by_layer.png`: Probe metrics by layer

## Summary

```json
{
  "example_count": 500,
  "figures": {
    "report_probe_helpful_harmless_off_diagonal_generated_first_third/auroc_by_layer": {
      "caption": "AUROC across captured layers for probe step report_probe_helpful_harmless_off_diagonal_generated_first_third.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_helpful_harmless_off_diagonal_generated_first_third/auroc_by_layer",
      "path": "assets/report_probe_helpful_harmless_off_diagonal_generated_first_third/auroc_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_helpful_harmless_off_diagonal_generated_first_third",
      "title": "AUROC by layer"
    },
    "report_probe_helpful_harmless_off_diagonal_generated_first_third/balanced_accuracy_by_layer": {
      "caption": "Balanced Accuracy across captured layers for probe step report_probe_helpful_harmless_off_diagonal_generated_first_third.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_helpful_harmless_off_diagonal_generated_first_third/balanced_accuracy_by_layer",
      "path": "assets/report_probe_helpful_harmless_off_diagonal_generated_first_third/balanced_accuracy_by_layer.png",
      "primary": true,
      "result_kind": "probe_result",
      "step_name": "report_probe_helpful_harmless_off_diagonal_generated_first_third",
      "title": "Balanced Accuracy by layer"
    },
    "report_probe_helpful_harmless_off_diagonal_generated_first_third/probe_metrics_by_layer": {
      "caption": "Available probe metrics across captured layers for step report_probe_helpful_harmless_off_diagonal_generated_first_third.",
      "chart_kind": "probe_metrics_by_layer",
      "figure_id": "report_probe_helpful_harmless_off_diagonal_generated_first_third/probe_metrics_by_layer",
      "path": "assets/report_probe_helpful_harmless_off_diagonal_generated_first_third/probe_metrics_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_helpful_harmless_off_diagonal_generated_first_third",
      "title": "Probe metrics by layer"
    },
    "report_probe_helpful_harmless_off_diagonal_generated_last_third/auroc_by_layer": {
      "caption": "AUROC across captured layers for probe step report_probe_helpful_harmless_off_diagonal_generated_last_third.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_helpful_harmless_off_diagonal_generated_last_third/auroc_by_layer",
      "path": "assets/report_probe_helpful_harmless_off_diagonal_generated_last_third/auroc_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_helpful_harmless_off_diagonal_generated_last_third",
      "title": "AUROC by layer"
    },
    "report_probe_helpful_harmless_off_diagonal_generated_last_third/balanced_accuracy_by_layer": {
      "caption": "Balanced Accuracy across captured layers for probe step report_probe_helpful_harmless_off_diagonal_generated_last_third.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_helpful_harmless_off_diagonal_generated_last_third/balanced_accuracy_by_layer",
      "path": "assets/report_probe_helpful_harmless_off_diagonal_generated_last_third/balanced_accuracy_by_layer.png",
      "primary": true,
      "result_kind": "probe_result",
      "step_name": "report_probe_helpful_harmless_off_diagonal_generated_last_third",
      "title": "Balanced Accuracy by layer"
    },
    "report_probe_helpful_harmless_off_diagonal_generated_last_third/probe_metrics_by_layer": {
      "caption": "Available probe metrics across captured layers for step report_probe_helpful_harmless_off_diagonal_generated_last_third.",
      "chart_kind": "probe_metrics_by_layer",
      "figure_id": "report_probe_helpful_harmless_off_diagonal_generated_last_third/probe_metrics_by_layer",
      "path": "assets/report_probe_helpful_harmless_off_diagonal_generated_last_third/probe_metrics_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_helpful_harmless_off_diagonal_generated_last_third",
      "title": "Probe metrics by layer"
    },
    "report_probe_helpful_harmless_off_diagonal_generated_middle_third/auroc_by_layer": {
      "caption": "AUROC across captured layers for probe step report_probe_helpful_harmless_off_diagonal_generated_middle_third.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_helpful_harmless_off_diagonal_generated_middle_third/auroc_by_layer",
      "path": "assets/report_probe_helpful_harmless_off_diagonal_generated_middle_third/auroc_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_helpful_harmless_off_diagonal_generated_middle_third",
      "title": "AUROC by layer"
    },
    "report_probe_helpful_harmless_off_diagonal_generated_middle_third/balanced_accuracy_by_layer": {
      "caption": "Balanced Accuracy across captured layers for probe step report_probe_helpful_harmless_off_diagonal_generated_middle_third.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_helpful_harmless_off_diagonal_generated_middle_third/balanced_accuracy_by_layer",
      "path": "assets/report_probe_helpful_harmless_off_diagonal_generated_middle_third/balanced_accuracy_by_layer.png",
      "primary": true,
      "result_kind": "probe_result",
      "step_name": "report_probe_helpful_harmless_off_diagonal_generated_middle_third",
      "title": "Balanced Accuracy by layer"
    },
    "report_probe_helpful_harmless_off_diagonal_generated_middle_third/probe_metrics_by_layer": {
      "caption": "Available probe metrics across captured layers for step report_probe_helpful_harmless_off_diagonal_generated_middle_third.",
      "chart_kind": "probe_metrics_by_layer",
      "figure_id": "report_probe_helpful_harmless_off_diagonal_generated_middle_third/probe_metrics_by_layer",
      "path": "assets/report_probe_helpful_harmless_off_diagonal_generated_middle_third/probe_metrics_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_helpful_harmless_off_diagonal_generated_middle_third",
      "title": "Probe metrics by layer"
    },
    "report_probe_helpful_harmless_off_diagonal_generated_total/auroc_by_layer": {
      "caption": "AUROC across captured layers for probe step report_probe_helpful_harmless_off_diagonal_generated_total.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_helpful_harmless_off_diagonal_generated_total/auroc_by_layer",
      "path": "assets/report_probe_helpful_harmless_off_diagonal_generated_total/auroc_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_helpful_harmless_off_diagonal_generated_total",
      "title": "AUROC by layer"
    },
    "report_probe_helpful_harmless_off_diagonal_generated_total/balanced_accuracy_by_layer": {
      "caption": "Balanced Accuracy across captured layers for probe step report_probe_helpful_harmless_off_diagonal_generated_total.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_helpful_harmless_off_diagonal_generated_total/balanced_accuracy_by_layer",
      "path": "assets/report_probe_helpful_harmless_off_diagonal_generated_total/balanced_accuracy_by_layer.png",
      "primary": true,
      "result_kind": "probe_result",
      "step_name": "report_probe_helpful_harmless_off_diagonal_generated_total",
      "title": "Balanced Accuracy by layer"
    },
    "report_probe_helpful_harmless_off_diagonal_generated_total/probe_metrics_by_layer": {
      "caption": "Available probe metrics across captured layers for step report_probe_helpful_harmless_off_diagonal_generated_total.",
      "chart_kind": "probe_metrics_by_layer",
      "figure_id": "report_probe_helpful_harmless_off_diagonal_generated_total/probe_metrics_by_layer",
      "path": "assets/report_probe_helpful_harmless_off_diagonal_generated_total/probe_metrics_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_helpful_harmless_off_diagonal_generated_total",
      "title": "Probe metrics by layer"
    },
    "report_probe_helpful_harmless_off_diagonal_prompt_end/auroc_by_layer": {
      "caption": "AUROC across captured layers for probe step report_probe_helpful_harmless_off_diagonal_prompt_end.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_helpful_harmless_off_diagonal_prompt_end/auroc_by_layer",
      "path": "assets/report_probe_helpful_harmless_off_diagonal_prompt_end/auroc_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_helpful_harmless_off_diagonal_prompt_end",
      "title": "AUROC by layer"
    },
    "report_probe_helpful_harmless_off_diagonal_prompt_end/balanced_accuracy_by_layer": {
      "caption": "Balanced Accuracy across captured layers for probe step report_probe_helpful_harmless_off_diagonal_prompt_end.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_helpful_harmless_off_diagonal_prompt_end/balanced_accuracy_by_layer",
      "path": "assets/report_probe_helpful_harmless_off_diagonal_prompt_end/balanced_accuracy_by_layer.png",
      "primary": true,
      "result_kind": "probe_result",
      "step_name": "report_probe_helpful_harmless_off_diagonal_prompt_end",
      "title": "Balanced Accuracy by layer"
    },
    "report_probe_helpful_harmless_off_diagonal_prompt_end/probe_metrics_by_layer": {
      "caption": "Available probe metrics across captured layers for step report_probe_helpful_harmless_off_diagonal_prompt_end.",
      "chart_kind": "probe_metrics_by_layer",
      "figure_id": "report_probe_helpful_harmless_off_diagonal_prompt_end/probe_metrics_by_layer",
      "path": "assets/report_probe_helpful_harmless_off_diagonal_prompt_end/probe_metrics_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_helpful_harmless_off_diagonal_prompt_end",
      "title": "Probe metrics by layer"
    },
    "report_probe_strong_harmless_generated_first_third/auroc_by_layer": {
      "caption": "AUROC across captured layers for probe step report_probe_strong_harmless_generated_first_third.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_strong_harmless_generated_first_third/auroc_by_layer",
      "path": "assets/report_probe_strong_harmless_generated_first_third/auroc_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_harmless_generated_first_third",
      "title": "AUROC by layer"
    },
    "report_probe_strong_harmless_generated_first_third/balanced_accuracy_by_layer": {
      "caption": "Balanced Accuracy across captured layers for probe step report_probe_strong_harmless_generated_first_third.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_strong_harmless_generated_first_third/balanced_accuracy_by_layer",
      "path": "assets/report_probe_strong_harmless_generated_first_third/balanced_accuracy_by_layer.png",
      "primary": true,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_harmless_generated_first_third",
      "title": "Balanced Accuracy by layer"
    },
    "report_probe_strong_harmless_generated_first_third/probe_metrics_by_layer": {
      "caption": "Available probe metrics across captured layers for step report_probe_strong_harmless_generated_first_third.",
      "chart_kind": "probe_metrics_by_layer",
      "figure_id": "report_probe_strong_harmless_generated_first_third/probe_metrics_by_layer",
      "path": "assets/report_probe_strong_harmless_generated_first_third/probe_metrics_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_harmless_generated_first_third",
      "title": "Probe metrics by layer"
    },
    "report_probe_strong_harmless_generated_last_third/auroc_by_layer": {
      "caption": "AUROC across captured layers for probe step report_probe_strong_harmless_generated_last_third.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_strong_harmless_generated_last_third/auroc_by_layer",
      "path": "assets/report_probe_strong_harmless_generated_last_third/auroc_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_harmless_generated_last_third",
      "title": "AUROC by layer"
    },
    "report_probe_strong_harmless_generated_last_third/balanced_accuracy_by_layer": {
      "caption": "Balanced Accuracy across captured layers for probe step report_probe_strong_harmless_generated_last_third.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_strong_harmless_generated_last_third/balanced_accuracy_by_layer",
      "path": "assets/report_probe_strong_harmless_generated_last_third/balanced_accuracy_by_layer.png",
      "primary": true,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_harmless_generated_last_third",
      "title": "Balanced Accuracy by layer"
    },
    "report_probe_strong_harmless_generated_last_third/probe_metrics_by_layer": {
      "caption": "Available probe metrics across captured layers for step report_probe_strong_harmless_generated_last_third.",
      "chart_kind": "probe_metrics_by_layer",
      "figure_id": "report_probe_strong_harmless_generated_last_third/probe_metrics_by_layer",
      "path": "assets/report_probe_strong_harmless_generated_last_third/probe_metrics_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_harmless_generated_last_third",
      "title": "Probe metrics by layer"
    },
    "report_probe_strong_harmless_generated_middle_third/auroc_by_layer": {
      "caption": "AUROC across captured layers for probe step report_probe_strong_harmless_generated_middle_third.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_strong_harmless_generated_middle_third/auroc_by_layer",
      "path": "assets/report_probe_strong_harmless_generated_middle_third/auroc_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_harmless_generated_middle_third",
      "title": "AUROC by layer"
    },
    "report_probe_strong_harmless_generated_middle_third/balanced_accuracy_by_layer": {
      "caption": "Balanced Accuracy across captured layers for probe step report_probe_strong_harmless_generated_middle_third.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_strong_harmless_generated_middle_third/balanced_accuracy_by_layer",
      "path": "assets/report_probe_strong_harmless_generated_middle_third/balanced_accuracy_by_layer.png",
      "primary": true,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_harmless_generated_middle_third",
      "title": "Balanced Accuracy by layer"
    },
    "report_probe_strong_harmless_generated_middle_third/probe_metrics_by_layer": {
      "caption": "Available probe metrics across captured layers for step report_probe_strong_harmless_generated_middle_third.",
      "chart_kind": "probe_metrics_by_layer",
      "figure_id": "report_probe_strong_harmless_generated_middle_third/probe_metrics_by_layer",
      "path": "assets/report_probe_strong_harmless_generated_middle_third/probe_metrics_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_harmless_generated_middle_third",
      "title": "Probe metrics by layer"
    },
    "report_probe_strong_harmless_generated_total/auroc_by_layer": {
      "caption": "AUROC across captured layers for probe step report_probe_strong_harmless_generated_total.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_strong_harmless_generated_total/auroc_by_layer",
      "path": "assets/report_probe_strong_harmless_generated_total/auroc_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_harmless_generated_total",
      "title": "AUROC by layer"
    },
    "report_probe_strong_harmless_generated_total/balanced_accuracy_by_layer": {
      "caption": "Balanced Accuracy across captured layers for probe step report_probe_strong_harmless_generated_total.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_strong_harmless_generated_total/balanced_accuracy_by_layer",
      "path": "assets/report_probe_strong_harmless_generated_total/balanced_accuracy_by_layer.png",
      "primary": true,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_harmless_generated_total",
      "title": "Balanced Accuracy by layer"
    },
    "report_probe_strong_harmless_generated_total/probe_metrics_by_layer": {
      "caption": "Available probe metrics across captured layers for step report_probe_strong_harmless_generated_total.",
      "chart_kind": "probe_metrics_by_layer",
      "figure_id": "report_probe_strong_harmless_generated_total/probe_metrics_by_layer",
      "path": "assets/report_probe_strong_harmless_generated_total/probe_metrics_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_harmless_generated_total",
      "title": "Probe metrics by layer"
    },
    "report_probe_strong_harmless_prompt_end/auroc_by_layer": {
      "caption": "AUROC across captured layers for probe step report_probe_strong_harmless_prompt_end.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_strong_harmless_prompt_end/auroc_by_layer",
      "path": "assets/report_probe_strong_harmless_prompt_end/auroc_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_harmless_prompt_end",
      "title": "AUROC by layer"
    },
    "report_probe_strong_harmless_prompt_end/balanced_accuracy_by_layer": {
      "caption": "Balanced Accuracy across captured layers for probe step report_probe_strong_harmless_prompt_end.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_strong_harmless_prompt_end/balanced_accuracy_by_layer",
      "path": "assets/report_probe_strong_harmless_prompt_end/balanced_accuracy_by_layer.png",
      "primary": true,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_harmless_prompt_end",
      "title": "Balanced Accuracy by layer"
    },
    "report_probe_strong_harmless_prompt_end/probe_metrics_by_layer": {
      "caption": "Available probe metrics across captured layers for step report_probe_strong_harmless_prompt_end.",
      "chart_kind": "probe_metrics_by_layer",
      "figure_id": "report_probe_strong_harmless_prompt_end/probe_metrics_by_layer",
      "path": "assets/report_probe_strong_harmless_prompt_end/probe_metrics_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_harmless_prompt_end",
      "title": "Probe metrics by layer"
    },
    "report_probe_strong_helpful_generated_first_third/auroc_by_layer": {
      "caption": "AUROC across captured layers for probe step report_probe_strong_helpful_generated_first_third.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_strong_helpful_generated_first_third/auroc_by_layer",
      "path": "assets/report_probe_strong_helpful_generated_first_third/auroc_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_helpful_generated_first_third",
      "title": "AUROC by layer"
    },
    "report_probe_strong_helpful_generated_first_third/balanced_accuracy_by_layer": {
      "caption": "Balanced Accuracy across captured layers for probe step report_probe_strong_helpful_generated_first_third.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_strong_helpful_generated_first_third/balanced_accuracy_by_layer",
      "path": "assets/report_probe_strong_helpful_generated_first_third/balanced_accuracy_by_layer.png",
      "primary": true,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_helpful_generated_first_third",
      "title": "Balanced Accuracy by layer"
    },
    "report_probe_strong_helpful_generated_first_third/probe_metrics_by_layer": {
      "caption": "Available probe metrics across captured layers for step report_probe_strong_helpful_generated_first_third.",
      "chart_kind": "probe_metrics_by_layer",
      "figure_id": "report_probe_strong_helpful_generated_first_third/probe_metrics_by_layer",
      "path": "assets/report_probe_strong_helpful_generated_first_third/probe_metrics_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_helpful_generated_first_third",
      "title": "Probe metrics by layer"
    },
    "report_probe_strong_helpful_generated_last_third/auroc_by_layer": {
      "caption": "AUROC across captured layers for probe step report_probe_strong_helpful_generated_last_third.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_strong_helpful_generated_last_third/auroc_by_layer",
      "path": "assets/report_probe_strong_helpful_generated_last_third/auroc_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_helpful_generated_last_third",
      "title": "AUROC by layer"
    },
    "report_probe_strong_helpful_generated_last_third/balanced_accuracy_by_layer": {
      "caption": "Balanced Accuracy across captured layers for probe step report_probe_strong_helpful_generated_last_third.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_strong_helpful_generated_last_third/balanced_accuracy_by_layer",
      "path": "assets/report_probe_strong_helpful_generated_last_third/balanced_accuracy_by_layer.png",
      "primary": true,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_helpful_generated_last_third",
      "title": "Balanced Accuracy by layer"
    },
    "report_probe_strong_helpful_generated_last_third/probe_metrics_by_layer": {
      "caption": "Available probe metrics across captured layers for step report_probe_strong_helpful_generated_last_third.",
      "chart_kind": "probe_metrics_by_layer",
      "figure_id": "report_probe_strong_helpful_generated_last_third/probe_metrics_by_layer",
      "path": "assets/report_probe_strong_helpful_generated_last_third/probe_metrics_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_helpful_generated_last_third",
      "title": "Probe metrics by layer"
    },
    "report_probe_strong_helpful_generated_middle_third/auroc_by_layer": {
      "caption": "AUROC across captured layers for probe step report_probe_strong_helpful_generated_middle_third.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_strong_helpful_generated_middle_third/auroc_by_layer",
      "path": "assets/report_probe_strong_helpful_generated_middle_third/auroc_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_helpful_generated_middle_third",
      "title": "AUROC by layer"
    },
    "report_probe_strong_helpful_generated_middle_third/balanced_accuracy_by_layer": {
      "caption": "Balanced Accuracy across captured layers for probe step report_probe_strong_helpful_generated_middle_third.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_strong_helpful_generated_middle_third/balanced_accuracy_by_layer",
      "path": "assets/report_probe_strong_helpful_generated_middle_third/balanced_accuracy_by_layer.png",
      "primary": true,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_helpful_generated_middle_third",
      "title": "Balanced Accuracy by layer"
    },
    "report_probe_strong_helpful_generated_middle_third/probe_metrics_by_layer": {
      "caption": "Available probe metrics across captured layers for step report_probe_strong_helpful_generated_middle_third.",
      "chart_kind": "probe_metrics_by_layer",
      "figure_id": "report_probe_strong_helpful_generated_middle_third/probe_metrics_by_layer",
      "path": "assets/report_probe_strong_helpful_generated_middle_third/probe_metrics_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_helpful_generated_middle_third",
      "title": "Probe metrics by layer"
    },
    "report_probe_strong_helpful_generated_total/auroc_by_layer": {
      "caption": "AUROC across captured layers for probe step report_probe_strong_helpful_generated_total.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_strong_helpful_generated_total/auroc_by_layer",
      "path": "assets/report_probe_strong_helpful_generated_total/auroc_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_helpful_generated_total",
      "title": "AUROC by layer"
    },
    "report_probe_strong_helpful_generated_total/balanced_accuracy_by_layer": {
      "caption": "Balanced Accuracy across captured layers for probe step report_probe_strong_helpful_generated_total.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_strong_helpful_generated_total/balanced_accuracy_by_layer",
      "path": "assets/report_probe_strong_helpful_generated_total/balanced_accuracy_by_layer.png",
      "primary": true,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_helpful_generated_total",
      "title": "Balanced Accuracy by layer"
    },
    "report_probe_strong_helpful_generated_total/probe_metrics_by_layer": {
      "caption": "Available probe metrics across captured layers for step report_probe_strong_helpful_generated_total.",
      "chart_kind": "probe_metrics_by_layer",
      "figure_id": "report_probe_strong_helpful_generated_total/probe_metrics_by_layer",
      "path": "assets/report_probe_strong_helpful_generated_total/probe_metrics_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_helpful_generated_total",
      "title": "Probe metrics by layer"
    },
    "report_probe_strong_helpful_prompt_end/auroc_by_layer": {
      "caption": "AUROC across captured layers for probe step report_probe_strong_helpful_prompt_end.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_strong_helpful_prompt_end/auroc_by_layer",
      "path": "assets/report_probe_strong_helpful_prompt_end/auroc_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_helpful_prompt_end",
      "title": "AUROC by layer"
    },
    "report_probe_strong_helpful_prompt_end/balanced_accuracy_by_layer": {
      "caption": "Balanced Accuracy across captured layers for probe step report_probe_strong_helpful_prompt_end.",
      "chart_kind": "probe_metric_by_layer",
      "figure_id": "report_probe_strong_helpful_prompt_end/balanced_accuracy_by_layer",
      "path": "assets/report_probe_strong_helpful_prompt_end/balanced_accuracy_by_layer.png",
      "primary": true,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_helpful_prompt_end",
      "title": "Balanced Accuracy by layer"
    },
    "report_probe_strong_helpful_prompt_end/probe_metrics_by_layer": {
      "caption": "Available probe metrics across captured layers for step report_probe_strong_helpful_prompt_end.",
      "chart_kind": "probe_metrics_by_layer",
      "figure_id": "report_probe_strong_helpful_prompt_end/probe_metrics_by_layer",
      "path": "assets/report_probe_strong_helpful_prompt_end/probe_metrics_by_layer.png",
      "primary": false,
      "result_kind": "probe_result",
      "step_name": "report_probe_strong_helpful_prompt_end",
      "title": "Probe metrics by layer"
    }
  },
  "input_count": 17,
  "step_summaries": {
    "report_probe_helpful_harmless_off_diagonal_generated_first_third": {
      "headline_metrics": {
        "best_layer": 36,
        "best_metric": "source_family_holdout_balanced_accuracy",
        "best_value": 0.6175595238095237,
        "example_count": 90,
        "group_count": null,
        "split_mode": "source_family_holdout"
      },
      "kind": "probe_result",
      "primary_figure_id": "report_probe_helpful_harmless_off_diagonal_generated_first_third/balanced_accuracy_by_layer",
      "step_name": "report_probe_helpful_harmless_off_diagonal_generated_first_third",
      "table_path": "tables/report_probe_helpful_harmless_off_diagonal_generated_first_third.json"
    },
    "report_probe_helpful_harmless_off_diagonal_generated_last_third": {
      "headline_metrics": {
        "best_layer": 44,
        "best_metric": "source_family_holdout_balanced_accuracy",
        "best_value": 0.6284722222222222,
        "example_count": 90,
        "group_count": null,
        "split_mode": "source_family_holdout"
      },
      "kind": "probe_result",
      "primary_figure_id": "report_probe_helpful_harmless_off_diagonal_generated_last_third/balanced_accuracy_by_layer",
      "step_name": "report_probe_helpful_harmless_off_diagonal_generated_last_third",
      "table_path": "tables/report_probe_helpful_harmless_off_diagonal_generated_last_third.json"
    },
    "report_probe_helpful_harmless_off_diagonal_generated_middle_third": {
      "headline_metrics": {
        "best_layer": 44,
        "best_metric": "source_family_holdout_balanced_accuracy",
        "best_value": 0.621031746031746,
        "example_count": 90,
        "group_count": null,
        "split_mode": "source_family_holdout"
      },
      "kind": "probe_result",
      "primary_figure_id": "report_probe_helpful_harmless_off_diagonal_generated_middle_third/balanced_accuracy_by_layer",
      "step_name": "report_probe_helpful_harmless_off_diagonal_generated_middle_third",
      "table_path": "tables/report_probe_helpful_harmless_off_diagonal_generated_middle_third.json"
    },
    "report_probe_helpful_harmless_off_diagonal_generated_total": {
      "headline_metrics": {
        "best_layer": 44,
        "best_metric": "source_family_holdout_balanced_accuracy",
        "best_value": 0.6453373015873016,
        "example_count": 90,
        "group_count": null,
        "split_mode": "source_family_holdout"
      },
      "kind": "probe_result",
      "primary_figure_id": "report_probe_helpful_harmless_off_diagonal_generated_total/balanced_accuracy_by_layer",
      "step_name": "report_probe_helpful_harmless_off_diagonal_generated_total",
      "table_path": "tables/report_probe_helpful_harmless_off_diagonal_generated_total.json"
    },
    "report_probe_helpful_harmless_off_diagonal_prompt_end": {
      "headline_metrics": {
        "best_layer": 16,
        "best_metric": "source_family_holdout_balanced_accuracy",
        "best_value": 0.6408730158730158,
        "example_count": 90,
        "group_count": null,
        "split_mode": "source_family_holdout"
      },
      "kind": "probe_result",
      "primary_figure_id": "report_probe_helpful_harmless_off_diagonal_prompt_end/balanced_accuracy_by_layer",
      "step_name": "report_probe_helpful_harmless_off_diagonal_prompt_end",
      "table_path": "tables/report_probe_helpful_harmless_off_diagonal_prompt_end.json"
    },
    "report_probe_strong_harmless_generated_first_third": {
      "headline_metrics": {
        "best_layer": 0,
        "best_metric": "source_family_holdout_balanced_accuracy",
        "best_value": 0.5506750126021369,
        "example_count": 500,
        "group_count": null,
        "split_mode": "source_family_holdout"
      },
      "kind": "probe_result",
      "primary_figure_id": "report_probe_strong_harmless_generated_first_third/balanced_accuracy_by_layer",
      "step_name": "report_probe_strong_harmless_generated_first_third",
      "table_path": "tables/report_probe_strong_harmless_generated_first_third.json"
    },
    "report_probe_strong_harmless_generated_last_third": {
      "headline_metrics": {
        "best_layer": 0,
        "best_metric": "source_family_holdout_balanced_accuracy",
        "best_value": 0.663826232994211,
        "example_count": 500,
        "group_count": null,
        "split_mode": "source_family_holdout"
      },
      "kind": "probe_result",
      "primary_figure_id": "report_probe_strong_harmless_generated_last_third/balanced_accuracy_by_layer",
      "step_name": "report_probe_strong_harmless_generated_last_third",
      "table_path": "tables/report_probe_strong_harmless_generated_last_third.json"
    },
    "report_probe_strong_harmless_generated_middle_third": {
      "headline_metrics": {
        "best_layer": 4,
        "best_metric": "source_family_holdout_balanced_accuracy",
        "best_value": 0.6090817732512264,
        "example_count": 500,
        "group_count": null,
        "split_mode": "source_family_holdout"
      },
      "kind": "probe_result",
      "primary_figure_id": "report_probe_strong_harmless_generated_middle_third/balanced_accuracy_by_layer",
      "step_name": "report_probe_strong_harmless_generated_middle_third",
      "table_path": "tables/report_probe_strong_harmless_generated_middle_third.json"
    },
    "report_probe_strong_harmless_generated_total": {
      "headline_metrics": {
        "best_layer": 44,
        "best_metric": "source_family_holdout_balanced_accuracy",
        "best_value": 0.6142569585650821,
        "example_count": 500,
        "group_count": null,
        "split_mode": "source_family_holdout"
      },
      "kind": "probe_result",
      "primary_figure_id": "report_probe_strong_harmless_generated_total/balanced_accuracy_by_layer",
      "step_name": "report_probe_strong_harmless_generated_total",
      "table_path": "tables/report_probe_strong_harmless_generated_total.json"
    },
    "report_probe_strong_harmless_prompt_end": {
      "headline_metrics": {
        "best_layer": 28,
        "best_metric": "source_family_holdout_balanced_accuracy",
        "best_value": 0.5735146366440623,
        "example_count": 500,
        "group_count": null,
        "split_mode": "source_family_holdout"
      },
      "kind": "probe_result",
      "primary_figure_id": "report_probe_strong_harmless_prompt_end/balanced_accuracy_by_layer",
      "step_name": "report_probe_strong_harmless_prompt_end",
      "table_path": "tables/report_probe_strong_harmless_prompt_end.json"
    },
    "report_probe_strong_helpful_generated_first_third": {
      "headline_metrics": {
        "best_layer": 36,
        "best_metric": "source_family_holdout_balanced_accuracy",
        "best_value": 0.5592526406446845,
        "example_count": 500,
        "group_count": null,
        "split_mode": "source_family_holdout"
      },
      "kind": "probe_result",
      "primary_figure_id": "report_probe_strong_helpful_generated_first_third/balanced_accuracy_by_layer",
      "step_name": "report_probe_strong_helpful_generated_first_third",
      "table_path": "tables/report_probe_strong_helpful_generated_first_third.json"
    },
    "report_probe_strong_helpful_generated_last_third": {
      "headline_metrics": {
        "best_layer": 8,
        "best_metric": "source_family_holdout_balanced_accuracy",
        "best_value": 0.5898776693344356,
        "example_count": 500,
        "group_count": null,
        "split_mode": "source_family_holdout"
      },
      "kind": "probe_result",
      "primary_figure_id": "report_probe_strong_helpful_generated_last_third/balanced_accuracy_by_layer",
      "step_name": "report_probe_strong_helpful_generated_last_third",
      "table_path": "tables/report_probe_strong_helpful_generated_last_third.json"
    },
    "report_probe_strong_helpful_generated_middle_third": {
      "headline_metrics": {
        "best_layer": 8,
        "best_metric": "source_family_holdout_balanced_accuracy",
        "best_value": 0.5741081161753798,
        "example_count": 500,
        "group_count": null,
        "split_mode": "source_family_holdout"
      },
      "kind": "probe_result",
      "primary_figure_id": "report_probe_strong_helpful_generated_middle_third/balanced_accuracy_by_layer",
      "step_name": "report_probe_strong_helpful_generated_middle_third",
      "table_path": "tables/report_probe_strong_helpful_generated_middle_third.json"
    },
    "report_probe_strong_helpful_generated_total": {
      "headline_metrics": {
        "best_layer": 16,
        "best_metric": "source_family_holdout_balanced_accuracy",
        "best_value": 0.6100958182916448,
        "example_count": 500,
        "group_count": null,
        "split_mode": "source_family_holdout"
      },
      "kind": "probe_result",
      "primary_figure_id": "report_probe_strong_helpful_generated_total/balanced_accuracy_by_layer",
      "step_name": "report_probe_strong_helpful_generated_total",
      "table_path": "tables/report_probe_strong_helpful_generated_total.json"
    },
    "report_probe_strong_helpful_prompt_end": {
      "headline_metrics": {
        "best_layer": 36,
        "best_metric": "source_family_holdout_balanced_accuracy",
        "best_value": 0.5922926906258534,
        "example_count": 500,
        "group_count": null,
        "split_mode": "source_family_holdout"
      },
      "kind": "probe_result",
      "primary_figure_id": "report_probe_strong_helpful_prompt_end/balanced_accuracy_by_layer",
      "step_name": "report_probe_strong_helpful_prompt_end",
      "table_path": "tables/report_probe_strong_helpful_prompt_end.json"
    }
  },
  "tables": {
    "report_probe_helpful_harmless_off_diagonal_generated_first_third": {
      "columns": [
        "auroc",
        "balanced_accuracy",
        "cv_auroc",
        "cv_balanced_accuracy",
        "layer",
        "n"
      ],
      "path": "tables/report_probe_helpful_harmless_off_diagonal_generated_first_third.json",
      "result_kind": "probe_result",
      "rows": 8,
      "step_name": "report_probe_helpful_harmless_off_diagonal_generated_first_third"
    },
    "report_probe_helpful_harmless_off_diagonal_generated_last_third": {
      "columns": [
        "auroc",
        "balanced_accuracy",
        "cv_auroc",
        "cv_balanced_accuracy",
        "layer",
        "n"
      ],
      "path": "tables/report_probe_helpful_harmless_off_diagonal_generated_last_third.json",
      "result_kind": "probe_result",
      "rows": 8,
      "step_name": "report_probe_helpful_harmless_off_diagonal_generated_last_third"
    },
    "report_probe_helpful_harmless_off_diagonal_generated_middle_third": {
      "columns": [
        "auroc",
        "balanced_accuracy",
        "cv_auroc",
        "cv_balanced_accuracy",
        "layer",
        "n"
      ],
      "path": "tables/report_probe_helpful_harmless_off_diagonal_generated_middle_third.json",
      "result_kind": "probe_result",
      "rows": 8,
      "step_name": "report_probe_helpful_harmless_off_diagonal_generated_middle_third"
    },
    "report_probe_helpful_harmless_off_diagonal_generated_total": {
      "columns": [
        "auroc",
        "balanced_accuracy",
        "cv_auroc",
        "cv_balanced_accuracy",
        "layer",
        "n"
      ],
      "path": "tables/report_probe_helpful_harmless_off_diagonal_generated_total.json",
      "result_kind": "probe_result",
      "rows": 8,
      "step_name": "report_probe_helpful_harmless_off_diagonal_generated_total"
    },
    "report_probe_helpful_harmless_off_diagonal_prompt_end": {
      "columns": [
        "auroc",
        "balanced_accuracy",
        "cv_auroc",
        "cv_balanced_accuracy",
        "layer",
        "n"
      ],
      "path": "tables/report_probe_helpful_harmless_off_diagonal_prompt_end.json",
      "result_kind": "probe_result",
      "rows": 8,
      "step_name": "report_probe_helpful_harmless_off_diagonal_prompt_end"
    },
    "report_probe_strong_harmless_generated_first_third": {
      "columns": [
        "auroc",
        "balanced_accuracy",
        "cv_auroc",
        "cv_balanced_accuracy",
        "layer",
        "n"
      ],
      "path": "tables/report_probe_strong_harmless_generated_first_third.json",
      "result_kind": "probe_result",
      "rows": 8,
      "step_name": "report_probe_strong_harmless_generated_first_third"
    },
    "report_probe_strong_harmless_generated_last_third": {
      "columns": [
        "auroc",
        "balanced_accuracy",
        "cv_auroc",
        "cv_balanced_accuracy",
        "layer",
        "n"
      ],
      "path": "tables/report_probe_strong_harmless_generated_last_third.json",
      "result_kind": "probe_result",
      "rows": 8,
      "step_name": "report_probe_strong_harmless_generated_last_third"
    },
    "report_probe_strong_harmless_generated_middle_third": {
      "columns": [
        "auroc",
        "balanced_accuracy",
        "cv_auroc",
        "cv_balanced_accuracy",
        "layer",
        "n"
      ],
      "path": "tables/report_probe_strong_harmless_generated_middle_third.json",
      "result_kind": "probe_result",
      "rows": 8,
      "step_name": "report_probe_strong_harmless_generated_middle_third"
    },
    "report_probe_strong_harmless_generated_total": {
      "columns": [
        "auroc",
        "balanced_accuracy",
        "cv_auroc",
        "cv_balanced_accuracy",
        "layer",
        "n"
      ],
      "path": "tables/report_probe_strong_harmless_generated_total.json",
      "result_kind": "probe_result",
      "rows": 8,
      "step_name": "report_probe_strong_harmless_generated_total"
    },
    "report_probe_strong_harmless_prompt_end": {
      "columns": [
        "auroc",
        "balanced_accuracy",
        "cv_auroc",
        "cv_balanced_accuracy",
        "layer",
        "n"
      ],
      "path": "tables/report_probe_strong_harmless_prompt_end.json",
      "result_kind": "probe_result",
      "rows": 8,
      "step_name": "report_probe_strong_harmless_prompt_end"
    },
    "report_probe_strong_helpful_generated_first_third": {
      "columns": [
        "auroc",
        "balanced_accuracy",
        "cv_auroc",
        "cv_balanced_accuracy",
        "layer",
        "n"
      ],
      "path": "tables/report_probe_strong_helpful_generated_first_third.json",
      "result_kind": "probe_result",
      "rows": 8,
      "step_name": "report_probe_strong_helpful_generated_first_third"
    },
    "report_probe_strong_helpful_generated_last_third": {
      "columns": [
        "auroc",
        "balanced_accuracy",
        "cv_auroc",
        "cv_balanced_accuracy",
        "layer",
        "n"
      ],
      "path": "tables/report_probe_strong_helpful_generated_last_third.json",
      "result_kind": "probe_result",
      "rows": 8,
      "step_name": "report_probe_strong_helpful_generated_last_third"
    },
    "report_probe_strong_helpful_generated_middle_third": {
      "columns": [
        "auroc",
        "balanced_accuracy",
        "cv_auroc",
        "cv_balanced_accuracy",
        "layer",
        "n"
      ],
      "path": "tables/report_probe_strong_helpful_generated_middle_third.json",
      "result_kind": "probe_result",
      "rows": 8,
      "step_name": "report_probe_strong_helpful_generated_middle_third"
    },
    "report_probe_strong_helpful_generated_total": {
      "columns": [
        "auroc",
        "balanced_accuracy",
        "cv_auroc",
        "cv_balanced_accuracy",
        "layer",
        "n"
      ],
      "path": "tables/report_probe_strong_helpful_generated_total.json",
      "result_kind": "probe_result",
      "rows": 8,
      "step_name": "report_probe_strong_helpful_generated_total"
    },
    "report_probe_strong_helpful_prompt_end": {
      "columns": [
        "auroc",
        "balanced_accuracy",
        "cv_auroc",
        "cv_balanced_accuracy",
        "layer",
        "n"
      ],
      "path": "tables/report_probe_strong_helpful_prompt_end.json",
      "result_kind": "probe_result",
      "rows": 8,
      "step_name": "report_probe_strong_helpful_prompt_end"
    }
  },
  "template": "default"
}
```
