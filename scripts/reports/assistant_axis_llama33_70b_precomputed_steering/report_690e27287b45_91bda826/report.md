# assistant_axis_llama33_70b_precomputed_steering

- template: `assistant_axis_llama33_70b_precomputed_steering`
- input_count: 5
- example_count: 1
- manifest: `assets/manifest.json`
- summary: `summary.json`

## Inputs

### load_calm_trait

- artifact_id: `assistant_axis_trait_coordinate_1_0819f1a9`
- artifact_kind: `assistant_axis_trait_coordinate`
- provenance: run `wr_e777c120a4bb_4559f201` / step `load_calm_trait` / index `0`
- runtime: runner `modal` / app `ap-UqH59r40Ur86TeuTCj6tAy`
- results: `results/load_calm_trait_results.json`

### calm_unit_direction

- artifact_id: `transform_1_c99276e9`
- artifact_kind: `transform`
- provenance: run `wr_e777c120a4bb_4559f201` / step `calm_unit_direction` / index `2`
- runtime: runner `modal` / app `ap-BfMpIBed4NCvA07hu9ytTA`
- results: `results/calm_unit_direction_results.json`

### baseline_generation

- artifact_id: `generation_run_1_14fe853df3c3`
- artifact_kind: `generation_run`
- provenance: run `wr_e777c120a4bb_4559f201` / step `baseline_generation` / index `1`
- runtime: runner `modal` / app `ap-p6tNt4I0P8bloCW9gZETpS`
- results: `results/baseline_generation_results.json`

### calm_steered_generation

- artifact_id: `patched_generation_1_9b6ea1cf`
- artifact_kind: `patched_generation`
- provenance: run `wr_96588bb43527_9c18a1da` / step `calm_steered_generation` / index `3`
- runtime: runner `modal` / app `ap-z4wubc5djsjo8RCqYarGlW`
- results: `results/calm_steered_generation_results.json`

### steering_summary

- artifact_id: `transform_1_c7124090`
- artifact_kind: `transform`
- provenance: run `wr_96588bb43527_9c18a1da` / step `steering_summary` / index `4`
- runtime: runner `modal` / app `ap-lfLCBztZfE6iW3lhTM2jzl`
- results: `results/steering_summary_results.json`

## Summary

```json
{
  "example_count": 1,
  "figures": {},
  "input_count": 5,
  "step_summaries": {},
  "tables": {},
  "template": "assistant_axis_llama33_70b_precomputed_steering"
}
```
