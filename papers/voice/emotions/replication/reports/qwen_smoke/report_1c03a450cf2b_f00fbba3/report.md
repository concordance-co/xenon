# voice_emotions_qwen_smoke

- template: `voice_emotions_qwen_smoke`
- input_count: 4
- example_count: 4
- manifest: `assets/manifest.json`
- summary: `summary.json`

## Inputs

### emotion_space

- artifact_id: `emotion_vector_space_1_ce103f73`
- artifact_kind: `emotion_vector_space`
- provenance: run `wr_85fcc186ab83_e27b2237` / step `emotion_space` / index `2`
- runtime: runner `modal` / app `ap-wNw6Fqvva1dvmSgR2iar14`
- results: `results/emotion_space_results.json`

### emotion_geometry

- artifact_id: `emotion_geometry_1_82191c10`
- artifact_kind: `emotion_geometry`
- provenance: run `wr_85fcc186ab83_e27b2237` / step `emotion_geometry` / index `3`
- runtime: runner `modal` / app `ap-YikqRyzjpL76xAjdrjGLn7`
- results: `results/emotion_geometry_results.json`

### score_emotions

- artifact_id: `emotion_score_1_bda88b3d`
- artifact_kind: `emotion_score`
- provenance: run `wr_85fcc186ab83_e27b2237` / step `score_emotions` / index `4`
- runtime: runner `modal` / app `ap-mDf5v3U8d7gELNXQl8EIoL`
- results: `results/score_emotions_results.json`

### happy_direction

- artifact_id: `emotion_direction_1_b56e7903`
- artifact_kind: `emotion_direction`
- provenance: run `wr_85fcc186ab83_e27b2237` / step `happy_direction` / index `5`
- runtime: runner `modal` / app `ap-IiA9s5qMcMuHctCRBuJoS5`
- results: `results/happy_direction_results.json`

## Summary

```json
{
  "example_count": 4,
  "figures": {},
  "input_count": 4,
  "step_summaries": {},
  "tables": {},
  "template": "voice_emotions_qwen_smoke"
}
```
