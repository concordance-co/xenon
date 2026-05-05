# Emotion Vectors

Goal: reproduce the Xenon-native core of the Transformer Circuits emotions work:
derive emotion concept vectors, score sections, inspect geometry, and export an
emotion direction for steering.

Smoke command:

```bash
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/emotions/specs/workflow.py
```

Expected artifacts: capture, emotion vector space, emotion scores, one exported
emotion direction, geometry diagnostics, steered generation, and report summary.

Real run default: `Qwen/Qwen3-8B`. Paper-faithful Claude-specific results are
not claimed by this open-source smoke path.

Data hooks:

- `emotion_probe_story_dataset(limit=...)` points at a public generated
  story/dialogue probe mirror with `real_emotion`, `displayed_emotion`, `topic`,
  and `text` columns.
- `emotion_contrast_dataset(records=...)` maps arbitrary agent logs, stories,
  or transcript sections into the same labeled vector-space workflow.
- `EmotionPrecomputedVectorSpaceSpec(...)` remains the path for released or
  user-owned precomputed emotion-space artifacts.

Paper-scale scaffold:

- `replication/` contains TODO-marked prompts, config, data manifest, report
  directories, and a workflow outline for recreating the paper's story-vector
  recipe without treating the work as a normal Xenon research phase.

Claim boundary: smoke vectors are fixture-only. Paper-level claims require
large labeled story/dialogue data, naturalistic transcript checks, preference
or behavior evaluations, and intervention controls.
