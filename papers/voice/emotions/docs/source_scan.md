# Source Scan

Source: Transformer Circuits, "Emotion Concepts and their Function in a Large
Language Model" (2026).

The paper identifies emotion concept vectors from synthetic stories/dialogues,
validates projection behavior on broad corpora and naturalistic transcripts,
studies geometry, and shows steering effects on expression and preferences.

Xenon mapping:

- `EmotionVectorSpaceSpec` derives concept vectors.
- `EmotionScoreSpec` projects captured sections onto selected emotions.
- `EmotionGeometrySpec` emits cosine/PCA/cluster diagnostics.
- `EmotionDirectionSpec` exports one concept for `AddDirectionPatch`.

Deviation: the workflow is residual-stream and fixture-based; it does not claim
Claude-specific internal emotion functionality.

Replication scaffold: `papers/voice/emotions/replication/` is the place to fill
paper-specific details manually before paper-scale generation/capture.
