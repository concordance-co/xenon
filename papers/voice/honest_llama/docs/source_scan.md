# Source Scan

Source: `likenneth/honest_llama`.

The source repo implements Inference-Time Intervention for LLaMA/Alpaca/Vicuna:
collect activations on TruthfulQA-style data, identify truthful directions
across selected attention heads, intervene during inference, and evaluate
truthfulness/helpfulness tradeoffs.

Xenon mapping:

- `TruthfulnessDirectionSpec` computes truthful-minus-untruthful residual
  directions.
- `TruthfulnessScoreSpec` scores validation answer sections.
- `TruthfulnessDirectionSelectionSpec` selects a residual layer by projection
  gap.
- `TruthfulnessAblationSubspaceSpec` supports `ProjectOutPatch` controls.
- `AddDirectionPatch` provides a residual steering analogue.

Deviation: the first Xenon implementation is residual-stream, not attention-head
ITI. Head-specific writes should become a later engine/intervention surface.
