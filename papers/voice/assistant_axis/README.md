# Assistant Axis / Persona Vectors

Goal: reproduce the Assistant Axis shape as a Xenon workflow: derive
`mean(default assistant) - mean(role-play personas)`, score conversation turns
for persona drift, and run a small add-direction steering demo.

Smoke command:

```bash
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/assistant_axis/specs/workflow.py
```

Expected artifacts: capture, assistant-axis direction, drift projection scores,
baseline generation, steered generation, and report summary.

Real run default: `Qwen/Qwen3-8B`. The released paper configs for larger models
remain available through the existing `AssistantAxis*` APIs.

Data/vector hooks:

- `assistant_axis_prompt_dataset(limit=...)` loads the released prompt/source
  dataset used for Assistant Axis-style derivation.
- `AssistantAxisPrecomputedCoordinateSpec(model_id="Qwen/Qwen3-32B")` loads the
  released assistant axis for supported models.
- `AssistantAxisTraitCoordinateSpec(model_id="llama-3.3-70b", trait="calm")`
  loads per-trait files from
  `lu-christina/assistant-axis-vectors/llama-3.3-70b/trait_vectors`.

Claim boundary: the smoke fixture proves only that the Xenon API wiring works.
Paper-level claims need generated role adherence scoring, turn-level behavioral
inspection, and real transcript projection.
