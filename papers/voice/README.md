# Voice Paper Replications

Xenon-native smoke replications for direction/intervention papers about model
"voice": persona, emotion, refusal, and truthfulness. These are public example
workflows, not normal `projects/<project>/phase_XX` research phases.

Default smoke workflows use `ToyEngine` and tiny in-code fixtures so tests and
`workflow plan` run without model downloads. Real runs should swap in
`VLLMEngine(model_id="Qwen/Qwen3-8B")`, move durable datasets to Neon, and keep
captures/generations in Modal volumes.

## Workflows

```bash
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/assistant_axis/specs/workflow.py
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/emotions/specs/workflow.py
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/refusal_direction/specs/workflow.py
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/honest_llama/specs/workflow.py
```

## Layout

- `assistant_axis/`: default-assistant vs role-play persona axis.
- `emotions/`: emotion concept vectors, scoring, geometry, and steering export.
- `refusal_direction/`: harmful-minus-harmless direction, selection, add-direction, and project-out smoke.
- `honest_llama/`: residual TruthfulQA-style truthful-minus-untruthful ITI smoke.
- `common/`: shared smoke constants and local runner fixtures.

## Lightweight Data Hooks

The smoke workflows keep fixtures in code, but the reusable APIs now point at
the paper-style sources where possible:

- Assistant Axis prompts and released vectors: `assistant_axis_prompt_dataset`,
  `AssistantAxisPrecomputedCoordinateSpec`, and
  `AssistantAxisTraitCoordinateSpec(model_id="llama-3.3-70b", trait="calm")`.
- Emotions: `emotion_probe_story_dataset` for a public generated-probe mirror,
  or `emotion_contrast_dataset(records=...)` for agent/product logs.
- Refusal Direction: `refusal_direction_split_dataset(split="train")` and
  `refusal_direction_processed_dataset("advbench")`.
- Honest LLaMA / ITI: `truthfulqa_generation_dataset()` and
  `truthfulqa_answer_contrast_dataset(prompt_template=...)`.

These helpers are deliberately prompt-template friendly. Users can keep the
same labels and vector workflow while changing the actual chat/completion
format to match their product.

## Claim Boundary

Smoke workflows check API plumbing and artifact shapes. They do not reproduce
paper-scale claims. A paper-scale result requires real datasets, behavioral
sanity checks, confound checks, selected loci that transfer to real data, and
intervention controls.
