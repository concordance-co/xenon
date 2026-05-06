# Voice Paper Replications

Xenon-native smoke replications for direction/intervention papers about model
"voice": persona, emotion, refusal, and truthfulness. These are public example
workflows, not normal `projects/<project>/phase_XX` research phases.

Each paper directory should expose four runnable surfaces: paper rerun, BYOD,
BYOT, and BYOP. Assistant Axis is the current reference implementation for
that shape.

## Workflows

```bash
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/assistant_axis/specs/paper_rerun.py
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/assistant_axis/specs/byod_axis.py
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/assistant_axis/specs/byot_score.py
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/assistant_axis/specs/byop_generate.py

uv run python -m pipelines_v2.cli workflow plan --file papers/voice/emotions/specs/workflow.py
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/refusal_direction/specs/workflow.py
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/honest_llama/specs/workflow.py
```

## Layout

- `assistant_axis/`: default-assistant vs role-play persona axis; reference
  package layout with paper/BYOD/BYOT/BYOP specs.
- `emotions/`: emotion concept vectors, scoring, geometry, and steering export.
- `refusal_direction/`: harmful-minus-harmless direction, selection, add-direction, and project-out smoke.
- `honest_llama/`: residual TruthfulQA-style truthful-minus-untruthful ITI smoke.
- `common/`: shared smoke constants and local runner fixtures.
- `schemas/`: method-level BYOD row contracts.
- `BYOD.md`: where to edit data recipes and how schemas map onto methods.

The paper directory names are historical. BYOD schemas are method schemas, not
domain schemas. For example, the emotion-concepts workflow can build concept
vectors over finance labels, support intents, or failure modes if the rows fit
the concept-vector-space schema.

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

Planning a workflow checks API plumbing and artifact shapes. It does not prove
paper-scale claims. A paper-scale result requires real datasets, behavioral
sanity checks, confound checks, selected loci that transfer to real data, and
intervention controls.
