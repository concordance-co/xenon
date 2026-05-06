# Assistant Axis / Persona Vectors

Goal: reproduce the Assistant Axis shape as a Xenon workflow: derive
`mean(default assistant) - mean(role-play personas)`, score conversation turns
for persona drift, and run a small add-direction steering demo.

Start here:

- `PAPER_IMPLEMENTATION.md`: paper-specific status, released vectors, source
  prompt data, and from-scratch rerun constants.
- `paper.py`: paper data/source-prompt edit surface.
- `judge.py`: OpenAI-compatible role-adherence judge for paper reruns.
- `method.py`: method/BYOD edit surface for non-paper domains.
- `runtime.py`: shared transforms/loaders used by runnable specs.
- `specs/paper_rerun.py`: paper-rerun scaffold against the released
  prompt-source data.
- `specs/byod_axis.py`: arbitrary default-vs-role data -> custom axis and
  scores.
- `specs/byot_score.py`: existing trace(s) -> released axis/trait scores.
- `specs/byop_generate.py`: prompt -> baseline generation, trait-steered
  generation, and scores.

## Commands

```bash
# paper-rerun surface; default smoke slice uses actual released HF prompt source
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/assistant_axis/specs/paper_rerun.py

# BYOD method surface; set ASSISTANT_AXIS_BYOD_JSONL for real rows
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/assistant_axis/specs/byod_axis.py

# BYOT trace scoring; set ASSISTANT_AXIS_BYOT_JSONL or ASSISTANT_AXIS_BYOT_TRACE
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/assistant_axis/specs/byot_score.py

# BYOP generation + steering + scoring; set ASSISTANT_AXIS_BYOP_PROMPT
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/assistant_axis/specs/byop_generate.py
```

Replace `plan` with `run --logging INFO` to launch. Defaults use
`ASSISTANT_AXIS_MODEL_KEY=llama_3_3_70b`; supported values are
`gemma_2_27b`, `qwen_3_32b`, and `llama_3_3_70b`.

## Runtime Knobs

- `ASSISTANT_AXIS_MODEL_KEY`: released model key.
- `ASSISTANT_AXIS_JUDGE_MODEL`: OpenAI-compatible judge model for paper rerun.
- `ASSISTANT_AXIS_JUDGE_API_KEY_ENV`: env var containing the judge API key.
- `ASSISTANT_AXIS_JUDGE_SECRET_NAME`: Modal secret name to bind for that env var.
- `ASSISTANT_AXIS_JUDGE_DRY_RUN`: set `1` to attach placeholder score-3 role
  labels without spending judge calls.
- `ASSISTANT_AXIS_TRAITS`: comma-separated BYOT trait vectors.
- `ASSISTANT_AXIS_BYOD_JSONL`: BYOD rows matching
  `papers/voice/schemas/assistant_axis_method.schema.json`.
- `ASSISTANT_AXIS_BYOT_JSONL`: BYOT rows matching
  `papers/voice/schemas/assistant_axis_trace.schema.json`.
- `ASSISTANT_AXIS_BYOT_TRACE`: one literal trace string for BYOT.
- `ASSISTANT_AXIS_BYOP_PROMPT`: one prompt for BYOP.
- `ASSISTANT_AXIS_STEERING_TRAIT`: released trait vector for BYOP steering.
- `ASSISTANT_AXIS_STEERING_STRENGTH`: add-direction steering strength.
- `ASSISTANT_AXIS_ROLE_LIMIT`, `ASSISTANT_AXIS_QUESTION_LIMIT`,
  `ASSISTANT_AXIS_INSTRUCTION_LIMIT`: paper-rerun smoke slice controls.
- `ASSISTANT_AXIS_MIN_ROLE_EXAMPLES_PER_ROLE`: derivation threshold. Use `50`
  for the full paper threshold after real judge labels are attached.

Paper data/vector hooks:

- `assistant_axis_prompt_dataset(limit=...)` loads the released prompt/source
  dataset used for Assistant Axis-style derivation.
- `paper.py` records the paper rerun defaults: 275 roles plus default,
  5 instruction variants, 240 extraction questions, vLLM sampling settings,
  response-activation pooling, judge model/score scale, and score-3 filtering.
- `AssistantAxisPrecomputedCoordinateSpec(model_id="Qwen/Qwen3-32B")` loads the
  released assistant axis for supported models.
- `AssistantAxisTraitCoordinateSpec(model_id="llama-3.3-70b", trait="calm")`
  loads per-trait files from
  `lu-christina/assistant-axis-vectors/llama-3.3-70b/trait_vectors`.

Claim boundary: released-vector scoring and add-direction steering are real
workflow surfaces. The paper-rerun workflow uses the released prompt-source
data, paper constants, and an env-configured 0-3 role-adherence judge. Full
paper-faithful recomputation means running that surface at paper scale with
`ASSISTANT_AXIS_MIN_ROLE_EXAMPLES_PER_ROLE=50`. Activation capping is documented
through released configs and still needs a dedicated first-class intervention
operator.
