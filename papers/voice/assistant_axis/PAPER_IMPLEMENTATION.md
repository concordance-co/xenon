# Assistant Axis Paper Implementation

Paper: Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish, and Jack
Lindsey, "The Assistant Axis: Situating and Stabilizing the Default Persona of
Language Models", arXiv:2601.10387, 2026.

Upstream surfaces:

- code: `safety-research/assistant-axis`
- released prompt-source table: `belmore/assistant-axis-vector-prompts`
- released vector artifacts: `lu-christina/assistant-axis-vectors`

This directory has two distinct surfaces:

1. **Paper implementation**: reproduce or reuse the Assistant Axis paper
   artifacts for the paper's released models and prompt/source data.
2. **Method BYOD**: use the same default-vs-role contrast method on other
   domains or datasets.

Do not conflate these. BYOD can use arbitrary grouped conditions. Paper
implementation should preserve the paper's source data, prompt expansion,
model/layer choices, filtering, and released artifact conventions.

## Current Status

### Off The Shelf: Released Vectors

Implemented.

The repo can load released Assistant Axis vectors and trait vectors from
`lu-christina/assistant-axis-vectors` through:

- `AssistantAxisPrecomputedCoordinateSpec`
- `AssistantAxisTraitCoordinateSpec`

Known released model configs:

| model | total layers | target layer | assistant axis | default vector | capping |
|---|---:|---:|---|---|---|
| `google/gemma-2-27b-it` | 46 | 22 | `gemma-2-27b/assistant_axis.pt` | `gemma-2-27b/default_vector.pt` | none |
| `Qwen/Qwen3-32B` | 64 | 32 | `qwen-3-32b/assistant_axis.pt` | `qwen-3-32b/default_vector.pt` | `qwen-3-32b/capping_config.pt`, experiment `layers_46:54-p0.25` |
| `meta-llama/Llama-3.3-70B-Instruct` | 80 | 40 | `llama-3.3-70b/assistant_axis.pt` | `llama-3.3-70b/default_vector.pt` | `llama-3.3-70b/capping_config.pt`, experiment `layers_56:72-p0.25` |

Validated example:

- workflow: `scripts/assistant_axis_llama33_70b_trait_trace_score.py`
- run: `wr_fe6ec0ba5ec8_5b28235a`
- result: one fake trace scored against Llama 3.3 70B assistant axis plus ten
  released trait coordinates.

### Source Prompt Data

Implemented as scaffold, not run at paper scale here.

`assistant_axis_prompt_dataset(...)` points at
`belmore/assistant-axis-vector-prompts`. This is the prompt/source table used
as the canonical local hook for paper-style derivation. The helper returns one
row per source role/trait/default source with nested prompt fields.

Paper source data shape:

- upstream code source: `data/roles/instructions/*.json`
- role sources: 275 character roles
- default source: one `default.json`
- instruction variants per source: 5
- extraction questions: 240
- default generations per source: `5 * 240 = 1200`
- released HF prompt-source dataset sha observed locally:
  `57424a9d6075a44196b935983ce1fa4e83191679`

Default assistant instructions:

```text

You are an AI assistant.
You are a large language model.
You are {model_name}.
Respond as yourself.
```

The first default instruction is the empty string.

Prompt assembly matches the upstream generator:

1. Replace `{model_name}` in the instruction with the upstream model short name
   (`Gemma`, `Qwen`, or `Llama` for the released configs).
2. If the tokenizer chat template supports system turns, pass the instruction as
   a system message and the extraction question as the user message.
3. If the tokenizer does not support system turns, concatenate instruction,
   blank line, and question into one user message.
4. For Qwen models, disable thinking in chat-template kwargs.

Generation defaults:

| setting | value |
|---|---:|
| `max_model_len` | 2048 |
| `temperature` | 0.7 |
| `top_p` | 0.9 |
| `max_tokens` | 512 |
| `question_count` | 240 |
| `prompt_indices` | `0,1,2,3,4` |

### From-Scratch Axis Derivation

Workflow scaffold implemented; expensive paper-scale rerun intentionally not
launched.

`AssistantAxisVectorSpec` computes:

```text
mean(default_response_activations) - mean(per_role_role_playing_vectors)
```

It supports:

- default-vs-role labels
- per-role aggregation
- optional adherence-score filtering
- selected model layers
- response-span pooling

Paper-faithful derivation details now encoded in `paper.py` and wired by
`specs/paper_rerun.py`:

- responses are generated for all role/default instruction-question pairs
- response activations are mean-pooled over assistant response turns
- upstream activation extraction defaults to all layers, max length 2048, batch
  size 16
- role adherence is judged by `judge.py`, defaulting to `gpt-4.1-mini` but
  configurable with `ASSISTANT_AXIS_JUDGE_MODEL`
- judge labels are 0, 1, 2, 3
- role vectors use only score-3 responses (`FULLY ROLE-PLAYING`)
- default vectors use all default activations without judge filtering
- minimum score-3 count per role is 50
- final axis is saved as a tensor with shape `(n_layers, hidden_dim)`

Current scaffold status:

- `specs/paper_rerun.py` loads the released HF prompt-source table,
  expands paper prompts, generates responses, runs the 0-3 role-adherence
  judge, converts judged responses into response-span capture rows, captures
  residual activations, derives an axis, loads the released axis, and scores
  the generated responses against both coordinates.
- For cheap dry runs, set `ASSISTANT_AXIS_JUDGE_DRY_RUN=1`; for paper-scale
  reruns, leave the judge enabled and set
  `ASSISTANT_AXIS_MIN_ROLE_EXAMPLES_PER_ROLE=50`.

Judge scale:

| score | paper meaning |
|---:|---|
| 0 | not role-playing; model refused while identifying as itself |
| 1 | not role-playing; model identifies as itself but attempts answer |
| 2 | model identifies as itself while showing some role attributes |
| 3 | fully playing the role, including refusals while still identifying as the role |

### Scoring And Monitoring

Implemented.

`ProjectionSpec` and `AssistantAxisScoreSpec` score captured spans against
released or recomputed coordinates. This is already usable for BYOP/BYOT/BYOD
when rows provide the required assistant-response section.

Runnable paper-package specs:

| surface | workflow | purpose |
|---|---|---|
| paper rerun | `specs/paper_rerun.py` | released prompt source -> generated responses -> recomputed axis -> released-axis comparison |
| BYOD | `specs/byod_axis.py` | arbitrary default-vs-role trace rows -> custom axis and scores |
| BYOT | `specs/byot_score.py` | existing trace rows -> released axis/trait scores |
| BYOP | `specs/byop_generate.py` | prompt -> baseline generation, trait-steered generation, and output scores |

### Steering / Capping

Partial.

The smoke workflow demonstrates add-direction steering with a released
precomputed trait vector:

- workflow: `scripts/assistant_axis_llama33_70b_precomputed_steering.py`
- run: `wr_96588bb43527_9c18a1da`
- vector: `calm`, Llama 3.3 70B, layer 40
- intervention: `AddDirectionPatch` at `resid_post`, strength `2.0`, prompt and
  decode tokens
- result: one baseline and one steered generation completed with non-empty patch
  stats for layer 40

The paper's activation capping behavior is represented as released capping
config metadata, but capping is not yet a first-class Xenon intervention
operator.

## What Counts As Paper-E2E

For this paper, paper-e2e means:

1. Load the released source prompt data.
2. Expand sources into concrete default and role-play prompt invocations.
3. Generate responses on a supported paper model with the defaults above.
4. Score/filter role adherence using the 0-3 judge scale above.
5. Capture response activations, normally all layers, and use the released
   model's target layer for paper comparison.
6. Compute the Assistant Axis vector with `AssistantAxisVectorSpec`.
7. Compare the recomputed vector to the released vector for that model.
8. Score heldout/probe traces and package a report.
9. Optionally run capping/steering once the intervention operator exists.

We are not claiming that complete loop has been rerun locally. The concrete
paper details, source hooks, model configs, vector artifacts, judge step,
thresholds, and prompt assembly semantics are now recorded so the rerun is
mechanical.

## Files To Edit

- `method.py`: method/BYOD recipe surface.
- `paper.py`: paper-specific source data, prompt expansion, and rerun
  constants.
- `runtime.py`: shared workflow transforms, env knobs, and dataset
  adapters.
- `specs/workflow.py`: tiny ToyEngine method smoke.
- `specs/paper_rerun.py`: actual paper-source rerun scaffold.
- `specs/byod_axis.py`: user data -> method axis/scores.
- `specs/byot_score.py`: trace data -> released coordinate scores.
- `specs/byop_generate.py`: prompt -> generation, steering, output scores.
- `scripts/assistant_axis_llama33_70b_trait_trace_score.py`: real released-vector
  trace scoring example.
- `scripts/assistant_axis_llama33_70b_precomputed_steering.py`: real
  released-vector steering example.
