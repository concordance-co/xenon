# BYOD Method Schemas

BYOD in this tree means "bring data that fits the method," not "bring the
paper's original semantic domain." The paper directories keep their historical
names, but their schemas are method-level contracts.

## Where To Edit

- `assistant_axis/method.py`:
  default-vs-role contrast data and precomputed axis/trait scoring.
- `emotions/synth_data.py`:
  concept-labeled vector spaces with optional neutral/background projection.
- `refusal_direction/synth_data.py`:
  two-pole contrast directions.
- `honest_llama/synth_data.py`:
  binary truth/correctness readouts.
- `schemas/*.schema.json`:
  row contracts for BYOD files and generated-data outputs.

These method/data files are intentionally small. Edit labels, topics, seed
items, counts, split policy, and prompt templates there before wiring a larger
workflow. The JSON schemas are the stable boundary: if your imported or
generated rows satisfy the relevant schema, the adapter can turn them into
Xenon `Dataset` examples with the right labels and token sections.

## Method Contracts

### Assistant Axis Method

Schema: `schemas/assistant_axis_method.schema.json`

Use for:

- scoring traces against released assistant-axis or trait coordinates
- deriving a new default-vs-role axis from your own grouped conditions

Required scoring row shape:

```json
{
  "example_id": "trace_001",
  "text": "Human: ...\n\nAssistant: ...",
  "assistant_response": {"char_start": 42, "char_end": 180},
  "model_id": "meta-llama/Llama-3.3-70B-Instruct"
}
```

Additional fields for deriving a new axis:

```json
{
  "axis_kind": "role",
  "role": "analyst_mode",
  "adherence_score": 3,
  "split": "train"
}
```

### Concept Vector Space Method

Schema: `schemas/concept_vector_space_method.schema.json`

Use for any labeled concept set. The paper default is emotions, but concepts
can be financial states, support intents, failure classes, styles, or other
semantic groups.

```json
{
  "example_id": "row_001",
  "text": "The market opened with widening credit spreads and defensive positioning.",
  "row_role": "concept",
  "concept": "risk_off",
  "split": "train",
  "topic": "morning market note"
}
```

Neutral/background rows use `row_role: "neutral"` and do not need `concept`.

### Contrast Direction Method

Schema: `schemas/contrast_direction_method.schema.json`

Use for any two-pole direction. The paper default is refusal, but the method
only needs a defined positive pole and negative pole.

```json
{
  "example_id": "contrast_001",
  "text": "Rendered prompt or trace text.",
  "label": "positive_pole",
  "positive_label": "positive_pole",
  "negative_label": "negative_pole",
  "split": "train"
}
```

### Binary Truth/Correctness Method

Schema: `schemas/binary_truth_readout_method.schema.json`

Use for true-vs-false, correct-vs-incorrect, or equivalent binary answer
contrasts.

```json
{
  "example_id": "claim_001_true",
  "prompt": "Question or claim prompt.",
  "answer": "Candidate answer text.",
  "label": "true",
  "claim_id": "claim_001",
  "split": "train"
}
```

## Adapter Rule

Adapters should do the boring work:

1. Validate rows against the method schema.
2. Render text if the schema stores prompt/answer separately.
3. Attach `token_sections` and `section_records` from char spans.
4. Preserve method labels under stable names.
5. Pass the resulting `Dataset` into the existing workflow specs.

Do not bake paper-domain assumptions into adapters. Put domain defaults in
the paper's method-data file, and keep workflow specs pointed at method labels.
