# Improve a behavior

Illustrative example; no experiment was run.

> “Can we use interpretability to make this assistant follow our output format?”

The underlying objective might be reliable structured output. It might also be
understanding why the model ignores a format instruction. The distinction affects the approach.
The agent can surface both interpretations and recommend one from the user's context.

If reliable output is the goal, compare the current prompt with candidate instructions
or an available structured-output feature. Use representative inputs and a parser-based evaluation.
Check content quality as well as format. A valid object with an empty answer can still fail the task.

This needs examples of the formats, inputs, and failures that matter in use.
The agent can inspect available traces and propose missing cases.
It need not make the user design the evaluation first.

If a candidate meets the user's needs, improving behavior may complete the task.
If important failures remain, internal measurements or interventions may help investigate them.
If the original goal was mechanism, a format improvement alone would not answer it.

## Connection to Xenon

Use a supported generation workflow when it fits the model and execution needs.
Keep the format parser and task-specific scoring in the research repo.
The [LLM guide](../../platform/LLM_RESEARCH.md) covers supported execution paths.
A different model API can be used directly when that better serves the evaluation.
Xenon need not perform every part of an investigation.
