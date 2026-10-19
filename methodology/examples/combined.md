# Combine behavioral and internal measurements

Illustrative example; no experiment was run.

> “Our model ignores a constraint when the prompt gets long. Can we find out why and fix it?”

The user wants both an explanation and an improvement.
The agent can retain both goals instead of choosing only a behavioral or internal approach.

Candidate comparisons include constraint position, prompt length, and distracting content.
Matched prompts can help distinguish these factors. The outcome measure needs to check
whether the response follows the constraint, not merely whether it mentions the instruction.

Prompt changes can test practical improvements. Captures and interventions can investigate a proposed
explanation for the same failures. The work can proceed together when the data and compute allow it.
There is no requirement to finish every behavioral test before inspecting internals.

Suppose a prompt change improves performance but the suspected internal edit has no clear effect.
The improvement and the unresolved explanation should be reported separately.
Suppose an edit affects behavior only on the analysis examples. That motivates a transfer check,
but it does not erase what those examples showed.

## Connection to Xenon

Use [configurable generation and edits](../../platform/LLM_RESEARCH.md) for supported models.
Keep task scoring and matched prompt construction in the research repo.
Check sequence length, capture volume, and compute settings for the requested workload.
A finding on a supported small model does not establish the same mechanism in the user's production model.
