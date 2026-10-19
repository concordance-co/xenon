# Compare two models

Illustrative example; no experiment was run.

> “What changed inside this model after instruction tuning?”

This can mean changes in behavior, changes in representations, or the connection between them.
The agent can propose a comparison that matches the user's interest.
It should not silently reduce an internal comparison to a behavior benchmark.

Before choosing an analysis, consider which inputs make the models comparable.
Identical text, each model's normal prompt format, and matched output positions answer different questions.
Token positions are not automatically equivalent across tokenizers or templates.

If the goal is practical behavior, paired outputs and task scores may be sufficient.
If the goal is representation change, aligned captures can support a crosscoder or another comparison.
Both views can be useful. Alignment does not establish that individual coordinates have the same meaning.

The data should cover the behaviors or contexts the user cares about.
A convenient collection of prompts may omit the conditions where the models differ.
The agent should identify that gap before treating the comparison as comprehensive.

## Connection to Xenon

The [crosscoder recipe](../../examples/qwen_crosscoder/README.md) supports the pinned
Qwen2.5-0.5B base/instruct pair, aligned capture, fitting, continuation, and feature edits.
Use its configuration for the study's data and budget.
A different model pair may need adapter work. State that separately from the experiment plan.
