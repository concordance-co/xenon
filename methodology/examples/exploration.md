# Explore learned features

Illustrative example; no experiment was run.

> “I want to see what kinds of features occur in these conversations.”

The request can be useful without a precise causal hypothesis.
The agent can propose a way to inspect feature activations across a relevant conversation sample.
It should explain what the sample covers and what the chosen representation exposes.

Possible approaches include direct activation summaries, clustering, or a compatible pretrained SAE.
The choice depends on the model, available dictionary, data, and desired view.
A new dictionary training project is not necessary merely because feature exploration was requested.

Inspect examples with high and low activation, not only a name assigned to the feature.
Compare across topics or formats when those differences might explain a pattern.
An observed pattern can motivate a later question without establishing a mechanism.

The agent can propose follow-up work when a pattern is useful or surprising.
It need not force each observation into a hypothesis catalog or open a new phase.
The requested coverage can justify a broad exploration from the start.

## Connection to Xenon

Use [native-site summaries](../../platform/NATIVE_SITES.md) or a supported
[SAE dictionary](../../pipelines_v2/representation/sae/README.md).
The existing SAE support uses a pinned GPT-2 layer-6 dictionary.
Do not assume it works with another model or layer.
Use [evidence views](../../platform/EVIDENCE.md) for supported summaries.
Keep custom exploration views with the study.
