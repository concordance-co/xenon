# Investigate a probe result

Illustrative example; no experiment was run.

> “This probe predicts the model's answer. Is it reading the representation that drives the answer?”

This is already a useful question. The agent does not need to invent a deeper one.
It can distinguish predicting an answer from identifying information the model uses.

One candidate approach is to compare examples where the answer differs while other
relevant properties remain similar. Probe evaluation on held-out examples can test prediction.
An intervention can then test whether changing the suspected representation affects the answer.

The intervention should address a specific alternative explanation.
For example, a matched control can help distinguish an answer-specific effect from general disruption.
A no-op can check the edit implementation. These checks answer different questions.
The agent should select controls for the proposed claim rather than apply every available control.

A failed intervention can reflect the site, edit, or representation choice.
It does not automatically establish that the probe found irrelevant information.
A successful intervention also needs interpretation in light of the controls and edit size.

## Connection to Xenon

Use the relevant probe operation and supported capture adapter.
The [native-site guide](../../platform/NATIVE_SITES.md) and
[Torch guide](../../platform/TORCH_ENGINE.md) describe available edit paths.
Check the model, coordinates, and token alignment before combining artifacts.
The researcher supplies the comparison data and outcome measure.
