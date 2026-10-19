# Configurable crosscoder recipe

A completed CPU/Modal CUDA recipe for aligned Qwen2.5-0.5B base/instruct capture, pinned-reference fitting, explicit checkpoint continuation, encode/decode and feature splices in both original models. Configuration controls the dataset, split, width, steps and resource budgets.

Full CPU/CUDA workflows and fresh-process reuse passed. No-op and independent delta checks verified the edit path. Token alignment does not imply neuron correspondence; reconstruction is lossy. Feature quality and scientific model differences need a separate study.

Use [the recipe guide](../README.md) and `workflow.py` for current execution instructions. Record new experiment outcomes in a new phase rather than extending this fixture’s old run history.
