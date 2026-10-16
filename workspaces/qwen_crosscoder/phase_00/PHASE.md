# Phase00 — crosscoder engineering proof

## Premise and scope

Verify paired original-model capture, safe resumable reference fitting, dictionary encode/decode and original-model feature edits. Latest user steering replaces T19's research campaign with this bounded plumbing proof. No model-difference, feature-quality, causal-mechanism or stability claim is tested.

## Frozen design

Pinned Qwen2.5-0.5B base/instruct; identical raw token IDs, no chat template. Four fixed prompts in workflow.py: first 3 train, final 1 held out. Layer 12 resid_pre is a convenient matched engineering site, not a discovered causal locus. CPU FP32 eager, seed 0, width 16, 20 steps, batch 16, Adam lr 0.001, L1 penalty0.001, warmup 1. Checkpoint after 10 steps; resume to 20. One preselected feature 0, amount 10, final prompt token, both original models.

## Verification

Independent original HF forward/capture/patch parity for both checkpoints; exact paired token/position/content identity; corrupted or foreign checkpoints rejected; uninterrupted and safely serialized/resumed fixture fitting must match complete state exactly. Real original-model no-op must preserve logits exactly. Feature delta and full reconstruction are measured separately; the delta oracle independently applies decoder-weight arithmetic at the original residual site, preserving FP32 bias rounding.

## Boundaries

Token alignment does not equate arbitrary neurons across architectures: both models have the same pinned dense architecture and ordered-side residual tensors. Reconstruction is lossy and never a no-op. A changed held-out logit proves hook plumbing only. Random/same-label controls, two-seed feature stability and hypothesis evaluation are deliberately deferred. Scientific verdict: not_tested.

Runtime, phase status and observed run IDs are recorded in the execution receipt and generated summary. No interrupted or failed process is promoted to canonical storage.

## General-purpose execution extension

The original four-prompt design above remains an example, not an API ceiling. The configurable workflow now accepts explicit dataset/split/fit settings and finite resource budgets on local CPU or a separately locked Modal CUDA runtime. The current engineering gate uses six prompts (four train/two held out), width 64, six fit steps with checkpoint at step 3, feature 40 and amount 10. CPU run `wr_scratch_4e03ab205d2c49af859831bd9990eab7` completed all eight stages with resolved provenance; no-op and independent delta-oracle errors were zero for both held-out prompts in both original models. No research claim or held-out retuning follows from these checks.

The matching six-prompt CUDA workflow completed as `wr_b8b8539a74b4_080b674f` with all eight identities resolved and zero no-op/direct-delta error. The final-source two-step reuse gate also passed: initial `wr_ab55ac0771b9_315ce039`, fresh-process reuse `wr_ab55ac0771b9_613efa70`, two CPU readiness calls and zero GPU execution calls. Details are recorded in `generalization_receipt.json`.
