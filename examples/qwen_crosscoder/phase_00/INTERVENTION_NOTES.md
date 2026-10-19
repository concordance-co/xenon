# Intervention notes

This is a plumbing proof, not a causal study. Layer 12 residual input is both the read and write locus; last prompt token only is edited. The original base and instruct checkpoints are loaded independently and verified byte-for-byte.

- No-op returns the original activation and must preserve logits exactly.
- Delta adds decode(one-hot feature 0×10)−decode(zero), preserving the baseline reconstruction error.
- Reconstruction replaces the token with decode(encode(paired activation)) on that model side; its effect is reported separately.
- Independent direct HF prehook uses the original decoder weight and explicit FP32 bias arithmetic; fixed rtol 1e-4/atol 1e-5, no tolerance retuning.

Feature0 and amount 10 are predetermined engineering settings. No feature selection, held-out tuning, scientific null interpretation or model-difference conclusion follows from this run.
