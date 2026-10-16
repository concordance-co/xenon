# Pretrained transcoder tracing plumbing

Premise: establish an executable pinned official PLT trace and preserve its distinct replacement-model semantics in C07.

Design: Qwen3-0.6B, official low-L0 PLTs, one `Hello` prompt, two selected features, one logit, backward batch one, FP32 A100-80GB. All error nodes and nonzero matrix edges survive import. Source and destination layer maps remain explicit. Model/transcoder preparation is a separate CPU step.

Verification: typed PLT and CLT-format fixtures; unknown schemas and mismatched requests rejected; real isolated interpreter import before GPU; one bounded actual trace; populated artifact-backed local report. No full research campaign. Same-prompt replacement diagnostic and optional original-model direction edit cannot establish fidelity or causal route quality.

Scientific verdict: **not tested**. Original-model route: **unvalidated**. Live CLT: **not tested**. Measured execution evidence belongs in the T20 receipt and actual artifacts. T06/T18 acceptance dependencies and independent review remain explicit.

Actual engineering gate (2026-10-16): normal workflow `wr_t20_qwen3_plt_smoke` passed all three steps on the first GPU canary. The official trace imported 61 nodes and 90 signed edges, retaining all 56 error nodes. GPU peak allocation was 43,015,907,840 bytes; aggregate tool-process peak RSS was 11,747,143,680 bytes. Model loading, attribution and same-prompt diagnostics took 17.19 seconds after verified cold asset reads; whole workflow took 351.97 seconds. Artifact: `circuit_trace_1_326bb8eaf356`.

The strict HF/TransformerLens forward parity gate did not pass (maximum absolute logit difference 1.55e-5, absolute tolerance 1e-5), so the optional original-model edit was skipped. Tolerance was not relaxed. Replacement diagnostics are displayed as same-prompt engineering observations only. No claim about circuit quality or held-out fidelity follows from this run.
