# Phase 18 Patching Method Redesign

## Why this needs to change

The current Phase 18 setup is a valid first causal probe, but it is not yet a strong best-practices activation patching experiment.

What we have been testing is mostly:

- a **necessity-style ablation**
- on a **market-only** prompt distribution
- using **section-mean subspace project-out**
- measured mostly with **final tool / token changes**

That is useful, but it leaves several methodological gaps called out in the patching review:

- no explicit **clean/source vs corrupt/base** contrast
- no **reverse/noising** test
- no proper **lambda sweep**
- limited **random-direction** coverage
- no **neighbor-PC / subspace-size** sweep support baked into the runner
- weak **distribution-shift diagnostics**
- weak **uncertainty reporting**


## Current Phase 18 claim scope

The current runs support a narrow claim:

- removing selected market subspace components from the market section can change downstream behavior

They do **not** yet support the stronger claim:

- this exact market variable is the unique causal mediator of the downstream behavior

That stronger claim needs paired source/base tests plus a broader control battery.


## What has been changed now

These changes are implemented in the current codebase.

### 1. Lambda-capable ablations

`strength` now scales:

- `project_out`
- `random_control`
- `swap_mean`

This makes proper lambda sweeps possible without changing the patch operator.

Relevant file:

- [vllm_market_patch.py](/Users/brockelmore/concordance/xenon/pipelines/interp/vllm_market_patch.py)


### 2. Explicit component selection

The runners now support explicit per-layer component selection instead of only:

- named single-PC directions
- or the first `k` PCs

This is required for:

- neighboring-PC controls
- subspace-size sweeps
- exact reproducible component sets

Supported forms:

- global: `0,1,2,3`
- per-layer: `4=0,1,2,3;35=0,1,2,3`

Relevant files:

- [synthetic_market_patching_runner.py](/Users/brockelmore/concordance/xenon/pipelines/interp/synthetic_market_patching_runner.py)
- [synthetic_market_behavior_runner.py](/Users/brockelmore/concordance/xenon/pipelines/interp/synthetic_market_behavior_runner.py)
- [modal_vllm_patching.py](/Users/brockelmore/concordance/xenon/pipelines/interp/modal_vllm_patching.py)


### 3. Better behavior-analysis summaries

Behavior analysis now reports more than one headline rate.

Added:

- bootstrap 95% CIs for core change rates
- median and mean deltas
- patch-applied / patch-skipped rates
- patch norm diagnostics from `patch_stats_json`
- per-family-variant summaries

This does not fix the causal design by itself, but it makes the readout much less brittle.

Relevant file:

- [synthetic_market_behavior_analysis.py](/Users/brockelmore/concordance/xenon/pipelines/interp/synthetic_market_behavior_analysis.py)


## Revised protocol

### Stage A. Localized necessity tests

Keep the current `project_out` path, but make it a real robustness battery:

1. Run the target component or subspace.
2. Run matched orthogonal random controls with multiple seeds.
3. Sweep lambda:
   - `0.0`
   - `0.25`
   - `0.5`
   - `0.75`
   - `1.0`
   - `1.5`
4. Sweep neighboring PCs and subspace size.

This tells us whether the effect is:

- target-specific
- smooth under intervention size
- stronger than generic perturbation


### Stage B. Paired source/base swap tests

This is the most important missing piece.

For each targeted hypothesis, build matched prompt pairs:

- same family
- same family variant
- same roster
- same context variant
- different value of the prompt-derived target property

Examples:

- leader-like contrast:
  - higher vs lower `vol_1h_max`
  - or higher vs lower `pct_1h_max`
- dispersion-like contrast:
  - higher vs lower `pct_1h_mad`
  - with `pct_1h_std` and `pct_1h_gap` tracked as alternatives

Then run both directions:

- **denoising / sufficiency**
  - patch clean/source mean into corrupt/base
- **noising / necessity**
  - patch corrupt/base mean into clean/source

The preferred operator here is not pure project-out.
It is:

- `swap_mean`
- or a coefficient/difference patch equivalent

This turns the experiment from “remove some representation” into “transfer a specific causal contrast.”


### Stage C. Better outcome metrics

Final tool identity is too coarse as the only readout.

Next runs should track:

- final tool name
- chosen asset token
- spend percent
- generated token count
- per-example restoration rates
- prompt-family-stratified restoration

If we can cheaply expose logprobs later, add:

- target-vs-foil logit or logprob margin

But that is optional for the next round. The paired-source/base design matters more.


### Stage D. Distribution diagnostics

Every intervention run should summarize:

- `delta_norm_std`
- `mean_norm_before / after`
- `mean_std_norm_before / after`
- random-control matched-norm checks

This is the minimum needed to catch “large off-manifold perturbation” problems.


## Recommended next experiments

### 1. L4 leader subspace robustness battery

- site: `market_mean @ L4`
- components:
  - target top-4
  - neighboring slices like `1..4`, `2..5`
- controls:
  - 5-10 random seeds
- lambda sweep:
  - `0.25` to `1.5`


### 2. L35 dispersion subspace robustness battery

Same structure as above.


### 3. First paired swap experiment

Build matched source/base prompt pairs for:

- leader-like prompt-derived property
- dispersion-like prompt-derived property

Run:

- clean -> corrupt
- corrupt -> clean

This is the experiment that upgrades the causal claim materially.


## Interpretation guardrails

Even after the redesign, we should still avoid overclaiming:

- PCA directions are variance directions, not guaranteed native causal variables.
- Subspace patching can still work through alternate pathways.
- Broad section-mean interventions are stronger for sufficiency than for localization.

So the right standard is:

- target beats matched random controls
- effect is stable across lambda
- effect is stronger than neighboring PCs
- denoising and noising both behave sensibly
- diagnostics stay within a reasonable distributional range

Only then should we treat the subspace as a serious causal object rather than a convenient descriptive basis.
