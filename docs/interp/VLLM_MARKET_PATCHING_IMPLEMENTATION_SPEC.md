# vLLM Market Patching Implementation Spec

## Goal

Add causal patching to the current vLLM worker stack so we can intervene on the discovered market-representation subspace and measure:

- downstream representation changes
- behavioral changes
- whether the subspace is strong enough to justify CLT work

This spec is for the current Qwen3 MoE + vLLM setup already used for capture.


## What We Already Have

The current stack already proves the two hard prerequisites:

1. We can run code directly on the model inside vLLM workers via `LLM.apply_model(...)`.
2. We can monkey-patch model forwards inside workers in eager mode.

That is already how router capture works now.

Relevant current code:

- [pipelines/interp/vllm_capture.py](/Users/brockelmore/concordance/xenon/pipelines/interp/vllm_capture.py)
- [pipelines/interp/vllm_qwen3_moe.py](/Users/brockelmore/concordance/xenon/pipelines/interp/vllm_qwen3_moe.py)
- [pipelines/interp/modal_synthetic_capture.py](/Users/brockelmore/concordance/xenon/pipelines/interp/modal_synthetic_capture.py)

Important constraints from the current setup:

- `enforce_eager=True` must stay on
- start with `max_num_seqs=1`
- the current capture path is prefill-only, so causal behavior runs need a separate generation path


## Core Design Choice

The discovered object is `market_mean`, which is a pooled section statistic, not a native token state.

So the intervention should not pretend that `market_mean` is a special token.

Instead, patch the token-level hidden states inside the market section in a way that directly changes the section mean.

### Tokenwise Mean-Shift Intervention

For a target layer and market token span:

- let `H` be the token hidden states in the market span, shape `(n_tokens, d_model)`
- let `mu = mean(H, dim=0)`
- let `c` be the Phase 17 centering vector for that layer/state
- let `B` be the orthonormal market basis matrix for that layer, shape `(d_model, k)`

Then:

- centered mean: `z = mu - c`
- subspace coefficients: `a = B^T z`

Interventions operate on `z`, then convert the mean change back into a uniform token delta applied to every market token.

This is the cleanest way to make a pooled subspace causal while minimally disturbing within-section token structure.


## Intervention Modes

Start with four modes.

### 1. `project_out`

Remove the component of the section mean in the target basis.

- `delta_mean = - B (B^T z)`
- apply `delta_mean` to every token in the market span

Use cases:

- full 4D subspace ablation
- leader-only ablation
- dispersion-only ablation

### 2. `add_direction`

Move the section mean along one basis direction or a small linear combination.

- `delta_mean = strength * direction`

Use cases:

- leader-up / leader-down
- dispersion-up / dispersion-down

### 3. `swap_mean`

Replace the current section mean with the mean from a donor prompt.

- `delta_mean = donor_mu - mu`

Use cases:

- matched prompt pair where only leader or dispersion differs
- strongest directional-control test

### 4. `random_control`

Same-norm intervention using a random orthogonal direction or subspace.

Use cases:

- mandatory control baseline


## Working Subspace

Start with the same practical working basis already discussed in the patching plan:

- `market_mean @ L4` leader axis
- `market_mean @ L35` dispersion axis
- plus the next two stable top-PC directions that complete the working `~4D` market subspace

Do not make the first implementation depend on final semantic naming of all four directions.

The implementation should support:

- 1D basis
- 2D basis
- full 4D basis


## Exact First Experiment

### Dataset

Use the improved DX-like synthetic prompts first.

Start with market-only prompts before moving to settings/context prompts.

### First interventions

Run:

1. full 4D `project_out`
2. leader-only `project_out`
3. dispersion-only `project_out`
4. leader-up / leader-down `add_direction`
5. dispersion-up / dispersion-down `add_direction`
6. matched random orthogonal controls

### First readouts

Lead with representational readouts before final action labels:

1. market row ranking shifts
2. score / geometry shifts
3. later section-state shifts
4. only then final action / tool-call changes


## Worker-Side Architecture

### New files

Add these modules:

- `pipelines/interp/vllm_market_patch.py`
- `pipelines/interp/vllm_patching_runner.py`
- `pipelines/interp/modal_vllm_patching.py`
- `pipelines/interp/synthetic_market_patching_analysis.py`

Optional later:

- `scripts/prepare_synthetic_market_phase18_patching.py`


## `vllm_market_patch.py`

This is the worker-side patching module, analogous to `vllm_qwen3_moe.py`.

### Responsibilities

1. find decoder layers
2. install patched forwards
3. maintain active patch spec state
4. apply tokenwise mean-shift interventions at target layers
5. restore original forwards

### Required functions

#### `find_decoder_layers(model) -> dict[int, Any]`

Return the transformer layers to patch.

For Qwen-style models this will likely traverse `model.model.layers`.

#### `init_market_patching(model) -> None`

For each decoder layer:

- save original `forward`
- install wrapped `forward`
- initialize disabled patch state

#### `set_patch_spec(model, patch_spec) -> None`

Activate a patch spec for the next request.

#### `clear_patch_spec(model) -> None`

Disable patching after each request.

#### `restore_original_forwards(model) -> None`

Restore the model to unpatched state.


## Decoder-Layer Wrapper Behavior

Patch at the decoder-layer output, not deep inside the MLP.

Why:

- it is closer to the residual states we actually analyze
- it is more model-agnostic
- it avoids tying the first implementation to one internal submodule path

### Wrapped forward logic

Pseudo-behavior:

1. call original layer `forward`
2. extract the hidden-state tensor from the output
3. if:
   - patching is enabled
   - current layer is targeted
   - current request has a valid market token span
   then apply the tokenwise mean-shift intervention to the market tokens
4. return the modified hidden states in the original output structure

The wrapper must handle both:

- tensor outputs
- tuple outputs where the first item is hidden states


## Patch Spec Format

Use a plain dataclass or serializable dict.

### Required fields

- `request_id`
- `mode`
- `target_layers`
- `section_name`
- `section_token_spans`
- `center_vectors`
- `basis_matrices`
- `strength`
- `direction_weights`
- `donor_means`
- `control_seed`

### Minimal concrete schema

```python
@dataclass(slots=True)
class PatchSpec:
    mode: str  # project_out | add_direction | swap_mean | random_control
    target_layers: list[int]
    section_name: str  # "market"
    token_span: tuple[int, int]  # half-open [start, end)
    center_by_layer: dict[int, torch.Tensor]  # (d_model,)
    basis_by_layer: dict[int, torch.Tensor]   # (d_model, k)
    strength: float = 1.0
    direction_weights_by_layer: dict[int, torch.Tensor] | None = None  # (k,)
    donor_mean_by_layer: dict[int, torch.Tensor] | None = None  # (d_model,)
    random_basis_by_layer: dict[int, torch.Tensor] | None = None
```


## Where Token Spans Come From

Do not rediscover section spans inside the worker.

The host should compute them ahead of time using the same prompt-structure logic already used for pooling.

Relevant existing structure code:

- [pipelines/interp/counterfactual/capture.py](/Users/brockelmore/concordance/xenon/pipelines/interp/counterfactual/capture.py)
- [pipelines/interp/decision_structure/core.py](/Users/brockelmore/concordance/xenon/pipelines/interp/decision_structure/core.py)

The patch runner should send, for each prompt:

- tokenized prompt
- market start token
- market end token

The intervention should use the exact same half-open market span used for later pooling.


## Basis Assets

The patch runner needs a basis artifact on disk.

Create a compact artifact derived from Phase 17:

- centering vector per target layer
- orthonormal basis matrix per target layer
- named directions for leader and dispersion
- optional donor mean library for matched prompt pairs

Suggested path:

- `data/analysis_results/synthetic_market_axis_decomposition/phase17_market_axis_decomposition_v1/patch_basis.pt`

This keeps patching decoupled from the larger JSON report outputs.


## `vllm_patching_runner.py`

This is the orchestration layer.

### Responsibilities

1. load patch basis artifact
2. load prompts and precomputed token spans
3. build patch specs
4. run worker-side patching requests
5. save:
   - patched generations
   - patched pooled states
   - intervention metadata

### Two modes

#### Mode A: representational patching

- prefill-only or short decode
- save downstream pooled states
- used for fast sanity checks

#### Mode B: behavioral patching

- allow the model to continue to its normal decision output
- save response / tool-call outcome
- used only after representational sanity checks pass


## `modal_vllm_patching.py`

This should mirror the current capture workers, but add:

- worker setup that initializes market patching via `apply_model`
- request method that:
  - sets a patch spec
  - runs one prompt
  - clears patch spec

Keep:

- `enforce_eager=True`
- `max_num_seqs=1`

For the first version, do not optimize for high throughput.

Correctness first.


## Exact Intervention Math

For a targeted layer:

1. get market token slice `H`
2. compute `mu = mean(H, dim=0)`
3. compute `z = mu - c`

### `project_out`

```python
proj = B @ (B.T @ z)
delta_mean = -proj
H_new = H + delta_mean
```

### `add_direction`

```python
direction = B @ w
delta_mean = strength * direction
H_new = H + delta_mean
```

### `swap_mean`

```python
delta_mean = donor_mu - mu
H_new = H + delta_mean
```

### `random_control`

Same as `add_direction` or `project_out`, but use a matched-norm orthogonal random direction/subspace.

### Why uniform token delta is the default

Applying the same delta to every market token:

- changes the pooled `market_mean` exactly by `delta_mean`
- preserves relative structure inside the section better than token-specific edits
- is simple enough to reason about causally

Later, if needed, we can add token-weighted interventions.


## First Metrics

### Representational

1. change in downstream section means
2. change in row ranking / leader identity
3. change in market geometry metrics
4. change in post-market settings / constraints representations

### Behavioral

1. tool-call type
2. chosen asset
3. chosen size / aggression
4. observe vs act


## Mandatory Sanity Checks

These gate the whole patching line.

### 1. Null patch

Patch a prompt with itself.

Expected:

- near-zero representation delta
- no material behavior change

### 2. Random orthogonal control

Run matched-norm random interventions.

Expected:

- meaningfully smaller effect than leader / dispersion / full-subspace patching

### 3. Layer specificity

Patch nearby weaker layers too.

Expected:

- strongest effect at the hypothesized target layers

### 4. Dose response

Use at least:

- `0.25`
- `0.5`
- `1.0`
- `2.0`

Expected:

- roughly monotonic effect size

### 5. Directional symmetry

Leader-up vs leader-down, dispersion-up vs dispersion-down.

Expected:

- opposite-signed movement on downstream readouts

### 6. Holdout prompts

Define the patch basis on one prompt set and intervene on held-out prompts.

Expected:

- same-sign effect on held-out data

### 7. Section-locality check

Patch only the market section.

Expected:

- no spurious edits outside the intended token span

### 8. Corruption check

Watch for generic breakdown:

- malformed output
- degenerate repeated text
- impossible tool-call format

If corruption dominates, the intervention is too blunt.


## Decision Rule For Moving To CLT

Do not move to CLT just because a probe-friendly subspace exists.

Move to CLT only if all of these are true:

1. Null patch is near zero.
2. Random orthogonal controls are much smaller than the real patch.
3. Directional patches have interpretable sign.
4. Effects survive held-out prompts.
5. At least one intervention changes a downstream representation strongly enough to matter.
6. At least one intervention changes behavior or a behavior-adjacent readout in a stable way.

Practical threshold:

- real patch effect at least `2x` random-control effect
- same-sign effect on at least `70%` of held-out prompts
- no obvious corruption signature


## CLT Stage

Only after patching passes the gate.

### CLT Goal

Use CLT-Forge to trace how the causal market subspace is built and propagated, not to discover a target from scratch.

### Initial CLT pilot

Train on:

- synthetic market-only prompts first
- layers around the strongest patch locations
- section states aligned to the market span

Questions:

1. do learned CLT features align with the causal leader / dispersion directions?
2. can attribution graphs show where those directions are constructed?
3. can they show where they are consumed downstream?

### CLT deliverables

- alignment scores between CLT features and the hand-built causal basis
- attribution graph from earlier market-processing layers into the patched layers
- candidate sparse features for later fine-grained interventions

If patching is weak, do not do this yet.


## Implementation Order

### Step 1

Build worker-side patch installation and clear/restore lifecycle.

### Step 2

Implement `project_out` for one layer and one span.

### Step 3

Run representational smoke tests on synthetic market-only prompts.

### Step 4

Add directional patch-in and random controls.

### Step 5

Run held-out synthetic evaluation.

### Step 6

Only then add behavioral generation mode.

### Step 7

If patching passes, start the CLT-Forge pilot.


## Explicit Non-Goals For Version 1

Do not try to do all of these in the first patching implementation:

- batched patching with `max_num_seqs > 1`
- real DX prompts first
- fine-grained token-specific steering
- CLT integration before causal gate
- generalized patching for every section type

Version 1 should only prove:

- we can causally intervene on the discovered market subspace
- and the effect is stronger than matched controls


## Immediate Next Build

If implemented now, the first concrete build should be:

1. `vllm_market_patch.py`
2. synthetic worker entrypoint for patched runs
3. `project_out` on the full 4D market subspace
4. representational smoke test on market-only synthetic prompts
5. null + random controls

That is the smallest build that can answer whether this line is worth pushing further.
