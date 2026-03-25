# Patching And CLT Plan

## Why This Is The Main Track

We already have a strong descriptive result:

- a market-only `market_mean` subspace that carries leader and dispersion signal
- a late dispersion-like axis that is not just literal standard deviation
- a market subspace that is broader than one PC and is roughly `~4D` in the stored top-PC readout

The next question is no longer:

- what does this subspace correlate with?

The next question is:

- is this subspace causal?
- and if it is causal, can we trace how it is built and used across layers?

So the primary track is:

1. patching on the discovered market subspace
2. only then, if patching succeeds, CLT-style circuit tracing


## Phase A: Patching

### A1. Intervention Target

Start with the `market_mean` market-representation subspace, not a downstream policy state.

Use a small basis built from the strongest discovered directions:

- leader-like direction
  - anchored on Phase 17 `market_mean`, `L4`, `PC1`
- dispersion-like direction
  - anchored on Phase 17 `market_mean`, `L35`, `PC1`
- add the next 2 stable top-PC directions from the same Phase 15/17 residualized discovery basis
  - do not assume they are semantically named yet
  - use them to complete the working `~4D` market subspace

The first patch target should be:

- pooled `market_mean`
- at a small number of layers where the subspace is strongest and cleanest

Initial layer candidates:

- early leader layer: around `L4`
- late dispersion layer: around `L35`
- one late broad-summary layer from the high-variance `market_mean` region, e.g. `L40-L42`


### A2. Intervention Types

Run both necessity and directional-control tests.

#### Necessity tests

1. Project-out / ablate the full discovered subspace.
2. Project-out leader only.
3. Project-out dispersion only.

#### Directional-control tests

1. Patch in a higher-leader activation pattern.
2. Patch in a lower-leader activation pattern.
3. Patch in a higher-dispersion activation pattern.
4. Patch in a lower-dispersion activation pattern.

The cleanest patch source is a matched prompt pair where only the target market property differs.


### A3. First Measurement Targets

Do not lead with final buy/sell labels. Lead with more local outcomes.

Measure:

1. Asset ranking shifts.
   - does the preferred asset move?
   - does the gap between top assets move?

2. Score / geometry shifts.
   - does the patched state move later market or post-market geometry in the expected direction?

3. Riskiness shifts.
   - does higher-dispersion patching make later states look more risk-sensitive or more conservative?

4. Trade aggressiveness shifts.
   - trade vs observe
   - more concentrated vs more diversified
   - larger vs smaller implied size preference

Only after that:

5. Final behavior shifts.
   - action choice
   - chosen asset
   - chosen size


### A4. Experimental Ladder

Start simple.

#### Stage 1: Synthetic causal smoke test

Use the clean synthetic market-only prompts first.

Goals:

- confirm the patching machinery works
- confirm leader and dispersion patches produce directional changes
- avoid policy confounds

#### Stage 2: Synthetic post-market prompts

Use the improved DX-like synthetic surface with settings / constraints present.

Goals:

- see whether the same market subspace still causally shapes later integration states
- measure whether the effect survives policy context

#### Stage 3: Real DX bridge

Use matched real prompts only after Stage 1 or 2 shows clean signal.

Goals:

- test whether the same interventions move real downstream representations
- only then claim the subspace is real-world causal


## Patching Sanity Checks

These are mandatory.

### S1. Null patch

Patch a prompt with itself.

Expected result:

- no meaningful change

If this fails, stop.


### S2. Random orthogonal control

Project out or patch in a matched-norm random subspace orthogonal to the discovered one.

Expected result:

- materially smaller effect than the real leader / dispersion subspace

If random controls move behavior as much as the target patch, stop.


### S3. Layer specificity

Run the same intervention at a nearby weaker layer.

Expected result:

- strongest effect at the hypothesized layer(s)
- weaker effect off-target

If every layer behaves the same, interpretation weakens sharply.


### S4. Dose response

Patch strengths:

- weak
- medium
- strong

Expected result:

- roughly monotonic downstream change

If only extreme patch strengths work, or sign flips are unstable, interpretation is weak.


### S5. Directional sign symmetry

Leader-up and leader-down should move outcomes in opposite directions.

Likewise for dispersion-up and dispersion-down.

If only one direction works, or both push the same way, investigate before proceeding.


### S6. Behavioral specificity

A leader patch should not just collapse the model into arbitrary extra action.

A dispersion patch should not just destroy everything and force observe.

Expected result:

- targeted representational / behavioral movement
- not generic corruption


### S7. Holdout prompts

Define the subspace on one set of prompts and test patching on held-out prompts.

Expected result:

- effect survives out-of-sample

If it only works on the prompts used to define the patch basis, do not move to CLT.


## Decision Rule: When Patching Is Strong Enough

Move forward only if all of the following are true:

1. Null and random-control patches are much smaller than the true patch.
2. Effects are directionally interpretable.
3. Effects replicate on held-out prompts.
4. The effect is visible in at least one downstream representational target, not just final action.
5. At least one of leader or dispersion produces a stable, nontrivial delta in behavior or post-market representation.

A good practical threshold:

- target patch effect at least `2x` the matched random-control effect
- same-sign effect on at least `70%` of held-out matched prompts
- no obvious corruption signature

If those are not met, do not move to CLT yet.


## Phase B: CLT-Forge

CLT work is conditional on patch success.

### B1. Why Use CLT Here

If patching shows the market subspace is causal, CLT becomes useful for a specific question:

- how is that causal subspace constructed and propagated across layers?

That is exactly where a cross-layer transcoder and attribution graph can help.


### B2. What Not To Do

Do not adopt CLT-Forge as the new main backbone before patching works.

Do not train a broad CLT just to explore blindly.

That would add tooling cost before we know the target subspace is worth tracing.


### B3. Minimal CLT Pilot

If the patch gate passes:

1. Start with a reduced model / reduced scope pilot.
2. Train on market-section activations first.
3. Focus on the same layers / states where patching was strongest.
4. Ask whether CLT features recover:
   - leader-like directions
   - dispersion-like directions
   - cross-layer attribution paths into later market or post-market states

Success criteria for the pilot:

- CLT features align with the known causal subspace
- attribution graph is stable across prompts
- graph edges are interpretable and not dominated by junk / formatting features


### B4. CLT Readiness Checklist

Only start CLT if:

- patching is clearly causal
- the target layers are narrowed down
- the prompt surface is stable enough
- activation caching for the needed states is reliable
- we have a small pilot compute budget first


## Immediate Next Steps

1. Define the working `~4D` market subspace from Phase 17 outputs.
2. Implement null, random-control, and project-out patches.
3. Run synthetic market-only patching first.
4. Evaluate representational and ranking deltas before final action deltas.
5. If clean, extend to post-market synthetic prompts.
6. Only then decide whether the CLT pilot starts.
