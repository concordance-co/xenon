---
name: activation-patching-causal-evals
description: Use when planning, running, or reviewing activation patching or interchange experiments for causal claims in mechanistic interpretability. Especially useful for choosing patch sites, designing paired examples, adding same-label controls, distinguishing read layers from write layers, and avoiding overclaiming from weak or lossy interventions.
---

# Activation Patching Causal Evals

Use this skill when the user wants to:

- test whether an internal representation is causally important
- design activation patching or interchange experiments
- improve a weak patching methodology
- interpret flip-rate results carefully
- add controls to causal claims

This skill is about **causal evaluation design**, not just how to run a patch operator.

## Core rule

`A readable signal is not automatically a writable signal.`

A layer that gives the best probe AUROC may be a bad place to intervene.

## Workflow

### 1. Start from a behavioral target, not a patch operator

Before deciding how to patch, define:

- what exact behavior should change?
- what counts as success?
- what counts as a malformed output?
- which direction matters most?

### 2. Choose the target site from the computation story

Prefer a site because you have a reason, not because it is convenient.

Useful evidence for choosing a site:

- span-local probes
- temporal / positional reasoning
- attention or routing evidence

Bad site selection:

- patch the last token because it is easy
- patch the highest-AUROC layer without asking whether the computation is already finished there

## Read layer vs write layer

Keep these separate:

- **read layer**
- **write layer**

These often differ.

Typical pattern:

1. earlier layer: weaker readout, higher causal leverage
2. later layer: stronger readout, lower causal leverage

## Patch operator choice

### Prefer full-state interchange over lossy summaries when possible

If your span contains structured, position-specific information, mean-based patches may be too weak.

Risky:

- span-mean translation
- class-mean direction only

Stronger:

- token-by-token full activation replacement
- matched donor-target interchange

### Patch one layer at a time first

Multi-layer patching can be useful later, but it is a poor first causal test.

Why:

- interactions become hard to attribute
- a positive result does not localize the site

## Paired example design

For clean interchange, use paired examples that match on as many nuisance dimensions as possible.

Ideal pairing:

- same template family
- same carrier family
- same action family
- same control map structure
- different value of the target variable

## Controls are mandatory

### Same-label control

Always test whether examples move under a same-label donor swap.

Interpretation:

- if cross-label effect is much larger than same-label effect, that supports directional causality
- if same-label effect is similar to main effect, the intervention may just be destabilizing the model

### Random or unmatched control

If feasible, also test:

- random donor from same label
- random donor from opposite label but unmatched

## Success criteria

Do not evaluate patching only with “some rows changed.”

Track:

- intended-direction flips
- reverse-direction flips
- malformed outputs
- overlap with same-label control flips

## How to interpret asymmetry

Asymmetric patching is common and can still be meaningful.

Example:

- removing a mismatch signal may be easy
- injecting that mismatch signal into a valid case may be hard

Do not force every result into a symmetric-axis interpretation.

## Common failure modes

### Same tiny set of rows always flips

Likely interpretation:

- borderline sensitivity, not broad causal control

### Late-layer patching yields garbage

Likely interpretation:

- you are patching a compressed summary state

### Mean-swap does almost nothing

Likely interpretation:

- the patch may be too lossy, not the hypothesis necessarily wrong

### High AUROC but no causal effect

Likely interpretation:

- strong readout from a downstream summary
- poor intervention site

## Default stance

Activation patching is strongest when it is treated as:

`a controlled causal experiment with proper baselines`

not:

`a cool qualitative demo where some examples changed.`
