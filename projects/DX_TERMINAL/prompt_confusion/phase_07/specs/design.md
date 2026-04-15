# Phase 07 Plan

Follows from Phase 06 v4 review (2026-04-15). v4 established robust
conflict-detection probing (0.849 bal_acc combined strict holdout) but
surfaced format-driven resolution behavior and several dataset design
issues. Phase 07 isolates format as a variable and builds toward a
resolution probe.

---

## Move 1: v5 dataset, minimal diff from v4

Rewrite setting variants to be verbal and imperative, matching strategy
format. Drop setting_v0's numeric scale entirely. Drop strategy_v3 and
setting_v3 -- they're behaviorally broken and you don't need 4 variants.
3 per side is plenty for lexical holdout, and fewer cells means you can
actually look at every one. Keep the system prompt resolution hint for
now -- changing format AND removing the hint simultaneously muddies
attribution. Keep everything else identical: same market contexts, same
pressure buckets, same section ordering, same matched-pair structure.

Before running capture, do a behavioral pre-screen -- generate on all
3x3 variant pairs, check that aligned rows produce expected behavior at
>90%, and check that no single variant dominates conflict resolution
the way setting_v0 did on v4. Kill bad variants before spending GPU on
activations.

## Move 2: v4 vs v5 comparison

Same probe pipeline, same layers, same holdout structure. Three
deliverables: behavioral distribution shift (does resolution become
less format-driven), probe accuracy comparison (does detection hold,
improve, degrade), and cosine similarity of probe directions at each
layer between v4 and v5. If the directions are highly aligned, the
model's conflict representation is format-invariant and you found
something real about semantic comprehension. If they rotate, the model
was partly encoding format mismatch, not content disagreement. Either
answer is interesting and slide-worthy.

## Move 2.5 (conditional): remove system prompt resolution hint

Remove the sentence "If STRATEGY and SETTINGS disagree, SETTINGS still
constrain the final execution" from the system prompt. Keep everything
else from v5 identical.

**When to proceed:** v5 resolution behavior leans toward follow_setting
consistently across variant pairs (not because one variant dominates).
That's the hint working as intended with the format confound removed.
Then you remove the hint and measure what shifts -- does it drift toward
50/50? Does strategy start winning? Does refusal spike? The shift is
the finding.

**When to hold:** v5 still shows the v4 pattern -- some variant pairs
at 90% follow_setting, others at 90% follow_strategy. That means
format-matching didn't fix it and there's still a wording-level
confound to resolve before hint removal is meaningful.

Can be collapsed into Move 2 as a single capture run (include both
system prompt variants as a dataset factor) if capture budget allows.

## Move 3: resolution probe (conditional on Move 2 / 2.5)

Change the probe target from `conflict_present` to
`resolution_direction` -- train on conflict rows only, predict whether
the model follows strategy vs setting. This only works if v5 (or
v5-no-hint) resolution behavior is clean enough that the labels mean
something -- not dominated by one format, one variant, or one section
order. If Move 2 / 2.5 gives you that, you can ask whether the model's
internal state at the last prompt token already predicts which source
it'll follow. That's the mechanistic resolution claim -- and the thing
that matters most for the product story: "We can predict which policy
source your agent will follow before it generates, from the activation
pattern alone."

---

## What this builds toward

- **For DX Terminal:** concrete findings about how policy format and
  structure affect agent behavior on Qwen3, translatable into agent
  design guidance.
- **For the SF talk:** v4 detection result + format-sensitivity finding
  is already a two-slide story. v5 comparison strengthens it. Move 3
  results (if ready) are the demo.
- **For Concordance:** prototype of the product motion -- "run your
  agent through our interp pipeline, get back engineering decisions
  about policy stack design."
