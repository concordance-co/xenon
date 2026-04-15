# Prompt Confusion Phase 06 Design

## Framing

Phase 05 closed out with one defensible mechanistic result (conflict
detection is real, depth-progressive, and linearly orthogonal to family
identity) and one retracted claim (family identity on the Phase 04 dataset
is fully lexical -- a CountVectorizer+LR text baseline classifies
`strategy_family` at 100%). The family-vs-vocabulary confound makes
branching-hypothesis work unanswerable on that data.

Phase 06 narrows the scientific question to the thing the DX Terminal
partnership actually cares about and the thing Phase 05 left
under-characterized:

> **When "what I should do" isn't clear, how does the model decide?**

Concretely, under a STRATEGY/SETTINGS conflict there are three observed
behaviors in Phase 04/05 data:

1. **Follow strategy.** The model picks the strategy-prescribed answer.
2. **Follow settings.** The model picks the settings-prescribed answer.
3. **Refuse.** The model declines to act (`size: "none"`).

Phase 05 silently dropped the refusal rows (15/36 on
`trade_size_force_large`, 6/36 on `trade_size_force_small`) when building
the side-following readout view. Phase 06 labels refusal in the readout
view when it occurs rather than dropping those rows. Beyond labeling,
Phase 06 does not deliberately engineer refusal -- it is a behavior we
observe, not a class we construct.

## Scope Decisions (from Phase 05 postmortem)

- **Drop family-identity claims entirely.** Phase 05 established these
  can't be separated from lexical leakage on the current generator.
- **Drop the cross-family transfer analysis** (asymmetric
  activity->size). It was a cross-axis transfer and is not meaningful
  inside one conflict axis.
- **Drop within-family-specific resolution analysis.** We don't know if
  the model has a real family-differentiated resolution mechanism, so
  probing per-family would bake in the same confound.
- **Drop the setting-value grid** (values 1-5 for graded conflict
  strength). Not needed for the decision question.
- **Drop scenario-frame invariance axes.** Not relevant to the DX
  Terminal partnership; would be fabricated for our purposes.
- **Keep: matched-pair scaffold, STRATEGY/SETTINGS section structure,
  lexical-holdout methodology, depth sweep.** These are what let Phase 05
  rehabilitate the detection claim at all.

Phase 06 is therefore a **targeted rebuild around a single conflict axis**,
not a generator augmentation.

## Dataset Redesign

### Pick one conflict axis: size

Between the two conflict axes in the current generator:

| Axis | Refusal rate (v3 dataset) |
|---|---|
| Size (`trade_size_force_*`) | 17-42% |
| Activity (`activity_force_*`) | ~0% |

Refusal shows up on the size axis and not on activity. It is an
interesting behavior to have visible in the data, and it is the
motivation for picking size over activity when we can only pick one.
We are not trying to maximize or minimize refusal, only to not actively
hide it the way the v3 readout view did.

Activity (or another second axis) returns in **Phase 07 as a
transferability test** of the Phase 06 findings -- not as a parallel
dimension in Phase 06 itself.

### Core prompt structure

Same as Phase 04/05: prompt contains a STRATEGY section and a SETTINGS
section. Under conflict, STRATEGY prescribes one size; SETTINGS prescribes
another. Under alignment, both prescribe the same size.

### Lexical variants: wide on both axes

- **4+ strategy lexical variants** per (family, side).
- **4+ setting lexical variants** per (family, side).
- Independent combinatorial coverage: every (strategy variant x setting
  variant) cell is represented.

Phase 05 had exactly 2 variants per axis, which gave one holdout split
per axis and could not distinguish lexical-holdout scarcity from real
asymmetry (specifically, strategy-holdout conflict detection was
dramatically weaker than settings-holdout). Four variants enable:

- Train on variants {v0, v1}, test on {v2, v3} (and three more holdout
  cells).
- Proper cross-validation over lexical variance.
- Distinction between "the probe's direction transfers but its threshold
  does not" (Phase 05's working interpretation) and "the probe relies on
  vocabulary we happened to hold out."

### Section order: swap baked in from day one

Half the rows STRATEGY-before-SETTINGS, half SETTINGS-before-STRATEGY.

Every Phase 04 prompt had STRATEGY first, which contaminated any
attention-based follow-up with recency bias toward SETTINGS. Phase 05
could not run logit attribution because the order-swap control was
missing. Phase 06 bakes it in so:

1. Attention-based analyses downstream are publishable without a
   separate capture pass for a control.
2. Order-shift on the resolution probe becomes a free behavioral
   measurement rather than a blocker.

### Refusal: labeled, not engineered

Side-followed labeling in the readout view carries
`follow_strategy` / `follow_setting` / `refuse`, not `strategy` /
`setting` with refusal silently dropped. That is the only change versus
Phase 04/05 on this axis -- we do not construct refusal-inducing
prompts, we only stop hiding refusal when it happens.

If refusal turns out to be rare or irrelevant on the new generator,
that's fine; the 3-class label collapses gracefully to 2-class in
analysis. If refusal turns out to be more structured than we expect, we
follow up then.

### Row count targets

Roughly 3-4x the v3 dataset (288 total / 144 conflict). Working target:

- Conflict rows: ~450.
- Aligned matched pairs: one per conflict row.
- Total dataset: ~900.

Rationale is statistical power on per-layer probes under 4x4 lexical
holdout, not a specific outcome-class count. Refine once we see the
actual refusal rate on the new generator via a smoke run.

## Dataset QA Gates (before capture)

All of these run against the generated rows before any model capture
time is spent. Each is a `TextBaselineSpec` over raw `user_text` with
`countvectorizer_logreg`, plus structural checks.

| Check | Target |
|---|---|
| `conflict_present` text baseline (lexical holdout) | <=60% bal_acc |
| `side_followed` (3-class) text baseline | <=55% bal_acc |
| Matched-pair integrity | every conflict row has one aligned partner with shared variants + order |
| Class support per (lexical-holdout cell x outcome class) | >=30 rows |
| Section-order balance | 50/50 within every (family, outcome class) cell |

Any gate failure returns to the generator, not to capture. Phase 05
burned analysis cycles on a dataset whose text baseline only got
inspected retrospectively.

### Dataset QA is itself a `pipelines_v2` workflow

The QA gates above should be implemented as a small `workflow.py`
targeting just the new dataset (no model capture, only text baselines +
structural asserts). Iterate on the generator against this workflow
before wiring it to any capture runner.

## Analyses (Phase 06 proper, after QA passes)

All analyses are straight ports of Phase 05 scripts, re-pointed at the
new dataset. New: the 3-class resolution probe.

### Detection (same methodology as Phase 05)

- `ProbeSpec` on `conflict_present`, swept across all captured layers.
- Lexical holdout across both strategy and settings variants (4 holdout
  cells per axis, 4 axes -> 4 holdout conditions each).
- Report bal_acc and AUROC.

### Threshold-shift characterization (live thread)

Phase 05 found that the probe's learned direction transfers across
lexical variants (AUROC stays high) while the calibrated threshold does
not (bal_acc drops). This is a soft caveat on the "lexical-invariant
conflict representation" claim.

With 4+ variants per axis we have enough holdout cells to actually
investigate this. How we investigate it depends on what the new data
shows -- candidate approaches (per-variant additive bias, lexical-
subspace residualization, something else) are a live thread, not a
pre-committed method.

### Resolution (new)

- 3-class `ProbeSpec` on `side_followed`: `follow_strategy` /
  `follow_setting` / `refuse`.
  - Global, not grouped by family.
  - Lexical holdout across strategy and settings variants.
  - Depth sweep.
  - Collapses gracefully to 2-class if refusal rate is low enough to
    not support a third class.
- Order-swap behavioral check: does the probe decode different
  resolutions for order-swapped matched pairs? Does the model actually
  resolve differently? (These are distinct questions -- one is about the
  probe, one is about the model.)

### What explicitly is not in Phase 06

Deferred to Phase 07 or later, once Phase 06 establishes whether the
clean dataset yields a defensible detection + resolution story:

- Second conflict axis (activity or other) as transferability test.
- Geometry: PCA/LDA/GeometrySpec over resolution-labeled activations.
- Logit attribution (STRATEGY vs SETTINGS token-span competition on the
  decisive output token). Infrastructure survives from Phase 04; we just
  haven't earned the right to run it yet.
- Causal patching and intervention experiments.

## Sequencing

| Step | Work | Gate |
|---|---|---|
| 1 | Generator rebuild (size-only, 4x4 variants, order swap, 3-class readout labels) | Dataset publishes to Neon as `conflict_probe_examples_v4` |
| 2 | Dataset QA workflow (text baselines + structural asserts) | All QA gates pass; otherwise return to step 1 |
| 3 | Capture | Runs on the existing Modal capture infra from Phase 04 |
| 4 | Detection analyses | Port Phase 05 `confound_battery` and `cross_family_transfer` scripts to the new dataset |
| 5 | Resolution analyses | New 3-class probe + refusal probe |
| 6 | Threshold-shift characterization | If detection result holds, fit and residualize the lexical additive component |

Phase 07 sequencing is **not** committed here. The shape of Phase 07
depends on Phase 06 outcomes:

- If detection + resolution probes look clean, Phase 07 adds a second
  conflict axis as a transferability test and re-runs the same protocol.
- If detection is clean but resolution is not, Phase 07 digs into
  resolution mechanism before adding an axis.
- If even detection doesn't cleanly beat the text baseline under the
  tighter QA gates, Phase 07 is dataset generator redesign, not more
  analysis.

## Known Carried Limitations

Tracked in `notes.md`:

- MoE logit attribution gap (attention pathway only).
- Compact activation file staleness (recompact on each new capture).
- Probe weight vectors across layers are not comparable.
- Balanced accuracy alone is misleading under covariate shift; always
  report AUROC alongside.
