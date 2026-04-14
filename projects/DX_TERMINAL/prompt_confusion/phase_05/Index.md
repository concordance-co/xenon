# Prompt Confusion Phase 05 -- Index

Status: **family-identity claim retracted; conflict-detection claim rehabilitated after proper residualization and lexical-holdout layer progression; minimal dataset redesign required to strengthen conflict work further, full redesign required for family-level work**

Continues from [Phase 04](../phase_04/Index.md). See Phase 04's combined
checkpoint report and stage 2 handoff for prior results.

Phase 05 working hypothesis: the model resolves conflicts by identifying
the conflict type and applying a family-specific resolution policy. Phase
05 investigates the structure of this branching mechanism. See
[design.md](specs/design.md) for the full framing and falsification
conditions.

## Specs

| File | What it covers |
|---|---|
| [design.md](specs/design.md) | Full Phase 05 design: framing, three directions, sequencing, expansion gate |
| [notes.md](notes.md) | Running log of known limitations and learnings |

## Dataset

Reuses the Phase 04 dataset (288 rows, `workflow_dataset_conflict_probe_v3_v1`
in Neon) for Directions 1, 2a, 2b, 2c, and the first pass of Direction 3.
Expansion is gated on Direction 1/2 results -- see design.md.

## Analyses

| Direction | Script | Status | Output |
|---|---|---|---|
| 1a + 1b | [`scripts/cross_family_transfer.py`](scripts/cross_family_transfer.py) | **Run 2026-04-14** | [`outputs/cross_family_transfer/summary.json`](outputs/cross_family_transfer/summary.json) |
| 2b | [`scripts/family_identity_probe.py`](scripts/family_identity_probe.py) | **Run 2026-04-14** | [`outputs/family_identity_probe/summary.json`](outputs/family_identity_probe/summary.json) |
| Confound battery | [`scripts/confound_battery.py`](scripts/confound_battery.py) | **Run 2026-04-14** | [`outputs/confound_battery/summary.json`](outputs/confound_battery/summary.json) |
| 2a + 2c | [`scripts/family_geometry.py`](scripts/family_geometry.py) | Deferred (dataset-limited) | -- |
| 3 | Not started | Deferred (dataset-limited) | -- |

Support scripts:

- [`scripts/inspect_router_data.py`](scripts/inspect_router_data.py) -- one-off: audits activation layout on the Modal volume
- [`scripts/recompact_router.py`](scripts/recompact_router.py) -- rebuilds router compact files over all 288 rows (see notes.md)
- [`scripts/fetch_summaries.py`](scripts/fetch_summaries.py) -- copies summary.json files from Modal to `outputs/`

All analysis scripts are Modal apps reading from the Phase 04 capture run
(`16474bceae4e`) and writing results to
`/data/analysis_results/prompt_confusion/phase_05/<analysis_name>/`.

Invocation pattern:

```bash
uv run --extra interp --extra modal modal run \
    projects/DX_TERMINAL/prompt_confusion/phase_05/scripts/<script>.py
```

## Findings (2026-04-14, post-confound-battery)

Initial Directions 1 + 2b results made it look like the model had very
strong family-identity and conflict-detection representations, with
"branching at recognition" as the working interpretation. The confound
battery changed that picture substantially. **See
[notes.md](notes.md) for the full reasoning.**

### What the confound battery showed

**Family identity (claim retracted).**
CountVectorizer + LogisticRegression on raw `user_text` decodes
`strategy_family` at **100.0%** balanced accuracy. Surface vocabularies
across size and activity families are disjoint. The 98.9% activation
probe adds no evidence for a mechanistic family representation -- a
text-feature baseline already saturates.

**Conflict detection (rehabilitated; real mechanistic signal that builds with depth and lives outside the family subspace).**

Within-family detection with lexical holdout (hold out v0 or v1 wording,
test on the other), swept across all 12 captured layers. Two metrics:
balanced accuracy (default-threshold prediction accuracy) and AUROC
(threshold-free discrimination). AUROC is the more conservative metric
for the rehabilitation claim because it is not sensitive to a
miscalibrated decision boundary.

| Condition | Metric | L0 | L24 | L36 | L40 | L44 |
|---|---|---|---|---|---|---|
| size / hold out setting | bal_acc | 0.71 | 0.65 | **0.79** | 0.71 | 0.74 |
| size / hold out setting | AUROC | 0.81 | 0.80 | **0.93** | 0.89 | 0.84 |
| activity / hold out setting | bal_acc | 0.61 | 0.71 | 0.61 | 0.64 | 0.65 |
| activity / hold out setting | AUROC | 0.62 | 0.79 | **0.90** | 0.73 | 0.68 |
| size / hold out strategy | bal_acc | 0.50 | 0.69 | 0.53 | 0.50 | 0.50 |
| size / hold out strategy | AUROC | 0.85 | 0.89 | 0.93 | **0.94** | 0.91 |
| activity / hold out strategy | bal_acc | 0.51 | 0.64 | 0.54 | 0.60 | 0.67 |
| activity / hold out strategy | AUROC | 0.61 | 0.71 | 0.78 | **0.80** | 0.77 |

Lexical baseline AUROC is 0.50 in every holdout condition (chance).

**Interpretation: the learned direction transfers; the calibrated threshold
does not.** AUROC 0.77-0.94 at deep layers means a probe trained on one
lexical half ranks conflict vs aligned examples very well in the other
half. Balanced accuracy near chance in some conditions reflects threshold
miscalibration under covariate shift, not absence of signal. The probe's
output looks like "conflict signal + lexical-variant-dependent additive
bias": the conflict component dominates the ranking, but an additive
lexical component shifts scores between variants so the default
threshold is wrong in the target half.

**Caveats:**
- L0 AUROC is surprisingly high in `size / hold out strategy` (0.85).
  Under strategy-holdout, settings wording still varies with the label
  (aligned vs conflict flip setting values), so token-level features in
  the embedding can distinguish classes without any mid-layer
  construction. Settings-holdout is the cleaner test.
- What we can claim: "conflict detection generalizes across lexical
  variants in ranking, with a lexical-dependent threshold shift."
  What we cannot claim: "the model has a purely lexical-invariant
  conflict representation" -- the threshold shift is evidence that
  some lexical dependence remains in the probe's output magnitude.

**Conflict detection is linearly orthogonal to the family subspace.**
Fit a 4-class family classifier, take its coefficient matrix as the family
subspace (rank 3), project activations onto the orthogonal complement,
then probe conflict. AUROC and balanced accuracy both reported:

| Layer | conf_raw bal / AUROC | conf_null bal / AUROC | ΔAUROC |
|---|---|---|---|
| L24 | 0.94 / 0.983 | 0.94 / 0.984 | -0.001 |
| L28 | 0.95 / 0.991 | 0.94 / 0.989 | +0.002 |
| L32 | 0.93 / 0.985 | 0.92 / 0.981 | +0.005 |
| L36 | 0.98 / 0.995 | 0.95 / 0.992 | +0.002 |
| L40 | 0.93 / 0.985 | 0.93 / 0.983 | +0.002 |
| L44 | 0.89 / 0.962 | 0.89 / 0.959 | +0.004 |

Projection removes the rank-3 family subspace. A refit classifier on the
null-space still finds some residual family structure (because linear
subspace projection only removes rank-3 family content, not higher-rank
correlations), which is why `family_null` accuracy stays above the 0.25
chance baseline. Despite this, **conflict AUROC is essentially unchanged
(all deltas within ±0.01) at every high-accuracy layer**. The conflict
signal lives substantially outside the linear family subspace.

**Cross-family transfer (asymmetric, not zero).**
At L36, regardless of regularization:
- `activity → size`: **75-77%** (moderate transfer, well above chance)
- `size → activity`: 55-57% (chance-ish)

Activity-trained probes learn something that partially generalizes to size.
The reverse does not. This is real shared signal that the earlier "0
transferability" framing missed.

### What this means for the hypothesis

- Family identity claim: **retracted.** Cannot distinguish from surface text.
- Conflict detection claim: **rehabilitated.** Real mechanistic signal
  (~65-79% at best layers under settings-holdout), with depth-progressive
  construction and linear independence from family identity.
- Branching-at-recognition hypothesis: **cannot be supported or rejected**
  on this dataset. Family-vs-vocabulary confound makes family-level
  claims unanswerable with Phase 04 data.
- Cross-family transfer: **asymmetric partial sharing.** Activity→size
  at L36 = 75-77%. Interpretation still open; needs more families.

### Why further structural work is paused

Directions 2a/2c (PCA/LDA geometry) and Direction 3 (logit attribution)
were scoped to characterize *family* branching. Given that family identity
on this dataset is fully lexical, geometry and attribution would describe
surface-text structure rather than a mechanistic branching feature.
Deferring until a dataset that dissociates family from vocabulary exists.

### Next session (planned 2026-04-15)

Two dataset updates, both intended as low-risk rebuilds of the Phase 04
generator rather than a full Phase 06 redesign. Both need careful
attention during the wording/value design; the capture-and-analysis
rerun afterward should be mechanical (existing Phase 05 scripts work
against any workflow that produces compatible activation compact
files).

1. **Harmonize vocabulary across families to remove lexical leakage.**
   The goal is that a `CountVectorizer + LogisticRegression` baseline
   on `user_text` can no longer decode `strategy_family` above chance
   (current: 100%). Size and activity families need to share
   substantial surface vocabulary so family identity is not a free
   lexical readout. Preserve the semantic role distinctions
   (STRATEGY vs SETTINGS, aligned vs strong-conflict) while
   neutralizing family-indicative tokens. A successful harmonization
   should also lower the ceiling on unholdout conflict-detection
   probes closer to the lexical-holdout numbers we already measured
   -- those two should converge when vocabulary is matched.

2. **Restore the full setting-value grid from the DX Terminal original
   dataset.** Phase 04 reduced settings to two extremes per family
   (values 1 and 5). Restoring intermediate values (2, 3, 4) gives a
   graded conflict-strength axis we can use for both dataset diversity
   and conflict-strength analyses. Review the original dataset's
   setting structure before redesign to understand what was there and
   how matched pairs were constructed with more than two values.

After both changes land: rerun capture on the new dataset, recompact
router activations over the full row set (see "Compact activation
files" known-limitation), and rerun the Phase 05 analysis scripts
(`cross_family_transfer`, `family_identity_probe`, `confound_battery`)
against the new activations to see whether the rehabilitated conflict
signal holds, tightens, or breaks under the improved dataset.

### Longer-term: Phase 06 dataset redesign (scoped)

The redesign needed depends on which claims we want to strengthen:

**Minimum, if only strengthening the conflict-detection claim:**
- More lexical variants per family (currently 2; target 4+) for robust
  lexical-holdout testing.
- Strategy-variant diversity in particular -- strategy-holdout was much
  weaker than settings-holdout in Phase 05, and we want to know whether
  that is a data-scarcity issue or a genuine asymmetry.
- More rows per (family, pressure) cell to reduce variance on the
  depth-progression curves.

**Full, if pursuing family-level / branching claims:**
- Matched vocabularies across families -- families should share substantial
  surface vocabulary and differ on the semantic dimension of interest,
  not on token identity.
- More family groupings (currently 2; need 3+) to triangulate
  shared-vs-specific claims beyond a binary test.
- Explicit lexical controls so every family claim can be paired with a
  matched lexical baseline.

## Reports

TBD
