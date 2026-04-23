# Phase 05 Notes

Running log of limitations, gotchas, and learnings as Phase 05 progresses.
Keep entries dated. Add things as they come up -- do not wait until the end
of the phase.

---

## Known Limitations Going In

### Dataset power

- Phase 04 dataset has 288 rows total, 144 conflict. Per-family:
  ~30 conflict rows per family after labelable-side filtering.
- **Baseline behavior is severely skewed in some families** (Phase 04 Stage 2):
  - `trade_size_force_large`: 0 strategy / 21 setting (untrainable for arbitration)
  - `activity_force_observe`: 30 strategy / 6 setting (near-untrainable)
  - `activity_force_trade`: 14 strategy / 22 setting (OK)
  - `trade_size_force_small`: 22 strategy / 7 setting / 1 neither (borderline)
- Implication: per-family arbitration probing (Direction 2b) is not feasible
  in Phase 05 without dataset expansion. Only `activity_force_trade` has a
  meaningful class split.
- 4-class LDA on 144 rows (~36 per class) is near the floor for stable
  results. Treat LDA as visualization, not quantitative claim.

### Section-order position bias

- Every Phase 04 prompt has STRATEGY before SETTINGS, always in the same order.
- Any attention-based analysis (Direction 3) will be contaminated by recency
  bias toward SETTINGS.
- Phase 05 includes a section-order swap as an in-scope control alongside
  the attention infra build. Do not publish attention-based conclusions
  without the swap as a control.
- **The swap requires new behavioral generation**, not just new LA captures.
  The model may resolve differently when section order changes, so the
  swapped rows need fresh side-following labels from a generation pass
  before LA can be interpreted on them.

### PCA/LDA normalization

- Residual stream norms grow with depth in transformer models.
- PCA is scale-sensitive: PC1 can be dominated by layer-driven norm rather
  than content.
- Always RMS-normalize activations before PCA/LDA in Phase 05. If you skip
  normalization, state it explicitly and caveat the result.

### Cosine similarity across layers is meaningless

- Probe weight vectors at different layers live in different semantic spaces.
- Only compare probes *at the same layer* in Direction 1's transfer analysis.

### Causal asymmetry from Phase 04

- Phase 04 Stage 2 found `strategy_push` moves behavior 9pp but `setting_push`
  only moves it 3pp. Activity families barely moved at all.
- Not a Phase 05 workstream, but: if the Direction 2a/2c geometry shows
  activity families cluster far from the SETTINGS direction used for patching,
  that is the likely mechanistic explanation. Flag this in any geometry
  interpretation.

### Family identity in Phase 04 dataset is 100% lexically decodable

- CountVectorizer + LogisticRegression on raw `user_text` classifies
  the 4 families at 100% balanced accuracy. Size families use
  "tier / allocation"; activity families use "entries / pace / waiting."
  Surface vocabularies are effectively disjoint.
- Consequence: any "family" probe on activations from this dataset is
  indistinguishable from a lexical baseline. Mechanistic family-identity
  claims cannot be made on the Phase 04 data.
- Any claim of "family-specific representation" on this dataset is really
  "representation aligned with a specific vocabulary cluster." A valid
  test of mechanistic family structure requires matched vocabularies
  across families.

### Phase 04 conflict detection was partially lexical, but what survives transfers in ranking

- Phase 04 reported 92% conflict detection at L36 using matched-pair-id
  grouped k-fold. That grouping splits pairs but does not hold out
  lexical variants.
- Under proper lexical holdout (Phase 05 confound battery test 3), both
  balanced accuracy and AUROC were reported. The two metrics tell
  complementary stories:
  - Balanced accuracy at deep layers: 0.50-0.79 depending on holdout
    condition. Large variance because balanced accuracy is threshold-
    sensitive and the probe's calibrated threshold does not fully
    transfer across lexical variants.
  - **AUROC at deep layers: 0.77-0.94 across all four holdout
    conditions.** AUROC is threshold-free and measures whether the
    probe's *direction* separates the two classes.
- Interpretation: the probe's learned direction transfers well across
  lexical halves -- it ranks conflict vs aligned examples correctly in
  the held-out half at AUROC 0.77-0.94. The probe's calibrated decision
  threshold does not transfer because the feature distribution shifts
  between lexical halves (covariate shift). Result is a threshold
  miscalibration that hurts balanced accuracy without reducing AUROC.
- What this supports claiming: "conflict detection generalizes across
  lexical variants in ranking (AUROC ≥ 0.77 at deep layers)."
- What this does not support claiming: "the model has a lexical-
  invariant conflict representation." The threshold shift is evidence
  that the probe's output has a lexical-dependent additive component --
  conflict signal dominates the ranking, but there's still lexical
  structure in the magnitude.
- The surviving signal has the classic constructed-feature depth profile
  (early dip, steady rise into L24-L40) in both bal_acc and AUROC views.
- The surviving signal is also linearly independent of family identity
  (confound battery test 4 with proper rank-3 subspace projection):
  family-residualization drops conflict AUROC by ≤0.01 at every
  high-accuracy layer.
- Takeaway: there is real, depth-progressive, family-orthogonal
  conflict signal that transfers across lexical variants in its ranking.
  The threshold component is lexically contingent and should be
  flagged as a residual lexical dependence. Report AUROC alongside
  balanced accuracy on any future holdout evaluation.

### Strategy-holdout is much weaker than settings-holdout

- Phase 05 confound battery test 3 ran lexical holdout across both
  `strategy_lexical_split` and `setting_lexical_split`. Settings-holdout
  conditions showed strong depth-progressive conflict signal (65-79%
  at L24-L36). Strategy-holdout conditions were much weaker and flatter
  (mostly 50-55% with a single L24 spike).
- Interpretation: strategy wording appears to carry more of the conflict
  semantics than setting wording, so holding it out strips more of what
  the activation probe relies on. Could also be that strategy variants
  v0/v1 are more lexically divergent than setting variants v0/v1, so
  the probe has less shared substrate to bridge.
- Either way: do not treat strategy-holdout and settings-holdout as
  interchangeable tests. Report them separately. If Phase 06 adds more
  strategy variants (4+), we'll be able to distinguish data-scarcity
  from a genuine asymmetry.

### Cross-family transfer is asymmetric, not zero

- First-pass reading of cross-family transfer called it a collapse to
  chance. Regularization sweep plus careful inspection of later layers
  changed the picture:
  - At L36, `activity → size` transfers at 75-77% balanced accuracy
    (well above chance) regardless of regularization strength.
  - `size → activity` stays at 55-57% (chance-ish).
- Interpretation is open: could be that activity-family conflict is a
  more general "action conflict" that subsumes size conflict, or could be
  an artifact of which family's probe latched onto features that happen
  to work both ways. A dataset with matched vocabularies would isolate
  whether the asymmetry is semantic or lexical.

### Compact activation files can be scoped narrower than the capture

- Per-example activations in `residual_stream/` and `router_logits/` are
  the source of truth. Compact per-layer files in `compact/` are a
  derived summary produced by the old compact-analysis path.
- Compact files carry whatever row set was used when compaction was run.
  Phase 04's router compact files were produced as part of the
  conflict-readout arbitration analysis (123 conflict-only rows), and
  those were the only router compact files on the volume at the start
  of Phase 05.
- Always inspect compact files before trusting row counts. Phase 05's
  `scripts/inspect_router_data.py` does this; `scripts/recompact_router.py`
  rebuilt the router compact over all 288 rows before Direction 1 and 2b
  ran.
- Pattern to follow: whenever a new phase analysis targets a different
  row slice than prior compactions, either reuse the existing compact
  and filter by `log_id`, or recompact -- but never assume the compact
  matches the full capture.

### Attention infrastructure is partially built, not from scratch

- Phase 04's `modal_conflict_arbitration_analysis.py` already runs an HF
  transformers pass in eager mode with `output_attentions=True` to get
  per-head attention weights. Attention mass analysis lives there.
- For Direction 3 logit attribution, we additionally need per-head value
  vectors V, which are not exposed by `output_attentions`. This requires
  forward hooks on the attention submodule's `v_proj` (or equivalent
  per-architecture). W_O and the unembedding are model weights, trivial.
- Plan: extend the existing eager HF pass, do not build attention capture
  from scratch.

### MoE logit attribution gap

- Qwen3-30B-A3B has MoE routing. Attention-based logit attribution captures
  only the attention pathway's contribution to the output logit. It misses
  the MLP/expert contribution.
- Claims from Direction 3 should be framed as "how STRATEGY and SETTINGS
  tokens compete in the attention pathway," not "in the model's computation."

---

## Learnings Log

Add dated entries below as Phase 05 work surfaces new findings, surprises,
or caveats.

### YYYY-MM-DD -- template entry

- Context: what were we doing
- What we found: the result or surprise
- Implication: what this changes for subsequent work
- Reference: paths to outputs, run IDs, report sections

### 2026-04-14 -- Router compact was scoped to 123 rows, not 288

- Context: first run of `family_identity_probe.py` returned
  `n_examples=288` on residual but `n_examples=123` on router.
- What we found: the per-example router_logits files exist for all 288
  rows, but the compact files had been generated by the Phase 04
  arbitration analysis over the 123-row conflict-readout slice and never
  regenerated for the full dataset. `scripts/recompact_router.py` rebuilt
  the router compact over all 288 rows.
- Implication: compact file row counts are not a capture invariant; they
  carry whatever slice was used at compaction time. Captured this as a
  standing caveat under "Known Limitations."
- Reference: `scripts/inspect_router_data.py`,
  `scripts/recompact_router.py`. Modal runs
  `ap-gGfUDmR2KwVVp44R7rmpb0` (inspect) and `ap-tKam8FeoSVuAn29g51kzQK`
  (recompact).

### 2026-04-14 -- Branching happens at recognition, not at resolution

- **SUPERSEDED by confound battery result below.** Keeping this entry
  for traceability; the hypothesis it lays out did not survive confound
  testing.
- Context: first results from Directions 1 (cross-family transfer) and
  2b (family identity probe).
- What we found:
  - Direction 2b: residual stream decodes `strategy_family` at 98.9%
    balanced accuracy, saturating at L24. Higher and earlier than Phase
    04's 92% conflict-detection peak at L36.
  - Direction 1a: within-family conflict detection is strong (96% at L24
    for size), but cross-family transfer appeared to collapse to chance
    at L24 (50-52%). Transfer delta -30 to -44pp across the strong layer
    band. Cosine similarity of size-trained vs activity-trained probe
    weights only 0.2-0.3 at best layers.
  - Direction 1b: similar pattern for arbitration.
  - Router shows essentially no cross-family transfer anywhere.
- Pre-confound interpretation: conflict detection itself is family-specific.
  Phase 04's 92% was a probe learning four family-specific conflict
  representations in parallel; branching happens at recognition.
- Why superseded: the family-identity result was 100% reproducible with
  a text-only lexical baseline, so the activation result was not
  mechanistic evidence. Once we ran proper lexical-holdout detection and
  a regularization sweep, the picture was much more nuanced than
  "family-specific recognition, zero sharing."
- Reference: `outputs/family_identity_probe/summary.json`,
  `outputs/cross_family_transfer/summary.json`. Modal runs
  `ap-NF2lkLyHV5osO7hXSbTwIa`, `ap-WV5Xb5JDcvylJTscEeVaWw`.

### 2026-04-14 -- Confound battery forces hypothesis retraction

- Context: ran `scripts/confound_battery.py` after concerns that the
  99% family-identity result was suspiciously high.
- What we found:
  - **Test 1 (lexical family identity):** CountVectorizer+LR on raw
    `user_text` decodes family at **100.0%** balanced accuracy. Size
    and activity families have disjoint surface vocabularies. The 98.9%
    activation result cannot be distinguished from a text-feature
    baseline.
  - **Test 3 (within-family lexical holdout):** lexical baseline is
    50% under holdout. Activation probe (residual) is 64-79% at L24-L36
    -- real semantic signal, but ~25-30pp lower than the matched-pair
    grouped baselines we reported earlier. A substantial fraction of
    Phase 04's "92% conflict detection" was lexical.
  - **Test 5 (regularization sweep):** at L36, `activity → size`
    transfer is 75-77% across all C values; `size → activity` stays at
    55-57%. Cross-family transfer is not zero; it is strongly asymmetric.
    Earlier "0 transferability" framing was wrong.
  - **Test 4 (family-residualized):** per-family mean-centering did not
    reduce conflict signal, but this was not a real residualization
    (family mean has no conflict information when each family is 50/50
    aligned/conflict). A proper family-null-subspace projection was not
    run; follow-up if we return to this dataset.
- Implications:
  - Family-identity claim: **retracted.** Nothing the activation probe
    showed is beyond what surface text already provides.
  - Conflict-detection claim: **downgraded.** Real, but honest
    mechanistic accuracy is ~60-70% after lexical holdout, not 92%.
  - Cross-family transfer: **asymmetric, not zero.** Activity→size
    generalizes at ~76% at L36; size→activity does not.
  - Branching-at-recognition hypothesis: cannot be evaluated on this
    dataset. The family-vs-vocabulary confound makes the central
    question unanswerable.
- Decision: pause Directions 2a/2c and 3. Geometry and logit attribution
  on a dataset where family ≡ vocabulary would describe surface-text
  structure, not mechanism. Scope a Phase 06 dataset redesign with
  matched vocabularies across families, more lexical variants (4+ per
  family), and more family groupings (3+).
- Reference: `outputs/confound_battery/summary.json`. Modal run
  `ap-Ti95UFqwYrCHoKuLfKMcSk` (confound battery),
  `ap-HGhJ7PisvYixiNIe92I5rJ` (summary fetch).

### 2026-04-14 -- Follow-up: layer progression + proper residualization rescue conflict detection

- Context: two targeted fixes to the confound battery after the initial
  run. (1) Test 3 rerun across all 12 captured layers instead of
  {0, 24, 36} to see whether surviving conflict signal builds with depth.
  (2) Test 4 replaced with a proper family-subspace projection -- fit a
  4-class family classifier, build the rank-3 projector onto its
  coefficient row-space, project activations into the orthogonal
  complement, then probe conflict.
- What we found:
  - **Test 3 depth progression:** under settings-holdout, conflict
    detection shows a clear constructed-feature profile. size/setting
    holdout: 71% L0 → dips → 65→75→78→79% L24-L36. activity/setting
    holdout: 61% L0 → dips → 57→71→72→74% L20-L32. Under
    strategy-holdout, signal is much weaker (50-55% mostly, one L24
    spike). The settings-holdout progression is the stronger evidence;
    strategy-holdout needs more lexical variants before we can separate
    scarcity from a real asymmetry.
  - **Test 4 proper residualization:** conflict detection survives
    removing the rank-3 family subspace with 0-2pp drop at every
    high-accuracy layer (L24 94%→94%, L32 93%→92%, L36 98%→96%, L40
    93%→93%, L44 89%→89%). Sanity check on the projection: family
    decodability dropped from ~100% to 48-78% (can't fully zero because
    a refit classifier still finds higher-rank residual structure, but
    the rank-3 principal family component is gone).
- Implications:
  - **Conflict detection claim rehabilitated.** Real mechanistic signal
    with depth progression and linear independence from family identity.
    Honest accuracy ~65-79% in stronger holdout conditions, not 92%
    but clearly non-trivial.
  - **Family-identity claim still dead.** Nothing changed on that axis.
  - **Phase 06 redesign scoping is now two-tier:** minimum redesign
    (more lexical variants per family) is enough to strengthen conflict
    work; full redesign (matched vocabularies, more families) is still
    required for any family-level or branching claims.
- Methodology notes carried forward:
  - Always run lexical-holdout tests across the full depth, not just
    a few representative layers -- the progression matters.
  - Family residualization should use a rank-based subspace projection
    on classifier coefficients, not per-family mean-centering (mean
    centering is a null operation for a balanced-class label like
    `conflict_present`).
  - Settings-holdout and strategy-holdout are not interchangeable.
    Report both, and be cautious about conclusions from only one.
- Reference: `outputs/confound_battery/summary.json` (overwritten from
  earlier run). Modal run `ap-Ut2LarT0BBKKfIoGqse0qU` (confound battery
  rerun), `ap-ybGK07zD9TbLDb3ForIIiM` (summary fetch).

### 2026-04-14 -- Follow-up 2: AUROC reveals threshold-miscalibration, not lack of signal

- Context: added AUROC metrics to the confound battery alongside
  balanced accuracy. All probe helpers (`_activation_probe`,
  `_lexical_probe`, `_activation_grouped_cv`) now return both; test 1
  additionally reports OVR-macro AUROC for the 4-class family classifier.
- What we found:
  - **Test 3:** AUROC at deep layers is 0.77-0.94 in every holdout
    condition. Balanced accuracy is 0.50-0.79 in the same conditions.
    The gap is largest in `size / hold out strategy` (bal_acc 0.50 vs
    AUROC 0.94 at L40). Interpretation: when balanced accuracy = 0.5
    and AUROC > 0.9, the probe is predicting everything as one class
    (TPR = 0, TNR = 1 ⇒ bal_acc = 0.5) while its scores still rank
    conflict examples above aligned ones. The direction transfers; the
    threshold does not.
  - **Test 4:** family-residualization moves conflict AUROC by at
    most 0.005 at every high-accuracy layer. Cleaner signal than the
    balanced-accuracy version (which varied by up to 2pp).
  - **Test 1:** lexical family-identity AUROC = 1.000. Same conclusion
    as balanced_accuracy = 1.000, but reported for completeness.
  - One caveat surfaced by AUROC: `size / hold out strategy` has
    AUROC = 0.85 at L0. Under strategy-holdout, settings wording still
    varies with the label (aligned/conflict flip setting values), so
    token-level features can distinguish classes at the embedding
    layer. Settings-holdout is the cleaner test for "conflict
    representation independent of text."
- Implications:
  - **Rehabilitation is stronger than the balanced-accuracy-only
    version suggested.** Conflict detection generalizes across
    lexical variants in ranking at AUROC ≥ 0.77 at depth, with
    family-residualized AUROC essentially unchanged.
  - **Threshold shift is a soft caution.** The probe's output has an
    additive lexical component. We cannot claim pure lexical
    invariance -- only that ranking transfers robustly.
  - **Methodology note:** AUROC and balanced accuracy are both worth
    reporting. AUROC is the more conservative and threshold-independent
    metric for "is there signal." Balanced accuracy is useful for
    deployability questions. Disagreement between the two is itself a
    finding about distributional shift.
- Reference: `outputs/confound_battery/summary.json` (overwritten).
  Modal runs `ap-gYVvF5yAoAR7tWKUOoljCc` (battery with AUROC),
  `ap-CZSu4sARzxDEed7mpS3SLt` (summary fetch).
