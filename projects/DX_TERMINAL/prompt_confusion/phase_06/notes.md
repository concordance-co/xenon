# Phase 06 Notes

Running log of limitations, gotchas, and learnings as Phase 06 progresses.
Keep entries dated. Add things as they come up -- do not wait until the end
of the phase.

---

## Known Limitations Going In

### Carried from Phase 05 (and still load-bearing)

- **AUROC and balanced accuracy disagree under covariate shift.** Phase 05
  lexical-holdout probes had AUROC 0.77-0.94 at depth but bal_acc as low
  as 0.50 in the same condition. The probe's direction transfers; the
  calibrated threshold does not. Always report both metrics on any
  lexical-holdout probe.
- **Probe weight vectors across layers are not comparable.** Cosine
  similarity is only meaningful within a layer.
- **Compact activation files are not a capture invariant.** They carry
  whatever row slice was used at compaction time. Recompact over the
  full row set whenever a new analysis targets a different slice. Phase
  05's `scripts/recompact_router.py` is the pattern; port or reuse for
  Phase 06.
- **MoE logit attribution gap.** Qwen3-30B-A3B attention-based logit
  attribution only captures the attention pathway. MLP/expert
  contributions are invisible. Any logit-attribution claim (deferred in
  Phase 06) must be scoped to "how tokens compete in the attention
  pathway."
- **Section-order position bias.** Phase 04/05 had STRATEGY before
  SETTINGS universally; this contaminated attention-based analyses with
  recency bias. Phase 06 bakes in 50/50 order swap, but this requires
  behavioral generation on both orders (the model may resolve
  differently when section order flips). Do not publish attention-based
  claims without treating order as a cross-checked variable.

### Phase 06-specific starting state

- **We are rebuilding the dataset generator, not augmenting.** The
  current `conflict_probe_examples_v3` table ties family identity to
  vocabulary so tightly that any family-level claim on that data is
  indistinguishable from a lexical baseline. Do not try to "patch" the
  v3 generator. Phase 06 publishes a new `conflict_probe_examples_v4`
  table.
- **Phase 04/05 readout view silently dropped refusal rows.**
  `workflow_dataset_conflict_probe_v3_conflict_readout_side_v1`
  contains 123 of 144 conflict rows; the 21 missing rows are all from
  size families where the model returned `size: "none"`. Drops are
  not random -- 15/36 on `trade_size_force_large`, 6/36 on
  `trade_size_force_small`, 0 on activity families. Phase 06 labels
  refusal in the readout view when it occurs rather than dropping
  those rows. We are not engineering refusal, just no longer hiding
  it.
- **Labels on the survivor slice were correctly derived.** Audited
  2026-04-15 by re-deriving side-followed from `generated_text` vs
  strategy/setting expected actions; 123/123 agreed. Methodology is
  sound, coverage is the issue.
- **Single-axis design is deliberate.** Phase 06 covers size only.
  Second axis returns in Phase 07 as transferability test. Do not add
  activity rows to the Phase 06 dataset.
- **Probe expectations should be calibrated to text-baseline gates, not
  to Phase 04/05 absolute numbers.** Phase 04's 92% conflict detection
  was partially lexical. Phase 05's honest post-holdout number was
  65-79% bal_acc / 0.77-0.94 AUROC at best layers. Phase 06's headline
  number will also be lower than pre-holdout absolutes; that's the
  point.

### Dataset QA gates are a `pipelines_v2` workflow, not an ad-hoc script

The QA gates listed in `design.md` (text baselines, matched-pair
integrity, class support, order balance) should be implemented as a
small `workflow.py` so they can be rerun after every generator change.
Do not let dataset QA slip into informal notebooks -- Phase 04's dataset
ended up with 100% lexical family decodability in part because the text
baseline was an afterthought.

---

## Learnings Log

### YYYY-MM-DD -- template entry

- Context: what were we doing
- What we found: the result or surprise
- Implication: what this changes for subsequent work
- Reference: paths to outputs, run IDs, report sections

### 2026-04-15 -- Final report compiled (md + typ + pdf)

- End-of-phase writeup at
  `projects/DX_TERMINAL/prompt_confusion/phase_06/reports/`:
  `phase_06_v4_report.md`, `phase_06_v4_report.typ`,
  `phase_06_v4_report.pdf`, plus `figures/` with 4 embedded PNGs.
- Figures: (1) depth sweep across all 4 holdout conditions,
  (2) behavioral outcome stacked bar by (family, pair),
  (3) force_small aligned refusal rate by pressure bucket,
  (4) 4x4 strategy-variant x setting-variant follow_setting heatmap
  per family.
- Report framing follows the four-tier claim structure we agreed on:
  confident (non-lexical constructed-feature detection at L28+
  plateau, 0.85 bal_acc / 0.88 AUROC under strict combined holdout);
  defensible-with-framing (resolution dominated by wording format,
  descriptive of v4 not a mechanistic claim); interesting-preliminary
  (77% aligned refusal as cross-section composition on force_small x
  setting_favored); not-yet-claimable (any resolution-direction
  representational claim, any causal claim). Novelty paragraph on
  "interpretability of policy conflict resolution as a function of
  prompt formatting."

### 2026-04-15 -- Variant x variant audit: numeric-scale setting wording has disproportionate behavioral authority

- Context: after classifying all 768 generations by outcome, the
  family-aggregate numbers (force_large conflict 52%/33%
  follow_setting/follow_strategy; force_small 14%/56%) looked clean.
  The audit classifier was spot-checked against a known example and
  verified correct.
- What we found: the aggregates average over 4x4 (strategy_v x
  setting_v) cells with qualitatively different behaviors.
  - `setting_v0` is the only variant written as an explicit numeric
    scale ("Trade size: N/5. Use the [small|large] allocation tier.").
    On force_small conflict it drives follow_setting in 22/24
    non-refuse cases. The other three setting variants (verbal-only,
    e.g. "execution size can use the large tier") drive
    follow_strategy in 86/92 non-refuse cases.
  - `strategy_v3` ("On a clear setup, size up/down rather than scale
    down/up") is systematically soft. On force_large conflict it
    yields 0/46 follow_strategy across all four setting variants;
    the wording is effectively ignored.
  - Despite the system prompt saying "SETTINGS still constrain the
    final execution," STRATEGY wins in most verbal-setting cells.
    The instruction does not reliably dominate.
- Implication: the family-aggregate follow_strategy / follow_setting
  numbers are misleading as "how the model arbitrates." The primary
  signal is wording-format authority (numeric scale vs verbal) rather
  than source identity. Any resolution claim on v4 must be broken out
  at the variant-pair level at minimum. This is descriptive of v4 and
  the specific wordings authored here -- it is not yet a general
  claim about the model.
- Reference:
  `projects/DX_TERMINAL/prompt_confusion/phase_06/reports/figures/fig4_variant_heatmap.png`.

### 2026-04-15 -- Combined strict lexical holdout: detection signal holds

- Context: extended the workflow with a `combined_lexical_split`
  column (`strict_train` if both strategy and setting splits are
  train, `strict_test` if both test, else `mixed`) and added a
  `ProbeSpec` + `TextBaselineSpec` using it. Tests whether single-axis
  holdouts were inflated by the other axis leaking.
- What we found: combined-holdout probe peaked at 0.849 bal_acc /
  0.876 AUROC at L40, vs single-axis strategy-holdout 0.867 / 0.941
  and settings-holdout 0.805 / 0.859. Text baseline at chance (0.50).
- Implication: confounding from the non-held-out axis existed but was
  small (about 2pt drop). The detection signal is not a lexical
  artifact even under strict both-axes holdout.
- Reference: the underlying probe artifact lived on the xenon-data volume; the
  local materialization was archived out of the live repo.

### 2026-04-15 -- v4 dataset design flaws surfaced by behavioral audit

Behavioral audit on the 768-row capture exposed several dataset design
issues that make the v4 generator unsuitable for clean
detection/resolution claims despite the probe numbers looking good.

Issues found:

1. **20% of aligned rows produced refusal behavior.** 78/384 aligned
   rows returned `{"action":"observe","size":"none"}`. Asymmetric by
   family: 10/192 (5%) on force_large, 68/192 (35%) on force_small.
   Concentrated on `setting_favored` market-pressure variant: 49/64
   (77%) aligned force_small rows refused under setting_favored.
   Root cause: conditional STRATEGY wording ("when an entry is
   justified, use the small size tier") combined with hedged MARKET
   language ("usable edge with some confirmation, but one caution
   remains") lets the model read the combination as "not justified,
   refuse" even when STRATEGY and SETTINGS agree.

2. **STRATEGY variants are sizing rules, not strategies.** All four
   variants per direction are size directives with conditional
   preambles, not actual trading strategies (what/when to trade).
   STRATEGY effectively became a second sizing channel, making the
   "conflict" purely about two sources giving disagreeing sizing
   instructions rather than a strategy/execution policy conflict.

3. **SETTINGS wordings duplicate scale semantics.** Variants like
   "Trade size: 1/5. Use the small allocation tier." redundantly
   explain what 1/5 means. The scale semantics (1=small, 3=medium,
   5=large) should live in the system prompt once, leaving SETTINGS
   lean ("Trade size: 1/5").

4. **System prompt pre-resolves the conflict.** Current sys text
   includes "If STRATEGY and SETTINGS disagree, SETTINGS still
   constrain the final execution." This contaminates the resolution
   question -- the model is explicitly told which source to prefer.
   Partially explains the 52% settings-follow rate on force_large
   conflict.

5. **System prompt grants an explicit refuse escape hatch.** "If no
   trade should be made, return {action:observe,...}". This enables
   the conditional-strategy refusal pattern found in issue (1).

6. **Variant v3 is systematically softer.** size_large_v3 + setting_5_v3
   combo on aligned rows produced 8 "other" outputs where the model
   chose small instead of large. My authored variants aren't
   behaviorally equivalent.

7. **Family x behavior asymmetry.** force_large conflict prefers
   settings (52% vs 33%); force_small conflict prefers strategy
   (56% vs 14%); force_small refuses far more than force_large
   across both pair types. Real behavior but heavily dataset-
   specific given issues 1-5.

Implications for analyses on this dataset:

- Conflict-detection probe results are robust (combined-holdout 0.85
  bal_acc / 0.88 AUROC at L40; text baseline at chance) but measure
  the representation of "prompt-level STRATEGY/SETTINGS wording
  disagreement" rather than the intended "conflict state."
- Resolution analyses on this data would be heavily contaminated by
  the sys-prompt pre-resolution, the refuse escape hatch, and the
  pressure x wording interaction.
- A later generator redesign was needed before additional mechanistic work on
  this axis. That follow-on work is no longer kept in this live repo.

Rough shape for v5:

- Unconditional STRATEGY wordings about what/when to trade, not
  sizing rules.
- Lean SETTINGS ("Trade size: N/5") with scale definition in sys
  prompt.
- System prompt neutral on conflict resolution; do not hint at
  which source wins.
- Explicit decision on refusal: remove the escape hatch OR make
  MARKET contexts unambiguously tradeable on aligned rows.

Reference: run `wr_60e5e0058e21_5a0025c8` (initial) and
`wr_cc10418ff064_f8b538db` (with combined-holdout probe). The corresponding
generation artifacts were archived out of the live repo.

### 2026-04-15 -- Reuse hazard: adding a label column invalidates capture reuse

- Adding `combined_lexical_split` to the dataset SQL (a pure
  label-derivation) caused `--reuse-completed` to re-run capture
  instead of reusing the existing artifact. The semantic hash is
  sensitive to the dataset's full column schema, not just the
  prompt inputs.
- Workaround: avoid adding new label columns mid-phase. If a new
  label is needed, either derive it downstream via `TransformSpec`
  or land it pre-capture.

### 2026-04-15 -- Phase 06 first-pass conflict detection: strong, symmetric, non-lexical

- Context: ran the combined `workflow.py` end-to-end
  (run_id `wr_60e5e0058e21_5a0025c8`). Engine config switched to
  `enforce_eager=False`, `max_num_seqs=16`,
  `enable_prefix_caching=True`, residual-only (no MoE router). Capture
  over 768 rows finished in 8m 46s on A100-80GB; total workflow wall
  clock ~10 min.
- Peak bal_acc / AUROC on `conflict_present` probe:
  - Strategy-holdout: **0.87 / 0.94** at L28-L40
  - Settings-holdout: **0.81 / 0.86** at L36-L40
  - Grouped-CV (no holdout): **0.95 / 0.98** at L40
- Text-baseline controls all at chance (<=0.55), so the activation
  signal is not lexical.
- What changed vs Phase 05:
  - Strategy-holdout: ~0.50-0.55 on v3 -> 0.87 peak on v4. The Phase
    05 weakness on strategy-holdout *was* a data-scarcity artifact of
    2 strategy variants; 4+ variants fixes it.
  - Settings-holdout: ~0.65-0.79 on v3 -> 0.81 peak on v4.
  - Grouped-CV: 0.92 on v3 was partially lexical; 0.95 on v4 is real
    activation signal since text baseline is at chance.
  - Threshold-shift gap (bal_acc vs AUROC under holdout): shrank from
    Phase 05's 30+pt gap to ~8pt here. The probe threshold transfers
    much more cleanly under v4's wider lexical coverage.
- Depth profile: clean constructed-feature shape -- L0 at chance,
  L4-L12 rising, L16-L24 strong, L28+ plateau.
- Implication: the narrow redesign (single axis, 4+ lexical variants,
  section-order swap) delivered the expected tightening. The
  conflict-detection claim from Phase 05 is now on solid footing.
  Resolution (side-followed) and threshold-shift characterization are
  the next live threads.
- Reference: Modal capture app `ap-...` (run_id above). The corresponding
  local report and artifact materializations were archived out of the live
  repo during the v1/v2 split.

### 2026-04-15 -- Phase 06 v4 generator: lexical-leakage gate PASSED

- Context: built `conflict_probe_examples_v4` (768 rows, 384 conflict,
  2-family size-only, 4x4 lexical variants, 50/50 section order) and ran
  the pre-capture QA workflow
  (`projects/DX_TERMINAL/prompt_confusion/phase_06/specs/workflow.py`).
- Text baselines on `conflict_present` (all ≤0.60 gate):
  - Strategy-holdout: bal_acc 0.50, AUROC 0.50
  - Settings-holdout: bal_acc 0.50, AUROC 0.50
  - Grouped-CV (no holdout): bal_acc 0.43, AUROC 0.39
- Structural checks: 384 matched pairs of exactly 2 rows each; 96 rows
  per (strategy_split x setting_split x conflict_present) cell; 96 rows
  per (family x conflict x section_order) cell. All above the ≥30
  floor; section order perfectly balanced.
- Implication: the generator does not leak `conflict_present` lexically.
  This is the primary pre-capture QA gate. Capture can proceed.
- Reference: Neon table `conflict_probe_examples_v4`. The old QA artifact
  materializations were archived out of the live repo.

### 2026-04-15 -- Phase 05 readout view drops 21 conflict rows, correlated with family + direction

- Context: label audit before Phase 06 design freeze. Queried Neon to
  verify `workflow_dataset_conflict_probe_v3_conflict_readout_side_v1`
  label correctness.
- What we found: 123/123 labeled rows are correctly derived from
  `generated_text`. But only 123 of 144 conflict rows are labeled --
  21 are missing because the model returned `size: "none"` (refusal).
  Drops are not random: 15/36 on `trade_size_force_large` (42%), 6/36
  on `trade_size_force_small` (17%), 0 on both activity families.
- Implication: Phase 05 resolution analyses implicitly ran on a
  survivor-biased slice where the model was willing to commit.
  Labeling refusal in the Phase 06 readout view (rather than dropping
  those rows) is the motivated fix. We are not engineering refusal --
  picking the size axis over activity is the only deliberate choice
  connected to it, and that's because refusal shows up naturally on
  size, not because we want to maximize it.
- Reference: direct Neon queries against
  `workflow_dataset_conflict_probe_v3_conflict_readout_side_v1` and
  `capture_outputs_conflict_probe_v3` (run `16474bceae4e`). Audit
  logic: `gen_action` / `gen_size` from `generated_text::jsonb` matched
  against `strategy_expected_*` / `setting_expected_*` fields on
  `conflict_probe_examples_v3`.
